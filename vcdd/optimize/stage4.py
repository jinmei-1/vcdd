"""Stage 4: virtual-cell-guided analog design over the stage-3 winners.

The question this stage answers is the one the earlier stages cannot: *how
should these compounds be changed*.  Stages 1-3 rank what already exists; stage
4 proposes edits and predicts what each edit buys.

Pipeline
--------
1. **Parents.** Top ``stage4.top_parents`` compounds by stage-3 objective.
2. **Generation.** Every configured generator proposes analogs. The constrained
   GA is additionally given the cell objective as its fitness function, so it
   searches for the readout the user asked to improve rather than for
   drug-likeness alone.
3. **Guardrails.** Every candidate is checked and the rejects are *kept in the
   output with their reason*, so the funnel is auditable.
4. **Re-docking.** Analogs and their parents are re-docked with one protocol,
   so ``delta_dock_score`` measures the structural change and not a change of
   settings.
5. **Cell re-scoring.** The same stage-3 objective, rebuilt from the same
   config, is applied to analogs and parents together.
6. **Ranking.** Pareto fronts over (docking, cell, QED, SA) plus a weighted
   geometric-mean design score.
7. **SAR mining.** Per-transformation effects aggregated across parents.

Honest limits, restated in ``stage4_summary.json`` so they travel with the
results:

* A read-across cell backend predicts a new analog from its LINCS neighbours.
  Small edits barely move the neighbour set, so ``delta_cell_objective`` is
  compressed and can be exactly zero.  The summary reports how many analogs
  actually moved; if that fraction is low, the cell term is not informative for
  this parent set and the ranking is effectively docking + property driven.
* Docking deltas of less than ~1 kcal/mol are inside the noise of the scoring
  function. They are a triage signal, not a prediction of potency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from vcdd.cell.base import (
    PerturbationRequest,
    Signature,
    get_backend_class,
    resolve_backends,
)
from vcdd.cell.objective import ObjectiveContext, aggregate_backends, score_signatures
from vcdd.docking.base import select_engine
from vcdd.optimize.generators import resolve_generators
from vcdd.optimize.guardrails import check_analog
from vcdd.optimize.pareto import (
    DESIGN_OBJECTIVES,
    crowding_distance,
    design_score,
    pareto_fronts,
)
from vcdd.optimize.sar import mine_sar
from vcdd.receptor.box import Box
from vcdd.runtime.hashing import hash_obj, param_hash
from vcdd.runtime.ledger import RunLedger, UnitKey
from vcdd.runtime.logging import get_logger, log_event
from vcdd.runtime.paths import atomic_write_text, ensure_dir

__all__ = ["run_stage4", "load_parents", "Stage4Result", "STAGE"]

LOG = get_logger()
STAGE = "stage4"

RECOMMENDATION_COLUMNS = [
    "rank", "parent_id", "parent_smiles", "analog_id", "analog_smiles",
    "generator", "transformation", "delta_dock_score", "delta_cell_objective",
    "delta_gene_of_interest", "gene_of_interest", "design_score", "pareto_rank",
    "dock_score", "cell_objective", "qed", "sa_score", "similarity_to_parent",
    "guardrail_flags", "out_of_domain",
]


@dataclass
class Stage4Result:
    out_dir: Path
    analogs: pd.DataFrame
    recommendations: pd.DataFrame
    sar_rules: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Stage4Result analogs={len(self.analogs)} "
            f"recommended={len(self.recommendations)} rules={len(self.sar_rules)}>"
        )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_parents(stage3_dir: Path, top_n: int) -> pd.DataFrame:
    """Top-ranked stage-3 compounds, falling back to stage 2 then stage 1."""
    stage3_dir = Path(stage3_dir)
    candidates = [
        (stage3_dir / "objective_scores.parquet", "stage3/objective_scores"),
        (stage3_dir.parent / "stage2" / "consensus_ranking.parquet", "stage2/consensus_ranking"),
        (stage3_dir.parent / "stage1" / "ranking.parquet", "stage1/ranking"),
    ]
    for path, label in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "objective" in df.columns:
            df = df.sort_values(["objective", "mol_id"], ascending=[False, True], kind="mergesort")
        elif "consensus_rank" in df.columns:
            df = df.sort_values(["consensus_rank", "mol_id"], kind="mergesort")
        elif "score" in df.columns:
            df = df.sort_values(["score", "mol_id"], ascending=[True, True], kind="mergesort")
        out = df.head(top_n).reset_index(drop=True)
        out.attrs["source"] = label
        LOG.info("stage4: %d parents from %s", len(out), label)
        return out
    raise FileNotFoundError(
        f"no stage-3/2/1 ranking found under {stage3_dir}; run the earlier stages first"
    )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Cell scoring helper shared by the GA fitness and the final re-score
# --------------------------------------------------------------------------- #


class _CellScorer:
    """Scores arbitrary SMILES through the stage-3 objective.

    Holds the resolved backends and the objective context open across many
    calls so the GA can use it as an inner-loop fitness without paying model
    setup on every generation.
    """

    def __init__(self, cfg, models: Sequence[Any], ctx: ObjectiveContext) -> None:
        self.cfg = cfg
        self.models = list(models)
        self.ctx = ctx
        self.n_calls = 0
        self.n_scored = 0
        self._cache: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return bool(self.models)

    def score_frame(self, mol_ids: Sequence[str], smiles: Sequence[str]) -> pd.DataFrame:
        """Full per-backend component table for these molecules."""
        cell = self.cfg.cell
        requests = [
            PerturbationRequest(
                mol_id=str(m), smiles=str(s), cell_line=cell.cell_line,
                dose_um=cell.dose_um, time_h=cell.time_h,
            )
            for m, s in zip(mol_ids, smiles)
        ]
        if not requests or not self.models:
            return pd.DataFrame()
        self.n_calls += 1
        self.n_scored += len(requests)
        rows: list[dict] = []
        for model in self.models:
            try:
                sigs: list[Signature] = model.predict(requests)
            except Exception as exc:  # noqa: BLE001 - one backend must not kill stage 4
                LOG.error("stage4: backend %s failed during re-score: %s", model.name, exc)
                continue
            scored, _ = score_signatures(sigs, self.ctx)
            for cs in scored:
                row = cs.to_row()
                detail = cs.gene_detail.get("per_gene", {}) if cs.gene_detail else {}
                for gene, info in detail.items():
                    row[f"gene::{gene}"] = info.get("score")
                rows.append(row)
        return pd.DataFrame(rows)

    def objective_by_mol(self, mol_ids: Sequence[str], smiles: Sequence[str]) -> dict[str, float]:
        frame = self.score_frame(mol_ids, smiles)
        if frame.empty:
            return {}
        agg = frame.groupby("mol_id")["objective"].mean()
        return {str(k): float(v) for k, v in agg.items()}

    def fitness(self, smiles: Sequence[str], parent_smiles: str) -> list[float]:
        """GA fitness: the cell objective, cached across generations."""
        todo = [s for s in smiles if s not in self._cache]
        if todo:
            ids = [f"ga{i:06d}" for i in range(len(todo))]
            got = self.objective_by_mol(ids, todo)
            by_id = {i: s for i, s in zip(ids, todo)}
            for mid, val in got.items():
                self._cache[by_id[mid]] = val
            for s in todo:
                self._cache.setdefault(s, float("nan"))
        return [self._cache.get(s, float("nan")) for s in smiles]

    def close(self) -> None:
        for model in self.models:
            try:
                model.close()
            except Exception:  # noqa: BLE001
                pass


def _build_cell_scorer(cfg) -> tuple[_CellScorer, list[dict], list[str]]:
    """Resolve the backends that can score novel structures."""
    cell = cfg.cell
    models, report = resolve_backends(cell.backends, cell)
    usable, notes = [], []
    for model in models:
        caps = get_backend_class(model.name).capabilities()
        if caps.reference_only:
            notes.append(f"{model.name}: reference-only, excluded from analog scoring")
            model.close()
            continue
        if not caps.accepts_novel_smiles:
            notes.append(f"{model.name}: cannot score novel structures, excluded")
            model.close()
            continue
        usable.append(model)
    if not usable:
        return _CellScorer(cfg, [], None), report, notes  # type: ignore[arg-type]

    genes = usable[0].genes()
    ctx = ObjectiveContext.build(
        cell.objective, genes, target_gene=cell.target_gene,
        activity_method=cell.activity_method,
    )
    notes.extend(ctx.warnings)
    return _CellScorer(cfg, usable, ctx), report, notes


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_stage4(
    cfg,
    receptor_pdbqt: str | Path | None,
    box: Box | None,
    stage3_dir: Path,
    out_dir: Path,
    ledger: RunLedger,
    force: bool = False,
) -> Stage4Result:
    """Generate, score, and rank analogs of the stage-3 winners."""
    t0 = time.time()
    out_dir = ensure_dir(Path(out_dir))
    s4 = cfg.stage4
    guard = s4.guardrails

    parents = load_parents(Path(stage3_dir), s4.top_parents)
    if parents.empty:
        raise ValueError("stage-4 parent set is empty; run stage 3 first")
    parents = parents.rename(columns={"mol_id": "parent_id", "smiles": "parent_smiles"})
    parent_smiles = dict(zip(parents["parent_id"], parents["parent_smiles"]))

    p_hash = param_hash(
        {
            "stage4": s4.model_dump(mode="json"),
            "cell": cfg.cell.model_dump(mode="json"),
            "n_parents": len(parents),
        }
    )
    input_hash = hash_obj(sorted(f"{k}|{v}" for k, v in parent_smiles.items()))

    # ---- cell scorer (also the GA fitness) --------------------------------
    scorer, backend_report, backend_notes = _build_cell_scorer(cfg)
    if scorer.available:
        LOG.info("stage4: cell scoring via %s", [m.name for m in scorer.models])
    else:
        LOG.warning(
            "stage4: no cell backend can score novel structures; the GA falls back "
            "to a drug-likeness fitness and delta_cell_objective will be NaN"
        )
    for note in backend_notes:
        LOG.info("stage4: %s", note)

    # ---- generation --------------------------------------------------------
    gens, gen_report = resolve_generators(s4.generators, s4, seed=s4.seed)
    if not gens:
        scorer.close()
        raise RuntimeError(
            f"no usable analog generator among {s4.generators}: "
            + "; ".join(f"{r['generator']}: {r['reason']}" for r in gen_report)
        )
    for g in gens:
        if hasattr(g, "set_fitness") and scorer.available:
            g.set_fitness(scorer.fitness)

    raw_path = out_dir / "analogs_raw.parquet"
    gen_unit = UnitKey(STAGE, "generate", input_hash, p_hash)
    if not force and ledger.is_done(gen_unit) and raw_path.exists():
        LOG.info("stage4: reusing cached analog proposals")
        raw = pd.read_parquet(raw_path)
    else:
        ledger.mark_running(gen_unit)
        raw = _generate(parents, gens, s4, guard)
        _write_parquet(raw, raw_path)
        ledger.mark_done(gen_unit, artifact=raw_path, n_items=len(raw))
        ledger.sync()

    n_pass = int(raw["guardrail_pass"].sum()) if not raw.empty else 0
    LOG.info(
        "stage4: %d proposals, %d passed guardrails (%d parents, %d generators)",
        len(raw), n_pass, len(parents), len(gens),
    )
    kept = raw[raw["guardrail_pass"]].copy() if not raw.empty else raw.copy()
    if kept.empty:
        scorer.close()
        raise RuntimeError(
            "every proposed analog failed the guardrails; loosen "
            "stage4.guardrails (similarity floors are the usual cause)"
        )
    kept["analog_id"] = [f"AN{i:06d}" for i in range(1, len(kept) + 1)]

    # ---- re-docking --------------------------------------------------------
    dock_path = out_dir / "redock.parquet"
    dock_unit = UnitKey(STAGE, "redock", hash_obj(sorted(kept["analog_smiles"])), p_hash)
    dock_df = pd.DataFrame()
    if s4.redock_mode == "none" or receptor_pdbqt is None or box is None:
        LOG.info("stage4: re-docking disabled (redock_mode=%s)", s4.redock_mode)
    elif not force and ledger.is_done(dock_unit) and dock_path.exists():
        LOG.info("stage4: reusing cached re-docking results")
        dock_df = pd.read_parquet(dock_path)
    else:
        ledger.mark_running(dock_unit)
        dock_df = _redock(cfg, kept, parent_smiles, receptor_pdbqt, box, out_dir)
        _write_parquet(dock_df.drop(columns=["pose_pdbqt"], errors="ignore"), dock_path)
        ledger.mark_done(dock_unit, artifact=dock_path, n_items=len(dock_df))
        ledger.sync()

    # ---- cell re-scoring ---------------------------------------------------
    cell_path = out_dir / "cell_scores.parquet"
    cell_unit = UnitKey(STAGE, "cell", hash_obj(sorted(kept["analog_smiles"])), p_hash)
    if scorer.available:
        if not force and ledger.is_done(cell_unit) and cell_path.exists():
            LOG.info("stage4: reusing cached cell re-scoring")
            cell_df = pd.read_parquet(cell_path)
        else:
            ledger.mark_running(cell_unit)
            cell_df = _rescore_cell(scorer, kept, parent_smiles)
            _write_parquet(cell_df, cell_path)
            ledger.mark_done(cell_unit, artifact=cell_path, n_items=len(cell_df))
            ledger.sync()
    else:
        cell_df = pd.DataFrame(columns=["mol_id", "cell_objective"])
    scorer.close()

    # ---- assemble ----------------------------------------------------------
    analogs, parent_baseline = _assemble(kept, parents, dock_df, cell_df, cfg)

    # ---- rank --------------------------------------------------------------
    objectives = {k: v for k, v in DESIGN_OBJECTIVES.items() if k in analogs.columns}
    analogs["design_score"], terms = design_score(analogs, s4.score_weights)
    for col in terms.columns:
        analogs[f"term_{col}"] = terms[col].to_numpy()
    analogs["pareto_rank"] = pareto_fronts(analogs, objectives).to_numpy()
    analogs["crowding_distance"] = crowding_distance(
        analogs, objectives, analogs["pareto_rank"]
    ).to_numpy()
    analogs = analogs.sort_values(
        ["pareto_rank", "design_score", "analog_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    analogs["rank"] = np.arange(1, len(analogs) + 1)

    # a parent's own design score, for "is this analog actually better?"
    analogs["delta_design_score"] = analogs["design_score"] - analogs["parent_id"].map(
        parent_baseline.set_index("parent_id")["parent_design_score"]
    )

    # ---- SAR ---------------------------------------------------------------
    sar = mine_sar(
        analogs,
        metrics=(
            "delta_cell_objective", "delta_dock_score",
            "delta_design_score", "delta_gene_of_interest",
        ),
        min_parents=s4.sar_min_parents,
    )

    # ---- outputs -----------------------------------------------------------
    _write_parquet(
        analogs.drop(columns=["transformations"], errors="ignore"),
        out_dir / "analogs.parquet",
    )
    _write_parquet(
        analogs[["analog_id", "parent_id", "analog_smiles", "pareto_rank",
                 "crowding_distance", "design_score",
                 *[c for c in objectives if c in analogs.columns]]],
        out_dir / "pareto.parquet",
    )
    recs = _recommendations(analogs)
    recs.to_csv(out_dir / "analog_recommendations.csv", index=False)
    sar.to_csv(out_dir / "sar_rules.csv", index=False)
    n_sdf = _write_sdf(analogs, dock_df, out_dir / "top_designs.sdf", top_n=min(100, len(analogs)))

    summary = _summarise(
        cfg, parents, raw, analogs, sar, gen_report, backend_report, backend_notes,
        parent_baseline, n_sdf, time.time() - t0,
    )
    atomic_write_text(out_dir / "stage4_summary.json", json.dumps(summary, indent=2, default=str))
    atomic_write_text(
        out_dir / "generator_report.json",
        json.dumps({"generators": gen_report, "cell_backends": backend_report,
                    "notes": backend_notes}, indent=2, default=str),
    )
    log_event(
        "stage4_complete", n_analogs=len(analogs), n_rules=len(sar),
        seconds=summary["seconds"],
    )
    return Stage4Result(out_dir, analogs, recs, sar, summary)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _generate(parents: pd.DataFrame, gens, s4, guard) -> pd.DataFrame:
    """Run every generator over every parent and guardrail the proposals."""
    from rdkit import Chem

    per_gen = max(1, int(np.ceil(s4.analogs_per_parent / len(gens))))
    rows: list[dict] = []
    for prow in parents.itertuples(index=False):
        pid, psmi = str(prow.parent_id), str(prow.parent_smiles)
        pmol = Chem.MolFromSmiles(psmi)
        if pmol is None:
            LOG.warning("stage4: parent %s has unparsable SMILES, skipped", pid)
            continue
        seen: set[str] = set()
        for gen in gens:
            floor = (
                guard.similarity_floor_ga
                if gen.name() == "constrained_ga"
                else guard.similarity_floor_enum
            )
            t_gen = time.time()
            try:
                proposals = gen.generate(pid, psmi, per_gen)
            except Exception as exc:  # noqa: BLE001 - a generator must not kill the stage
                LOG.error("stage4: generator %s failed on %s: %s", gen.name(), pid, exc)
                continue
            for prop in proposals:
                rep = check_analog(prop.smiles, pmol, guard, similarity_floor=floor)
                if rep.smiles in seen:
                    continue
                seen.add(rep.smiles)
                row = rep.to_row()
                row.update(
                    parent_id=pid,
                    parent_smiles=psmi,
                    generator=prop.generator,
                    transformation=prop.transformation,
                    transformations=list(prop.meta.get("transformations") or [prop.transformation]),
                    similarity_floor=floor,
                )
                rows.append(row)
            LOG.debug(
                "stage4: %s proposed %d for %s in %.1fs",
                gen.name(), len(proposals), pid, time.time() - t_gen,
            )
    return pd.DataFrame(rows)


def _redock(cfg, kept: pd.DataFrame, parent_smiles: dict, receptor_pdbqt, box, out_dir: Path):
    """Dock analogs and parents together under one protocol.

    There is only one docking protocol in 0.2.0: stage 1's. Stage 2 no longer
    docks, so ``redock_mode`` is ``stage1`` or ``none``; the config validator
    rejects ``stage2`` with a pointer rather than letting it silently mean
    something else.

    Re-docking the parents alongside the analogs is not redundant. Vina-GPU 2.1
    exposes no seed, so its search is not bit-reproducible; a delta computed
    against a parent score stored from an earlier run would mix the structural
    change with run-to-run search noise.
    """
    from vcdd.screening.stage1 import build_dock_params, dock_molecules

    s4 = cfg.stage4
    engine = select_engine(cfg.stage1.engine, cfg.stage1.engine_selection, prefer_fast=True)
    params = build_dock_params(cfg.stage1)
    ids = list(kept["analog_id"]) + [f"PARENT::{p}" for p in parent_smiles]
    smis = list(kept["analog_smiles"]) + list(parent_smiles.values())
    LOG.info(
        "stage4: re-docking %d analogs + %d parents (%s protocol, variant=%s thread=%d)",
        len(kept), len(parent_smiles), s4.redock_mode, params["variant"], params["thread"],
    )
    df = dock_molecules(
        ids, smis, receptor_pdbqt, box, engine.name, params,
        n_conformers=getattr(cfg.stage1, "n_conformers", 1),
        minimize=getattr(cfg.stage1, "minimize_conformer", True),
        seed=cfg.stage1.seed,
        n_workers=cfg.n_workers,
        keep_poses=True,
        batch_size=getattr(cfg.stage1, "gpu_batch_size", None),
    )
    n_ok = int(df["score"].notna().sum())
    LOG.info("stage4: re-docked %d/%d successfully", n_ok, len(df))
    return df


def _rescore_cell(scorer: _CellScorer, kept: pd.DataFrame, parent_smiles: dict) -> pd.DataFrame:
    """Score analogs and parents through the stage-3 objective in one batch."""
    ids = list(kept["analog_id"]) + [f"PARENT::{p}" for p in parent_smiles]
    smis = list(kept["analog_smiles"]) + list(parent_smiles.values())
    frame = scorer.score_frame(ids, smis)
    if frame.empty:
        return pd.DataFrame(columns=["mol_id", "cell_objective"])
    gene_cols = [c for c in frame.columns if c.startswith("gene::")]
    agg = frame.groupby("mol_id", as_index=False).agg(
        cell_objective=("objective", "mean"),
        cell_gene_score=("gene_score", "mean"),
        cell_pathway_score=("pathway_score", "mean"),
        cell_target_consistency=("target_consistency", "mean"),
        cell_specificity=("specificity", "mean"),
        max_tanimoto=("max_tanimoto", "max"),
        out_of_domain=("out_of_domain", "any"),
        n_backends=("backend", "nunique"),
        **{c: (c, "mean") for c in gene_cols},
    )
    return agg


def _assemble(kept, parents, dock_df, cell_df, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join docking + cell results onto analogs and compute deltas vs parent."""
    analogs = kept.copy()

    dock_map: dict[str, float] = {}
    if not dock_df.empty:
        dock_map = {
            str(r.mol_id): (float(r.score) if pd.notna(r.score) else np.nan)
            for r in dock_df.itertuples(index=False)
        }
    analogs["dock_score"] = analogs["analog_id"].map(dock_map)

    cell_map: dict[str, dict] = {}
    gene_cols: list[str] = []
    if not cell_df.empty:
        gene_cols = [c for c in cell_df.columns if c.startswith("gene::")]
        cell_map = cell_df.set_index("mol_id").to_dict("index")
    for col in ["cell_objective", "cell_gene_score", "cell_pathway_score",
                "cell_target_consistency", "cell_specificity", "max_tanimoto",
                "out_of_domain", *gene_cols]:
        analogs[col] = [cell_map.get(a, {}).get(col, np.nan) for a in analogs["analog_id"]]

    # ---- parent baselines under the identical protocol ----
    base_rows = []
    for pid, psmi in zip(parents["parent_id"], parents["parent_smiles"]):
        key = f"PARENT::{pid}"
        rec = cell_map.get(key, {})
        row = {
            "parent_id": pid,
            "parent_smiles": psmi,
            "parent_dock_score": dock_map.get(key, np.nan),
            "parent_cell_objective": rec.get("cell_objective", np.nan),
        }
        for gc in gene_cols:
            row[f"parent_{gc}"] = rec.get(gc, np.nan)
        base_rows.append(row)
    baseline = pd.DataFrame(base_rows)

    # parent QED / SA on the same footing as the analogs
    from rdkit import Chem
    from rdkit.Chem import QED

    from vcdd.optimize.guardrails import sa_score

    qeds, sas = [], []
    for smi in baseline["parent_smiles"]:
        m = Chem.MolFromSmiles(str(smi))
        qeds.append(float(QED.qed(m)) if m is not None else np.nan)
        sas.append(sa_score(m) if m is not None else np.nan)
    baseline["parent_qed"], baseline["parent_sa_score"] = qeds, sas

    pd_score, _ = design_score(
        baseline.rename(
            columns={
                "parent_dock_score": "dock_score",
                "parent_cell_objective": "cell_objective",
                "parent_qed": "qed",
                "parent_sa_score": "sa_score",
            }
        ),
        cfg.stage4.score_weights,
    )
    baseline["parent_design_score"] = pd_score.to_numpy()

    idx = baseline.set_index("parent_id")
    analogs["parent_dock_score"] = analogs["parent_id"].map(idx["parent_dock_score"])
    analogs["parent_cell_objective"] = analogs["parent_id"].map(idx["parent_cell_objective"])
    analogs["delta_dock_score"] = analogs["dock_score"] - analogs["parent_dock_score"]
    analogs["delta_cell_objective"] = (
        analogs["cell_objective"] - analogs["parent_cell_objective"]
    )
    analogs["delta_qed"] = analogs["qed"] - analogs["parent_id"].map(idx["parent_qed"])
    analogs["delta_sa_score"] = analogs["sa_score"] - analogs["parent_id"].map(idx["parent_sa_score"])

    # ---- the single "gene of interest" the user is steering ----
    goi = _gene_of_interest(cfg, gene_cols)
    analogs["gene_of_interest"] = goi or ""
    if goi and f"gene::{goi}" in analogs.columns:
        analogs["delta_gene_of_interest"] = analogs[f"gene::{goi}"] - analogs["parent_id"].map(
            idx[f"parent_gene::{goi}"]
        )
    else:
        analogs["delta_gene_of_interest"] = np.nan
    return analogs, baseline


def _gene_of_interest(cfg, gene_cols: Sequence[str]) -> str | None:
    """The primary steered gene: first requested up-gene that was actually scored."""
    scored = {c.split("::", 1)[1] for c in gene_cols}
    for name in list(cfg.cell.objective.genes_up) + list(cfg.cell.objective.genes_down):
        if name in scored:
            return name
    return next(iter(sorted(scored)), None)


def _recommendations(analogs: pd.DataFrame) -> pd.DataFrame:
    out = analogs.copy()
    out["guardrail_flags"] = out.get("guardrail_reasons", "")
    cols = [c for c in RECOMMENDATION_COLUMNS if c in out.columns]
    recs = out[cols].copy()
    for c in recs.columns:
        if pd.api.types.is_float_dtype(recs[c]):
            recs[c] = recs[c].round(4)
    return recs


def _write_sdf(analogs: pd.DataFrame, dock_df: pd.DataFrame, path: Path, top_n: int) -> int:
    """Write the top designs as 3D SDF, preferring the docked pose."""
    from rdkit import Chem

    from vcdd.docking.ligprep import embed_conformers, pdbqt_to_mol

    poses: dict[str, str] = {}
    if not dock_df.empty and "pose_pdbqt" in dock_df.columns:
        poses = {
            str(r.mol_id): r.pose_pdbqt
            for r in dock_df.itertuples(index=False)
            if isinstance(getattr(r, "pose_pdbqt", None), str)
        }

    ensure_dir(path.parent)
    writer = Chem.SDWriter(str(path))
    n = 0
    try:
        for row in analogs.head(top_n).itertuples(index=False):
            smi = str(row.analog_smiles)
            mol = None
            pose = poses.get(str(row.analog_id))
            if pose:
                mol = pdbqt_to_mol(pose, template_smiles=smi)
            if mol is None:
                # embed_conformers returns a single Mol carrying N conformers,
                # not a list of Mols.
                prepared = embed_conformers(smi, n_conformers=1, minimize=True, seed=42)
                mol = (
                    prepared
                    if prepared is not None and prepared.GetNumConformers() > 0
                    else None
                )
            if mol is None:
                continue
            mol = Chem.Mol(mol)
            mol.SetProp("_Name", str(row.analog_id))
            for field_name in (
                "parent_id", "generator", "transformation", "dock_score",
                "delta_dock_score", "cell_objective", "delta_cell_objective",
                "design_score", "pareto_rank", "qed", "sa_score",
                "similarity_to_parent", "pose_source",
            ):
                if field_name == "pose_source":
                    mol.SetProp("pose_source", "docked" if pose else "rdkit_embed")
                    continue
                val = getattr(row, field_name, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    mol.SetProp(field_name, str(val))
            writer.write(mol)
            n += 1
    finally:
        writer.close()
    return n


def _summarise(
    cfg, parents, raw, analogs, sar, gen_report, backend_report, backend_notes,
    baseline, n_sdf, seconds,
) -> dict:
    reasons: dict[str, int] = {}
    if not raw.empty:
        for txt in raw.loc[~raw["guardrail_pass"], "guardrail_reasons"]:
            for r in str(txt).split(";"):
                if r:
                    reasons[r] = reasons.get(r, 0) + 1

    d_cell = pd.to_numeric(analogs.get("delta_cell_objective"), errors="coerce").dropna()
    d_dock = pd.to_numeric(analogs.get("delta_dock_score"), errors="coerce").dropna()
    moved = float((d_cell.abs() > 1e-6).mean()) if d_cell.size else 0.0

    warnings: list[str] = list(backend_notes)
    if d_cell.size and moved < 0.5:
        warnings.append(
            f"only {moved:.0%} of analogs changed the cell objective at all; with a "
            "read-across backend, small edits map to the same LINCS neighbours, so "
            "the ranking here is effectively docking + property driven"
        )
    if d_dock.size and float(d_dock.abs().median()) < 1.0:
        warnings.append(
            f"median |delta_dock_score| is {float(d_dock.abs().median()):.2f} kcal/mol, "
            "inside docking noise; treat docking deltas as triage only"
        )

    def _stats(s: pd.Series) -> dict:
        if not s.size:
            return {"n": 0}
        return {
            "n": int(s.size), "min": float(s.min()), "median": float(s.median()),
            "max": float(s.max()), "n_improved": int((s > 0).sum()),
        }

    return {
        "n_parents": int(len(parents)),
        "parent_source": parents.attrs.get("source"),
        "n_proposed": int(len(raw)),
        "n_guardrail_pass": int(len(analogs)),
        "guardrail_rejection_reasons": dict(
            sorted(reasons.items(), key=lambda kv: -kv[1])
        ),
        "generators": gen_report,
        "generators_used": sorted(analogs["generator"].unique()) if not analogs.empty else [],
        "analogs_per_generator": (
            analogs["generator"].value_counts().to_dict() if not analogs.empty else {}
        ),
        "cell_backends": [r for r in backend_report if r.get("available")],
        "redock_mode": cfg.stage4.redock_mode,
        "score_weights": dict(cfg.stage4.score_weights),
        "gene_of_interest": (
            analogs["gene_of_interest"].iloc[0] if not analogs.empty else None
        ),
        "delta_cell_objective": _stats(d_cell),
        "delta_dock_score": {
            **_stats(d_dock),
            "n_improved": int((d_dock < 0).sum()) if d_dock.size else 0,
        },
        "frac_analogs_with_cell_change": round(moved, 4),
        "n_pareto_front1": int((analogs["pareto_rank"] == 1).sum()) if not analogs.empty else 0,
        "n_sar_rules": int(len(sar)),
        "n_sdf_written": int(n_sdf),
        "parent_baseline": baseline.round(4).to_dict("records"),
        "warnings": warnings,
        "seconds": round(seconds, 2),
    }
