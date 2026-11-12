"""Stage 3: virtual-cell perturbation scoring of the stage-2 shortlist.

Stage 2 says which compounds *bind*.  Stage 3 asks the different question the
user actually cares about -- what does the compound do to the cell -- by
running one or more compound-perturbation models over the shortlist and scoring
the predicted transcriptional response against a multi-objective definition.

Outputs (under ``<run>/stage3``):

``signatures/<backend>.parquet``
    Long format ``mol_id, backend, gene, value``; the raw predictions.
``activity/<backend>_tf.parquet`` / ``activity/<backend>_pathway.parquet``
    Inferred TF and pathway activity, when a network is available.
``objective_scores.parquet``
    One row per compound: components, aggregate, rank, confidence flags.
``objective_by_backend.parquet``
    The same before backends are averaged, for auditing disagreement.
``backend_report.json``
    Which backends ran, which skipped and why, with install hints.
``stage3_summary.json``
    Counts, timings, objective distribution and the context warnings.

Resume: one ledger unit per backend.  Re-running skips a backend whose
signatures are already on disk with an unchanged input and parameter hash, then
always recomputes the (cheap) objective so a changed objective definition takes
effect without re-predicting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config.schema import RunConfig
from ..runtime.hashing import hash_obj, param_hash
from ..runtime.ledger import RunLedger, UnitKey
from ..runtime.logging import get_logger, log_event
from ..runtime.paths import atomic_write_text, ensure_dir
from . import assets
from .base import (
    PerturbationRequest,
    get_backend_class,
    Signature,
    resolve_backends,
    signature_meta_frame,
    signatures_to_frame,
)
from .objective import ObjectiveContext, aggregate_backends, score_signatures

log = get_logger()

__all__ = ["run_stage3", "load_shortlist", "Stage3Result"]

STAGE = "stage3"


def load_shortlist(stage2_dir: Path, top_n: int) -> pd.DataFrame:
    """Read the stage-2 consensus ranking, falling back to stage 1.

    Stage 3 is expensive per compound relative to how many compounds are worth
    asking about, so it deliberately consumes a shortlist rather than a library.
    """
    consensus = stage2_dir / "consensus_ranking.parquet"
    if consensus.exists():
        df = pd.read_parquet(consensus)
        rank_col = "consensus_rank" if "consensus_rank" in df.columns else None
        if rank_col:
            df = df.sort_values(rank_col)
        source = "stage2_consensus"
    else:
        alt = stage2_dir / "scores.parquet"
        if alt.exists():
            df = pd.read_parquet(alt).sort_values("dock_score")
            source = "stage2_scores"
        else:
            s1 = stage2_dir.parent / "stage1" / "ranking.parquet"
            if not s1.exists():
                raise FileNotFoundError(
                    f"no stage-2 ranking at {consensus} and no stage-1 ranking at {s1}"
                )
            df = pd.read_parquet(s1).sort_values(["score", "mol_id"], kind="mergesort")
            source = "stage1_ranking"
    df = df[df["smiles"].notna()].head(max(int(top_n), 0)).reset_index(drop=True)
    df.attrs["source"] = source
    return df


class Stage3Result:
    """Summary handle returned by :func:`run_stage3`."""

    def __init__(self, out_dir: Path, scores: pd.DataFrame, report: list[dict], summary: dict):
        self.out_dir = out_dir
        self.scores = scores
        self.backend_report = report
        self.summary = summary

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Stage3Result n={len(self.scores)} "
            f"backends={self.summary.get('backends_used')}>"
        )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def _activity_frames(
    signatures: Sequence[Signature], cfg, out_dir: Path, backend: str
) -> dict:
    """Compute and persist TF / pathway activity for a backend's signatures."""
    from . import protein_activity

    usable = [s for s in signatures if s.ok]
    info: dict = {}
    if not usable or not cfg.protein_activity:
        return info
    genes = usable[0].genes
    if not all(s.genes == genes for s in usable):
        return {"skipped": "inconsistent gene spaces"}
    mat = np.column_stack([s.values for s in usable])
    ids = [s.mol_id for s in usable]
    act_dir = out_dir / "activity"
    for label, fn in (
        ("tf", protein_activity.tf_activity),
        ("pathway", protein_activity.pathway_activity),
    ):
        try:
            res = fn(mat, genes, ids, method=cfg.activity_method)
        except assets.ResourceMissing as exc:
            info[f"{label}_error"] = str(exc)
            continue
        if res.n_regulators == 0:
            info[f"{label}_error"] = "no regulator had enough measured targets"
            continue
        _write_parquet(res.to_long(), act_dir / f"{backend}_{label}.parquet")
        info[label] = {
            "n_regulators": res.n_regulators,
            "n_genes_used": res.n_genes_used,
            "coverage": round(res.coverage, 3),
            "method": res.method,
            "dropped_small_regulons": res.dropped_small,
        }
    return info


def run_stage3(
    cfg: RunConfig,
    stage2_dir: Path,
    out_dir: Path,
    ledger: RunLedger,
    force: bool = False,
) -> Stage3Result:
    """Predict cellular responses for the shortlist and score the objective."""
    t0 = time.time()
    out_dir = ensure_dir(Path(out_dir))
    cell_cfg = cfg.cell

    shortlist = load_shortlist(Path(stage2_dir), cell_cfg.top_n)
    if shortlist.empty:
        raise ValueError("stage-3 shortlist is empty; run stage 2 first")
    log.info(
        "stage3: %d compounds from %s", len(shortlist), shortlist.attrs.get("source")
    )

    requests = [
        PerturbationRequest(
            mol_id=str(row.mol_id),
            smiles=str(row.smiles),
            cell_line=cell_cfg.cell_line,
            dose_um=cell_cfg.dose_um,
            time_h=cell_cfg.time_h,
        )
        for row in shortlist.itertuples()
    ]
    input_hash = hash_obj([r.key() + "|" + r.smiles for r in requests])
    p_hash = param_hash(
        {
            "cell": cell_cfg.model_dump(mode="json", exclude={"objective"}),
            "n_requests": len(requests),
        }
    )

    models, report = resolve_backends(cell_cfg.backends, cell_cfg)
    if not models:
        hints = "; ".join(
            f"{r['backend']}: {r.get('reason', '')}" for r in report if not r["available"]
        )
        raise RuntimeError(
            f"no usable cell backend among {cell_cfg.backends}. {hints}. "
            "Use `backends: [mock]` to exercise the pipeline without a real model."
        )

    # ------------------------------------------------------------ predict
    sig_dir = ensure_dir(out_dir / "signatures")
    per_backend: dict[str, list[Signature]] = {}
    activity_info: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for model in models:
        name = model.name
        unit = UnitKey(STAGE, f"backend:{name}", input_hash, p_hash)
        sig_path = sig_dir / f"{name}.parquet"
        meta_path = _meta_path(sig_path)
        if not force and ledger.is_done(unit) and sig_path.exists() and meta_path.exists():
            log.info("stage3: reusing cached signatures for %s", name)
            per_backend[name] = _signatures_from_frame(
                pd.read_parquet(sig_path), pd.read_parquet(meta_path), name
            )
            timings[name] = 0.0
            continue

        ledger.mark_running(unit)
        t_backend = time.time()
        try:
            sigs = model.predict(requests)
            if len(sigs) != len(requests):
                raise RuntimeError(
                    f"{name} returned {len(sigs)} signatures for {len(requests)} requests"
                )
            frame = signatures_to_frame(sigs)
            _write_parquet(frame, sig_path)
            _write_parquet(signature_meta_frame(sigs), _meta_path(sig_path))
            activity_info[name] = _activity_frames(sigs, cell_cfg, out_dir, name)
            per_backend[name] = sigs
            timings[name] = time.time() - t_backend
            n_ok = sum(1 for s in sigs if s.ok)
            ledger.mark_done(unit, artifact=sig_path, n_items=n_ok)
            log.info(
                "stage3: %s produced %d/%d signatures in %.1fs",
                name, n_ok, len(sigs), timings[name],
            )
        except Exception as exc:  # noqa: BLE001 - one backend must not kill stage 3
            ledger.mark_failed(unit, f"{type(exc).__name__}: {exc}")
            log.error("stage3: backend %s failed: %s", name, exc)
            for rec in report:
                if rec["backend"] == name:
                    rec["available"] = False
                    rec["reason"] = f"failed at predict: {exc}"
        finally:
            model.close()

    if not per_backend:
        raise RuntimeError("every cell backend failed; see backend_report.json")

    # ----------------------------------------------------------- objective
    gene_space = next(iter(per_backend.values()))[0].genes
    ctx = ObjectiveContext.build(
        cell_cfg.objective,
        gene_space,
        target_gene=cell_cfg.target_gene,
        activity_method=cell_cfg.activity_method,
    )
    for w in ctx.warnings:
        log.warning("stage3 objective: %s", w)

    # Reference-only backends (e.g. target_knockdown) return the same signature
    # for every compound. Averaging them in would add a constant to the ranking
    # and score target_consistency against itself, so they are materialised for
    # provenance but excluded here.
    reference_only = {
        name
        for name in per_backend
        if get_backend_class(name).capabilities().reference_only
    }
    scoring_backends = [n for n in per_backend if n not in reference_only]
    if not scoring_backends:
        raise RuntimeError(
            "every usable backend is reference-only "
            f"({sorted(reference_only)}); add a compound-responsive backend such "
            "as lincs_readacross or mock"
        )

    rows: list[dict] = []
    obj_diagnostics: dict[str, dict] = {}
    for name in scoring_backends:
        sigs = per_backend[name]
        scored, diag = score_signatures(sigs, ctx)
        obj_diagnostics[name] = diag
        for cs in scored:
            row = cs.to_row()
            row["gene_detail"] = json.dumps(cs.gene_detail, default=float)
            row["pathway_detail"] = json.dumps(cs.pathway_detail, default=float)
            rows.append(row)

    by_backend = pd.DataFrame(rows)
    _write_parquet(by_backend, out_dir / "objective_by_backend.parquet")

    numeric_cols = [
        "gene_score", "pathway_score", "target_consistency", "specificity",
        "objective", "max_tanimoto", "frac_genes_moved", "stress_score",
    ]
    slim = by_backend[["mol_id", "backend", *numeric_cols, "out_of_domain"]]
    combined = aggregate_backends(slim)

    meta_cols = ["mol_id", "smiles"]
    for extra in ("dock_score", "consensus_score", "consensus_rank", "stage1_score", "score"):
        if extra in shortlist.columns and extra not in meta_cols:
            meta_cols.append(extra)
    scores = shortlist[meta_cols].merge(combined, on="mol_id", how="left")
    scores = scores.sort_values(
        ["objective", "mol_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scores["objective_rank"] = np.arange(1, len(scores) + 1)
    _write_parquet(scores, out_dir / "objective_scores.parquet")

    # ------------------------------------------------------------- report
    atomic_write_text(
        out_dir / "backend_report.json",
        json.dumps(
            {
                "requested": list(cell_cfg.backends),
                "used": sorted(per_backend),
                "records": report,
                "timings_s": {k: round(v, 2) for k, v in timings.items()},
                "activity": activity_info,
            },
            indent=2,
            default=str,
        ),
    )

    finite = scores["objective"].dropna()
    summary = {
        "n_compounds": int(len(scores)),
        "shortlist_source": shortlist.attrs.get("source"),
        "backends_requested": list(cell_cfg.backends),
        "backends_used": sorted(per_backend),
        "backends_scored": sorted(scoring_backends),
        "backends_reference_only": sorted(reference_only),
        "cell_line": cell_cfg.cell_line,
        "dose_um": cell_cfg.dose_um,
        "time_h": cell_cfg.time_h,
        "gene_space": cell_cfg.gene_space,
        "n_genes": len(gene_space),
        "objective_weights": dict(cell_cfg.objective.weights),
        "genes_up": list(cell_cfg.objective.genes_up),
        "genes_down": list(cell_cfg.objective.genes_down),
        "tf_routed_requests": sorted(ctx.tf_requests),
        "objective_stats": {
            "n_scored": int(finite.size),
            "min": float(finite.min()) if finite.size else None,
            "median": float(finite.median()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        },
        "n_out_of_domain": int(scores["out_of_domain"].fillna(False).sum())
        if "out_of_domain" in scores
        else 0,
        "warnings": ctx.warnings,
        "objective_diagnostics": obj_diagnostics,
        "seconds": round(time.time() - t0, 2),
    }
    atomic_write_text(
        out_dir / "stage3_summary.json", json.dumps(summary, indent=2, default=str)
    )
    log_event(
        "stage3_complete",
        n=len(scores),
        backends=sorted(per_backend),
        seconds=summary["seconds"],
    )
    return Stage3Result(out_dir, scores, report, summary)


def _meta_path(sig_path: Path) -> Path:
    return sig_path.with_name(sig_path.stem + "_meta.parquet")


def _signatures_from_frame(
    frame: pd.DataFrame, meta: pd.DataFrame, backend: str
) -> list[Signature]:
    """Rebuild Signature objects from a cached long-format parquet + metadata."""
    if frame.empty:
        return []
    by_id = {str(r["mol_id"]): r for _, r in meta.iterrows()}
    genes = frame[frame["mol_id"] == frame["mol_id"].iloc[0]]["gene"].tolist()
    out: list[Signature] = []
    for mol_id, sub in frame.groupby("mol_id", sort=False):
        mol_id = str(mol_id)
        row = by_id.get(mol_id)
        vals = sub.set_index("gene").reindex(genes)["value"].to_numpy(dtype=np.float32)
        meta_dict: dict = {}
        if row is not None:
            try:
                meta_dict = json.loads(row["meta_json"]) or {}
            except (TypeError, ValueError):
                meta_dict = {}
        meta_dict["from_cache"] = True
        out.append(
            Signature(
                mol_id=mol_id,
                smiles=str(row["smiles"]) if row is not None else "",
                genes=genes,
                values=vals,
                backend=backend,
                cell_line=str(row["cell_line"]) if row is not None else "",
                dose_um=float(row["dose_um"]) if row is not None else 0.0,
                time_h=float(row["time_h"]) if row is not None else 0.0,
                value_scale=str(row["value_scale"]) if row is not None else "zscore",
                meta=meta_dict,
                error=(
                    None
                    if row is None or pd.isna(row["error"])
                    else str(row["error"])
                ),
            )
        )
    return out
