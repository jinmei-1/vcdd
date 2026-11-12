"""Stage 2: Boltz co-folding and binding prediction over the stage-1 shortlist.

**Read this before interpreting a stage-2 table.** In 0.1.0 stage 2 re-docked
the shortlist harder and rescored the pose. In 0.2.0 it does something
categorically different: it sends the receptor *sequence* and the ligand
*SMILES* to the Boltz API, which predicts the complex from scratch and returns
its own confidence metrics.

Three consequences that change how the numbers should be read:

* **Boltz does not rescore the stage-1 pose.** It never sees it. The stage-2
  structure is a Boltz prediction, not a refined docking pose. This is a
  feature - a second empirical scoring function agreeing with Vina mostly
  measures their shared lineage, whereas a co-folding model is genuinely
  orthogonal evidence - but it is not "the same pose, scored better".
* **Nothing constrains the ligand to the box** except the pocket residues we
  pass. A confident prediction at the wrong site looks exactly like a hit, so
  :mod:`vcdd.refine.boltz.poseqc` measures the predicted ligand against the
  stage-1 box and the result is reported, not hidden.
* **``dock_score`` is stage 1's score, carried forward.** Stage 2 does not dock,
  so there is no new docking score. The column is kept, and kept equal to
  ``stage1_score``, because stages 3 and 4 and the report read it. Any apparent
  "improvement" from stage 1 to stage 2 in a 0.2.0 run would be a bug.

It also costs real money, per molecule, so the order of operations is: ask the
API for a cost estimate (free, no GPU, not billed), enforce the ceiling, and
only then submit. Every submission carries a deterministic idempotency key, so
an interrupted stage 2 re-attaches to its runs instead of paying twice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vcdd.receptor.box import Box
from vcdd.refine.boltz import local as boltz_local
from vcdd.refine.boltz import poseqc as bposeqc
from vcdd.refine.boltz import sab, screen
from vcdd.refine.boltz.client import make_client, resolve_credentials, sdk_version
from vcdd.refine.boltz.target import build_target
from vcdd.refine.consensus import build_consensus, rank_correlation_with_stage1
from vcdd.runtime.hashing import hash_obj, param_hash
from vcdd.runtime.ledger import RunLedger, UnitKey
from vcdd.runtime.logging import get_logger, log_event
from vcdd.runtime.paths import atomic_write_text, ensure_dir

LOG = get_logger()

#: Columns every stage-2 run emits, in this order, whichever backend ran.
BASE_COLUMNS = [
    "mol_id", "smiles", "stage1_score", "stage1_rank",
    "binding_confidence", "optimization_score", "structure_confidence",
    "iptm", "ptm", "complex_plddt", "complex_iplddt",
    "adme_lipophilicity", "adme_permeability", "adme_solubility",
    # Boltz-2's affinity head, when it was requested (local backend only, and
    # only with ``stage2.local_boltz.affinity``). ``affinity_pred_value`` is a
    # potency value, log10(IC50) with IC50 in uM - comparable between binders,
    # which is what analog ranking needs - while
    # ``affinity_probability_binary`` is the binder/decoy probability. Both stay
    # None when the head did not run, so the schema is the same either way.
    "affinity_pred_value", "affinity_probability_binary",
    "ligand_in_box", "centroid_offset_A",
    "boltz_run_id", "boltz_result_id", "warnings", "error",
]

#: Emitted for downstream compatibility. See the module docstring: stage 2 does
#: not dock in 0.2.0, so ``dock_score`` is stage 1's score unchanged.
CARRY_FORWARD = ["dock_score", "consensus_score", "consensus_rank", "stage1_score"]


# --------------------------------------------------------------------- inputs
def _receptor_ca_coords(receptor_path: str | Path) -> np.ndarray:
    """C-alpha trace of the receptor, ordered by residue number.

    This is the frame the docking box lives in.  Co-folding backends return
    models in their own frame, so pose QC has to superpose onto this trace before
    it can ask whether a predicted ligand sits in the box.  Residues are sorted
    by ``(resSeq, iCode)`` because prepared PDBQTs are not necessarily written in
    residue order.
    """
    per_chain: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    try:
        lines = Path(receptor_path).read_text(errors="replace").splitlines()
    except OSError:
        return np.zeros((0, 3), dtype=float)
    for raw in lines:
        if not raw.startswith("ATOM") or len(raw) < 54:
            continue
        if raw[12:16].strip().upper() != "CA":
            continue
        try:
            key = (int(raw[22:26]), raw[26].strip())
            xyz = (float(raw[30:38]), float(raw[38:46]), float(raw[46:54]))
        except ValueError:
            continue
        chain_id = (raw[21] or " ").strip() or "A"
        per_chain.setdefault(chain_id, {})[key] = xyz
    if not per_chain:
        return np.zeros((0, 3), dtype=float)
    largest = max(per_chain.values(), key=len)
    return np.array([largest[k] for k in sorted(largest)], dtype=float)


def _reference_ligand_smiles(cfg: Any) -> str | None:
    """SMILES of the co-crystal ligand, when one can be read with bond orders.

    A PDB block has no bond orders, so a SMILES derived from one is a guess -
    and a wrong reference ligand is a worse hint than none. Only SDF/MOL/MOL2
    are used; anything else is skipped with a warning.
    """
    path = getattr(cfg.receptor, "reference_ligand_path", None)
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    if suffix not in {".sdf", ".mol", ".mol2"}:
        LOG.warning(
            "stage2: reference ligand %s has no bond orders (%s); not sending it as a Boltz "
            "reference. Supply an SDF/MOL if you want the hint.", path, suffix or "no suffix",
        )
        return None
    try:
        from rdkit import Chem

        from vcdd.receptor.redock import _load_reference_mol

        mol = _load_reference_mol(path, getattr(cfg.receptor, "reference_ligand", None))
        if mol is None:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("stage2: could not derive reference-ligand SMILES from %s (%s)", path, exc)
        return None


def _shortlist(stage1_dir: str | Path, top_n: int) -> pd.DataFrame:
    from vcdd.screening.stage1 import load_ranking

    ranking = load_ranking(stage1_dir)
    if ranking.empty:
        raise RuntimeError("stage 1 ranking is empty; nothing to refine")
    shortlist = ranking.head(min(int(top_n), len(ranking))).copy()
    shortlist["stage1_score"] = pd.to_numeric(shortlist["score"], errors="coerce")
    shortlist["stage1_rank"] = (
        pd.to_numeric(shortlist["rank"], errors="coerce")
        if "rank" in shortlist.columns
        else pd.Series(range(1, len(shortlist) + 1), index=shortlist.index)
    )
    return shortlist.reset_index(drop=True)


# -------------------------------------------------------------------- ranking
def build_ranking(
    scores: pd.DataFrame,
    fusion: str = "boltz_only",
    gate_threshold: float = 0.7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank the shortlist under the configured fusion policy.

    All three policies emit ``consensus_score`` on the project-wide
    *lower-is-better* convention, plus a 1-based ``consensus_rank``. Molecules
    with no Boltz result keep their row and sort last: dropping them would hide
    a systematically rejected chemotype.
    """
    out = scores.copy()
    bc = pd.to_numeric(out.get("binding_confidence"), errors="coerce")
    s1 = pd.to_numeric(out.get("stage1_score"), errors="coerce")
    has_boltz = bc.notna()
    report: dict[str, Any] = {
        "fusion": fusion,
        "n_with_boltz": int(has_boltz.sum()),
        "n_without_boltz": int((~has_boltz).sum()),
    }

    if fusion == "rank_fusion":
        # build_consensus expects lower-is-better columns; binding confidence is
        # higher-is-better, so it is negated rather than re-ranked by hand.
        out["_rank_input_stage1"] = s1
        out["_rank_input_boltz"] = -bc
        ranked, agreement = build_consensus(
            out, ["_rank_input_stage1", "_rank_input_boltz"],
            method="borda", min_agreement=0.0, dedupe=False,
        )
        report.update(agreement)
        ranked = ranked.drop(columns=["_rank_input_stage1", "_rank_input_boltz"])
        order = ranked["consensus_score"]
        ranked = ranked.assign(_order=order).sort_values(
            ["_order", "mol_id"], na_position="last", kind="mergesort",
        ).drop(columns=["_order"])
    elif fusion == "gate":
        passed = bc.fillna(-np.inf) >= float(gate_threshold)
        out["gate_pass"] = passed
        out["consensus_score"] = s1
        report["gate_threshold"] = float(gate_threshold)
        report["n_gate_pass"] = int(passed.sum())
        ranked = out.assign(_fail=~passed).sort_values(
            ["_fail", "consensus_score", "mol_id"], na_position="last", kind="mergesort",
        ).drop(columns=["_fail"])
    else:  # boltz_only
        if fusion != "boltz_only":
            LOG.warning("stage2: unknown fusion %r; falling back to boltz_only", fusion)
        out["consensus_score"] = -bc
        ranked = out.sort_values(
            ["consensus_score", "mol_id"], na_position="last", kind="mergesort",
        )

    # ``fusion`` is what the config asked for; ``policy`` is what actually ran.
    # They differ only when an unrecognised name fell back to boltz_only, and
    # that is precisely the case a reader of the summary must be able to see.
    report["policy"] = fusion if fusion in ("boltz_only", "rank_fusion", "gate") else "boltz_only"

    ranked = ranked.reset_index(drop=True)
    ranked["consensus_rank"] = range(1, len(ranked) + 1)
    # dock_score is stage 1's score. Stage 2 does not dock in 0.2.0.
    ranked["dock_score"] = pd.to_numeric(ranked["stage1_score"], errors="coerce")
    return ranked, report


# ----------------------------------------------------------------------- main
def run_stage2(
    cfg: Any,
    receptor_pdbqt: str | Path,
    box: Box,
    stage1_dir: str | Path,
    out_dir: str | Path,
    ledger: RunLedger,
    force: bool = False,
    ensemble_pdbqt: list[str] | None = None,
    allow_spend: bool = False,
    client: Any | None = None,
    pose_fetch: Any = None,
) -> dict[str, Any]:
    """Screen the stage-1 shortlist through Boltz and write the ranking.

    ``client`` is injectable so the tests can drive the whole stage against a
    scripted transport behind the real SDK, and ``pose_fetch`` is the same seam
    for the structure downloads, which go to a signed URL rather than the API.
    ``allow_spend`` is ``--yes-spend``.
    """
    out = ensure_dir(out_dir)
    batch_dir = ensure_dir(out / "batches")
    boltz_dir = ensure_dir(out / "boltz")
    poses_dir = out / "poses"
    s2 = cfg.stage2
    t_start = time.time()

    if ensemble_pdbqt:
        LOG.warning(
            "stage2: receptor.ensemble_paths is ignored in 0.2.0 - Boltz predicts the complex "
            "from sequence, so alternative receptor conformations have nothing to dock into."
        )

    shortlist = _shortlist(stage1_dir, s2.top_n)
    mol_ids = [str(m) for m in shortlist["mol_id"]]
    smiles = [str(s) for s in shortlist["smiles"]]

    # ---- target ------------------------------------------------------------
    is_local = s2.backend == "local"
    ref_smiles = _reference_ligand_smiles(cfg) if s2.use_reference_ligand and not is_local else None
    target, provenance = build_target(
        receptor_pdbqt, box,
        pocket_from_box=s2.pocket_from_box,
        pocket_margin=s2.pocket_margin_a,
        reference_ligands=[ref_smiles] if ref_smiles else None,
    )
    if provenance.get("warning"):
        LOG.warning("stage2: %s", provenance["warning"])
    LOG.info(
        "stage2: target = %d chain(s), %d residue(s), pocket %s",
        len(provenance.get("chains", [])),
        sum(c.get("length", 0) for c in provenance.get("chains", [])),
        provenance.get("pocket_sizes") or "not constrained",
    )

    # ---- credentials -------------------------------------------------------
    creds = resolve_credentials(s2.api_key_env)
    if is_local:
        if client is None:
            exe = boltz_local.find_binary(s2.local_boltz.binary)
            if exe is None:
                raise RuntimeError(
                    f"stage2.backend='local' but the '{s2.local_boltz.binary}' executable was not "
                    "found on PATH. Install the boltz package (`pip install boltz`) and check "
                    "stage2.local_boltz.binary."
                )
            LOG.info(
                "stage2: local boltz predict via %s (version %s)",
                exe, boltz_local.version(s2.local_boltz.binary) or "unknown",
            )
    else:
        if client is None:
            creds.require()
            client = make_client(
                s2.api_key_env,
                request_timeout_s=float(s2.request_timeout_s),
                max_retries=2,
            )
        if creds.present and not creds.livemode:
            LOG.warning(
                "stage2: $%s holds a TEST-MODE key. The full lifecycle runs and the numbers look "
                "real, but they are synthetic: no GPU, no billing, no science. Every artefact from "
                "this run is labelled livemode=false.", s2.api_key_env,
            )

    molecule_filters = (
        {"boltz_smarts_catalog_filter_level": s2.molecule_filter_level}
        if s2.molecule_filter_level != "disabled" and not is_local else None
    )

    local_tool = boltz_local.version(s2.local_boltz.binary) if is_local else None
    phash = param_hash({
        "backend": s2.backend, "target": target, "filters": molecule_filters,
        "num_samples": s2.num_samples,
        "tool": local_tool if is_local else sdk_version(),
        "local_boltz": {
            "binary": s2.local_boltz.binary,
            "devices": list(s2.local_boltz.devices),
            "cache_dir": s2.local_boltz.cache_dir,
            "half_precision": s2.local_boltz.half_precision,
            "extra_args": list(s2.local_boltz.extra_args),
        } if is_local else None,
    })
    chunk_size = max(1, int(s2.chunk_size))
    chunks = [
        (i, list(range(start, min(start + chunk_size, len(mol_ids)))))
        for i, start in enumerate(range(0, len(mol_ids), chunk_size))
    ]

    # ---- spend gate --------------------------------------------------------
    pending = [
        (ci, idx) for ci, idx in chunks
        if force or not (
            ledger.is_done(UnitKey("stage2", f"chunk_{ci:05d}",
                                   hash_obj([mol_ids[i] for i in idx]), phash))
            and (batch_dir / f"chunk_{ci:05d}.parquet").exists()
        )
    ]
    estimates: list[dict[str, Any]] = []
    if pending and is_local:
        LOG.info(
            "stage2: %d molecule(s) in %d pending chunk(s); local inference - no API, "
            "no spend gate",
            sum(len(idx) for _, idx in pending), len(pending),
        )
    elif pending:
        estimates = _estimate(client, s2, target, molecule_filters, mol_ids, smiles, pending)
        total = screen.check_spend(estimates, float(s2.max_spend_usd), allow=allow_spend)
        LOG.info(
            "stage2: %d molecule(s) in %d pending chunk(s); API estimate $%.2f "
            "(ceiling $%.2f)%s",
            sum(len(idx) for _, idx in pending), len(pending), total,
            float(s2.max_spend_usd), " [--yes-spend]" if allow_spend else "",
        )
    else:
        LOG.info("stage2: every chunk already complete; nothing to submit")

    # ---- submit / poll / collect ------------------------------------------
    runs_path = boltz_dir / "runs.json"
    all_runs: list[screen.ScreenRun] = screen.load_runs(runs_path) if not force else []
    known = {r.run_id for r in all_runs}
    result_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for ci, idx in chunks:
        tag = f"chunk_{ci:05d}"
        chunk_ids = [mol_ids[i] for i in idx]
        chunk_smiles = [smiles[i] for i in idx]
        key = UnitKey("stage2", tag, hash_obj(chunk_ids), phash)
        parq = batch_dir / f"{tag}.parquet"
        rej_parq = batch_dir / f"{tag}_rejected.parquet"

        if not force and ledger.is_done(key) and parq.exists():
            result_rows.extend(pd.read_parquet(parq).to_dict("records"))
            if rej_parq.exists():
                rejected.extend(pd.read_parquet(rej_parq).to_dict("records"))
            LOG.info("stage2: %s already complete (%d molecules) - skipping", tag, len(chunk_ids))
            continue

        ledger.mark_running(key)
        est = next((e for c, e in zip([c for c, _ in pending], estimates) if c == ci), None)
        t0 = time.time()
        try:
            rows, runs, rej = _run_chunk(
                client, s2, target, provenance, chunk_ids, chunk_smiles, ci,
                molecule_filters, est,
                local_out_dir=boltz_dir / "local",
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            ledger.mark_failed(key, f"{type(exc).__name__}: {exc}")
            ledger.sync()
            raise

        pd.DataFrame(rows).to_parquet(parq, index=False)
        if rej:
            pd.DataFrame(rej).to_parquet(rej_parq, index=False)
        screen.append_results_jsonl(boltz_dir / "results.jsonl", rows)
        for r in runs:
            if r.run_id not in known:
                all_runs.append(r)
                known.add(r.run_id)
        screen.save_runs(runs_path, all_runs)
        result_rows.extend(rows)
        rejected.extend(rej)
        ledger.mark_done(key, artifact=parq, n_items=len(rows))
        ledger.sync()
        LOG.info("stage2: %s done - %d/%d returned in %.1fs",
                 tag, len(rows), len(chunk_ids), time.time() - t0)

    # ---- assemble ----------------------------------------------------------
    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = results.drop_duplicates(subset=["mol_id"], keep="last")
    rej_df = pd.DataFrame(rejected)
    scores = shortlist[["mol_id", "smiles", "stage1_score", "stage1_rank"]].copy()
    scores["mol_id"] = scores["mol_id"].astype(str)
    if not results.empty:
        drop = [c for c in ("smiles", "created_at", "boltz_smiles") if c in results.columns]
        scores = scores.merge(results.drop(columns=drop), on="mol_id", how="left")
    if not rej_df.empty:
        reasons = rej_df.drop_duplicates(subset=["mol_id"], keep="last").set_index("mol_id")["reason"]
        scores["error"] = scores["mol_id"].map(reasons)
        rej_df.to_parquet(boltz_dir / "rejected.parquet", index=False)
    if "error" not in scores.columns:
        scores["error"] = None
    missing = scores["binding_confidence"].isna() if "binding_confidence" in scores else pd.Series(
        [True] * len(scores), index=scores.index)
    scores.loc[missing & scores["error"].isna(), "error"] = "no_result_returned"

    # ---- pose QC -----------------------------------------------------------
    qc_rows: list[dict[str, Any]] = []
    if is_local and not s2.local_boltz.keep_raw_outputs:
        LOG.warning(
            "stage2: stage2.local_boltz.keep_raw_outputs=false deletes the predicted "
            "structures, so ligand-in-box pose QC is skipped."
        )
    elif s2.keep_structures_top_k and not results.empty:
        ranked_for_qc, _ = build_ranking(scores, s2.fusion, s2.gate_threshold)
        # Co-folding returns a model in its own coordinate frame, so the box test
        # is only meaningful after superposing the predicted receptor onto the
        # receptor the box was defined on.  Without this the QC reports every
        # ligand as hundreds of Angstroms from the box, whatever it actually did.
        reference_ca = _receptor_ca_coords(receptor_pdbqt)
        if is_local:
            # For local runs the "URL" column holds the full-complex CIF on
            # disk, in which Boltz tags the ligand HETATM. The fetch seam reads
            # the file; hetatm_only measures only the ligand's heavy atoms.
            qc_rows = bposeqc.qc_poses(
                ranked_for_qc.to_dict("records"), box, poses_dir,
                top_k=int(s2.keep_structures_top_k),
                hetatm_only=True,
                timeout_s=float(s2.local_boltz.timeout_s),
                fetch=pose_fetch if pose_fetch is not None else boltz_local.read_local_file,
                reference_ca=reference_ca,
            )
        else:
            qc_rows = bposeqc.qc_poses(
                ranked_for_qc.to_dict("records"), box, poses_dir,
                top_k=int(s2.keep_structures_top_k),
                timeout_s=float(s2.request_timeout_s),
                reference_ca=reference_ca,
                **({"fetch": pose_fetch} if pose_fetch is not None else {}),
            )
    qc_df = pd.DataFrame(qc_rows) if qc_rows else pd.DataFrame(columns=bposeqc.QC_COLUMNS)
    qc_df.to_parquet(out / "pose_qc.parquet", index=False)
    qc_stats = bposeqc.qc_summary(qc_rows)
    if qc_rows:
        qc_join = qc_df.set_index("mol_id")
        for col in ("ligand_in_box", "centroid_offset_A"):
            scores[col] = scores["mol_id"].map(qc_join[col])

    for col in BASE_COLUMNS:
        if col not in scores.columns:
            scores[col] = None
    # ``stage3.load_shortlist`` falls back to sorting scores.parquet by
    # ``dock_score`` when the ranking is missing. In 0.1.0 that column existed
    # because stage 2 re-docked; in 0.2.0 it must be carried forward explicitly
    # or the fallback raises a bare KeyError on an otherwise complete run.
    scores["dock_score"] = pd.to_numeric(scores["stage1_score"], errors="coerce")
    scores = scores[BASE_COLUMNS + [c for c in scores.columns if c not in BASE_COLUMNS]]
    scores.to_parquet(out / "scores.parquet", index=False)

    # ---- ranking -----------------------------------------------------------
    consensus, fusion_report = build_ranking(scores, s2.fusion, s2.gate_threshold)
    consensus.to_parquet(out / "consensus_ranking.parquet", index=False)
    rho_stage1 = rank_correlation_with_stage1(consensus, "stage1_score")

    # ---- summary -----------------------------------------------------------
    n_scored = int(pd.to_numeric(scores["binding_confidence"], errors="coerce").notna().sum())
    best_bc = pd.to_numeric(scores["binding_confidence"], errors="coerce").max()
    spent = sum(
        v for v in (screen._as_float(e.get("estimated_cost_usd")) for e in estimates)
        if v is not None
    )
    summary = {
        "backend": s2.backend,
        "engine": all_runs[0].engine if all_runs else None,
        "engine_version": all_runs[0].engine_version if all_runs else None,
        "sdk_version": None if is_local else sdk_version(),
        "local_boltz": {
            "binary": s2.local_boltz.binary,
            "version": local_tool,
            "devices": list(s2.local_boltz.devices),
            "max_concurrency": int(s2.local_boltz.max_concurrency),
            "num_samples": int(s2.num_samples),
        } if is_local else None,
        "livemode": bool(all_runs[0].livemode) if all_runs else creds.livemode,
        "api_key_env": s2.api_key_env,
        "n_input": int(len(scores)),
        "n_scored": n_scored,
        "n_failed": int(len(scores) - n_scored),
        "n_rejected": int(len(rej_df)),
        "estimated_cost_usd": round(spent, 4) if estimates else 0.0,
        "n_runs": len(all_runs),
        "fusion": fusion_report,
        "gate_threshold": float(s2.gate_threshold),
        "best_binding_confidence": float(best_bc) if pd.notna(best_bc) else None,
        "spearman_vs_stage1": rho_stage1,
        "pose_qc": qc_stats,
        "target": provenance,
        "seconds": round(time.time() - t_start, 1),
        "consensus_ranking_path": str(out / "consensus_ranking.parquet"),
        "poses_path": str(poses_dir),
        "note": (
            "dock_score is stage 1's Vina-GPU score carried forward; stage 2 does not dock. "
            "Boltz predicts the complex from sequence + SMILES rather than rescoring the "
            "stage-1 pose."
            + (
                " backend=local: boltz-2 ran on this machine (no API billing, no spend gate); "
                "pocket residues and reference ligands were not sent to the model, so "
                "ligand_in_box is the only site check."
                if is_local else ""
            )
        ),
    }
    atomic_write_text(out / "stage2_summary.json", json.dumps(summary, indent=2, default=str))
    log_event(
        f"stage2 complete: {n_scored}/{len(scores)} scored by {s2.backend} in "
        f"{summary['seconds']}s; est ${summary['estimated_cost_usd']}; "
        f"ligand-in-box {qc_stats.get('fraction_in_box')}",
        stage="stage2",
        **{k: v for k, v in summary.items() if k not in {"target", "fusion"}},
    )
    if not summary["livemode"]:
        LOG.warning("stage2: results are TEST-MODE synthetic values - do not act on them.")
    if qc_stats.get("fraction_in_box") is not None and qc_stats["fraction_in_box"] < 0.5:
        LOG.warning(
            "stage2: only %.0f%% of the %d measured poses put the ligand inside the stage-1 "
            "box (median receptor C-alpha RMSD %.2f A). Either the pocket constraint is too weak "
            "or the model prefers a different site; treat the binding-confidence ranking as "
            "site-agnostic until you look at the structures.",
            100 * qc_stats["fraction_in_box"], int(qc_stats.get("n_checked") or 0),
            float(qc_stats.get("median_receptor_ca_rmsd_A") or 0.0),
        )
    if qc_stats.get("n_frame_unaligned"):
        LOG.warning(
            "stage2: %d of %d pose(s) could not be placed in the receptor's frame (the predicted "
            "protein chain does not match the receptor), so no box distance is reported for them "
            "- see pose_qc.parquet's pose_error column.",
            int(qc_stats["n_frame_unaligned"]),
            int(qc_stats["n_frame_unaligned"]) + int(qc_stats.get("n_checked") or 0),
        )
    if rho_stage1 is not None and rho_stage1 > 0.95:
        LOG.warning(
            "stage2: the Boltz ranking correlates with stage 1 at rho=%.2f - it barely reordered "
            "the shortlist, so it bought little here beyond a second opinion.", rho_stage1,
        )
    return summary


# ------------------------------------------------------------------- backends
def _estimate(
    client: Any,
    s2: Any,
    target: dict[str, Any],
    molecule_filters: dict[str, Any] | None,
    mol_ids: list[str],
    smiles: list[str],
    pending: list[tuple[int, list[int]]],
) -> list[dict[str, Any]]:
    """One cost estimate per pending chunk. Free, no GPU, not billed."""
    estimates: list[dict[str, Any]] = []
    for ci, idx in pending:
        chunk_mols = screen.make_molecules([mol_ids[i] for i in idx], [smiles[i] for i in idx])
        if s2.backend == "library_screen":
            estimates.append(screen.estimate_cost(
                client, target, chunk_mols, molecule_filters, s2.workspace_id))
            continue
        # structure_and_binding is priced per sample by receptor crop size, so the
        # per-molecule cost is molecule-independent: estimate once and multiply
        # rather than making one free-but-rate-limited call per molecule.
        payload = sab.build_input(
            target["entities"], chunk_mols[0]["smiles"],
            pocket_residues=target.get("pocket_residues"),
            num_samples=s2.num_samples,
        )
        one = sab.estimate_cost(client, payload, workspace_id=s2.workspace_id)
        unit = screen._as_float(one.get("estimated_cost_usd"))
        estimates.append({
            "estimated_cost_usd": None if unit is None else f"{unit * len(chunk_mols):.4f}",
            "disclaimer": one.get("disclaimer"),
            "breakdown": {**one.get("breakdown", {}), "num_units": len(chunk_mols)},
        })
    return estimates


def _run_chunk(
    client: Any,
    s2: Any,
    target: dict[str, Any],
    provenance: dict[str, Any],
    chunk_ids: list[str],
    chunk_smiles: list[str],
    chunk_index: int,
    molecule_filters: dict[str, Any] | None,
    estimate: dict[str, Any] | None,
    local_out_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[screen.ScreenRun], list[dict[str, Any]]]:
    """Submit one chunk on the configured backend and collect its results."""
    est_usd = (estimate or {}).get("estimated_cost_usd")

    if s2.backend == "local":
        if provenance.get("pocket_sizes") or provenance.get("reference_ligands"):
            LOG.warning(
                "stage2: backend='local' cannot send pocket residues or reference ligands to "
                "boltz predict (the YAML has no such fields), so the stage-1 docking box is "
                "not passed to the model. The ligand-in-box pose QC is the check that the "
                "predicted site matches the docked one."
            )
        return boltz_local.screen_molecules(
            s2.local_boltz, target["entities"], chunk_ids, chunk_smiles,
            out_dir=local_out_dir, chunk_index=chunk_index,
            num_samples=int(s2.num_samples),
            # ``--force`` must reach boltz: its output directory is keyed by
            # molecule id and boltz skips predictions that already exist there.
            force=force,
        )

    if s2.backend == "structure_and_binding":
        chain_ids = [c for e in target["entities"] for c in e["chain_ids"]]
        rows, runs, rejected = sab.screen_molecules(
            client, target["entities"], chunk_ids, chunk_smiles,
            pocket_residues=target.get("pocket_residues"),
            ligand_chain_id=sab.pick_ligand_chain_id(chain_ids),
            num_samples=int(s2.num_samples),
            workspace_id=s2.workspace_id,
            poll_interval_s=float(s2.poll_interval_s),
            estimates=[est_usd] * len(chunk_ids),
        )
        return rows, runs, rejected

    molecules = screen.make_molecules(chunk_ids, chunk_smiles)
    run = screen.start_chunk(
        client, target, molecules, chunk_index,
        molecule_filters=molecule_filters,
        workspace_id=s2.workspace_id,
        estimated_cost_usd=est_usd,
    )
    LOG.info("stage2: chunk %d submitted as run %s (%d molecules)",
             chunk_index, run.run_id, len(molecules))
    run = screen.poll_until_terminal(
        client, run,
        poll_interval_s=float(s2.poll_interval_s),
        workspace_id=s2.workspace_id,
        on_progress=_log_progress,
    )
    if run.status == "failed" and not run.progress.get("num_molecules_screened"):
        LOG.error("stage2: run %s failed - %s", run.run_id, run.error)
        return [], [run], screen.reconcile_missing(molecules, [], run)
    rows = screen.fetch_results(client, run.run_id, workspace_id=s2.workspace_id)
    rejected = screen.reconcile_missing(molecules, [r["mol_id"] for r in rows], run)
    if run.status == "stopped":
        LOG.warning("stage2: run %s was stopped; keeping the %d partial result(s)",
                    run.run_id, len(rows))
    return rows, [run], rejected


_LAST_PROGRESS: dict[str, int] = {}


def _log_progress(run: screen.ScreenRun) -> None:
    done = run.progress.get("num_molecules_screened") or 0
    total = run.progress.get("total_molecules_to_screen") or run.n_molecules
    if done and done != _LAST_PROGRESS.get(run.run_id):
        _LAST_PROGRESS[run.run_id] = done
        LOG.info("stage2: run %s %d/%d screened", run.run_id, done, total)


# --------------------------------------------------------------------- loaders
def load_consensus(stage2_dir: str | Path) -> pd.DataFrame:
    path = Path(stage2_dir) / "consensus_ranking.parquet"
    if not path.exists():
        raise FileNotFoundError(f"stage 2 consensus not found: {path}. Run `vcdd stage2` first.")
    return pd.read_parquet(path)
