"""Stage 1: fast high-throughput docking over the whole library.

Design points that matter at 10^6 scale:

* **Shard-level resume.** One ledger unit per shard. A killed run resumes at the
  shard boundary and the resulting ranking is bit-for-bit identical to an
  uninterrupted run *in composition* (a unit test asserts this on the mock
  engine).
* **Measured, not assumed, throughput.** A pilot slice is docked first and
  ``throughput.json`` records seconds/ligand and a projected ETA.
* **Failure isolation.** Ligand preparation or docking failures are recorded per
  molecule with a reason and never abort the shard.
* **Flat memory.** Shards stream in and out; nothing accumulates except the
  final per-shard score table.

What 0.2.0 changed
------------------
Stage 1 is GPU-only, and a GPU docking program is *batch-native*: it takes a
directory of ligands and keeps the device saturated. The 0.1.0 structure - one
ligand per task, fanned out over a process pool - would launch one GPU process
per molecule and be slower than the CPU it replaced.

So the pool is now used for what it is actually good at (RDKit conformer
generation and PDBQT writing, which are CPU-bound and embarrassingly parallel)
and the docking call is made once per batch in the parent process. That split
lives in :func:`execute_jobs`, and every caller - the shard loop, the pilot,
stage 4's re-dock, and active learning - goes through it, so there is exactly
one place where the batch/serial decision is made.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from vcdd.docking.base import DockParams, get_engine, select_engine
from vcdd.docking.ligprep import prepare_ligand
from vcdd.io.library import load_shard_index
from vcdd.receptor.box import Box
from vcdd.runtime.hashing import hash_obj, param_hash
from vcdd.runtime.ledger import RunLedger, UnitKey
from vcdd.runtime.logging import get_logger, log_event
from vcdd.runtime.parallel import map_items, resolve_workers, shard_indices
from vcdd.runtime.paths import atomic_write_text, ensure_dir

LOG = get_logger()

SCORE_COLUMNS = [
    "mol_id", "smiles", "score", "engine", "seconds", "n_rotatable", "error", "shard",
]


# --------------------------------------------------------------------- worker
@dataclass
class _DockJob:
    """Picklable unit of work for the process pool."""

    mol_id: str
    smiles: str
    receptor_pdbqt: str
    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    engine_name: str
    params: dict[str, Any]
    n_conformers: int
    minimize: bool
    seed: int
    keep_pose: bool = False


def _prep_one(job: _DockJob) -> dict[str, Any]:
    """Ligand preparation only. Runs inside a worker process.

    Split out of :func:`_dock_one` so the batch path can parallelise the CPU
    work (RDKit embedding, PDBQT conversion) without also fanning out the GPU
    call. Returns the PDBQT text rather than a path: workers and parent do not
    share a filesystem assumption, and the text is a few kB.
    """
    from vcdd.runtime.logging import worker_quiet

    worker_quiet()
    t0 = time.time()
    prepared = prepare_ligand(
        job.mol_id, job.smiles, n_conformers=job.n_conformers,
        minimize=job.minimize, seed=job.seed,
    )
    return {
        "mol_id": job.mol_id,
        "smiles": job.smiles,
        "pdbqt": prepared.pdbqt if prepared.ok else None,
        "n_rotatable": prepared.n_rotatable,
        "error": None if prepared.ok else (prepared.error or "ligand_prep_failed"),
        "prep_seconds": time.time() - t0,
    }


def _dock_one(job: _DockJob) -> dict[str, Any]:
    """Prepare + dock a single ligand in one worker process.

    The serial path, used by engines that are not batch-native (currently only
    the ``mock`` test double). Kept because the contract must still work for a
    CPU-shaped engine added later.
    """
    from vcdd.runtime.logging import worker_quiet

    worker_quiet()
    t0 = time.time()
    prep = _prep_one(job)
    if prep["error"] is not None:
        return {
            "mol_id": job.mol_id, "smiles": job.smiles, "score": None,
            "engine": job.engine_name, "seconds": time.time() - t0,
            "n_rotatable": prep["n_rotatable"], "error": prep["error"],
        }

    engine = get_engine(job.engine_name)
    box = Box(center=job.box_center, size=job.box_size, source="stage1")
    params = DockParams(**job.params)
    result = engine.dock(prep["pdbqt"] or "", job.receptor_pdbqt, box, params, mol_id=job.mol_id)
    row = {
        "mol_id": job.mol_id,
        "smiles": job.smiles,
        "score": result.score,
        "engine": result.engine,
        "seconds": time.time() - t0,
        "n_rotatable": prep["n_rotatable"],
        "error": result.error,
    }
    if job.keep_pose:
        row["pose_pdbqt"] = result.pose_pdbqt
    return row


def _build_jobs(df: pd.DataFrame, common: dict[str, Any]) -> list[_DockJob]:
    return [
        _DockJob(mol_id=str(row.mol_id), smiles=str(row.smiles), **common)
        for row in df.itertuples(index=False)
    ]


# ------------------------------------------------------------------ dispatch
def execute_jobs(
    jobs: list[_DockJob],
    n_workers: int,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run a list of dock jobs, choosing the batch or serial path by capability.

    This is the single dispatch point for all of stage 1. It branches on
    ``capabilities().batch_native`` and never on the engine name.

    Batch path (Vina-GPU): prepare every ligand in the process pool, then hand
    chunks of ``batch_size`` prepared ligands to ``engine.dock_batch`` from the
    parent process, so the GPU sees one long-lived process per chunk instead of
    one per molecule.

    Serial path: one ``_dock_one`` task per ligand in the pool, as in 0.1.0.

    Returns one row per input job, in input order. Rows always carry ``mol_id``,
    ``smiles``, ``score``, ``engine``, ``seconds``, ``n_rotatable`` and
    ``error``; a failure is a row with ``score=None`` and a reason, never a
    missing row.
    """
    if not jobs:
        return []

    engine = get_engine(jobs[0].engine_name)
    caps = engine.capabilities()
    if not caps.batch_native:
        results = map_items(_dock_one, jobs, n_workers=n_workers)
        rows: list[dict[str, Any]] = []
        for r in results:
            if r.ok and r.value is not None:
                rows.append(r.value)
            else:
                job = jobs[r.index]
                rows.append({
                    "mol_id": job.mol_id, "smiles": job.smiles, "score": None,
                    "engine": job.engine_name, "seconds": 0.0, "n_rotatable": 0,
                    "error": r.error or "worker_failed",
                })
        return rows

    # ---- batch-native ------------------------------------------------------
    chunk = int(batch_size or caps.preferred_batch_size or 1)
    chunk = max(1, chunk)
    box = Box(center=jobs[0].box_center, size=jobs[0].box_size, source="stage1")
    params = DockParams(**jobs[0].params)
    receptor = jobs[0].receptor_pdbqt

    prep_results = map_items(_prep_one, jobs, n_workers=n_workers)
    preps: list[dict[str, Any]] = []
    for r in prep_results:
        job = jobs[r.index]
        if r.ok and r.value is not None:
            preps.append(r.value)
        else:
            preps.append({
                "mol_id": job.mol_id, "smiles": job.smiles, "pdbqt": None,
                "n_rotatable": 0, "error": r.error or "worker_failed",
                "prep_seconds": 0.0,
            })

    rows = [
        {
            "mol_id": p["mol_id"], "smiles": p["smiles"], "score": None,
            "engine": engine.name, "seconds": round(float(p["prep_seconds"]), 4),
            "n_rotatable": p["n_rotatable"],
            # A ligand that never reached the GPU is not 'not docked yet'; it
            # failed preparation, and the reason has to survive to the table.
            "error": p["error"] or "not_docked",
        }
        for p in preps
    ]

    dockable = [i for i, p in enumerate(preps) if p["pdbqt"]]
    for start in range(0, len(dockable), chunk):
        idxs = dockable[start : start + chunk]
        items = [(preps[i]["mol_id"], preps[i]["pdbqt"] or "") for i in idxs]
        t0 = time.time()
        try:
            batch = engine.dock_batch(items, receptor, box, params)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the shard
            LOG.warning("stage1: dock_batch failed for %d ligands - %s", len(items), exc)
            for i in idxs:
                rows[i]["error"] = f"batch_failed:{type(exc).__name__}"
            continue
        per_ligand = (time.time() - t0) / max(1, len(items))
        if len(batch) != len(items):  # pragma: no cover - contract violation
            LOG.error("stage1: engine %s returned %d results for %d ligands",
                      engine.name, len(batch), len(items))
        for i, res in zip(idxs, batch):
            rows[i]["score"] = res.score
            rows[i]["engine"] = res.engine or engine.name
            rows[i]["error"] = res.error
            rows[i]["seconds"] = round(float(preps[i]["prep_seconds"]) + per_ligand, 4)
    return rows


def build_dock_params(s1: Any) -> dict[str, Any]:
    """Config -> :class:`DockParams` kwargs for stage 1.

    One function so the shard loop, the pilot, the param hash and stage 4 cannot
    drift apart.
    """
    return {
        "variant": s1.variant,
        "thread": s1.thread,
        "search_depth": s1.search_depth,
        "rilc_bfgs": s1.rilc_bfgs,
        "opencl_binary_path": s1.opencl_binary_path,
        "seed": s1.seed,
        "timeout_s": s1.timeout_s,
        "batch_timeout_s": s1.batch_timeout_s,
    }


def dock_molecules(
    mol_ids: Sequence[str],
    smiles: Sequence[str],
    receptor_pdbqt: str | Path,
    box: Box,
    engine_name: str,
    params: dict[str, Any],
    n_conformers: int = 1,
    minimize: bool = True,
    seed: int = 42,
    n_workers: int | None = None,
    keep_poses: bool = False,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Dock an explicit list of SMILES with a fixed protocol.

    The public entry point for docking outside the shard machinery -- stage 4
    uses it to re-dock generated analogs *and their parents* under identical
    settings.  Re-docking the parents is not redundant: with a stochastic GPU
    search and no seed, a delta computed against a stored score from an earlier
    run would confound the structural change with run-to-run search noise.

    ``keep_poses`` currently only has an effect on the serial path; the batch
    path returns scores and leaves poses on the engine, which is what stage 1
    needs and what keeps a million-molecule run's memory flat.
    """
    common = {
        "receptor_pdbqt": str(receptor_pdbqt),
        "box_center": tuple(box.center),
        "box_size": tuple(box.size),
        "engine_name": engine_name,
        "params": params,
        "n_conformers": n_conformers,
        "minimize": minimize,
        "seed": seed,
        "keep_pose": keep_poses,
    }
    jobs = [
        _DockJob(mol_id=str(m), smiles=str(s), **common)
        for m, s in zip(mol_ids, smiles)
    ]
    if not jobs:
        cols = ["mol_id", "smiles", "score", "engine", "seconds", "n_rotatable", "error"]
        return pd.DataFrame(columns=cols + (["pose_pdbqt"] if keep_poses else []))
    rows = execute_jobs(jobs, n_workers=resolve_workers(n_workers), batch_size=batch_size)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- pilot
def measure_throughput(
    jobs: list[_DockJob],
    n_workers: int,
    total_molecules: int,
    out_path: Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Dock a pilot slice, record real throughput, and project an ETA.

    0.2.0 drops the per-core figures. With a batch-native GPU engine the CPU
    workers only prepare ligands, so ``seconds_per_ligand_core`` measured the
    wrong resource entirely and ``projected_core_hours`` was a number with no
    referent. Wall clock and ligands/hour are what a GPU run is actually sized
    on.
    """
    t0 = time.time()
    rows = execute_jobs(jobs, n_workers=n_workers, batch_size=batch_size)
    elapsed = time.time() - t0

    ok = [r for r in rows if r.get("score") is not None]
    failed = len(rows) - len(ok)
    per_ligand_wall = elapsed / max(1, len(jobs))
    undersized = batch_size is not None and len(jobs) < batch_size
    throughput = {
        "pilot_size": len(jobs),
        "pilot_seconds": round(elapsed, 2),
        "pilot_failures": failed,
        "n_workers": n_workers,
        "batch_size": batch_size,
        "seconds_per_ligand_wall": round(per_ligand_wall, 4),
        "ligands_per_hour": round(3600.0 / per_ligand_wall, 1) if per_ligand_wall > 0 else None,
        "total_molecules": total_molecules,
        "projected_hours": round(total_molecules * per_ligand_wall / 3600.0, 3),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "Measured on this machine with these exact parameters. Use for the ETA and for "
                "sizing a production run; do not reuse across different hardware or settings.",
    }
    if undersized:
        throughput["warning"] = (
            f"pilot of {len(jobs)} ligands is smaller than one GPU batch ({batch_size}), so it "
            f"pays the receptor-grid and OpenCL-kernel setup over too few molecules. The real "
            f"throughput of a full run will be higher than this projection - treat "
            f"projected_hours as an upper bound."
        )
    atomic_write_text(out_path, json.dumps(throughput, indent=2))
    log_event(
        f"stage1 pilot: {len(jobs)} ligands in {elapsed:.1f}s "
        f"-> {throughput['ligands_per_hour']} ligands/h, ETA {throughput['projected_hours']} h "
        f"for {total_molecules} molecules",
        stage="stage1",
        **throughput,
    )
    if undersized:
        LOG.warning("stage1: %s", throughput["warning"])
    return throughput


# --------------------------------------------------------------------- driver
def run_stage1(
    cfg: Any,
    receptor_pdbqt: str | Path,
    box: Box,
    library_dir: str | Path,
    out_dir: str | Path,
    ledger: RunLedger,
    force: bool = False,
) -> dict[str, Any]:
    """Dock every library shard and write ``stage1/ranking.parquet``."""
    out = ensure_dir(out_dir)
    scores_dir = ensure_dir(out / "scores")
    s1 = cfg.stage1

    engine = select_engine(s1.engine, s1.engine_selection, prefer_fast=True)
    caps = engine.capabilities()
    if engine.name != s1.engine:
        LOG.warning("stage1: using engine %r (auto-selected) instead of %r", engine.name, s1.engine)

    params = build_dock_params(s1)
    # Validate the box against the compiled binary once, here, rather than
    # discovering it a million ligands into the run.
    if caps.max_box_edge is not None and max(box.size) >= caps.max_box_edge:
        from vcdd.docking.engines.vina_gpu import check_box, resolve_variant

        check_box(box, resolve_variant(params.get("variant")))

    common = {
        "receptor_pdbqt": str(receptor_pdbqt),
        "box_center": tuple(box.center),
        "box_size": tuple(box.size),
        "engine_name": engine.name,
        "params": params,
        "n_conformers": s1.n_conformers,
        "minimize": s1.minimize_conformer,
        "seed": s1.seed,
    }
    phash = param_hash({**params, "engine": engine.name, "n_conformers": s1.n_conformers,
                        "minimize": s1.minimize_conformer, "box": [list(box.center), list(box.size)]})

    index = load_shard_index(library_dir)
    if index.empty:
        raise RuntimeError("library index is empty; run `vcdd prep-library` first")

    my_shards = shard_indices(len(index), cfg.shard, cfg.nshards)
    index = index.iloc[my_shards]
    total_molecules = int(index["n_molecules"].sum())
    n_workers = resolve_workers(cfg.n_workers)
    batch_size = s1.gpu_batch_size if caps.batch_native else None

    LOG.info(
        "stage1: %d molecules in %d shard(s) | engine=%s variant=%s thread=%s | "
        "%d prep workers | gpu batch=%s",
        total_molecules, len(index), engine.name, params.get("variant"), params.get("thread"),
        n_workers, batch_size,
    )
    if caps.needs_gpu:
        LOG.info("stage1: engine %s requires a GPU (%s)", engine.name, caps.notes)
    if not caps.deterministic:
        LOG.warning(
            "stage1: engine %s is not bit-reproducible - Vina-GPU 2.1 exposes no --seed, so "
            "re-running yields slightly different scores and near-ties may reorder. The "
            "protocol is reproducible; the exact score is not.", engine.name,
        )

    # ---- optional surrogate active learning -------------------------------
    if s1.surrogate_al:
        from vcdd.screening.active_learning import run_active_learning

        return run_active_learning(
            cfg=cfg, common=common, index=index, out_dir=out, ledger=ledger,
            n_workers=n_workers, engine_name=engine.name, param_hash_value=phash,
            force=force, batch_size=batch_size,
        )

    # ---- pilot -------------------------------------------------------------
    throughput_path = out / "throughput.json"
    if s1.pilot_size > 0 and (force or not throughput_path.exists()):
        first_shard = pd.read_parquet(index.iloc[0]["path"])
        pilot_df = first_shard.head(min(s1.pilot_size, len(first_shard)))
        measure_throughput(_build_jobs(pilot_df, common), n_workers, total_molecules,
                           throughput_path, batch_size=batch_size)

    # ---- shards ------------------------------------------------------------
    t_start = time.time()
    n_done = 0
    n_failed = 0
    for _, shard_row in index.iterrows():
        shard_id = int(shard_row["shard"])
        shard_path = Path(shard_row["path"])
        out_path = scores_dir / f"shard_{shard_id:05d}.parquet"
        key = UnitKey("stage1", f"shard_{shard_id:05d}", hash_obj(str(shard_path)), phash)

        if not force and ledger.is_done(key) and out_path.exists():
            done_df = pd.read_parquet(out_path)
            n_done += len(done_df)
            LOG.info("stage1: shard %d already complete (%d molecules) - skipping", shard_id, len(done_df))
            continue

        ledger.mark_running(key)
        df = pd.read_parquet(shard_path)
        jobs = _build_jobs(df, common)
        t0 = time.time()
        rows = [{**row, "shard": shard_id}
                for row in execute_jobs(jobs, n_workers=n_workers, batch_size=batch_size)]
        shard_df = pd.DataFrame(rows)
        for col in SCORE_COLUMNS:
            if col not in shard_df.columns:
                shard_df[col] = None
        shard_df = shard_df[SCORE_COLUMNS]
        shard_df.to_parquet(out_path, index=False)

        ok_n = int(shard_df["score"].notna().sum())
        n_done += ok_n
        n_failed += len(shard_df) - ok_n
        ledger.mark_done(key, artifact=out_path, n_items=len(shard_df))
        ledger.sync()
        LOG.info(
            "stage1: shard %d done - %d/%d scored in %.1fs",
            shard_id, ok_n, len(shard_df), time.time() - t0,
        )

    # ---- ranking -----------------------------------------------------------
    ranking = collect_ranking(scores_dir, out / "ranking.parquet")
    elapsed = time.time() - t_start
    summary = {
        "engine": engine.name,
        "variant": params.get("variant"),
        "thread": params.get("thread"),
        "search_depth": params.get("search_depth"),
        "deterministic": caps.deterministic,
        "gpu_batch_size": batch_size,
        "n_molecules": total_molecules,
        "n_scored": int(len(ranking)),
        "n_failed": n_failed,
        "seconds": round(elapsed, 1),
        "best_score": float(ranking["score"].min()) if len(ranking) else None,
        "median_score": float(ranking["score"].median()) if len(ranking) else None,
        "ranking_path": str(out / "ranking.parquet"),
        "surrogate_al": False,
    }
    log_event(
        f"stage1 complete: {summary['n_scored']}/{total_molecules} scored in {elapsed:.1f}s "
        f"(best {summary['best_score']})",
        stage="stage1", **summary,
    )
    return summary


def collect_ranking(
    scores_dir: str | Path, out_path: str | Path, pattern: str = "shard_*.parquet"
) -> pd.DataFrame:
    """Concatenate per-chunk score tables into one ranking (best score first).

    ``pattern`` exists because the active-learning path checkpoints its slices as
    ``al_*.parquet`` rather than ``shard_*.parquet``; passing the wrong glob here
    silently yields an empty ranking.
    """
    frames = [pd.read_parquet(p) for p in sorted(Path(scores_dir).glob(pattern))]
    if not frames:
        empty = pd.DataFrame(columns=SCORE_COLUMNS + ["rank"])
        empty.to_parquet(out_path, index=False)
        return empty
    df = pd.concat(frames, ignore_index=True)
    scored = df[df["score"].notna()].copy()
    # Deterministic tie-break on mol_id so the ranking never depends on
    # scheduler order or shard completion order.
    scored = scored.sort_values(["score", "mol_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)
    scored.to_parquet(out_path, index=False)
    return scored


def load_ranking(stage1_dir: str | Path) -> pd.DataFrame:
    path = Path(stage1_dir) / "ranking.parquet"
    if not path.exists():
        raise FileNotFoundError(f"stage 1 ranking not found: {path}. Run `vcdd stage1` first.")
    return pd.read_parquet(path)
