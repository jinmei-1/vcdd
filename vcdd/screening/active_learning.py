"""Optional surrogate active learning for stage 1 (Deep-Docking style).

Default OFF. Turning it on trades *measured* recall for wall clock: instead of
docking every molecule, it docks a random seed set, fits a cheap ligand-only
regressor on Morgan fingerprints, and docks only the molecules the surrogate
ranks highly, iterating a few times.

The honest bookkeeping this module insists on:

* A **held-out random slice is always docked** and never used for training, so
  recall of the top-k is *measured* on this library rather than quoted from a
  paper. ``active_learning.json`` records it.
* Every score in ``ranking.parquet`` is a **real docking score**. Surrogate
  predictions decide *what to dock*, never what to report.
* Molecules that were never docked are written to ``not_docked.parquet`` with
  their surrogate prediction, so the coverage gap is explicit rather than
  looking like a completed screen.

References for the approach: Gentile et al., Deep Docking (ACS Cent. Sci. 2020;
Nat. Protoc. 2022); Graff et al., MolPAL (Chem. Sci. 2021).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vcdd.runtime.hashing import hash_obj
from vcdd.runtime.ledger import RunLedger, UnitKey
from vcdd.runtime.logging import get_logger, log_event
from vcdd.runtime.paths import atomic_write_text, ensure_dir

LOG = get_logger()


# ------------------------------------------------------------------ features
def morgan_matrix(smiles: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Binary Morgan fingerprints as a dense float32 matrix."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        out[i] = fp.astype(np.float32)
    return out


def _fit_surrogate(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    if kind == "gbt":
        model = GradientBoostingRegressor(random_state=seed)
    else:
        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    model.fit(x, y)
    return model


# --------------------------------------------------------------------- driver
def run_active_learning(
    cfg: Any,
    common: dict[str, Any],
    index: pd.DataFrame,
    out_dir: str | Path,
    ledger: RunLedger,
    n_workers: int,
    engine_name: str,
    param_hash_value: str,
    force: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Iterative surrogate-guided stage 1. Returns the same summary as the brute force path.

    ``batch_size`` is passed straight through to :func:`vcdd.screening.stage1.execute_jobs`
    so a GPU engine sees whole slices, not one process per molecule.
    """
    from vcdd.screening.stage1 import SCORE_COLUMNS, _build_jobs, collect_ranking, execute_jobs

    out = ensure_dir(out_dir)
    scores_dir = ensure_dir(out / "scores")
    s1 = cfg.stage1
    rng = np.random.default_rng(s1.seed)

    pool = pd.concat([pd.read_parquet(p) for p in index["path"]], ignore_index=True)
    pool = pool.sort_values("mol_id", kind="mergesort").reset_index(drop=True)
    n_total = len(pool)
    LOG.info("stage1[AL]: pool of %d molecules, %d iterations", n_total, s1.al_iterations)

    def _dock_subset(subset: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Dock a slice, checkpointing it as one ledger unit."""
        path = scores_dir / f"al_{tag}.parquet"
        key = UnitKey("stage1", f"al_{tag}", hash_obj(sorted(subset["mol_id"].tolist())), param_hash_value)
        if not force and ledger.is_done(key) and path.exists():
            LOG.info("stage1[AL]: %s already docked (%d) - skipping", tag, len(subset))
            return pd.read_parquet(path)
        ledger.mark_running(key)
        jobs = _build_jobs(subset, common)
        t0 = time.time()
        # Route through the same dispatcher as the brute-force path: with a
        # batch-native GPU engine, an AL slice is exactly the shape that must
        # NOT be fanned out one-ligand-per-process.
        rows = [{**row, "shard": -1}
                for row in execute_jobs(jobs, n_workers=n_workers, batch_size=batch_size)]
        df = pd.DataFrame(rows)
        for col in SCORE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[SCORE_COLUMNS]
        df.to_parquet(path, index=False)
        ledger.mark_done(key, artifact=path, n_items=len(df))
        ledger.sync()
        LOG.info("stage1[AL]: %s docked %d molecules in %.1fs", tag, len(df), time.time() - t0)
        return df

    # ---- held-out slice: never trained on, used to measure recall ----------
    n_holdout = max(20, int(round(s1.al_holdout_fraction * n_total)))
    n_holdout = min(n_holdout, max(0, n_total - 1))
    holdout_idx = rng.choice(n_total, size=n_holdout, replace=False) if n_holdout else np.array([], dtype=int)
    holdout = pool.iloc[sorted(holdout_idx)]
    remaining = pool.drop(index=holdout.index)

    holdout_scores = _dock_subset(holdout, "holdout") if n_holdout else pd.DataFrame(columns=SCORE_COLUMNS)

    # ---- iteration 0: random seed set --------------------------------------
    n_seed = max(20, int(round(s1.al_initial_fraction * n_total)))
    n_seed = min(n_seed, len(remaining))
    seed_pos = rng.choice(len(remaining), size=n_seed, replace=False)
    docked = _dock_subset(remaining.iloc[sorted(seed_pos)], "iter00")
    undocked = remaining[~remaining["mol_id"].isin(docked["mol_id"])]

    history: list[dict[str, Any]] = [{"iteration": 0, "n_docked": int(len(docked)), "kind": "random_seed"}]

    # ---- surrogate-guided iterations ---------------------------------------
    last_pred: dict[str, float] = {}
    n_select = max(20, int(round(s1.al_select_fraction * n_total)))
    for it in range(1, s1.al_iterations + 1):
        train = docked[docked["score"].notna()]
        if len(train) < 20 or undocked.empty:
            LOG.warning("stage1[AL]: stopping at iteration %d (train=%d, pool=%d)",
                        it, len(train), len(undocked))
            break
        x_train = morgan_matrix(train["smiles"].tolist())
        y_train = train["score"].to_numpy(dtype=float)
        model = _fit_surrogate(x_train, y_train, s1.al_model, s1.seed + it)

        x_pool = morgan_matrix(undocked["smiles"].tolist())
        pred = model.predict(x_pool)
        last_pred = dict(zip(undocked["mol_id"].tolist(), (float(v) for v in pred)))
        take = min(n_select, len(undocked))
        # Lower predicted score = better; argsort ascending, deterministic ties.
        order = np.lexsort((undocked["mol_id"].to_numpy(), pred))[:take]
        chosen = undocked.iloc[sorted(order)]

        new_scores = _dock_subset(chosen, f"iter{it:02d}")
        docked = pd.concat([docked, new_scores], ignore_index=True)
        undocked = undocked[~undocked["mol_id"].isin(docked["mol_id"])]

        # Surrogate quality on the *previous* holdout, measured not assumed.
        metrics = _surrogate_metrics(model, holdout_scores)
        history.append({
            "iteration": it, "n_docked": int(len(new_scores)), "kind": "surrogate_selected",
            "train_size": int(len(train)), "pool_remaining": int(len(undocked)), **metrics,
        })
        LOG.info("stage1[AL]: iteration %d docked %d | holdout spearman=%s recall@1%%=%s",
                 it, len(new_scores), metrics.get("holdout_spearman"), metrics.get("holdout_recall_top1pct"))

    # ---- assemble ----------------------------------------------------------
    all_docked = pd.concat([holdout_scores, docked], ignore_index=True) if len(holdout_scores) else docked
    all_docked = all_docked.drop_duplicates(subset="mol_id", keep="first")
    # NB: the AL slices are checkpointed as ``al_*.parquet``; the brute-force glob
    # (``shard_*.parquet``) would match nothing and hand back an empty ranking.
    ranking = collect_ranking(scores_dir, out / "ranking.parquet", pattern="al_*.parquet")

    never = pool[~pool["mol_id"].isin(all_docked["mol_id"])].copy()
    if len(never):
        never["surrogate_predicted_score"] = never["mol_id"].map(last_pred).astype(float)
        never[["mol_id", "smiles", "surrogate_predicted_score"]].to_parquet(
            out / "not_docked.parquet", index=False
        )

    fraction = len(all_docked) / max(1, n_total)
    report = {
        "enabled": True,
        "model": s1.al_model,
        "iterations": s1.al_iterations,
        "n_total": n_total,
        "n_docked": int(len(all_docked)),
        "n_never_docked": int(len(never)),
        "fraction_docked": round(fraction, 4),
        "speedup_vs_brute_force": round(1.0 / fraction, 2) if fraction else None,
        "history": history,
        "caveat": (
            "Scores are real docking scores for docked molecules only. Molecules in "
            "not_docked.parquet were never evaluated: the ranking is a ranking of the docked "
            "subset, not of the library. Check holdout_recall_top1pct before trusting it."
        ),
    }
    atomic_write_text(out / "active_learning.json", json.dumps(report, indent=2))
    log_event(
        f"stage1[AL] complete: docked {report['n_docked']}/{n_total} "
        f"({report['speedup_vs_brute_force']}x fewer dockings)",
        stage="stage1", **{k: v for k, v in report.items() if k != "history"},
    )

    return {
        "engine": engine_name,
        "variant": s1.variant,
        "thread": s1.thread,
        "search_depth": s1.search_depth,
        "gpu_batch_size": batch_size,
        "n_molecules": n_total,
        "n_scored": int(len(ranking)),
        "n_failed": int(len(all_docked) - len(ranking)),
        "seconds": None,
        "best_score": float(ranking["score"].min()) if len(ranking) else None,
        "median_score": float(ranking["score"].median()) if len(ranking) else None,
        "ranking_path": str(out / "ranking.parquet"),
        "surrogate_al": True,
        "active_learning": report,
    }


def _surrogate_metrics(model: Any, holdout: pd.DataFrame) -> dict[str, Any]:
    """Spearman + top-k recall of the surrogate on the never-trained-on slice."""
    if holdout is None or holdout.empty:
        return {}
    valid = holdout[holdout["score"].notna()]
    if len(valid) < 10:
        return {}
    x = morgan_matrix(valid["smiles"].tolist())
    pred = model.predict(x)
    true = valid["score"].to_numpy(dtype=float)

    from scipy.stats import spearmanr

    rho = float(spearmanr(pred, true).statistic)
    k = max(1, int(round(0.01 * len(true))))
    top_true = set(np.argsort(true, kind="mergesort")[:k].tolist())
    top_pred = set(np.argsort(pred, kind="mergesort")[:k].tolist())
    recall = len(top_true & top_pred) / len(top_true)
    return {
        "holdout_n": int(len(valid)),
        "holdout_spearman": round(rho, 3),
        "holdout_recall_top1pct": round(recall, 3),
    }
