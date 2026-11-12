"""Consensus scoring across several scoring functions.

Why consensus at all: individual empirical scoring functions correlate poorly
with measured affinity and disagree with each other, but their *errors* are only
partly shared, so rank-level agreement is more informative than any single
number. Consensus is a variance-reduction trick, not an accuracy fix - a
compound ranked first by consensus is still a hypothesis.

Two combination rules are provided because they fail differently:

* **z-score mean** keeps the magnitude of the separation, so one scorer that
  reports a compound as wildly better can pull it up. Sensitive to outliers and
  to differences in each scorer's spread.
* **Borda count** (mean rank) throws magnitude away and keeps only the ordering.
  Robust to outliers, blind to "this one is *much* better".

``both`` reports each separately plus their mean rank, and always reports the
inter-scorer agreement so a low-confidence consensus is visible rather than
hidden behind a single tidy number.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def zscore(values: pd.Series) -> pd.Series:
    """Standardise, tolerating NaN and a zero-variance column."""
    v = pd.to_numeric(values, errors="coerce")
    mu = v.mean(skipna=True)
    sd = v.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - mu) / sd


def rank_ascending(values: pd.Series) -> pd.Series:
    """Competition-free average ranks, best (most negative) first.

    NaN gets the worst possible rank rather than being dropped, so a scorer that
    failed on a compound penalises it instead of silently promoting it.
    """
    v = pd.to_numeric(values, errors="coerce")
    ranks = v.rank(method="average", ascending=True, na_option="bottom")
    return ranks


def zscore_consensus(df: pd.DataFrame, score_cols: list[str]) -> pd.Series:
    """Mean of per-scorer z-scores. Lower is better."""
    if not score_cols:
        return pd.Series(np.full(len(df), np.nan), index=df.index)
    z = pd.DataFrame({c: zscore(df[c]) for c in score_cols})
    return z.mean(axis=1, skipna=True)


def borda_consensus(df: pd.DataFrame, score_cols: list[str]) -> pd.Series:
    """Mean rank across scorers (Borda count). Lower is better."""
    if not score_cols:
        return pd.Series(np.full(len(df), np.nan), index=df.index)
    r = pd.DataFrame({c: rank_ascending(df[c]) for c in score_cols})
    return r.mean(axis=1)


def scorer_agreement(df: pd.DataFrame, score_cols: list[str]) -> dict[str, Any]:
    """Mean pairwise Spearman rho between scoring functions.

    This is the number that tells you whether the consensus means anything. Two
    scorers correlating at 0.1 are not corroborating each other.
    """
    from scipy.stats import spearmanr

    pairs: dict[str, float] = {}
    usable = [c for c in score_cols if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3]
    for a, b in combinations(usable, 2):
        va = pd.to_numeric(df[a], errors="coerce")
        vb = pd.to_numeric(df[b], errors="coerce")
        mask = va.notna() & vb.notna()
        if mask.sum() < 3:
            continue
        rho = spearmanr(va[mask], vb[mask]).statistic
        if np.isfinite(rho):
            pairs[f"{a}~{b}"] = round(float(rho), 3)
    mean_rho = round(float(np.mean(list(pairs.values()))), 3) if pairs else None
    return {"pairwise_spearman": pairs, "mean_spearman": mean_rho, "n_scorers": len(usable)}


def drop_duplicate_scores(
    df: pd.DataFrame, score_cols: list[str], tol: float = 0.02
) -> tuple[list[str], dict[str, str]]:
    """Remove columns that are numerically the same opinion twice.

    This is not hypothetical: rescoring a Vina-docked pose with the Vina
    function reproduces the docking score to three decimals. Leaving both in the
    consensus silently double-weights one scoring function and makes the
    inter-scorer agreement look better than it is.

    Returns the columns to keep and a ``dropped -> duplicate_of`` map.
    """
    keep: list[str] = []
    dropped: dict[str, str] = {}
    for col in score_cols:
        v = pd.to_numeric(df[col], errors="coerce")
        duplicate_of = None
        for kept in keep:
            w = pd.to_numeric(df[kept], errors="coerce")
            mask = v.notna() & w.notna()
            if mask.sum() == 0:
                continue
            if float((v[mask] - w[mask]).abs().mean()) <= tol:
                duplicate_of = kept
                break
        if duplicate_of is None:
            keep.append(col)
        else:
            dropped[col] = duplicate_of
    return keep, dropped


def build_consensus(
    df: pd.DataFrame,
    score_cols: list[str],
    method: str = "both",
    min_agreement: float = 0.3,
    id_col: str = "mol_id",
    dedupe: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add consensus columns and return ``(ranked_df, agreement_report)``.

    All ``score_cols`` must already be *lower-is-better*. Higher-is-better
    quantities - Boltz ``binding_confidence``, for instance - must be negated by
    the caller before they get here.
    """
    out = df.copy()
    present = [c for c in score_cols if c in out.columns]
    duplicates: dict[str, str] = {}
    if dedupe and len(present) > 1:
        present, duplicates = drop_duplicate_scores(out, present)
    agreement = scorer_agreement(out, present)
    agreement["scorers_in_consensus"] = present
    if duplicates:
        agreement["dropped_duplicates"] = duplicates

    out["consensus_zscore"] = zscore_consensus(out, present)
    out["consensus_borda"] = borda_consensus(out, present)

    if method == "zscore":
        primary = out["consensus_zscore"]
    elif method == "borda":
        primary = out["consensus_borda"]
    else:  # both: rank-average the two consensus rules themselves
        primary = (
            rank_ascending(out["consensus_zscore"]) + rank_ascending(out["consensus_borda"])
        ) / 2.0
    out["consensus_score"] = primary

    # Per-compound disagreement: spread of that compound's per-scorer ranks.
    if len(present) >= 2:
        ranks = pd.DataFrame({c: rank_ascending(out[c]) for c in present})
        out["rank_spread"] = ranks.max(axis=1) - ranks.min(axis=1)
        out["rank_std"] = ranks.std(axis=1, ddof=0)
    else:
        out["rank_spread"] = 0.0
        out["rank_std"] = 0.0

    mean_rho = agreement.get("mean_spearman")
    low_conf = mean_rho is not None and mean_rho < min_agreement
    out["low_confidence_consensus"] = bool(low_conf)
    agreement["min_agreement_threshold"] = min_agreement
    agreement["low_confidence"] = bool(low_conf)
    if low_conf:
        agreement["interpretation"] = (
            f"Scoring functions agree at mean Spearman {mean_rho} (< {min_agreement}). "
            "The consensus ordering is weakly supported; treat the shortlist as a set, not a ranking."
        )
    elif mean_rho is not None:
        agreement["interpretation"] = (
            f"Scoring functions agree at mean Spearman {mean_rho}; the consensus is meaningful "
            "as a triage ordering but is still not a measure of affinity."
        )
    else:
        agreement["interpretation"] = "Fewer than two usable scoring functions: no consensus computed."

    sort_cols = ["consensus_score", id_col] if id_col in out.columns else ["consensus_score"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out["consensus_rank"] = range(1, len(out) + 1)
    return out, agreement


def rank_correlation_with_stage1(df: pd.DataFrame, stage1_col: str = "stage1_score") -> float | None:
    """How much stage 2 reordered stage 1. Low values mean stage 2 earned its cost."""
    from scipy.stats import spearmanr

    if stage1_col not in df.columns or "consensus_score" not in df.columns:
        return None
    a = pd.to_numeric(df[stage1_col], errors="coerce")
    b = pd.to_numeric(df["consensus_score"], errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return None
    rho = spearmanr(a[mask], b[mask]).statistic
    return round(float(rho), 3) if np.isfinite(rho) else None
