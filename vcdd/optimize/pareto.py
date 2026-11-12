"""Multi-objective ranking for generated analogs.

Stage 4 has to reconcile objectives that genuinely conflict.  Adding polarity
to improve the predicted cell response usually costs affinity; growing a
molecule to fill the pocket costs QED and synthetic accessibility.  Collapsing
that straight into one number hides the trade-off and lets a single dominant
term decide the answer.

So two rankings are produced side by side:

``pareto_rank``
    Non-dominated sorting (NSGA-II fronts).  Rank 1 is the set of designs that
    nothing else beats on every objective at once.  This is the honest answer
    to "which analogs are worth looking at" -- it makes no claim about how much
    a unit of docking score is worth in units of cell objective.
``design_score``
    A weighted geometric mean of the same objectives, normalised to [0, 1] on
    **absolute** scales.  This is the tie-break and the sort key for the
    recommendation table, because a chemist needs an ordered list.

The geometric mean is deliberate: unlike a weighted sum it cannot be rescued by
one excellent term, so a design that is inert in the cell model does not reach
the top of the list on docking score alone.

Normalisation is absolute rather than batch-relative, matching the stage-3
objective.  A relative normalisation would make an analog's score depend on
which other analogs happened to be generated alongside it, so parents and
analogs -- and separate runs -- would not be comparable.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DESIGN_OBJECTIVES",
    "normalise_docking",
    "normalise_sa",
    "normalise_objectives",
    "design_score",
    "pareto_fronts",
    "crowding_distance",
]

#: column -> direction. ``max`` means larger is better.
DESIGN_OBJECTIVES: dict[str, str] = {
    "dock_score": "min",       # kcal/mol, more negative is better
    "cell_objective": "max",   # [0, 1] from stage 3
    "qed": "max",              # [0, 1]
    "sa_score": "min",         # 1 (easy) .. 10 (hard)
}

#: Docking scores are mapped linearly onto [0, 1] between these anchors.
#: -4 kcal/mol is around the noise floor for a drug-sized ligand; -12 is an
#: excellent Vina score. Anything outside is clipped rather than extrapolated.
DOCK_ANCHORS = (-4.0, -12.0)


def normalise_docking(score: np.ndarray | pd.Series) -> np.ndarray:
    """Map a Vina-style affinity to [0, 1], 0 at -4 kcal/mol, 1 at -12."""
    lo, hi = DOCK_ANCHORS
    v = np.asarray(score, dtype=float)
    return np.clip((lo - v) / (lo - hi), 0.0, 1.0)


def normalise_sa(sa: np.ndarray | pd.Series) -> np.ndarray:
    """Map Ertl-Schuffenhauer SA (1..10) to [0, 1], 1 = trivially makeable."""
    v = np.asarray(sa, dtype=float)
    return np.clip((10.0 - v) / 9.0, 0.0, 1.0)


def normalise_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """Return the [0, 1] normalised terms that feed the design score."""
    out = pd.DataFrame(index=df.index)
    if "dock_score" in df:
        out["docking"] = normalise_docking(df["dock_score"])
    if "cell_objective" in df:
        out["cell"] = np.clip(pd.to_numeric(df["cell_objective"], errors="coerce"), 0.0, 1.0)
    if "qed" in df:
        out["qed"] = np.clip(pd.to_numeric(df["qed"], errors="coerce"), 0.0, 1.0)
    if "sa_score" in df:
        out["sa"] = normalise_sa(df["sa_score"])
    return out


def design_score(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    floor: float = 1e-3,
) -> tuple[pd.Series, pd.DataFrame]:
    """Weighted geometric mean of the normalised objectives.

    Terms that are missing for a row (an analog that failed to dock, say) are
    dropped from *that row's* mean and the remaining weights renormalised, so a
    missing term neither zeroes the score nor silently counts as average. The
    number of terms actually used is returned alongside, because a score built
    from two terms is not the same evidence as one built from four.
    """
    terms = normalise_objectives(df)
    if terms.empty:
        return pd.Series(np.nan, index=df.index), terms

    w = {k: float(v) for k, v in weights.items() if k in terms.columns and float(v) > 0}
    if not w:
        return pd.Series(np.nan, index=df.index), terms

    cols = list(w)
    vals = terms[cols].to_numpy(dtype=float)
    wt = np.array([w[c] for c in cols], dtype=float)

    valid = np.isfinite(vals)
    safe = np.clip(np.nan_to_num(vals, nan=1.0), floor, 1.0)
    wsum = (valid * wt).sum(axis=1)
    logsum = (valid * wt * np.log(safe)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(wsum > 0, np.exp(logsum / np.where(wsum > 0, wsum, 1.0)), np.nan)

    terms = terms.copy()
    terms["n_terms_used"] = valid.sum(axis=1)
    return pd.Series(score, index=df.index, name="design_score"), terms


# --------------------------------------------------------------------------- #
# Non-dominated sorting
# --------------------------------------------------------------------------- #


def _to_maximisation(df: pd.DataFrame, objectives: Mapping[str, str]) -> np.ndarray:
    """Stack objective columns as a maximisation matrix, NaN -> worst value."""
    cols = []
    for name, direction in objectives.items():
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        v = -v if direction == "min" else v
        finite = v[np.isfinite(v)]
        worst = finite.min() - 1.0 if finite.size else 0.0
        cols.append(np.where(np.isfinite(v), v, worst))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def pareto_fronts(
    df: pd.DataFrame,
    objectives: Mapping[str, str] | None = None,
) -> pd.Series:
    """Assign each row an NSGA-II front index (1 = non-dominated).

    O(n^2) in the number of rows, which is the right complexity here: stage 4
    ranks hundreds to low thousands of analogs, not millions, and the exact
    fronts matter more than the constant factor.
    """
    objectives = {
        k: v for k, v in (objectives or DESIGN_OBJECTIVES).items() if k in df.columns
    }
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=int, name="pareto_rank")
    if not objectives:
        return pd.Series(np.ones(n, dtype=int), index=df.index, name="pareto_rank")

    mat = _to_maximisation(df, objectives)
    # a dominates b iff a >= b on every objective and a > b on at least one
    ge = (mat[:, None, :] >= mat[None, :, :] - 1e-12).all(axis=2)
    gt = (mat[:, None, :] > mat[None, :, :] + 1e-12).any(axis=2)
    dominates = ge & gt

    n_dominated_by = dominates.sum(axis=0)  # how many rows dominate each row
    ranks = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    counts = n_dominated_by.copy()
    front = 1
    while remaining.any():
        current = remaining & (counts <= 0)
        if not current.any():  # numerical safety valve: never loop forever
            ranks[remaining] = front
            break
        ranks[current] = front
        remaining &= ~current
        # rows dominated only by this front are freed for the next one
        counts = counts - dominates[current].sum(axis=0)
        front += 1
    return pd.Series(ranks, index=df.index, name="pareto_rank")


def crowding_distance(
    df: pd.DataFrame,
    objectives: Mapping[str, str] | None = None,
    ranks: Sequence[int] | pd.Series | None = None,
) -> pd.Series:
    """NSGA-II crowding distance within each front (higher = more distinctive).

    Used to break ties inside front 1 so the recommendation table spans the
    trade-off surface instead of returning twenty near-identical designs.
    """
    objectives = {
        k: v for k, v in (objectives or DESIGN_OBJECTIVES).items() if k in df.columns
    }
    n = len(df)
    out = np.zeros(n, dtype=float)
    if n == 0 or not objectives:
        return pd.Series(out, index=df.index, name="crowding_distance")
    if ranks is None:
        ranks = pareto_fronts(df, objectives)
    rank_arr = np.asarray(ranks, dtype=int)
    mat = _to_maximisation(df, objectives)

    for front in np.unique(rank_arr):
        idx = np.flatnonzero(rank_arr == front)
        if idx.size <= 2:
            out[idx] = np.inf
            continue
        for j in range(mat.shape[1]):
            order = idx[np.argsort(mat[idx, j], kind="mergesort")]
            span = mat[order[-1], j] - mat[order[0], j]
            out[order[0]] = out[order[-1]] = np.inf
            if span <= 0:
                continue
            for k in range(1, len(order) - 1):
                out[order[k]] += (mat[order[k + 1], j] - mat[order[k - 1], j]) / span
    return pd.Series(out, index=df.index, name="crowding_distance")
