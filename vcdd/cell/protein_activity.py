"""Transcription-factor and pathway activity inference (decoupleR-style).

The user asked for compounds that drive *genes or proteins* to a high level.
No compound-perturbation virtual-cell model predicts protein abundance, so this
module supplies the closest defensible readout: footprint-based **activity**
inferred from the predicted transcriptome.  A TF whose activating targets go up
and repressed targets go down is inferred to be more active, whether or not its
own transcript moved.

This is a *transform*, not a predictor -- it turns a :class:`Signature` into
regulator activities.  It is deliberately not registered as a stage-3 backend,
because on its own it predicts nothing.

Two networks ship in the bundle:

CollecTRI
    signed TF -> target regulon (``mor`` = +1 activation, -1 repression).
PROGENy
    14 pathway footprints with continuous, responsiveness-derived weights.

Methods mirror decoupleR: ``ulm`` (univariate linear model t-value; the
default), ``wsum`` (weighted sum with permutation normalisation) and ``wmean``.

Important caveat to carry into any interpretation: an inferred activity is a
statement about the *downstream footprint*, not a measurement of protein level,
phosphorylation or localisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from . import assets

__all__ = [
    "ActivityResult",
    "build_network_matrix",
    "run_ulm",
    "run_wsum",
    "run_wmean",
    "infer_activity",
    "tf_activity",
    "pathway_activity",
]

Method = Literal["ulm", "wsum", "wmean"]

#: Regulons with fewer measured targets than this are dropped: a t-value from
#: three points is noise dressed as inference.
MIN_TARGETS = 5


@dataclass
class ActivityResult:
    """Regulator x sample activity scores."""

    scores: pd.DataFrame  # index = regulator, columns = sample ids
    method: str
    network: str
    n_regulators: int
    n_genes_used: int
    coverage: float  # fraction of network genes present in the input
    dropped_small: int = 0

    def top(self, sample: str, n: int = 10, direction: str = "up") -> list[tuple[str, float]]:
        if sample not in self.scores.columns:
            return []
        col = self.scores[sample].dropna().sort_values(ascending=direction != "up")
        return list(col.head(n).items())

    def to_long(self) -> pd.DataFrame:
        out = self.scores.reset_index().melt(
            id_vars=self.scores.index.name or "index",
            var_name="mol_id",
            value_name="activity",
        )
        out = out.rename(columns={out.columns[0]: "regulator"})
        out["method"] = self.method
        out["network"] = self.network
        return out


def build_network_matrix(
    net: pd.DataFrame,
    genes: Sequence[str],
    source_col: str,
    target_col: str,
    weight_col: str,
    min_targets: int = MIN_TARGETS,
) -> tuple[np.ndarray, list[str], int]:
    """Dense ``(n_genes, n_regulators)`` weight matrix restricted to ``genes``.

    Regulators are dropped when too few of their targets are measured, which is
    the dominant failure mode when scoring against the 978-gene landmark space.
    """
    index = {g: i for i, g in enumerate(genes)}
    net = net[net[target_col].isin(index)]
    if net.empty:
        return np.zeros((len(genes), 0), dtype=np.float32), [], 0

    counts = net.groupby(source_col)[target_col].nunique()
    keep = counts[counts >= min_targets].index
    dropped = int(len(counts) - len(keep))
    net = net[net[source_col].isin(keep)]
    regulators = sorted(net[source_col].unique().tolist())
    if not regulators:
        return np.zeros((len(genes), 0), dtype=np.float32), [], dropped

    reg_index = {r: j for j, r in enumerate(regulators)}
    mat = np.zeros((len(genes), len(regulators)), dtype=np.float32)
    rows = net[target_col].map(index).to_numpy()
    cols = net[source_col].map(reg_index).to_numpy()
    vals = net[weight_col].astype(np.float32).to_numpy()
    # Duplicate (regulator, target) edges are summed, matching decoupleR.
    np.add.at(mat, (rows, cols), vals)
    return mat, regulators, dropped


def run_ulm(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Univariate linear model t-values, vectorised over regulators/samples.

    For each regulator the network weight vector is the predictor and the
    signature is the response; the slope's t-statistic is the activity score.
    """
    n = mat.shape[0]
    if n < 3:
        return np.full((mat.shape[1], y.shape[1]), np.nan, dtype=np.float32)
    xc = mat - mat.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    sxx = (xc**2).sum(axis=0)  # (n_reg,)
    sxy = xc.T @ yc  # (n_reg, n_samples)
    syy = (yc**2).sum(axis=0)  # (n_samples,)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = sxy / sxx[:, None]
        sse = syy[None, :] - beta * sxy
        sse = np.clip(sse, 0.0, None)
        se = np.sqrt(sse / ((n - 2) * sxx[:, None]))
        t = beta / se
    t[~np.isfinite(t)] = np.nan
    return t.astype(np.float32)


def run_wsum(
    mat: np.ndarray, y: np.ndarray, n_perm: int = 100, seed: int = 42
) -> np.ndarray:
    """Weighted sum, z-normalised against permuted gene labels."""
    obs = mat.T @ y
    if n_perm <= 0:
        return obs.astype(np.float32)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, *obs.shape), dtype=np.float32)
    idx = np.arange(mat.shape[0])
    for k in range(n_perm):
        rng.shuffle(idx)
        null[k] = mat[idx].T @ y
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def run_wmean(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weighted mean: weighted sum divided by total absolute weight."""
    denom = np.abs(mat).sum(axis=0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (mat.T @ y) / denom
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def infer_activity(
    values: np.ndarray,
    genes: Sequence[str],
    net: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_col: str,
    sample_ids: Sequence[str],
    method: Method = "ulm",
    network_name: str = "network",
    min_targets: int = MIN_TARGETS,
    seed: int = 42,
) -> ActivityResult:
    """Infer regulator activity for one or more signatures.

    ``values`` is ``(n_genes,)`` or ``(n_genes, n_samples)`` aligned to ``genes``.
    """
    y = np.asarray(values, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != len(genes):
        raise ValueError(f"values has {y.shape[0]} rows but {len(genes)} genes given")

    mat, regulators, dropped = build_network_matrix(
        net, genes, source_col, target_col, weight_col, min_targets
    )
    if not regulators:
        return ActivityResult(
            pd.DataFrame(index=pd.Index([], name="regulator"), columns=list(sample_ids)),
            method,
            network_name,
            0,
            0,
            0.0,
            dropped,
        )

    if method == "ulm":
        scores = run_ulm(mat, y)
    elif method == "wsum":
        scores = run_wsum(mat, y, seed=seed)
    elif method == "wmean":
        scores = run_wmean(mat, y)
    else:  # pragma: no cover - guarded by the config Literal
        raise ValueError(f"unknown activity method '{method}'")

    n_used = int((np.abs(mat).sum(axis=1) > 0).sum())
    coverage = n_used / max(1, net[target_col].nunique())
    df = pd.DataFrame(scores, index=pd.Index(regulators, name="regulator"), columns=list(sample_ids))
    return ActivityResult(df, method, network_name, len(regulators), n_used, coverage, dropped)


def tf_activity(
    values: np.ndarray,
    genes: Sequence[str],
    sample_ids: Sequence[str],
    method: Method = "ulm",
    min_targets: int = MIN_TARGETS,
    seed: int = 42,
) -> ActivityResult:
    """Infer TF activity from CollecTRI."""
    return infer_activity(
        values,
        genes,
        assets.load_collectri(),
        source_col="tf",
        target_col="target",
        weight_col="mor",
        sample_ids=sample_ids,
        method=method,
        network_name="collectri",
        min_targets=min_targets,
        seed=seed,
    )


def pathway_activity(
    values: np.ndarray,
    genes: Sequence[str],
    sample_ids: Sequence[str],
    method: Method = "ulm",
    min_targets: int = MIN_TARGETS,
    seed: int = 42,
) -> ActivityResult:
    """Infer PROGENy pathway activity."""
    return infer_activity(
        values,
        genes,
        assets.load_progeny(),
        source_col="pathway",
        target_col="gene_symbol",
        weight_col="weight",
        sample_ids=sample_ids,
        method=method,
        network_name="progeny",
        min_targets=min_targets,
        seed=seed,
    )
