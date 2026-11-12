"""Mine transferable SAR rules from the generated analog set.

A list of good analogs is a result; a rule is advice.  "Replacing aryl-Cl with
aryl-CF3 improved the predicted cell objective in 7 of 9 parents where it was
applicable" tells a chemist what to try on the *next* molecule, which the
per-compound table does not.

The unit of analysis is the ``transformation`` label carried by every
:class:`~vcdd.optimize.generators.base.Proposal`.  Two guards keep the output
from becoming noise:

* **Per-parent collapse before aggregation.** A transform that matches eight
  aromatic positions on one molecule would otherwise contribute eight
  correlated observations and look far better supported than it is.  Each
  parent contributes exactly one number: the median delta over its analogs.
* **A minimum number of distinct parents** (``stage4.sar_min_parents``). A rule
  seen in one molecule is an anecdote about that molecule.

The reported p-value is an exact two-sided sign test on the number of parents
improved.  It asks only "is the direction consistent across parents", makes no
distributional assumption, and is *not* corrected for the number of transforms
tested -- with 20-30 transforms these are screening statistics for ranking
hypotheses, not confirmatory tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["SARRule", "sign_test", "mine_sar", "explode_transformations", "format_rule"]


@dataclass
class SARRule:
    """One aggregated transformation rule."""

    transformation: str
    metric: str
    n_analogs: int
    n_parents: int
    n_parents_improved: int
    median_delta: float
    mean_delta: float
    iqr_delta: float
    frac_parents_improved: float
    p_sign_test: float
    verdict: str

    def to_row(self) -> dict:
        return {
            "transformation": self.transformation,
            "metric": self.metric,
            "n_analogs": self.n_analogs,
            "n_parents": self.n_parents,
            "n_parents_improved": self.n_parents_improved,
            "frac_parents_improved": round(self.frac_parents_improved, 3),
            "median_delta": self.median_delta,
            "mean_delta": self.mean_delta,
            "iqr_delta": self.iqr_delta,
            "p_sign_test": self.p_sign_test,
            "verdict": self.verdict,
        }


def sign_test(n_improved: int, n_total: int) -> float:
    """Exact two-sided binomial test against p = 0.5. Ties count as failures."""
    if n_total <= 0:
        return float("nan")
    k = min(n_improved, n_total - n_improved)
    tail = sum(math.comb(n_total, i) for i in range(0, k + 1)) / (2.0**n_total)
    return float(min(1.0, 2.0 * tail))


def explode_transformations(analogs: pd.DataFrame) -> pd.DataFrame:
    """One row per (analog, individual transformation).

    GA lineages chain several edits, so their ``transformation`` label is a
    compound string.  Mining those verbatim would produce one unique rule per
    lineage and nothing would ever reach the parent threshold.  Exploding the
    ``transformations`` list recovers the individual edits, at the cost of
    attributing the whole delta to each edit in the chain -- so a rule built
    mostly from long lineages is weaker evidence than its counts suggest.
    ``mean_lineage_length`` in the output records that.
    """
    rows = []
    for rec in analogs.to_dict("records"):
        ops = rec.get("transformations")
        if ops is None or (isinstance(ops, float) and pd.isna(ops)):
            ops = []
        if isinstance(ops, str):
            ops = [ops]
        ops = [str(o) for o in list(ops) if str(o).strip()]
        if not ops:
            label = str(rec.get("transformation") or "").strip()
            ops = [label] if label else []
        if not ops:
            continue
        for op in ops:
            rows.append({**{k: v for k, v in rec.items() if k != "transformations"},
                         "op": op, "lineage_length": len(ops)})
    return pd.DataFrame(rows)


def mine_sar(
    analogs: pd.DataFrame,
    metrics: Sequence[str] = ("delta_cell_objective", "delta_dock_score", "delta_design_score"),
    min_parents: int = 2,
    higher_is_better: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Aggregate per-transformation effects across parents.

    ``analogs`` needs ``parent_id``, a ``transformation`` label (or a
    ``transformations`` list), and the delta columns named in ``metrics``.
    """
    if analogs.empty:
        return pd.DataFrame(columns=[f.name for f in SARRule.__dataclass_fields__.values()])

    direction = {
        "delta_cell_objective": True,
        "delta_dock_score": False,  # kcal/mol: more negative is better
        "delta_design_score": True,
        "delta_gene_of_interest": True,
        "delta_qed": True,
        "delta_sa_score": False,
        **(higher_is_better or {}),
    }

    long = explode_transformations(analogs)
    if long.empty:
        return pd.DataFrame(columns=[f.name for f in SARRule.__dataclass_fields__.values()])

    out: list[dict] = []
    for metric in metrics:
        if metric not in long.columns:
            continue
        up_is_good = direction.get(metric, True)
        sub = long[["op", "parent_id", metric, "lineage_length"]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            continue

        # one observation per parent, then aggregate over parents
        per_parent = (
            sub.groupby(["op", "parent_id"], as_index=False)
            .agg(delta=(metric, "median"), n=(metric, "size"),
                 lineage=("lineage_length", "mean"))
        )
        for op, grp in per_parent.groupby("op"):
            n_parents = int(grp["parent_id"].nunique())
            if n_parents < min_parents:
                continue
            deltas = grp["delta"].to_numpy(dtype=float)
            improved = int((deltas > 0).sum() if up_is_good else (deltas < 0).sum())
            q1, q3 = np.percentile(deltas, [25, 75])
            rule = SARRule(
                transformation=op,
                metric=metric,
                n_analogs=int(grp["n"].sum()),
                n_parents=n_parents,
                n_parents_improved=improved,
                median_delta=float(np.median(deltas)),
                mean_delta=float(np.mean(deltas)),
                iqr_delta=float(q3 - q1),
                frac_parents_improved=improved / n_parents,
                p_sign_test=sign_test(improved, n_parents),
                verdict="",
            )
            rule.verdict = format_rule(rule, up_is_good)
            row = rule.to_row()
            row["mean_lineage_length"] = round(float(grp["lineage"].mean()), 2)
            out.append(row)

    if not out:
        return pd.DataFrame(columns=[*SARRule.__dataclass_fields__, "mean_lineage_length"])
    res = pd.DataFrame(out)
    # Rank by consistency first, then effect size: a large median from two
    # parents is a weaker recommendation than a smaller one from ten.
    res["_abs"] = res["median_delta"].abs()
    res = res.sort_values(
        ["metric", "frac_parents_improved", "n_parents", "_abs"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop(columns="_abs").reset_index(drop=True)
    return res


_METRIC_LABEL = {
    "delta_cell_objective": "the virtual-cell objective",
    "delta_dock_score": "the docking score",
    "delta_design_score": "the overall design score",
    "delta_gene_of_interest": "the gene of interest",
    "delta_qed": "QED",
    "delta_sa_score": "synthetic accessibility",
}


def format_rule(rule: SARRule, up_is_good: bool) -> str:
    """Render one rule as a sentence a chemist can act on."""
    label = _METRIC_LABEL.get(rule.metric, rule.metric)
    direction = "improved" if rule.frac_parents_improved >= 0.5 else "hurt"
    n_shown = (
        rule.n_parents_improved
        if direction == "improved"
        else rule.n_parents - rule.n_parents_improved
    )
    sign = "+" if rule.median_delta >= 0 else ""
    strength = (
        "consistent" if rule.frac_parents_improved >= 0.8 and rule.n_parents >= 3
        else "suggestive" if rule.frac_parents_improved >= 0.6
        else "inconsistent"
    )
    return (
        f"{rule.transformation}: {direction} {label} in {n_shown}/{rule.n_parents} "
        f"parents (median {sign}{rule.median_delta:.4g}, sign-test p={rule.p_sign_test:.3g}) "
        f"[{strength}]"
    )
