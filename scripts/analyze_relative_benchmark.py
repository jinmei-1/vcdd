#!/usr/bin/env python
"""Score the scoring functions: can stage 1 / stage 2 make relative calls?

The campaign's claim is relative ("better than the positive control"), so the
question this answers is whether the scores rank *known* binders correctly.
Inputs are the benchmark run's own outputs:

* benchmark/stage1/ranking.parquet          - Vina-GPU docking
* benchmark/stage2/scores.parquet           - Boltz-2 confidence + affinity
* benchmark/known_inhibitors.csv            - measured IC50 (the ground truth)

Reported per score: Spearman rho against measured pIC50 (actives only),
ROC-AUC and Mann-Whitney U for actives vs the >10 uM negatives, the tier
medians, and where AZD7648 lands. A score that cannot put a 0.6 nM compound
above micromolar negatives cannot support a relative claim.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
ACTIVE_TIERS = ("potent", "mid", "weak")


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx, ry = x.rank(), y.rank()
    d2 = ((rx - ry) ** 2).sum()
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    # two-sided p via the t approximation, adequate for n ~ 45
    if abs(rho) >= 1:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # Student-t survival function is not in the stdlib; use the normal limit
    # for n > 30 and report the p as approximate otherwise.
    p = math.erfc(abs(t) / math.sqrt(2)) if n > 30 else float("nan")
    return rho, p


def roc_auc(pos: pd.Series, neg: pd.Series) -> float:
    """P(score of a random active is stronger) - lower value = stronger."""
    wins = sum(1 for a in pos for b in neg if a < b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def main() -> None:
    truth = pd.read_csv(BENCH / "known_inhibitors.csv")
    truth["mol_id"] = truth["mol_id"].astype(str)
    frames = {"measured": truth.set_index("mol_id")}
    for label, path in (("dock", ROOT / "dnapkcs_benchmark/stage1/ranking.parquet"),
                        ("stage2", ROOT / "dnapkcs_benchmark/stage2/scores.parquet")):
        if path.exists():
            frames[label] = pd.read_parquet(path).set_index("mol_id")
        else:
            print(f"! missing {path} - run the benchmark first")
            return
    data = frames["measured"].join(frames["dock"][["score"]].rename(columns={"score": "dock"}), how="left")
    s2 = frames["stage2"]
    for col in ("binding_confidence", "iptm", "affinity_pred_value", "affinity_probability_binary"):
        data[col] = s2[col] if col in s2.columns else float("nan")
    data.to_csv(BENCH / "benchmark_results.csv")

    act = data[data.tier.isin(ACTIVE_TIERS)]
    inact = data[data.tier == "inactive"]
    azd = data.loc["AZD7648"] if "AZD7648" in data.index else None
    scores = {"dock (kcal/mol)": "dock", "affinity (log10 IC50 uM)": "affinity_pred_value",
              "P(binder)": "affinity_probability_binary"}

    print(f"actives n={len(act)}, inactives n={len(inact)}")
    print("legend: ROC-AUC 0.5 = coin flip, 1.0 = every active ranked above every inactive;")
    print("        rho > 0 = the score orders the known inhibitors correctly\n")
    rows = []
    for label, col in scores.items():
        a, i = act[col].dropna(), inact[col].dropna()
        if a.empty or i.empty:
            print(f"{label:26s} unavailable")
            continue
        higher_is_better = col == "affinity_probability_binary"
        # one orientation for everything: larger "strength" = stronger binder
        strength = data[col] if higher_is_better else -data[col]
        sa, si = strength.loc[a.index], strength.loc[i.index]
        auc = (sum(1 for x in sa for y in si if x > y)
               + 0.5 * sum(1 for x in sa for y in si if x == y)) / (len(sa) * len(si))
        rho, p = spearman(act["pIC50"], strength.loc[act.index])
        rows.append({"score": label, "roc_auc": auc, "spearman_vs_pIC50": rho, "p": p,
                     **{f"median_{t}": data[data.tier == t][col].median()
                        for t in ACTIVE_TIERS + ("inactive",)}})
        pct = None
        if azd is not None and not math.isnan(strength.loc["AZD7648"]):
            stronger = (strength.drop("AZD7648") > strength.loc["AZD7648"]).sum()
            pct = 100 * stronger / (len(data) - 1)
        print(f"{label:26s} ROC-AUC {auc:.3f}  rho(pIC50) {rho:+.3f}"
              + (f" (p={p:.3f})" if not math.isnan(p) else "")
              + (f"  | out-scored by {pct:.0f}% of the set" if pct is not None else ""))

    print("\ntier medians")
    for t in ACTIVE_TIERS + ("inactive",):
        sub = data[data.tier == t]
        print(f"  {t:9s} n={len(sub):2d}  dock {sub.dock.median():+6.2f}"
              f"  affinity {sub.affinity_pred_value.median():+6.3f}"
              f"  P(binder) {sub.affinity_probability_binary.median():.3f}")
    if azd is not None:
        print(f"  {'AZD7648':9s} n= 1  dock {azd.dock:+6.2f}  affinity {azd.affinity_pred_value:+6.3f}"
              f"  P(binder) {azd.affinity_probability_binary:.3f}  (measured 0.63 nM)")

    print("\nrank of the known-potent tier by affinity (1 = strongest)")
    if data["affinity_pred_value"].isna().all():
        print("  affinity columns are empty - the run did not produce affinity output")
        return
    order = data.sort_values("affinity_pred_value")
    for t in ACTIVE_TIERS + ("inactive",):
        ranks = [i + 1 for i, m in enumerate(order.index) if data.loc[m, "tier"] == t]
        if ranks:
            print(f"  {t:9s} ranks {min(ranks)}-{max(ranks)} (median {int(pd.Series(ranks).median())})")
    if "AZD7648" in order.index:
        print(f"  AZD7648   rank {list(order.index).index('AZD7648') + 1} of {len(order)}")
    print(f"\n-> {BENCH / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
