"""Pick the Stage-3 shortlist from a finished Stage-2 run: structure first, then affinity.

The rule this implements, in the order it applies:

1. **Structural validity gate (hard filter, thresholds pre-registered).** A complex
   whose predicted ligand sits outside the ATP box, or whose pocket is not
   reproduced, cannot carry a meaningful affinity value - the affinity head would
   be scoring a pose the model never found. Defaults: at least 80% of ligand heavy
   atoms inside the docking box, and ATP-site C-alpha RMSD below 2.0 A against
   7OTW. Both are fixed before the affinity numbers are looked at, and the yield is
   reported either way.
2. **Affinity ranking (soft order) among the survivors.** Ascending
   ``affinity_pred_value`` (= log10(IC50/uM), so more negative is stronger).

Why not rank by ``binding_confidence``: over the original 500 complexes that score
spans 0.820-0.937, and the configured 0.70 gate passes 100% of molecules, so
ranking on it orders noise. The structural metrics do separate complexes
(pocket RMSD 1.36-2.37 A at the 5th-95th percentile), which is why they are the
gate and confidence is only reported.

The script also computes the two controls that decide whether an affinity ranking
may be claimed at all:

* the **positive control's percentile** under the same rule (AZD7648 is co-folded
  inside the set), and
* the **molecular-weight confound** (the affinity head tracks size: r = +0.45 over
  the 69-molecule benchmark) plus the **overlap** between the affinity and
  confidence top-N, which says how much the criterion choice actually matters.

Usage::

    python scripts/select_stage2_shortlist.py --run dnapkcs_affinity500 --top-n 200

Writes ``<run>/stage2/consensus_ranking.parquet`` reordered under the chosen
policy (the previous ranking is kept as ``consensus_ranking.<old policy>.parquet``
so both are available), plus ``<run>/stage2/selection_report.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Pre-registered structural thresholds. Changing them after seeing the affinity
# values would make the shortlist unfalsifiable, so they are constants here and
# are echoed into the report.
MIN_IN_BOX_FRAC = 0.80
MAX_POCKET_RMSD_A = 2.0


def molecular_weights(smiles: pd.Series) -> pd.Series:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    def one(s: str) -> float:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Descriptors.MolWt(mol) if mol else float("nan")

    return smiles.map(one)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dnapkcs_affinity500")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--rank-by", choices=("affinity", "confidence"), default="affinity")
    ap.add_argument("--min-in-box-frac", type=float, default=MIN_IN_BOX_FRAC)
    ap.add_argument("--max-pocket-rmsd", type=float, default=MAX_POCKET_RMSD_A)
    ap.add_argument("--control", default="AZD7648", help="positive control mol_id")
    ap.add_argument("--apply", action="store_true",
                    help="reorder consensus_ranking.parquet so stage 3 consumes this rule")
    args = ap.parse_args()

    run = ROOT / args.run
    s2 = run / "stage2"
    ranking_path = s2 / "consensus_ranking.parquet"
    qc_path = s2 / "complex_analysis.csv"
    for p in (ranking_path, qc_path):
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(ROOT)} - run stage 2 and "
                             f"scripts/analyze_stage2_complexes.py first")

    rank = pd.read_parquet(ranking_path)
    qc = pd.read_csv(qc_path)
    keep = ["mol_id", "ligand_in_box_frac", "pocket_ca_rmsd_A", "receptor_ca_rmsd_A",
            "ligand_centroid_to_azd7648_A", "n_contact_residues",
            "contact_residues_in_box_pocket_frac"]
    df = rank.merge(qc[[c for c in keep if c in qc.columns]], on="mol_id", how="left")
    n_total = len(df)

    if "affinity_pred_value" not in df.columns:
        raise SystemExit(
            "no affinity_pred_value column: this stage-2 run was made without the "
            "affinity head. Affinity cannot be added afterwards - re-run stage 2 "
            "with stage2.local_boltz.affinity: true (see paper/Methods_3.2_boltz2.md)")

    aff = pd.to_numeric(df["affinity_pred_value"], errors="coerce")
    err = df.get("affinity_error")
    aff_ok = aff.notna() & (err.isna() if err is not None else True)

    # ---- gate 1: structural validity
    in_box = pd.to_numeric(df["ligand_in_box_frac"], errors="coerce")
    pocket = pd.to_numeric(df["pocket_ca_rmsd_A"], errors="coerce")
    struct_ok = (in_box >= args.min_in_box_frac) & (pocket < args.max_pocket_rmsd)
    passed = struct_ok & aff_ok

    # ---- gate 2: rank the survivors
    order_col = "affinity_pred_value" if args.rank_by == "affinity" else "binding_confidence"
    ascending = args.rank_by == "affinity"          # more negative log10(IC50) = stronger
    df["_affinity"] = aff
    df["_qc_pass"] = passed
    df["_rank_value"] = pd.to_numeric(df[order_col], errors="coerce")
    df["_fail"] = ~passed
    ordered = df.sort_values(["_fail", "_rank_value", "mol_id"], ascending=[True, ascending, True],
                             na_position="last", kind="mergesort").reset_index(drop=True)
    ordered["consensus_rank"] = range(1, len(ordered) + 1)
    ordered["selection_policy"] = f"{args.rank_by}_after_structure_qc"
    shortlist = ordered.head(args.top_n)

    # ---- control 1: where does the known nanomolar inhibitor land?
    control = ordered[ordered.mol_id == args.control]
    control_info = None
    if len(control):
        pos = int(control.index[0]) + 1
        control_info = {
            "mol_id": args.control,
            "rank": pos, "of": n_total,
            "percentile_from_top": round(100.0 * pos / n_total, 1),
            "in_shortlist": bool(pos <= args.top_n),
            "qc_pass": bool(control._qc_pass.iloc[0]),
            "affinity_pred_value": (None if pd.isna(control._affinity.iloc[0])
                                    else round(float(control._affinity.iloc[0]), 3)),
            "binding_confidence": (None if pd.isna(control["binding_confidence"].iloc[0])
                                   else round(float(control["binding_confidence"].iloc[0]), 3)),
        }

    # ---- control 2: is the ranking a size ranking, and does the criterion matter?
    mw = molecular_weights(df["smiles"])
    diag: dict[str, object] = {}
    both = pd.DataFrame({"mw": mw, "aff": aff, "bc": pd.to_numeric(df["binding_confidence"], errors="coerce")}).dropna(subset=["mw", "aff"])
    if len(both) > 5:
        try:
            from scipy import stats
            rho, p = stats.spearmanr(both.mw, both.aff)
            diag["spearman_affinity_vs_mw"] = round(float(rho), 3)
            diag["p_spearman_affinity_vs_mw"] = float(f"{p:.2e}")
            diag["mw_median"] = round(float(both.mw.median()), 1)
        except Exception:  # noqa: BLE001
            pass
    aff_top = set(ordered[ordered._qc_pass].head(args.top_n).mol_id)
    conf_order = df[df._qc_pass].sort_values("binding_confidence", ascending=False, kind="mergesort")
    conf_top = set(conf_order.head(args.top_n).mol_id)
    if aff_top and conf_top:
        diag["overlap_top_n_affinity_vs_confidence"] = len(aff_top & conf_top)
        diag["jaccard"] = round(len(aff_top & conf_top) / len(aff_top | conf_top), 3)

    report = {
        "run": args.run,
        "policy": f"{args.rank_by}_after_structure_qc",
        "top_n": args.top_n,
        "rank_by": args.rank_by,
        "gate": {"min_in_box_frac": args.min_in_box_frac,
                 "max_pocket_rmsd_A": args.max_pocket_rmsd,
                 "n_total": n_total,
                 "n_struct_pass": int(struct_ok.sum()),
                 "n_affinity_ok": int(aff_ok.sum()),
                 "n_pass": int(passed.sum()),
                 "yield": round(float(passed.mean()), 3),
                 "note": ("thresholds are fixed in the script, not tuned after seeing "
                          "the affinity values")},
        "positive_control": control_info,
        "confound_checks": diag,
        "shortlist_head": shortlist[["consensus_rank", "mol_id", "affinity_pred_value",
                                     "binding_confidence", "ligand_in_box_frac",
                                     "pocket_ca_rmsd_A"]].head(10).to_dict("records"),
    }
    (s2 / "selection_report.json").write_text(json.dumps(report, indent=2, default=str))

    if args.apply:
        backup = s2 / "consensus_ranking.boltz_only.parquet"
        if not backup.exists():
            shutil.copy2(ranking_path, backup)
        cols = [c for c in ordered.columns if not c.startswith("_")]
        ordered[cols].to_parquet(ranking_path, index=False)

    # ------------------------------------------------------------------ printout
    g = report["gate"]
    print(f"{n_total} complexes | structure gate: {g['n_struct_pass']} pass "
          f"(in-box >= {args.min_in_box_frac}, pocket RMSD < {args.max_pocket_rmsd} A) | "
          f"affinity available: {g['n_affinity_ok']} | usable: {g['n_pass']} ({g['yield']:.0%})")
    print(f"ranked by {args.rank_by}; top {args.top_n} selected"
          + ("  [consensus_ranking.parquet rewritten]" if args.apply else
             "  [dry run: pass --apply to write]"))
    if control_info:
        print(f"\npositive control {control_info['mol_id']}: rank {control_info['rank']}/{n_total} "
              f"(top {control_info['percentile_from_top']}%), "
              f"affinity {control_info['affinity_pred_value']}, "
              f"in shortlist: {control_info['in_shortlist']}")
        if not control_info["in_shortlist"]:
            print("  !! the nanomolar reference inhibitor does not survive the rule: the "
                  "shortlist is then 'molecules predicted to beat a 0.63 nM inhibitor', "
                  "which this calibration cannot support. Say so in the paper.")
    if "spearman_affinity_vs_mw" in diag:
        print(f"\naffinity vs molecular weight: Spearman rho = {diag['spearman_affinity_vs_mw']:+.3f} "
              f"(p = {diag['p_spearman_affinity_vs_mw']})")
        if abs(diag["spearman_affinity_vs_mw"]) > 0.3:
            print("  !! strong size dependence: this ranking is partly a size ranking. "
                  "Enable --affinity_mw_correction and re-run, or rank within MW strata.")
    if "overlap_top_n_affinity_vs_confidence" in diag:
        print(f"top-{args.top_n} overlap with the confidence-based rule: "
              f"{diag['overlap_top_n_affinity_vs_confidence']}/{args.top_n} "
              f"(Jaccard {diag['jaccard']})")
    print(f"\nwrote {s2 / 'selection_report.json'}")


if __name__ == "__main__":
    main()
