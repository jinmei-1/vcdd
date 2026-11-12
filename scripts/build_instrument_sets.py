#!/usr/bin/env python
"""Build the phase-0 instrument sets: noise floor and receptor-offset test.

Both are unconditional in the redesigned workflow (paper/workflow_redesign.md,
steps 0.3 and 3.1): whatever the campaign claims, the docking score's run-to-run
noise and the effect of changing receptor preparation have to be measured on the
actual molecules before any ranking statement.

* noise floor  - 100 molecules stratified over the campaign score range, to be
                 docked three times with one frozen receptor; the spread between
                 replicates is the score's resolution.
* offset test  - 2000 molecules stratified over the campaign score range, to be
                 docked once with the frozen receptor and paired against their
                 campaign scores; if the shift is a near-constant offset the
                 campaign ranking survives, if it is molecule-dependent the whole
                 library has to be re-docked (62 h).

Outputs live under ``instrument/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "instrument"
CAMPAIGN = ROOT / "dnapkcs_run"


def stratified(scores: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n molecules spread evenly across the campaign score range.

    Rank-based rather than quantile-based: docking scores are quantised to 0.1
    and heavily duplicated, which collapses qcut's bins and silently samples one
    region of the distribution.
    """
    import numpy as np

    ordered = scores.sort_values("score").reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(ordered) - 1, n).astype(int))
    return ordered.iloc[idx]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ranking = pd.read_parquet(CAMPAIGN / "stage1" / "ranking.parquet")
    ranking = ranking[ranking.score.notna()][["mol_id", "smiles", "score"]]
    print(f"campaign ranking: {len(ranking)} scored molecules, "
          f"{ranking.score.min():.1f} .. {ranking.score.max():.1f}")

    for name, n in (("noise_floor_100", 100), ("offset_test_2000", 2000)):
        if name == "offset_test_2000":
            # half spread over the whole range to test whether the receptor shift
            # is a constant offset, half dense in the decision region (top 5000)
            # because that is where the shortlist cut at -10.4 is decided
            picked = pd.concat([stratified(ranking, 1000),
                                stratified(ranking.nsmallest(5000, "score"), 1000)])
            picked = picked.drop_duplicates("mol_id")
        else:
            picked = stratified(ranking, n)
        smi = OUT / f"{name}.smi"
        smi.write_text("\n".join(f"{r.smiles} {r.mol_id}" for r in picked.itertuples()) + "\n")
        picked[["mol_id", "score"]].to_csv(OUT / f"{name}_campaign_scores.csv", index=False)
        print(f"  {smi.name}: {len(picked)} molecules, campaign score "
              f"{picked.score.min():.1f} .. {picked.score.max():.1f}")

    base = yaml.safe_load((ROOT / "run_redock_harmonized.yaml").read_text())
    # the redock gate never ran because no reference ligand path was configured
    base["receptor"]["reference_ligand_path"] = "DNA-PKcs/MBW_ligand.pdb"
    base["receptor"]["redock_validation"] = True
    base["receptor"]["redock_fail_hard"] = False

    for name, title in (
        ("noise_floor_100", "100-molecule subset, dock three times with the frozen receptor"),
        ("offset_test_2000", "2000-molecule subset, paired against the campaign scores"),
    ):
        cfg = dict(base)
        cfg["name"] = f"dnapkcs-{name.replace('_', '-')}"
        cfg["library"] = dict(base["library"], path=f"instrument/{name}.smi")
        path = ROOT / f"run_{name}.yaml"
        path.write_text(
            f"# Phase-0 instrument run: {title}.\n"
            "# Receptor is frozen (see paper/docking_receptor_preparation.md) and the native-ligand\n"
            "# redock gate is enabled via reference_ligand_path.\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"  wrote {path.name}")

    gate = dict(base)
    gate["name"] = "dnapkcs-redock-gate"
    gate["library"] = dict(base["library"], path="instrument/noise_floor_100.smi")
    (ROOT / "run_redock_gate.yaml").write_text(
        "# Receptor-only run: clean the receptor, build the box, run the native-ligand (MBW)\n"
        "# redock gate. The campaign skipped this gate because reference_ligand_path was null.\n"
        + yaml.safe_dump(gate, sort_keys=False, allow_unicode=True))
    print("  wrote run_redock_gate.yaml")

    from vcdd.config.loader import load_config
    for f in ("run_redock_gate.yaml", "run_noise_floor_100.yaml", "run_offset_test_2000.yaml"):
        c = load_config(ROOT / f)
        print(f"  {f}: library={c.library.path} redock={c.receptor.redock_validation} "
              f"ref_ligand={c.receptor.reference_ligand_path}")


if __name__ == "__main__":
    main()
