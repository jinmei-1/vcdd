"""Inter-model structural consistency of co-folded stage-2 receptors.

The manuscript states that "inter-model comparisons show high internal
consistency (~0.7 A for the whole chain and ~0.8 A for the ATP site)". Nothing in
the repository computed that number, so this script does, and it is what the
sentence should be checked against.

It superposes every predicted receptor chain onto every other one (all pairs, no
reference structure involved) and reports the distribution of pairwise C-alpha
RMSD over the whole 423-residue construct and over the 39 box-defined ATP-site
residues. A tight distribution means Boltz-2 returns a conserved fold and a
conserved pocket regardless of which ligand it is co-folding with; a wide one
means the receptor geometry moves with the ligand and per-complex comparisons
are not like-for-like.

    python scripts/stage2_intermodel.py --run dnapkcs_affinity500
    python scripts/stage2_intermodel.py --run dnapkcs_run        # the original 500

Writes ``<run>/stage2/intermodel_consistency.json`` plus a per-pair CSV.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage2_complexes import (  # noqa: E402
    BOX, REF_PDB, ca_matrix, kabsch, split_chains,
)

DEFAULT_PATTERN = ("stage2/boltz/local/*/out/boltz_results_input/predictions/"
                   "input/input_model_0.cif")


def pocket_residues(box_path: Path, ref_pdb: Path) -> set[int]:
    """The 39 box-defined ATP-site residues, exactly as complex_analysis does."""
    import gemmi

    box = json.loads(box_path.read_text())
    lo = np.array(box["center"]) - np.array(box["size"]) / 2.0
    hi = np.array(box["center"]) + np.array(box["size"]) / 2.0
    ref_chain = gemmi.read_structure(str(ref_pdb))[0][0]
    return {
        i for i, res in enumerate(ref_chain)
        if any((not a.element.is_hydrogen)
               and lo[0] <= a.pos.x <= hi[0] and lo[1] <= a.pos.y <= hi[1]
               and lo[2] <= a.pos.z <= hi[2] for a in res)
    }


def model_ca(cif: Path) -> np.ndarray | None:
    """C-alpha coordinates of the receptor chain of one predicted complex."""
    import gemmi

    try:
        chains = split_chains(gemmi.read_structure(str(cif))[0])
    except Exception:  # noqa: BLE001
        return None
    if not chains:
        return None
    # the receptor is the longer chain; the ligand is a single residue
    chain = max(chains, key=len)
    if len(chain) < 50:
        return None
    return ca_matrix(chain)


def summarise(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {}
    return {"median": round(float(np.median(v)), 3),
            "mean": round(float(v.mean()), 3),
            "iqr_low": round(float(np.percentile(v, 25)), 3),
            "iqr_high": round(float(np.percentile(v, 75)), 3),
            "p90": round(float(np.percentile(v, 90)), 3),
            "max": round(float(v.max()), 3), "n_pairs": int(v.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dnapkcs_run",
                    help="run directory containing stage2/boltz/...")
    ap.add_argument("--pattern", default=None, help="override the prediction glob")
    ap.add_argument("--max-models", type=int, default=0,
                    help="0 = all; otherwise use the first N models")
    args = ap.parse_args()

    run = ROOT / args.run
    pattern = args.pattern or str(run / DEFAULT_PATTERN)
    cifs = [Path(p) for p in sorted(_glob.glob(pattern))]
    if not cifs:
        raise SystemExit(f"no predicted complexes under {pattern}")
    if args.max_models:
        cifs = cifs[: args.max_models]

    pocket = pocket_residues(BOX, REF_PDB)
    coords, names = [], []
    for cif in cifs:
        ca = model_ca(cif)
        if ca is None:
            continue
        coords.append(ca)
        names.append(cif.parents[4].name)

    n = len(coords)
    if n < 2:
        raise SystemExit(f"only {n} usable receptor chain(s); need at least 2")
    width = min(c.shape[0] for c in coords)
    stack = np.stack([c[:width] for c in coords])          # (n, L, 3)
    pocket_mask = np.array([i < width and i in pocket for i in range(width)])
    if pocket_mask.sum() < 5:
        raise SystemExit("ATP-site residue indices do not overlap the predicted chain")

    whole, site, pairs = [], [], []
    for i, j in combinations(range(n), 2):
        a, b = stack[i], stack[j]
        _, _, rms_whole = kabsch(a, b)
        _, _, rms_site = kabsch(a[pocket_mask], b[pocket_mask])
        whole.append(rms_whole)
        site.append(rms_site)
        pairs.append((names[i], names[j], round(rms_whole, 3), round(rms_site, 3)))

    whole, site = np.array(whole), np.array(site)
    out = {
        "run": args.run,
        "n_models": n,
        "n_residues": int(width),
        "n_pocket_residues": int(pocket_mask.sum()),
        "whole_chain_ca_rmsd_A": summarise(whole),
        "atp_site_ca_rmsd_A": summarise(site),
        "note": ("Pairwise superposition of predicted receptor chains only: no crystal "
                 "structure is involved, so this measures internal consistency, not "
                 "accuracy. Compare with the accuracy numbers in complex_analysis.csv."),
    }
    dest = run / "stage2"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "intermodel_consistency.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(pairs, columns=["model_a", "model_b", "ca_rmsd_A", "atp_site_rmsd_A"]).to_csv(
        dest / "intermodel_pairs.csv", index=False)

    w, s = out["whole_chain_ca_rmsd_A"], out["atp_site_ca_rmsd_A"]
    print(f"{n} models x {width} C-alpha, {pocket_mask.sum()} ATP-site residues, "
          f"{w['n_pairs']} pairs")
    print(f"  whole chain : {w['median']:.2f} A median (IQR {w['iqr_low']:.2f}-{w['iqr_high']:.2f}, "
          f"max {w['max']:.2f})")
    print(f"  ATP site    : {s['median']:.2f} A median (IQR {s['iqr_low']:.2f}-{s['iqr_high']:.2f}, "
          f"max {s['max']:.2f})")
    print(f"  wrote {dest / 'intermodel_consistency.json'}")


if __name__ == "__main__":
    main()
