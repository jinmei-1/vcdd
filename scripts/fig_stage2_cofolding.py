#!/usr/bin/env python3
"""Figure for Results 3.2 (Stage 2, co-folding) from the corrected Stage-2 run.

Panel A: distribution of Boltz-2 composite confidence and ipTM for the 500
         shortlisted molecules, with the configured 0.70 gate.
Panel B: fold fidelity after superposing each predicted receptor onto the
         crystal frame - global (423 C-alpha) versus ATP-site (39 C-alpha) RMSD -
         with the distance of the predicted ligand centroid to the co-crystal
         inhibitor AZD7648 as an inset. The model never received the docking box,
         the docking pose or the reference ligand, so panel B is independent
         localisation evidence, not a pose-validation metric.

Writes paper_figs/Fig3_stage2_iptm.png (600 dpi), the file embedded by
scripts/gen_main_paper.py. Run:  python scripts/fig_stage2_cofolding.py
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/home/geng/vcdd-0.3.0/tmp/mplcache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/geng/vcdd-0.3.0")
FIG = ROOT / "paper_figs"
FIG.mkdir(exist_ok=True)
C_BLUE, C_ORANGE, C_GREEN, C_RED = "#3B6FB6", "#E8833A", "#3E9B4F", "#C2453C"

conf = pd.read_parquet(ROOT / "dnapkcs_run/stage2/consensus_ranking.parquet")
geo = pd.read_csv(ROOT / "dnapkcs_run/stage2/complex_analysis.csv")
GATE = 0.70

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 4.3))

# ---------------------------------------------------------------- A confidence
bins = np.arange(0.60, 1.005, 0.01)
axa.hist(conf["binding_confidence"], bins=bins, color=C_GREEN, alpha=0.85,
         edgecolor="white", lw=0.4, label="composite confidence")
axa.hist(conf["iptm"], bins=bins, color=C_BLUE, alpha=0.45,
         edgecolor="white", lw=0.4, label="ipTM")
axa.axvline(GATE, color=C_RED, ls="--", lw=1.6,
            label=f"gate {GATE:.2f}: {(conf['binding_confidence'] >= GATE).sum()}/500 pass")
axa.set_xlabel("Boltz-2 confidence")
axa.set_ylabel("Complexes")
axa.set_title("A  Co-folding confidence, top-500 shortlist", loc="left", fontsize=10.5)
axa.legend(frameon=False, fontsize=8.5, loc="upper left")
axa.set_xlim(0.6, 1.0)

# ---------------------------------------------------------------- B geometry
gx, gy = geo["receptor_ca_rmsd_A"], geo["pocket_ca_rmsd_A"]
axb.scatter(gx, gy, s=13, color=C_GREEN, alpha=0.55, edgecolor="none")
lim = [min(gx.min(), gy.min()) - 0.3, max(gx.max(), gy.max()) + 0.3]
axb.plot(lim, lim, color="#888888", lw=1.0, ls="--")
axb.text(lim[0] + 0.12, lim[1] - 0.25, "equal fold fidelity", fontsize=8,
         color="#666666", rotation=38, ha="left", va="top")
axb.axhline(2.0, color=C_ORANGE, ls=":", lw=1.3,
            label=f"ATP site < 2.0 Å: {(gy < 2.0).sum()}/500")
axb.set_xlabel("Receptor Cα RMSD vs 7OTW 3701–4123 (Å)")
axb.set_ylabel("ATP-site Cα RMSD (39 residues, Å)")
axb.set_title("B  ATP site is reproduced far better than the whole fold",
              loc="left", fontsize=10.5)
axb.legend(frameon=False, fontsize=8.5, loc="upper left")
axb.set_xlim(lim)
axb.set_ylim(lim)

inset = axb.inset_axes([0.58, 0.52, 0.39, 0.42])
d = geo["ligand_centroid_to_azd7648_A"]
inset.hist(d, bins=np.arange(0, 22, 1.0), color=C_BLUE, alpha=0.85,
           edgecolor="white", lw=0.3)
inset.axvline(d.median(), color=C_RED, ls="--", lw=1.2)
inset.text(0.96, 0.9, f"median {d.median():.1f} Å\n{int((d < 5).sum())}/500 < 5 Å",
           transform=inset.transAxes, ha="right", va="top", fontsize=7.5)
inset.set_xlabel("ligand–AZD7648 centroid distance (Å)", fontsize=7.5)
inset.tick_params(labelsize=7)
inset.set_yticks([])

plt.tight_layout()
out = FIG / "Fig3_stage2_iptm.png"
plt.savefig(out, dpi=600)
plt.close()
print(f"wrote {out}")
print(f"  confidence median {conf['binding_confidence'].median():.3f} "
      f"(min {conf['binding_confidence'].min():.3f}), pass gate "
      f"{int((conf['binding_confidence'] >= GATE).sum())}/{len(conf)}")
print(f"  global Cα RMSD {gx.median():.2f} Å vs ATP site {gy.median():.2f} Å; "
      f"pocket < 2.0 Å in {int((gy < 2.0).sum())}/500")
print(f"  ligand centroid to AZD7648: median {d.median():.2f} Å, "
      f"{int((d < 5).sum())}/500 below 5 Å")
