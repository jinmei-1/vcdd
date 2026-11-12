#!/usr/bin/env python3
"""ACS Graphical Abstract (TOC) figure — 7x2.75 inch, 600 dpi."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/home/geng/vcdd-0.3.0/tmp/mplcache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from pathlib import Path

ROOT = Path("/home/geng/vcdd-0.3.0")
OUT = ROOT / "paper_figs/TOC_GraphicalAbstract.png"

REC = [
    "COC1CCC(n2c(=O)n(C)c3cnc4ccc(-c5ncc(C(N)=S)cc5C)cc4c32)CC1",   # VCDD00052532
    "[O-][N+]1=C(n2ncc3c4ccccc4c4ccccc4c32)CCC1",                    # VCDD00168468
    "Cc1ccc(-n2cc(-c3ccc(F)cc3)c3c2nc(N)n2[nH]c(=O)nc32)cc1C",       # VCDD00071494
]

# RDKit 2D images of the three recommended molecules
mols = [Chem.MolFromSmiles(s) for s in REC]
for m in mols:
    AllChem.Compute2DCoords(m)
imgs = [Draw.MolToImage(m, size=(330, 250)) for m in mols]

fig = plt.figure(figsize=(7.0, 2.75), dpi=600)
fig.patch.set_facecolor("white")

# ---------- left: pipeline ----------
ax1 = fig.add_axes([0.005, 0.06, 0.34, 0.88]); ax1.axis("off")
steps = [("Library", "200,000", "#3B6FB6"),
         ("Docking", "Vina-GPU", "#3B6FB6"),
         ("Co-folding", "Boltz-2", "#3E9B4F"),
         ("Virtual cell", "XPert · 978 genes", "#E8833A"),
         ("Optimize", "GA + R-group", "#C2453C")]
y = 0.92
for title, sub, c in steps:
    box = FancyBboxPatch((0.03, y - 0.155), 0.94, 0.15, boxstyle="round,pad=0.005",
                         linewidth=1.2, edgecolor=c, facecolor="white")
    ax1.add_patch(box)
    ax1.text(0.5, y - 0.09, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=c)
    ax1.text(0.5, y - 0.125, sub, ha="center", va="center", fontsize=5.4, color="#333333")
    y -= 0.18
ax1.set_xlim(0, 1); ax1.set_ylim(-0.05, 1.0)

# ---------- middle: molecules ----------
ax2 = fig.add_axes([0.36, 0.04, 0.44, 0.92]); ax2.axis("off")
ax2.set_xlim(0, 3); ax2.set_ylim(0, 1)
for i, img in enumerate(imgs):
    x = i * 0.06 + 0.02
    ax2.imshow(img, extent=[i + x, i + 0.98 - x * 0.4, 0.06, 0.94])
ax2.text(1.5, 0.985, "Recommended DNA-PKcs inhibitors", ha="center", fontsize=6.2,
         fontweight="bold", color="#111111")

# ---------- right: DDR concept ----------
ax3 = fig.add_axes([0.82, 0.06, 0.17, 0.88]); ax3.axis("off")
ax3.text(0.5, 0.85, "DNA-PKcs", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#C2453C")
ax3.text(0.5, 0.72, "NHEJ · DSB repair", ha="center", fontsize=5.5, color="#444444")
ax3.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.66), arrowprops=dict(arrowstyle="-|>", color="#C2453C", lw=1.6))
ax3.text(0.5, 0.46, "DNA repair ↓", ha="center", fontsize=7.5, fontweight="bold", color="#C2453C")
ax3.text(0.5, 0.33, "DDR activation ↑", ha="center", fontsize=6.0, color="#3E9B4F")
ax3.text(0.5, 0.18, "γH2AX foci", ha="center", fontsize=5.0, color="#666666")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

fig.savefig(OUT, dpi=600, facecolor="white")
print("saved:", OUT, OUT.stat().st_size // 1024, "KB")