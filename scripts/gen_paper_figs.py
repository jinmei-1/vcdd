#!/usr/bin/env python3
"""Generate 6 publication figures (PNG, 300 dpi) from pipeline results."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/home/geng/vcdd-0.3.0/tmp/mplcache")
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/geng/vcdd-0.3.0")
FIG = ROOT / "paper_figs"
FIG.mkdir(exist_ok=True)
C_BLUE, C_ORANGE, C_GREEN, C_RED = "#3B6FB6", "#E8833A", "#3E9B4F", "#C2453C"

# Stage-4 counts are read from the run so the flow diagram cannot keep stale
# numbers after a re-run.
S4_SUMMARY = json.loads((ROOT / "dnapkcs_run/stage4/stage4_summary.json").read_text())
N_GUARDRAIL_PASS = S4_SUMMARY["n_guardrail_pass"]

# ------------------------------------------------------------------ Fig 1 pipeline
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
stages = [
    ("Compound Library", "200,268 generated\nsmall molecules", 1, C_BLUE),
    ("Stage 1 · Docking", "Vina-GPU 2.1\n167,342 scored", 22, C_BLUE),
    ("Stage 2 · Co-folding", "Boltz-2 (local, 2×GPU)\ntop 500 complexes", 43, C_GREEN),
    ("Stage 3 · Virtual Cell", "XPert transcriptomics\n200 compounds × 978 genes", 64, C_ORANGE),
    ("Stage 4 · Optimization", f"GA + R-group\n{N_GUARDRAIL_PASS} refined analogs", 85, C_RED),
]
for title, sub, x, c in stages:
    box = FancyBboxPatch((x, 14), 14.5, 26, boxstyle="round,pad=0.5",
                         linewidth=1.4, edgecolor=c, facecolor="white", mutation_aspect=1.2)
    ax.add_patch(box)
    ax.text(x + 7.25, 32.5, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=c)
    for i, ln in enumerate(sub.split("\n")):
        ax.text(x + 7.25, 26 - i * 3.2, ln, ha="center", va="center", fontsize=8.8)
for x1, x2 in [(15.7, 21.7), (36.7, 42.7), (57.7, 63.7), (78.7, 84.7)]:
    ax.add_patch(FancyArrowPatch((x1, 27), (x2, 27), arrowstyle="-|>",
                                 mutation_scale=28, lw=2.2, color="#555555"))
ax.text(2.5, 44.5, "Virtual-Cell-Guided Drug Discovery Workflow (vcdd)",
        fontsize=12.5, fontweight="bold")
ax.text(2.0, 2.0, "Every stage is resumable at unit granularity (SQLite ledger); engines are "
                   "pluggable (Vina-GPU / Boltz API / local Boltz / arbitrary virtual-cell backends).",
        fontsize=8.6, color="#555555")
plt.tight_layout(); plt.savefig(FIG / "Fig1_pipeline.png", dpi=600, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ Fig 2 stage1 score
# Screening framing: the library is enriched in favorably docking molecules, and the
# shortlist that advances is drawn from the extreme favorable tail.  Panel A shows the
# bulk distribution with the shortlist cutoff; panel B shows the cumulative fraction at
# or below each score, so the "how many molecules are actually good" question is read
# directly off the curve.  The log-scale inset keeps the sparse positive tail visible.
s1 = pd.read_parquet(ROOT / "dnapkcs_run/stage1/ranking.parquet")
sc = s1["score"].to_numpy()
cut500 = float(np.sort(sc)[499])            # 500th best score = Stage-2 shortlist cutoff
cut200 = float(np.sort(sc)[199])            # 200th best score = Stage-3 cutoff
fig, (ax, axc) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                              gridspec_kw={"width_ratios": [1.35, 1.0]})

ax.hist(sc[(sc > -12.2) & (sc < -5.0)], bins=90, color=C_BLUE, alpha=0.9,
        edgecolor="white", lw=0.3)
ax.axvspan(-12.3, cut500, color=C_ORANGE, alpha=0.20, zorder=0)
ax.axvline(np.median(sc), color=C_RED, ls="--", lw=1.6,
           label=f"library median {np.median(sc):.1f}")
ax.axvline(-9.0, color="#6E6E6E", ls=":", lw=1.4,
           label=f"score $\\leq$ -9.0: {(sc <= -9.0).sum():,} ({(sc <= -9.0).mean()*100:.1f}%)")
ax.annotate(f"top-500 shortlist\n$\\leq$ {cut500:.1f} kcal/mol\n(0.3% of scored library)",
            xy=(cut500, ax.get_ylim()[1] * 0.30), xycoords="data",
            xytext=(0.30, 0.72), textcoords="axes fraction",
            fontsize=8, ha="left", va="center", color="#8A4B12",
            arrowprops=dict(arrowstyle="-|>", color="#8A4B12", lw=1.2,
                            connectionstyle="arc3,rad=-0.15"))
ax.set_xlim(-12.3, -5.0)
ax.set_xlabel("Vina-GPU score (kcal/mol)"); ax.set_ylabel("Molecules")
ax.set_title(f"A  Library-wide docking (n = {len(sc):,})", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
axi = ax.inset_axes([0.60, 0.32, 0.37, 0.40])
axi.hist(sc, bins=60, color=C_BLUE)
axi.set_yscale("log"); axi.tick_params(labelsize=6)
axi.set_title("full range (log)", fontsize=7)
axi.axvline(np.median(sc), color=C_RED, ls="--", lw=1.0)

order = np.sort(sc)
cum = 100.0 * np.arange(1, len(order) + 1) / len(order)
axc.plot(order, cum, color=C_BLUE, lw=2.0)
axc.set_xlim(-12.0, -7.0); axc.set_ylim(0, 100)
for thr, col, off in [(-9.0, "#6E6E6E", (7, 6)), (cut500, C_ORANGE, (7, 3))]:
    frac = 100.0 * float((sc <= thr).mean())
    axc.axvline(thr, color=col, ls=":", lw=1.4)
    axc.plot([thr], [frac], "o", color=col, ms=5)
    axc.annotate(f"{thr:.1f}: {frac:.1f}%", xy=(thr, frac), xytext=off,
                 textcoords="offset points", fontsize=8, color=col, ha="left")
axc.axhline(0.3, color="#999999", lw=0.8, ls="-")
axc.set_xlabel("Vina-GPU score (kcal/mol)")
axc.set_ylabel("Library at or below (%)")
axc.set_title("B  Cumulative enrichment", loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "Fig2_stage1_scores.png", dpi=600); plt.close()
print(f"Fig2: median {np.median(sc):.2f}, cut500 {cut500:.2f}, cut200 {cut200:.2f}")

# ------------------------------------------------------------------ Fig 3 stage2 co-folding
# Confidence alone does not say where the ligand went, so panel B adds the
# superposition-based geometry from the corrected run (see
# scripts/fig_stage2_cofolding.py, which writes exactly this figure).
s2 = pd.read_parquet(ROOT / "dnapkcs_run/stage2/consensus_ranking.parquet")
s2geo = pd.read_csv(ROOT / "dnapkcs_run/stage2/complex_analysis.csv")
GATE2 = 0.70
fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.3))
b2 = np.arange(0.60, 1.005, 0.01)
ax.hist(s2["binding_confidence"], bins=b2, color=C_GREEN, alpha=0.85, edgecolor="white", lw=0.4,
        label="composite confidence")
ax.hist(s2["iptm"], bins=b2, color=C_BLUE, alpha=0.45, edgecolor="white", lw=0.4, label="ipTM")
ax.axvline(GATE2, color=C_RED, ls="--", lw=1.6,
           label=f"gate {GATE2:.2f}: {(s2['binding_confidence']>=GATE2).sum()}/{len(s2)} pass")
ax.set_xlabel("Boltz-2 confidence"); ax.set_ylabel("Complexes"); ax.set_xlim(0.6, 1.0)
ax.set_title("A  Co-folding confidence, top-500 shortlist", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
gx, gy = s2geo["receptor_ca_rmsd_A"], s2geo["pocket_ca_rmsd_A"]
axb.scatter(gx, gy, s=13, color=C_GREEN, alpha=0.55, edgecolor="none")
lim2 = [min(gx.min(), gy.min()) - 0.3, max(gx.max(), gy.max()) + 0.3]
axb.plot(lim2, lim2, color="#888888", lw=1.0, ls="--")
axb.axhline(2.0, color=C_ORANGE, ls=":", lw=1.3, label=f"ATP site < 2.0 Å: {(gy<2.0).sum()}/500")
axb.set_xlim(lim2); axb.set_ylim(lim2)
axb.set_xlabel("Receptor Cα RMSD vs 7OTW 3701–4123 (Å)")
axb.set_ylabel("ATP-site Cα RMSD (39 residues, Å)")
axb.set_title("B  ATP site is reproduced far better than the whole fold", loc="left", fontsize=10.5)
axb.legend(frameon=False, fontsize=8.5, loc="upper left")
d2 = s2geo["ligand_centroid_to_azd7648_A"]
ins = axb.inset_axes([0.58, 0.52, 0.39, 0.42])
ins.hist(d2, bins=np.arange(0, 22, 1.0), color=C_BLUE, alpha=0.85, edgecolor="white", lw=0.3)
ins.axvline(d2.median(), color=C_RED, ls="--", lw=1.2)
ins.text(0.96, 0.9, f"median {d2.median():.1f} Å\n{int((d2<5).sum())}/500 < 5 Å",
         transform=ins.transAxes, ha="right", va="top", fontsize=7.5)
ins.set_xlabel("ligand–AZD7648 centroid distance (Å)", fontsize=7.5); ins.set_yticks([])
ins.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "Fig3_stage2_iptm.png", dpi=600); plt.close()
print(f"Fig3: confidence median {s2['binding_confidence'].median():.3f}, "
      f"pass gate {int((s2['binding_confidence']>=GATE2).sum())}/{len(s2)}; "
      f"global {gx.median():.2f} Å vs ATP site {gy.median():.2f} Å")

# ------------------------------------------------------------------ Fig 4 stage3 objective
s3 = pd.read_parquet(ROOT / "dnapkcs_run/stage3/objective_scores.parquet")
s3s = s3.sort_values("objective", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar(range(len(s3s)), s3s["objective"], color=C_ORANGE, width=1.0)
ax.axhline(s3["objective"].median(), color=C_RED, ls="--", lw=1.4,
           label=f"median {s3['objective'].median():.2f}")
ax.set_xlabel("Stage-3 rank (200 compounds)"); ax.set_ylabel("Objective score")
ax.set_title("Stage 3 — Virtual-cell (XPert) objective: 2×gene_score + target_consistency")
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG / "Fig4_stage3_objective.png", dpi=600); plt.close()

# ------------------------------------------------------------------ Fig 5 top50 DDR
t50 = pd.read_csv(ROOT / "dnapkcs_run/stage3/top50_ddr_inhibitors.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
im = ax.scatter(t50["gene_score"], t50["repair_inhibition"], c=t50["ddr_inhibition"],
                cmap="viridis", s=60, edgecolor="white", lw=0.5)
ax.set_xlabel("DDR activation (CDKN1A/GADD45A, gene_score)")
ax.set_ylabel("Repair-inhibition signature (−Δlog2FC of 16 repair genes)")
ax.set_title("Top-50 DDR-inhibitory candidates")
cb = plt.colorbar(im, ax=ax); cb.set_label("ddr_inhibition")
# annotate top3
for _, r in t50.head(3).iterrows():
    ax.annotate(f"#{int(r['ddr_rank'])} {r['mol_id']}", (r["gene_score"], r["repair_inhibition"]),
                xytext=(8, 6), textcoords="offset points", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "Fig5_top50_ddr.png", dpi=600); plt.close()

# ------------------------------------------------------------------ Fig 6 stage4
s4 = pd.read_parquet(ROOT / "dnapkcs_run/stage4/analogs.parquet")
fig, ax = plt.subplots(figsize=(7.4, 4.8))
cols = {"constrained_ga": C_BLUE, "rgroup": C_GREEN}
for g, c in cols.items():
    sub = s4[s4["generator"] == g]
    ax.scatter(sub["delta_dock_score"], sub["delta_cell_objective"], s=22, alpha=0.7,
               color=c, label=g, edgecolor="none")
ax.axhline(0, color="#888888", lw=0.8); ax.axvline(0, color="#888888", lw=0.8)
ax.set_xlabel("Δ docking score (kcal/mol, vs parent)")
ax.set_ylabel("Δ cell objective (vs parent)")
ax.set_title(f"Stage 4 — {len(s4)} generated analogs relative to 10 parents")
ax.legend(frameon=False)
ins = ax.inset_axes([0.60, 0.08, 0.36, 0.30])
for g, c in cols.items():
    sub = s4[s4["generator"] == g]
    ins.scatter(sub["delta_dock_score"], sub["delta_cell_objective"], s=8, alpha=0.6, color=c)
ins.axhline(0, color="#888888", lw=0.6); ins.axvline(0, color="#888888", lw=0.6)
ins.set_xticks([]); ins.set_yticks([])
plt.tight_layout(); plt.savefig(FIG / "Fig6_stage4_improvements.png", dpi=600); plt.close()

print("figs done:", sorted(p.name for p in FIG.glob("*.png")))