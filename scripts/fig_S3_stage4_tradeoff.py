#!/usr/bin/env python3
"""Figure S3 - Stage 4 trade-off between the docking and cellular objectives.

Scatter of every retained analog against its own parent: x = delta docking score
(negative = tighter predicted binding), y = delta cell objective (positive =
stronger predicted cellular response). The upper-left quadrant is the set that
improved both objectives, which the manuscript reports as 9 of 427.

The nine dual-improvers are labelled in a column placed in the empty upper-left
region of the axes, each connected to its point by a leader line.  The column is
ordered by the points' own y values, which makes the assignment monotone and
therefore guarantees the leader lines cannot cross each other.

Writes SVG (vector, submission quality), PNG (600 dpi) and a CSV of the plotted
values.

    cd /home/geng/vcdd-0.3.0
    ~/miniconda3/envs/vcdd/bin/python scripts/fig_S2_stage4_tradeoff.py
"""
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/home/geng/vcdd-0.3.0/tmp/mplcache")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/geng/vcdd-0.3.0")
RUN = ROOT / "dnapkcs_affinity500" / "stage4"
FIG = ROOT / "paper_figs"
FIG.mkdir(exist_ok=True)

C_GA = "#3B6FB6"       # constrained genetic algorithm
C_RG = "#E8833A"       # R-group enumeration
C_BOTH = "#0F766E"     # analogs improving both objectives
C_BAND = "#B8B8B8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "none",   # keep text as text so the journal can restyle it
})


def load() -> pd.DataFrame:
    k = pd.read_parquet(RUN / "analogs.parquet")
    k = k[["analog_id", "parent_id", "generator", "delta_dock_score",
           "delta_cell_objective", "delta_design_score", "design_score"]].copy()
    # delta dock == 0 counts as "not improved": docking is unchanged, not better
    k["both"] = (k.delta_dock_score < 0) & (k.delta_cell_objective > 0)
    return k


def main() -> None:
    k = load()
    n = len(k)
    both = k[k.both].sort_values("delta_cell_objective")
    noise = float(k.delta_dock_score.abs().median())

    cell_only = k[(k.delta_cell_objective > 0) & (k.delta_dock_score >= 0)]
    dock_only = k[(k.delta_cell_objective <= 0) & (k.delta_dock_score < 0)]
    neither = k[(k.delta_cell_objective <= 0) & (k.delta_dock_score >= 0)]
    n_flat = int((k.delta_dock_score == 0).sum())

    x, y = k.delta_dock_score.to_numpy(), k.delta_cell_objective.to_numpy()
    xlo, xhi = -1.20, 3.20
    ylo, yhi = -0.225, 0.085

    fig, ax = plt.subplots(figsize=(7.8, 6.0))

    # favourable half-planes; the overlap (upper left) reads darkest
    ax.axvspan(xlo, 0, color="#3E9B4F", alpha=0.05, lw=0, zorder=0)
    ax.axhspan(0, yhi, color="#3E9B4F", alpha=0.05, lw=0, zorder=0)
    ax.axvspan(xlo, 0, ymin=(0 - ylo) / (yhi - ylo), ymax=1,
               color="#3E9B4F", alpha=0.075, lw=0, zorder=0)

    # docking noise floor: median |delta dock| across the retained set
    ax.axvspan(-noise, noise, facecolor=C_BAND, alpha=0.22, lw=0, zorder=0)
    for edge in (-noise, noise):
        ax.axvline(edge, color=C_BAND, lw=0.7, ls=(0, (4, 3)), zorder=1)

    ax.axhline(0, color="#6E6E6E", lw=0.9, zorder=1)
    ax.axvline(0, color="#6E6E6E", lw=0.9, zorder=1)

    for g, c, lab in (("constrained_ga", C_GA, "Constrained GA"),
                      ("rgroup", C_RG, "R-group enumeration")):
        s = k[k.generator == g]
        ax.scatter(s.delta_dock_score, s.delta_cell_objective, s=19, c=c,
                   alpha=0.60, edgecolors="none", zorder=2,
                   label=f"{lab}  (n = {len(s)})")

    ax.scatter(both.delta_dock_score, both.delta_cell_objective, s=86,
               facecolors="none", edgecolors=C_BOTH, linewidths=1.5, zorder=4,
               label=f"Improved both objectives  (n = {len(both)})")
    ax.scatter(both.delta_dock_score, both.delta_cell_objective, s=18,
               c=C_BOTH, edgecolors="none", zorder=5)

    # ---- label column for the nine, in the empty upper-left region ----
    # Leader lines are drawn as ordinary low-zorder lines and the text
    # separately at high zorder, so the marker cloud always renders on top of
    # the leaders: a leader that grazes a neighbouring point is then visually
    # harmless instead of drawing over it.
    col_x = -1.13                       # left edge of the label column
    col_y0, col_y1 = 0.005, 0.077       # spans the clear space above the cluster
    text_w = 0.42                       # label width in data units
    ys = np.linspace(col_y0, col_y1, len(both))
    for (_, r), ylab in zip(both.iterrows(), ys):
        ax.plot([col_x + text_w, r.delta_dock_score],
                [ylab, r.delta_cell_objective],
                color=C_BOTH, lw=0.55, alpha=0.80, solid_capstyle="round",
                zorder=1.6, clip_on=False)
        ax.text(col_x, ylab, r.analog_id, fontsize=6.8, color=C_BOTH,
                ha="left", va="center", zorder=8)

    # ---- quadrant counts ----
    box = dict(boxstyle="round,pad=0.38", facecolor="white",
               edgecolor="#C4C4C4", linewidth=0.7, alpha=0.94)
    ax.text(0.985, 0.975,
            f"Both improved: $\\bf{{{len(both)}}}$ of {n}\n"
            f"Cell only: {len(cell_only)}\n"
            f"Docking only: {len(dock_only)}\n"
            f"Neither: {len(neither)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
            color="#3A3A3A", bbox=box, zorder=8, linespacing=1.5)

    ax.text(0.985, 0.028,
            f"docking unchanged ($\\Delta$dock = 0): {n_flat}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
            color="#6E6E6E", zorder=8)

    # ---- direction cue for the noise band (the axis labels already carry the
    # favourable-direction arrows, so no second arrow is drawn at the top: it
    # would cut straight through the label column) ----
    ax.text(-noise - 0.05, ylo + 0.008,
            f"median |$\\Delta$dock| = {noise:.2f} kcal mol$^{{-1}}$",
            ha="right", va="bottom", fontsize=6.9, color="#7A7A7A",
            rotation=90, zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("$\\Delta$ dock  (analog $-$ parent, kcal mol$^{-1}$)"
                  "          $\\leftarrow$ favourable", fontsize=9.6)
    ax.set_ylabel("$\\Delta$ cell  ($S_{cell}$ analog $-$ parent)"
                  "          favourable $\\uparrow$", fontsize=9.6)
    ax.set_title(f"Stage 4 analogs relative to their own parent  (n = {n})",
                 fontsize=10.6, pad=10)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.8, labelsize=8.4)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.105),
                    ncol=3, frameon=True, framealpha=0.95, fontsize=8.2,
                    edgecolor="#CCCCCC", borderpad=0.6, columnspacing=1.4,
                    handletextpad=0.5)
    leg.get_frame().set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(FIG / "FigS3_stage4_tradeoff.svg")
    fig.savefig(FIG / "FigS3_stage4_tradeoff.png", dpi=600)
    fig.canvas.draw()
    # geometry, so the raster can be checked without opening it
    bb = ax.get_position()
    geo = {"axes_bbox_fig_fraction": [round(v, 4) for v in bb.bounds],
           "xlim": [xlo, xhi], "ylim": [ylo, yhi],
           "n": n, "both": len(both), "cell_only": len(cell_only),
           "dock_only": len(dock_only), "neither": len(neither),
           "docking_unchanged": n_flat, "noise_median_abs_ddock": noise}
    (FIG / "FigS3_stage4_tradeoff.geometry.json").write_text(
        json.dumps(geo, indent=2), encoding="utf-8")
    plt.close(fig)

    k.sort_values("delta_design_score", ascending=False).to_csv(
        FIG / "FigS3_stage4_tradeoff.csv", index=False)

    print(f"n = {n}   both = {len(both)}   cell-only = {len(cell_only)}   "
          f"dock-only = {len(dock_only)}   neither = {len(neither)}   "
          f"flat = {n_flat}")
    print(f"quadrant sum = {len(both) + len(cell_only) + len(dock_only) + len(neither)}"
          f"  (must equal {n})")
    print("wrote", *(p.name for p in sorted(FIG.glob("FigS2_*"))))


if __name__ == "__main__":
    main()
