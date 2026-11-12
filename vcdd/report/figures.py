"""SVG figures for the run report.

Every function takes a run directory, returns ``None`` when its input stage did
not run, and otherwise writes ``report/figures/<name>.svg`` and returns the
path. Nothing here raises on missing data: a partial run must still produce a
report.

Text on these plots is kept to axis labels, units and legends. Interpretation
belongs in the report body, not burned into the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

#: Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

__all__ = [
    "build_all",
    "stage1_score_distribution",
    "stage1_vs_boltz",
    "boltz_confidence_distribution",
    "consensus_agreement",
    "stage3_objective_components",
    "stage3_objective_vs_docking",
    "stage4_delta_scatter",
    "stage4_sar_rules",
    "PALETTE",
]


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{name}.svg"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    return svg


def _read(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return None if df.empty else df
    except Exception:  # noqa: BLE001 - a corrupt intermediate must not kill the report
        return None


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


def stage1_score_distribution(run_dir: Path, out_dir: Path) -> Path | None:
    df = _read(run_dir / "stage1" / "ranking.parquet")
    if df is None or "score" not in df:
        return None
    s = pd.to_numeric(df["score"], errors="coerce").dropna()
    if s.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), width_ratios=[1.3, 1])

    ax1.hist(s, bins=min(60, max(10, int(np.sqrt(len(s))))), color=PALETTE[0], alpha=0.85)
    ax1.set_xlabel("Stage-1 docking score (kcal/mol)")
    ax1.set_ylabel("Compounds")
    q = float(np.quantile(s, 0.01))
    ax1.axvline(q, color=PALETTE[1], ls="--", lw=1.2, label="top 1%")
    ax1.legend(frameon=False, fontsize=8)

    ranks = np.arange(1, len(s) + 1)
    ax2.plot(ranks, np.sort(s.to_numpy()), color=PALETTE[0], lw=1.4)
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Score (kcal/mol)")
    ax2.set_xscale("log" if len(s) > 500 else "linear")
    return _save(fig, out_dir, "stage1_scores")


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #


def stage1_vs_boltz(run_dir: Path, out_dir: Path) -> Path | None:
    """Stage-1 docking score against the Boltz binding confidence.

    Two independent readouts of the same molecule, so the interesting region is
    the *disagreement*: high-confidence compounds that stage 1 ranked poorly.
    Points are coloured by whether the predicted pose stayed inside the docking
    box, because a confident complex outside the box is not the same hypothesis.
    """
    df = _read(run_dir / "stage2" / "consensus_ranking.parquet")
    if df is None:
        df = _read(run_dir / "stage2" / "scores.parquet")
    if df is None or not {"stage1_score", "binding_confidence"} <= set(df.columns):
        return None
    cols = ["stage1_score", "binding_confidence"]
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    if "ligand_in_box" in df.columns:
        d["ligand_in_box"] = df["ligand_in_box"]
    d = d.dropna(subset=cols)
    if len(d) < 3:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    in_box = d.get("ligand_in_box")
    if in_box is not None and in_box.notna().any():
        for flag, colour, label in (
            (True, PALETTE[0], "pose in box"), (False, PALETTE[1], "pose outside box"),
        ):
            sub = d[in_box.fillna(-1) == flag]
            if not sub.empty:
                ax.scatter(sub["stage1_score"], sub["binding_confidence"], s=26,
                           color=colour, alpha=0.8, edgecolor="none", label=label)
        unchecked = d[in_box.isna()]
        if not unchecked.empty:
            ax.scatter(unchecked["stage1_score"], unchecked["binding_confidence"], s=22,
                       color="0.65", alpha=0.7, edgecolor="none", label="not pose-checked")
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.scatter(d["stage1_score"], d["binding_confidence"], s=26, color=PALETTE[0],
                   alpha=0.8, edgecolor="none")

    ax.axhline(0.7, color="0.6", lw=1.0, ls="--")
    rho = d["stage1_score"].corr(-d["binding_confidence"], method="spearman")
    ax.set_xlabel("Stage-1 docking score (kcal/mol)")
    ax.set_ylabel("Boltz binding confidence")
    ax.set_title(f"Spearman rho = {rho:.2f}", fontsize=9, loc="left")
    return _save(fig, out_dir, "stage1_vs_boltz")


def boltz_confidence_distribution(run_dir: Path, out_dir: Path) -> Path | None:
    """Where the shortlist sits relative to the 0.7 high-confidence triage line.

    A screen where everything clears 0.7, or where nothing does, is telling you
    about the target and the API defaults rather than about the compounds.
    """
    df = _read(run_dir / "stage2" / "consensus_ranking.parquet")
    if df is None:
        df = _read(run_dir / "stage2" / "scores.parquet")
    if df is None or "binding_confidence" not in df.columns:
        return None
    bc = pd.to_numeric(df["binding_confidence"], errors="coerce").dropna()
    if len(bc) < 3:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(bc, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.axvline(0.7, color=PALETTE[1], lw=1.2, ls="--")
    ax.set_xlabel("Boltz binding confidence")
    ax.set_ylabel("Compounds")
    ax.set_xlim(0.0, 1.0)
    n_hi = int((bc >= 0.7).sum())
    ax.set_title(f"{n_hi}/{len(bc)} at or above 0.7", fontsize=9, loc="left")
    return _save(fig, out_dir, "boltz_confidence_distribution")


def consensus_agreement(run_dir: Path, out_dir: Path) -> Path | None:
    df = _read(run_dir / "stage2" / "consensus_ranking.parquet")
    if df is None:
        return None
    cols = [c for c in df.columns if c.startswith("norm_")]
    if len(cols) < 2:
        return None
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        return None
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.85 * len(cols) + 2.2, 0.85 * len(cols) + 1.6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("norm_", "") for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    return _save(fig, out_dir, "consensus_agreement")


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #


def stage3_objective_components(run_dir: Path, out_dir: Path, top_n: int = 20) -> Path | None:
    df = _read(run_dir / "stage3" / "objective_scores.parquet")
    if df is None or "objective" not in df:
        return None
    comps = [c for c in ["gene_score", "pathway_score", "target_consistency", "specificity"]
             if c in df.columns]
    if not comps:
        return None
    d = df.sort_values("objective", ascending=False).head(top_n).iloc[::-1]
    y = np.arange(len(d)) * 1.25  # gap between compounds
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(d) + 2.0))
    width = 0.9 / len(comps)
    for i, c in enumerate(comps):
        ax.barh(y + i * width - 0.45 + width / 2,
                pd.to_numeric(d[c], errors="coerce").fillna(0.0),
                height=width, color=PALETTE[i], label=c)
    ax.set_yticks(y, d["mol_id"].astype(str), fontsize=7)
    ax.set_ylabel("Compound")
    ax.set_xlabel("Component score (0-1)")
    ax.set_xlim(0, 1)
    ax.margins(y=0.02)
    ax.xaxis.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncols=len(comps),
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    return _save(fig, out_dir, "stage3_components")


def stage3_objective_vs_docking(run_dir: Path, out_dir: Path) -> Path | None:
    df = _read(run_dir / "stage3" / "objective_scores.parquet")
    if df is None or "objective" not in df:
        return None
    xcol = "dock_score" if "dock_score" in df.columns else "stage1_score"
    if xcol not in df.columns:
        return None
    d = df[[xcol, "objective"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        return None
    ood = df.get("out_of_domain")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    if ood is not None:
        mask = ood.reindex(d.index).fillna(False).astype(bool)
        ax.scatter(d.loc[~mask, xcol], d.loc[~mask, "objective"], s=30,
                   color=PALETTE[0], alpha=0.85, edgecolor="none", label="in domain")
        if mask.any():
            ax.scatter(d.loc[mask, xcol], d.loc[mask, "objective"], s=34,
                       facecolor="none", edgecolor=PALETTE[1], lw=1.2,
                       label="out of applicability domain")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(d[xcol], d["objective"], s=30, color=PALETTE[0], alpha=0.85,
                   edgecolor="none")
    rho = d[xcol].corr(d["objective"], method="spearman")
    ax.set_xlabel("Docking score (kcal/mol)")
    ax.set_ylabel("Virtual-cell objective")
    ax.set_title(f"Spearman rho = {rho:.2f}", fontsize=9, loc="left")
    return _save(fig, out_dir, "stage3_objective_vs_docking")


# --------------------------------------------------------------------------- #
# Stage 4
# --------------------------------------------------------------------------- #


def stage4_delta_scatter(run_dir: Path, out_dir: Path) -> Path | None:
    df = _read(run_dir / "stage4" / "analogs.parquet")
    if df is None:
        return None
    if not {"delta_dock_score", "delta_cell_objective"} <= set(df.columns):
        return None
    d = df.copy()
    d["delta_dock_score"] = pd.to_numeric(d["delta_dock_score"], errors="coerce")
    d["delta_cell_objective"] = pd.to_numeric(d["delta_cell_objective"], errors="coerce")
    d = d.dropna(subset=["delta_dock_score", "delta_cell_objective"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, (gen, grp) in enumerate(d.groupby("generator")):
        ax.scatter(grp["delta_dock_score"], grp["delta_cell_objective"],
                   s=32, alpha=0.85, edgecolor="none",
                   color=PALETTE[i % len(PALETTE)], label=str(gen))
    ax.axhline(0, color="0.7", lw=0.9)
    ax.axvline(0, color="0.7", lw=0.9)
    # Shade the region where a docking gain exceeds scoring-function noise, but
    # only when something actually lands there.
    lo = float(d["delta_dock_score"].min())
    if lo < -1.0:
        ax.axvspan(lo - 0.2, -1.0, color=PALETTE[2], alpha=0.07)
    ax.set_xlabel(r"$\Delta$ docking score vs parent (kcal/mol)")
    ax.set_ylabel(r"$\Delta$ virtual-cell objective vs parent")
    ax.set_xlim(lo - 0.2, float(d["delta_dock_score"].max()) + 0.2)
    ax.margins(y=0.08)
    ax.legend(frameon=False, fontsize=8, title="generator", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return _save(fig, out_dir, "stage4_deltas")


def stage4_sar_rules(run_dir: Path, out_dir: Path, metric: str = "delta_cell_objective",
                     top_n: int = 14) -> Path | None:
    df = _read(run_dir / "stage4" / "sar_rules.csv")
    if df is None or "metric" not in df.columns:
        return None
    d = df[df["metric"] == metric].copy()
    if d.empty:
        return None
    d = d.sort_values("median_delta", ascending=False).head(top_n).iloc[::-1]
    colours = [PALETTE[2] if v > 0 else PALETTE[1] for v in d["median_delta"]]
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(d) + 1.4))
    ax.barh(np.arange(len(d)), d["median_delta"], color=colours, alpha=0.9)
    ax.set_yticks(np.arange(len(d)), d["transformation"].astype(str), fontsize=8)
    ax.axvline(0, color="0.4", lw=1.0)
    ax.set_xlabel(f"Median {metric.replace('_', ' ')} across parents")
    for i, (v, n) in enumerate(zip(d["median_delta"], d["n_parents"])):
        ax.text(v + (0.002 if v >= 0 else -0.002), i, f"n={int(n)}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7, color="0.3")
    return _save(fig, out_dir, "stage4_sar")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

_BUILDERS: list[tuple[str, Any]] = [
    ("stage1_scores", stage1_score_distribution),
    ("stage1_vs_boltz", stage1_vs_boltz),
    ("boltz_confidence_distribution", boltz_confidence_distribution),
    ("consensus_agreement", consensus_agreement),
    ("stage3_components", stage3_objective_components),
    ("stage3_objective_vs_docking", stage3_objective_vs_docking),
    ("stage4_deltas", stage4_delta_scatter),
    ("stage4_sar", stage4_sar_rules),
]


def build_all(run_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Build every figure whose input exists. Returns ``{name: svg path}``."""
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "report" / "figures"
    made: dict[str, Path] = {}
    for name, fn in _BUILDERS:
        try:
            path = fn(run_dir, out)
        except Exception:  # noqa: BLE001 - one bad figure must not kill the report
            path = None
        if path is not None:
            made[name] = path
    return made
