#!/usr/bin/env python3
"""Figure S2 - 2D structures of the nine Stage 4 analogs that improved both
the docking and the cellular objective.

Companion to ``scripts/fig_stage3_parents.py`` (Figure 4) and
``scripts/fig_stage4_improved17.py`` (Figure 5) and built the same way, so the
three figures share a visual language.  Structures are RDKit vector paths
(submission quality) rather than embedded bitmaps.

The defining property of these nine is that *all* of them improved both primary
objectives (delta dock < 0 and delta cell > 0), so the accent outline is used
for the narrower distinction instead: the four that also exceeded their own
parent on the composite design score.  The subtitle states this explicitly.

A PNG preview is composed with PIL using the identical layout, because no SVG
rasteriser is installed on this machine.

Outputs into ``paper_figs/``:
  FigS2_dual_improvers9_2D.svg   vector, submission-ready
  FigS2_dual_improvers9_2D.png   raster preview (300 dpi at 2x)
  FigS2_dual_improvers9_2D.csv   the numbers behind the labels
"""
import io
import os

os.environ.setdefault("MPLCONFIGDIR", "/home/geng/vcdd-0.3.0/tmp/mplcache")

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path("/home/geng/vcdd-0.3.0")
RUN = ROOT / "dnapkcs_affinity500"
FIG = ROOT / "paper_figs"
FIG.mkdir(exist_ok=True)

# Layout, shared by the SVG and the PNG so the two cannot drift apart.
# Nine panels fill a 3 x 3 grid exactly, so no cell is left empty.
PANEL_W, PANEL_H = 520, 368
LABEL_H = 100
GAP_X, GAP_Y = 16, 30
MARGIN = 24
HEADER_H = 116
NCOL, NROW = 3, 3
GRID_W = NCOL * PANEL_W + (NCOL - 1) * GAP_X
GRID_H = NROW * (PANEL_H + LABEL_H) + (NROW - 1) * GAP_Y
W = 2 * MARGIN + GRID_W
H = HEADER_H + GRID_H + MARGIN

C_DARK, C_GREY, C_BORDER = "#1F2A37", "#6B7280", "#D1D5DB"
C_ACCENT = "#0F766E"   # also exceeded the parent on Dscore

TITLE = "Nine Stage 4 analogs that improved both the docking and cell objectives"
SUB1 = ("Ranked by Dscore.  Each panel gives the analog Dscore, its change versus the parent, "
        "and the changes in the two primary objectives.")
SUB2 = ("All nine lowered Δdock and raised Δcell; panels outlined in teal also exceeded "
        "their parent on Dscore.")


def load_dual() -> pd.DataFrame:
    """Analogs that improved both primary objectives, best Dscore first."""
    a = pd.read_parquet(RUN / "stage4" / "analogs.parquet")
    d = a[(a["delta_dock_score"] < 0) & (a["delta_cell_objective"] > 0)].copy()
    if d.empty:
        raise SystemExit("no analog improved both objectives")
    if d["analog_smiles"].isna().any():
        raise SystemExit("a dual improver has no SMILES")
    d["dscore_up"] = d["delta_design_score"] > 0
    return d.sort_values("design_score", ascending=False).reset_index(drop=True)


def labels_for(row: pd.Series, rank: int) -> tuple[str, str, str]:
    return (
        f"{rank}.  {row['analog_id']}",
        f"Dscore {row['design_score']:.4f}    ΔDscore {row['delta_design_score']:+.4f}",
        f"parent {row['parent_id']}    Δdock {row['delta_dock_score']:+.2f}    "
        f"Δcell {row['delta_cell_objective']:+.4f}",
    )


def panel_origin(i: int) -> tuple[int, int]:
    row, col = divmod(i, NCOL)
    return (MARGIN + col * (PANEL_W + GAP_X),
            HEADER_H + row * (PANEL_H + LABEL_H + GAP_Y))


def render_svg(mol: Chem.Mol) -> str:
    """Inner XML of one molecule drawn on a PANEL_W x PANEL_H canvas."""
    drawer = rdMolDraw2D.MolDraw2DSVG(PANEL_W, PANEL_H)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.4
    opts.minFontSize = 15
    opts.maxFontSize = 19
    opts.padding = 0.07
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    start = svg.index(">", svg.index("<svg")) + 1
    return svg[start:svg.rindex("</svg>")]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


def build_svg(d: pd.DataFrame) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="{W / 2:.0f}" y="46" text-anchor="middle" '
        f'font-family="{FONT}" font-size="30" font-weight="bold" fill="{C_DARK}">'
        f'{esc(TITLE)}</text>',
        f'<text x="{W / 2:.0f}" y="80" text-anchor="middle" '
        f'font-family="{FONT}" font-size="19" fill="{C_GREY}">{esc(SUB1)}</text>',
        f'<text x="{W / 2:.0f}" y="104" text-anchor="middle" '
        f'font-family="{FONT}" font-size="19" fill="{C_ACCENT}">{esc(SUB2)}</text>',
    ]
    for i, (_, row) in enumerate(d.iterrows()):
        x, y = panel_origin(i)
        line1, line2, line3 = labels_for(row, i + 1)
        mol = Chem.MolFromSmiles(row["analog_smiles"])
        if mol is None:
            raise SystemExit(f"unparseable SMILES for {row['analog_id']}")
        AllChem.Compute2DCoords(mol)
        accent = bool(row["dscore_up"])
        stroke = C_ACCENT if accent else C_BORDER
        width = 2.4 if accent else 1
        c1 = C_ACCENT if accent else C_DARK
        parts += [
            f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H + LABEL_H}" '
            f'fill="none" stroke="{stroke}" stroke-width="{width}"/>',
            f'<g transform="translate({x},{y})">{render_svg(mol)}</g>',
            f'<text x="{x + 12}" y="{y + PANEL_H + 28}" font-family="{FONT}" '
            f'font-size="22" font-weight="bold" fill="{c1}">{esc(line1)}</text>',
            f'<text x="{x + 12}" y="{y + PANEL_H + 57}" font-family="{FONT}" '
            f'font-size="19" fill="{C_DARK}">{esc(line2)}</text>',
            f'<text x="{x + 12}" y="{y + PANEL_H + 84}" font-family="{FONT}" '
            f'font-size="16" fill="{C_GREY}">{esc(line3)}</text>',
        ]
    parts.append("</svg>")
    return "\n".join(parts)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(ttf / name), size)


def build_png(d: pd.DataFrame, path: Path, scale: int = 2) -> None:
    """Same layout as the SVG, rasterised with PIL."""
    img = Image.new("RGB", (W * scale, H * scale), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_sub = font(30 * scale, True), font(19 * scale)
    f_id, f_l2, f_l3 = font(22 * scale, True), font(19 * scale), font(16 * scale)
    draw.text((W * scale / 2, 46 * scale), TITLE, font=f_title, fill=C_DARK, anchor="mm")
    draw.text((W * scale / 2, 80 * scale), SUB1, font=f_sub, fill=C_GREY, anchor="mm")
    draw.text((W * scale / 2, 104 * scale), SUB2, font=f_sub, fill=C_ACCENT, anchor="mm")
    for i, (_, row) in enumerate(d.iterrows()):
        x, y = panel_origin(i)
        mol = Chem.MolFromSmiles(row["analog_smiles"])
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(PANEL_W * scale, PANEL_H * scale)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.4 * scale
        opts.minFontSize = 15 * scale
        opts.maxFontSize = 19 * scale
        opts.padding = 0.07
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        panel = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
        img.paste(panel, (x * scale, y * scale))
        accent = bool(row["dscore_up"])
        stroke = C_ACCENT if accent else C_BORDER
        width = 2.4 if accent else 1
        draw.rectangle([x * scale, y * scale, (x + PANEL_W) * scale,
                        (y + PANEL_H + LABEL_H) * scale],
                       outline=stroke, width=max(1, int(width * scale)))
        line1, line2, line3 = labels_for(row, i + 1)
        c1 = C_ACCENT if accent else C_DARK
        draw.text(((x + 12) * scale, (y + PANEL_H + 20) * scale), line1, font=f_id, fill=c1)
        draw.text(((x + 12) * scale, (y + PANEL_H + 49) * scale), line2, font=f_l2, fill=C_DARK)
        draw.text(((x + 12) * scale, (y + PANEL_H + 76) * scale), line3, font=f_l3, fill=C_GREY)
    img.save(path, dpi=(300, 300))


def main() -> None:
    d = load_dual()
    n_up = int(d["dscore_up"].sum())
    print(f"{len(d)} analogs improved both objectives; {n_up} of them also "
          f"exceeded their parent on Dscore")

    svg_path = FIG / "FigS2_dual_improvers9_2D.svg"
    svg_path.write_text(build_svg(d), encoding="utf-8")
    print(f"wrote {svg_path.relative_to(ROOT)}  "
          f"({svg_path.stat().st_size / 1024:.0f} KB, vector)")

    png_path = FIG / "FigS2_dual_improvers9_2D.png"
    build_png(d, png_path)
    print(f"wrote {png_path.relative_to(ROOT)}  "
          f"({png_path.stat().st_size / 1024:.0f} KB preview)")

    cols = ["analog_id", "parent_id", "generator", "design_score",
            "delta_design_score", "delta_dock_score", "delta_cell_objective",
            "dscore_up", "analog_smiles"]
    csv = FIG / "FigS2_dual_improvers9_2D.csv"
    d[cols].to_csv(csv, index=False)
    print(f"wrote {csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
