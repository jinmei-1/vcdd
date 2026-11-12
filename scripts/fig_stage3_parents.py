#!/usr/bin/env python3
"""Figure: 2D structures of the ten Stage 3 parents carried into Stage 4.

The figure is assembled directly from RDKit's SVG renderer, so the structures
are true vector paths (submission quality) rather than embedded bitmaps.  Each
panel is labelled with the Stage 3 rank, the compound ID, the composite
virtual-cell objective, its three components, the post-hoc repair-inhibition
readout, and the Stage 1 docking score.

A PNG preview is composed with PIL using the identical layout, because no
SVG rasteriser is installed on this machine.

Outputs into ``paper_figs/``:
  Fig_stage3_parents_2D.svg   vector, submission-ready
  Fig_stage3_parents_2D.pdf   vector (via cairosvg if available, else skipped)
  Fig_stage3_parents_2D.png   raster preview
  Fig_stage3_parents_2D.csv   the numbers behind the labels
"""
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

N_PARENTS = 10
REPAIR_GENES = [
    "BRCA1", "RAD51C", "CHEK1", "CHEK2", "MSH6", "LIG1", "PCNA", "RFC2",
    "RFC5", "RPA1", "RPA2", "RPA3", "DDB2", "TP53BP1", "TP53", "TOP2A",
]

# Layout, shared by the SVG and the PNG so the two cannot drift apart.
PANEL_W, PANEL_H = 520, 400
LABEL_H = 84
GAP_X, GAP_Y = 14, 30
MARGIN = 24
HEADER_H = 104
NCOL, NROW = 5, 2
GRID_W = NCOL * PANEL_W + (NCOL - 1) * GAP_X
GRID_H = NROW * (PANEL_H + LABEL_H) + (NROW - 1) * GAP_Y
W = 2 * MARGIN + GRID_W
H = HEADER_H + GRID_H + MARGIN

C_DARK, C_GREY, C_BORDER = "#1F2A37", "#6B7280", "#D1D5DB"


def load_parents() -> pd.DataFrame:
    obj = pd.read_parquet(RUN / "stage3" / "objective_scores.parquet")
    sig = pd.read_parquet(RUN / "stage3" / "signatures" / "xpert.parquet")
    repair = (
        -sig[sig["gene"].isin(REPAIR_GENES)].groupby("mol_id")["value"].mean()
    ).rename("repair")
    top = obj.nlargest(N_PARENTS, "objective").merge(repair, on="mol_id", how="left")
    if top["smiles"].isna().any():
        raise SystemExit("a parent has no SMILES in objective_scores")
    return top.sort_values("objective", ascending=False).reset_index(drop=True)


def labels_for(row: pd.DataFrame, rank: int) -> tuple[str, str, str]:
    return (
        f"{rank}.  {row['mol_id']}",
        f"Objective {row['objective']:.3f}    DDR {row['gene_score']:.3f}    "
        f"Vina {row['stage1_score']:.1f}",
        f"PRKDC {row['target_consistency']:.3f}    spec {row['specificity']:.3f}    "
        f"repair {row['repair']:.2f}",
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
    opts.minFontSize = 16
    opts.maxFontSize = 20
    opts.padding = 0.07
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    start = svg.index(">", svg.index("<svg")) + 1
    return svg[start:svg.rindex("</svg>")]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(top: pd.DataFrame) -> str:
    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="{W / 2:.0f}" y="46" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
        f'font-size="30" font-weight="bold" fill="{C_DARK}">'
        f'Ten Stage 3 parents advanced to Stage 4 analog design</text>',
        f'<text x="{W / 2:.0f}" y="80" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
        f'font-size="20" fill="{C_GREY}">'
        f'Ranked by the composite virtual-cell objective; DDR, PRKDC and '
        f'specificity are its three components, and repair is the post-hoc '
        f'repair-inhibition readout</text>',
    ]
    for i, (_, row) in enumerate(top.iterrows()):
        x, y = panel_origin(i)
        line1, line2, line3 = labels_for(row, i + 1)
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise SystemExit(f"unparseable SMILES for {row['mol_id']}")
        AllChem.Compute2DCoords(mol)
        parts += [
            f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H + LABEL_H}" '
            f'fill="none" stroke="{C_BORDER}" stroke-width="1"/>',
            f'<g transform="translate({x},{y})">{render_svg(mol)}</g>',
            f'<text x="{x + 10}" y="{y + PANEL_H + 26}" '
            f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
            f'font-size="22" font-weight="bold" fill="{C_DARK}">{esc(line1)}</text>',
            f'<text x="{x + 10}" y="{y + PANEL_H + 52}" '
            f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
            f'font-size="19" fill="{C_DARK}">{esc(line2)}</text>',
            f'<text x="{x + 10}" y="{y + PANEL_H + 74}" '
            f'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" '
            f'font-size="17" fill="{C_GREY}">{esc(line3)}</text>',
        ]
    parts.append("</svg>")
    return "\n".join(parts)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(ttf / name), size)


def build_png(top: pd.DataFrame, path: Path, scale: int = 2) -> None:
    """Same layout as the SVG, rasterised with PIL."""
    img = Image.new("RGB", (W * scale, H * scale), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_sub = font(30 * scale, True), font(20 * scale)
    f_id, f_l2, f_l3 = font(22 * scale, True), font(19 * scale), font(17 * scale)
    draw.text((W * scale / 2, 46 * scale), "Ten Stage 3 parents advanced to Stage 4 "
              "analog design", font=f_title, fill=C_DARK, anchor="mm")
    draw.text((W * scale / 2, 80 * scale),
              "Ranked by the composite virtual-cell objective; DDR, PRKDC and specificity "
              "are its three components, and repair is the post-hoc repair-inhibition readout",
              font=f_sub, fill=C_GREY, anchor="mm")
    for i, (_, row) in enumerate(top.iterrows()):
        x, y = panel_origin(i)
        mol = Chem.MolFromSmiles(row["smiles"])
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(PANEL_W * scale, PANEL_H * scale)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.4 * scale
        opts.minFontSize = 16 * scale
        opts.maxFontSize = 20 * scale
        opts.padding = 0.07
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        import io

        panel = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
        img.paste(panel, (x * scale, y * scale))
        draw.rectangle([x * scale, y * scale, (x + PANEL_W) * scale,
                        (y + PANEL_H + LABEL_H) * scale], outline=C_BORDER, width=scale)
        line1, line2, line3 = labels_for(row, i + 1)
        draw.text(((x + 10) * scale, (y + PANEL_H + 18) * scale), line1, font=f_id, fill=C_DARK)
        draw.text(((x + 10) * scale, (y + PANEL_H + 44) * scale), line2, font=f_l2, fill=C_DARK)
        draw.text(((x + 10) * scale, (y + PANEL_H + 66) * scale), line3, font=f_l3, fill=C_GREY)
    img.save(path, dpi=(300, 300))


def main() -> None:
    top = load_parents()
    svg_path = FIG / "Fig_stage3_parents_2D.svg"
    svg_path.write_text(build_svg(top), encoding="utf-8")
    print(f"wrote {svg_path.relative_to(ROOT)}  ({svg_path.stat().st_size / 1024:.0f} KB, vector)")

    png_path = FIG / "Fig_stage3_parents_2D.png"
    build_png(top, png_path)
    print(f"wrote {png_path.relative_to(ROOT)}  ({png_path.stat().st_size / 1024:.0f} KB preview)")

    try:
        import cairosvg

        pdf = FIG / "Fig_stage3_parents_2D.pdf"
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf))
        print(f"wrote {pdf.relative_to(ROOT)}")
    except ImportError:
        print("cairosvg not installed; SVG is the vector deliverable "
              "(Inkscape/Illustrator convert it to PDF/EPS)")

    cols = ["mol_id", "objective", "gene_score", "target_consistency",
            "specificity", "repair", "stage1_score"]
    csv = FIG / "Fig_stage3_parents_2D.csv"
    top[cols].to_csv(csv, index=False)
    print(f"wrote {csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
