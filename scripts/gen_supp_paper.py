#!/usr/bin/env python3
"""Supplementary Materials (Word) — full data appendix for the JCIM manuscript."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path("/home/geng/vcdd-0.3.0")
OUT = ROOT / "paper/Supplementary_Materials.docx"
ROOT.joinpath("paper").mkdir(exist_ok=True)

doc = Document()
for s in doc.sections:
    s.left_margin = Cm(1.8); s.right_margin = Cm(1.8)

def cn(run, text=None, bold=False, italic=False, size=None):
    if text is None:
        text = run.text
    run.text = text
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.bold = bold; run.italic = italic
    if size:
        run.font.size = Pt(size)

def para(text="", size=10.5, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.space_after = Pt(5)
    if text:
        cn(p.add_run(text), text, bold=bold, italic=italic, size=size)
    return p

def h1(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(5)
    cn(p.add_run(text), text, bold=True, size=13)
    return p

def h2(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(7); p.paragraph_format.space_after = Pt(4)
    cn(p.add_run(text), text, bold=True, italic=True, size=11)
    return p

def table_from(df, col_widths=None, font_size=8):
    tb = doc.add_table(rows=1, cols=len(df.columns))
    tb.style = "Table Grid"; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cn(tb.rows[0].cells[i].paragraphs[0].add_run(str(c)), str(c), bold=True, size=font_size)
    for _, r in df.iterrows():
        cells = tb.add_row().cells
        for i, v in enumerate(r):
            cn(cells[i].paragraphs[0].add_run(str(v)), str(v), size=font_size)
    return tb

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(p.add_run("Supplementary Materials"), bold=True, size=16)
para("Supporting data for: Virtual-Cell-Guided Drug Discovery: An End-to-End Platform and Its "
     "Application to DNA-PKcs Inhibitors Targeting DNA Damage Repair.", size=11)

# ---------------- S1 ----------------
h1("S1. Pipeline statistics")
h2("S1.1 Stage 1 — Vina-GPU docking (n = 167,342)")
s1 = pd.read_parquet(ROOT / "dnapkcs_run/stage1/ranking.parquet")
q = s1["score"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(2)
table_from(pd.DataFrame({"quantile": ["min", "5%", "25%", "median", "75%", "95%", "max"],
                         "score": q.values}))
para("Counts: score < −9.0: 28,167; < −8.0: 99,818; > −6.0: 933. Library: 200,000 generative "
     "drug-like molecules; 32,658 failed (non-dockable or sanitization rejections).")

h2("S1.2 Stage 2 — Boltz-2 co-folding (n = 500)")
s2 = pd.read_parquet(ROOT / "dnapkcs_run/stage2/consensus_ranking.parquet")
q2 = s2["iptm"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(3)
table_from(pd.DataFrame({"quantile": ["min", "5%", "25%", "median", "75%", "95%", "max"],
                         "iptm": q2.values}))
para("Counts: composite confidence 0.82–0.937 (all 500 clear the 0.70 gate); ipTM ≥ 0.9: 374; "
     "≥ 0.8: 486; ≥ 0.6: 500. All 500 complexes predicted, 0 failures. "
     "Runtime ≈7.9 h wall time on 2× RTX 3090 Ti.")

h2("S1.3 Stage 3 — virtual-cell objective (n = 200)")
s3 = pd.read_parquet(ROOT / "dnapkcs_run/stage3/objective_scores.parquet")
for col in ["objective", "gene_score", "target_consistency", "specificity"]:
    q3 = s3[col].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(3)
    table_from(pd.DataFrame({"metric": [col], "min": [q3.iloc[0]], "25%": [q3.iloc[1]],
                             "median": [q3.iloc[2]], "75%": [q3.iloc[3]], "max": [q3.iloc[4]]}))

# ---------------- S2 ----------------
h1("S2. Full configuration (run_dnapkcs.yaml)")
cfg = (ROOT / "run_dnapkcs.yaml").read_text()
for ln in cfg.splitlines():
    para(ln, size=8.5, align=WD_ALIGN_PARAGRAPH.LEFT)

# ---------------- S3 ----------------
h1("S3. Software environment")
env = [
    ["Python", "3.11"],
    ["vcdd", "0.2.0"],
    ["Vina-GPU", "2.1 (quickvina2_gpu, source build)"],
    ["Boltz", "2.2.1"],
    ["cuequivariance", "0.11.1 (torch + ops_cu12 + ops_torch_cu12)"],
    ["PyTorch", "2.13.0 (CUDA 13)"],
    ["RDKit", "release (2024.x)"],
    ["XPert", "l1000_mdmt_warm_split.pth"],
    ["Uni-Mol", "mol_pre_all_h_220816.pt"],
    ["python-docx / matplotlib", "for reporting"],
    ["Hardware", "2× NVIDIA RTX 3090 Ti 24 GB, AMD/Intel host"],
    ["MSA", "ColabFold REST (single fetch) + offline .a3m, sanitized"],
]
table_from(pd.DataFrame(env, columns=["Component", "Version/Notes"]))

# ---------------- S4 ----------------
h1("S4. Top-50 DDR-inhibitory candidates (complete table)")
t50 = pd.read_csv(ROOT / "dnapkcs_run/stage3/top50_ddr_inhibitors.csv")
cols = ["ddr_rank", "mol_id", "smiles", "dock_score", "iptm", "gene_score",
        "target_consistency", "repair_inhibition", "ddr_inhibition", "objective", "out_of_domain"]
table_from(t50[cols], font_size=7)

# ---------------- S5 ----------------
h1("S5. Stage 4 — generative optimization")
h2("S5.1 Generator summary")
s4 = pd.read_parquet(ROOT / "dnapkcs_run/stage4/analogs.parquet")
s4["_impr_obj"] = (s4["cell_objective"] > s4["parent_cell_objective"]).astype(int)
s4["_impr_dock"] = (s4["dock_score"] > s4["parent_dock_score"]).astype(int)
g = s4.groupby("generator").agg(n=("analog_id", "count"),
                                improved_obj=("_impr_obj", "sum"),
                                improved_dock=("_impr_dock", "sum"))
g["frac_improved_obj"] = (g["improved_obj"] / g["n"]).round(3)
g["frac_improved_dock"] = (g["improved_dock"] / g["n"]).round(3)
table_from(g.reset_index())
h2("S5.2 Delta distributions (388 guardrail-passing analogs)")
delt = pd.DataFrame({
    "metric": ["Δ objective", "Δ dock (kcal/mol)", "Δ QED", "Δ SA score"],
    "min": [s4["delta_cell_objective"].min(), s4["delta_dock_score"].min(),
            s4["delta_qed"].min(), s4["delta_sa_score"].min()],
    "median": [s4["delta_cell_objective"].median(), s4["delta_dock_score"].median(),
               s4["delta_qed"].median(), s4["delta_sa_score"].median()],
    "max": [s4["delta_cell_objective"].max(), s4["delta_dock_score"].max(),
            s4["delta_qed"].max(), s4["delta_sa_score"].max()],
}).round(3)
table_from(delt)
h2("S5.3 SAR transformation buckets (counts and objective improvements)")
import re
def bucket(t):
    t = str(t); pats = []
    if "C -> N" in t: pats.append("ring C→N")
    if "N -> C" in t: pats.append("ring N→C")
    if "C-H -> C-CH3" in t or "C-H -> C-Cl" in t: pats.append("aryl H→Me/Cl")
    if "methyl -> ethyl" in t: pats.append("Me→Et")
    if "crossover" in t: pats.append("crossover")
    if "delete terminal" in t: pats.append("delete terminal")
    if "add methyl" in t: pats.append("add methyl")
    return " / ".join(pats) if pats else "other"
s4["bucket"] = s4["transformation"].map(bucket)
s4["_impr"] = (s4["cell_objective"] > s4["parent_cell_objective"]).astype(int)
bs = s4.groupby("bucket").agg(n=("analog_id", "count"), improved=("_impr", "sum")).sort_values("n", ascending=False)
table_from(bs.reset_index())

h2("S5.4 Guardrail rules (applied to every generated analog)")
para("• No PAINS alerts; • maximum ring size ≤ 6; • no bridgehead or spiro atoms; • heavy-atom "
     "count within +3/−2 of parent; • similarity to parent ≥ 0.5 (similarity_floor); • canonical "
     "sanitization and valence checks via RDKit. 462 proposed → 388 passed (84.0%).")

# ---------------- S6 ----------------
h1("S6. Scoring definitions")
para("• gene_score: fraction/level of DDR-activation consensus genes (CDKN1A, GADD45A) up-regulated "
     "by the predicted perturbation (XPert log2FC), scaled 0–1.", size=10.5)
para("• target_consistency: agreement of the predicted transcriptional profile with the "
     "PRKDC-knockout consensus signature (landmark genes), scaled 0–1.", size=10.5)
para("• objective = 2 × gene_score + target_consistency (Stage-3 ranking).", size=10.5)
para("• repair_inhibition = −mean(Δlog2FC) over 16 core repair genes {BRCA1, RAD51C, CHEK1, CHEK2, "
     "MSH6, LIG1, PCNA, RFC2, RFC5, RPA1, RPA2, RPA3, DDB2, TP53BP1, TP53, TOP2A}. Positive = repair "
     "gene suppression.", size=10.5)
para("• ddr_inhibition = 0.6 × z(repair_inhibition) + 0.4 × z(gene_score); top-50 by ddr_inhibition "
     "retained.", size=10.5)
para("• design_score (Stage 4) = weighted sum of normalized docking score, cell objective, QED, and "
     "(1 − SA). Pareto front defined on (design_score, cell_objective).", size=10.5)

doc.save(OUT)
print("saved:", OUT, OUT.stat().st_size // 1024, "KB")