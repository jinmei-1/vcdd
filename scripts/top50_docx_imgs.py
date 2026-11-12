#!/usr/bin/env python3
"""DNA-PKcs Top50 DDR-inhibition candidates report WITH 2D structures (Word)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path("/home/geng/vcdd-0.3.0")
CSV = ROOT / "dnapkcs_run/stage3/top50_ddr_inhibitors.csv"
OUT = ROOT / "dnapkcs_run/stage3/DNA-PKcs_前50_DDR抑制候选_含结构图.docx"

top = pd.read_csv(CSV)

doc = Document()
for s in doc.sections:
    s.left_margin = Cm(1.5); s.right_margin = Cm(1.5)

def cn(run, text=None, bold=False, size=None, color=None):
    if text is None:
        text = run.text
    run.text = text
    run.font.name = "Liberation Sans"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color

def para(text, size=10.5, bold=False, align=None):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    cn(p.add_run(text), text, bold=bold, size=size)
    return p

def mol_image(smiles, w=300, h=225):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    img = Draw.MolToImage(m, size=(w, h))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---- title ----
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(t.add_run("DNA-PKcs 抑制型候选 Top50（含 2D 结构）"), bold=True, size=18)
st = doc.add_paragraph(); st.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(st.add_run("加入化合物后抑制 DNA 损伤修复 —— 前 50 化合物（每候选附二维结构图）"),
   size=11, color=RGBColor(0x44, 0x44, 0x44))

para("一、筛选逻辑", 12, bold=True)
para("目标表型：化合物处理引起 DNA 损伤修复抑制。评分维度三合一：① DDR 激活签名 gene_score"
     "（CDKN1A/GADD45A 上调）；② DNA 修复抑制签名 repair_inhibition（16 个核心修复基因 BRCA1, "
     "RAD51C, CHEK1/2, MSH6, LIG1, PCNA, RFC2/5, RPA1/2/3, DDB2, TP53BP1, TP53, TOP2A 平均下调，"
     "正值=修复被抑制）；③ 靶点一致性 target_consistency（PRKDC 敲除共识）。")
para("综合分 ddr_inhibition = 0.6×修复抑制 + 0.4×DDR激活，取前 50。数据链：20 万分子 → Vina-GPU 初筛 → "
     "Boltz-2 精筛 top500 → XPert 转录组预测 200×978 基因（A375，10 µM，24 h）。")

para("二、Top 50 候选（附二维结构）", 12, bold=True)
para("每行为一个候选：左列为二维结构图，右列为编号、SMILES、对接/结构/转录组分数。", size=9.5)

for _, r in top.iterrows():
    buf = mol_image(r["smiles"])
    row_tb = doc.add_table(rows=1, cols=2)
    row_tb.style = "Table Grid"; row_tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0, c1 = row_tb.rows[0].cells
    c0.width = Cm(5.2); c1.width = Cm(11.5)
    if buf is not None:
        run = c0.paragraphs[0].add_run()
        run.add_picture(buf, width=Cm(4.2), height=Cm(3.2))
    c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    lines = [
        f"#{int(r['ddr_rank'])}  {r['mol_id']}",
        f"SMILES: {r['smiles']}",
        f"修复抑制 {r['repair_inhibition']:+.2f}  |  DDR激活 {r['gene_score']:.2f}  |  "
        f"PRKDC一致 {r['target_consistency']:.2f}",
        f"dock {r['dock_score']:.1f} kcal/mol  |  ipTM {r['iptm']:.2f}  |  "
        f"objective {r['objective']:.3f}",
    ]
    for li, ln in enumerate(lines):
        p = c1.paragraphs[0] if li == 0 else c1.add_paragraph()
        cn(p.add_run(ln), ln, bold=(li == 0), size=9 if li else 10)

para("三、结论与用法", 12, bold=True)
para("Top50 化合物同时满足：DNA 修复核心基因显著下调、损伤响应基因上调、与 DNA-PKcs 敲除表型一致。"
     "建议：① 前 10 名优先做 DNA-PKcs 激酶 IC50 与 γH2AX 焦点实验；② 与放疗/化疗联用增敏设计；"
     "③ 结构聚类选骨架代表做 SAR。")

doc.save(OUT)
print("saved:", OUT, OUT.stat().st_size // 1024, "KB")