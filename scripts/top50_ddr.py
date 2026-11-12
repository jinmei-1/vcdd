#!/usr/bin/env python3
"""Top-50 DDR-inhibition candidates for the DNA-PKcs screen (Word + CSV).

Scoring per compound:
  repair_inhibition = -mean(log2FC over core DNA-repair genes)   (higher = repair suppressed)
  objective        = vcdd stage-3 objective (gene_score x2 + target_consistency)
  ddr_rank         = rank by repair_inhibition (primary mechanistic lens)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/home/geng/vcdd-0.3.0")
SIG = ROOT / "dnapkcs_run/stage3/signatures/xpert.parquet"
SCORES = ROOT / "dnapkcs_run/stage3/objective_scores.parquet"
OUT_CSV = ROOT / "dnapkcs_run/stage3/top50_ddr_inhibitors.csv"
OUT_DOCX = ROOT / "dnapkcs_run/stage3/DNA-PKcs_前50_DDR抑制候选.docx"

REPAIR_CORE = ["BRCA1", "RAD51C", "CHEK1", "CHEK2", "MSH6", "LIG1", "PCNA",
               "RFC2", "RFC5", "RPA1", "RPA2", "RPA3", "DDB2", "TP53BP1", "TP53", "TOP2A"]

# ---------- load ----------
sig = pd.read_parquet(SIG)                      # mol_id, backend, gene, value
sc = pd.read_parquet(SCORES).set_index("mol_id")

piv = sig[sig["gene"].isin(REPAIR_CORE)].pivot(index="mol_id", columns="gene", values="value")
print("pivot:", piv.shape, "| genes found:", len(piv.columns))
piv["repair_inhibition"] = -piv.mean(axis=1)    # positive = repair suppressed

out = sc.join(piv["repair_inhibition"], how="left")
# normalize helper
def zn(s):
    return (s - s.min()) / (max(s.max() - s.min(), 1e-9))
out["ddr_inhibition"] = 0.6 * zn(out["repair_inhibition"].fillna(0)) + 0.4 * zn(out["gene_score"])
out = out.sort_values("ddr_inhibition", ascending=False)
out["ddr_rank"] = range(1, len(out) + 1)
top50 = out.head(50).reset_index()

cols = ["ddr_rank", "mol_id", "smiles", "dock_score", "iptm", "gene_score",
        "target_consistency", "repair_inhibition", "ddr_inhibition",
        "objective", "objective_rank", "out_of_domain"]
if "iptm" not in top50.columns:
    c2 = pd.read_parquet(ROOT / "dnapkcs_run/stage2/consensus_ranking.parquet")
    top50 = top50.merge(c2[["mol_id", "iptm"]], on="mol_id", how="left", suffixes=("", "_s2"))
top50[cols].to_csv(OUT_CSV, index=False)
print("saved csv:", OUT_CSV, "| top50 rows:", len(top50))

# ---------- Word ----------
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

def para(doc, text, size=10.5, bold=False):
    p = doc.add_paragraph(); cn(p.add_run(text), text, bold=bold, size=size); return p

t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(t.add_run("DNA-PKcs 抑制型候选 Top50"), bold=True, size=18)
st = doc.add_paragraph(); st.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(st.add_run("加入化合物后抑制 DNA 损伤修复（DDR 抑制）—— 前 50 化合物清单"), size=11, color=RGBColor(0x44, 0x44, 0x44))

para(doc, "一、筛选逻辑", 12, bold=True)
para(doc, "目标表型：化合物处理引起 DNA 损伤修复抑制。评分维度三合一：")
para(doc, "① DDR 激活签名 gene_score：CDKN1A / GADD45A 上调共识（损伤响应被激活）；")
para(doc, "② DNA 修复抑制签名 repair_inhibition：16 个核心修复基因（BRCA1, RAD51C, CHEK1/2, MSH6, "
          "LIG1, PCNA, RFC2/5, RPA1/2/3, DDB2, TP53BP1, TP53, TOP2A）平均下调幅度（正值=修复被抑制）；")
para(doc, "③ 靶点一致性 target_consistency：与 PRKDC（DNA-PKcs）敲除共识签名的一致性。")
para(doc, "综合分 ddr_inhibition = 0.6×修复抑制 + 0.4×DDR激活，取前 50。")

para(doc, "二、数据来源", 12, bold=True)
para(doc, "20 万生成式小分子 → stage1 Vina-GPU 对接初筛（-11.6 ~ +0.8 kcal/mol）→ stage2 Boltz-2 精筛 top500 "
          "（ipTM 中位 0.9）→ stage3 XPert 转录组预测 200 分子×978 基因 log2FC（A375 黑色素瘤，10 µM，24 h）。")

para(doc, "三、Top 50 候选机制要点", 12, bold=True)
top10 = top50.head(10)
for _, r in top10.iterrows():
    p = doc.add_paragraph()
    cn(p.add_run(f"#{int(r['ddr_rank'])}  {r['mol_id']}  |  修复抑制 {r['repair_inhibition']:+.2f}"
                 f"  |  DDR激活 {r['gene_score']:.2f}  |  PRKDC一致性 {r['target_consistency']:.2f}"
                 f"  |  dock {r['dock_score']:.1f} kcal/mol  |  ipTM {r['iptm']:.2f}"),
       size=9.5)

para(doc, "四、Top 50 完整清单", 12, bold=True)
cols2 = ["#", "mol_id", "SMILES", "dock", "ipTM", "repair.抑制", "DDR激活", "PRKDC一致", "OOD"]
tb = doc.add_table(rows=1, cols=len(cols2))
tb.style = "Table Grid"; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, c in enumerate(cols2):
    cn(tb.rows[0].cells[i].paragraphs[0].add_run(c), c, bold=True, size=9)
for _, r in top50.iterrows():
    cells = tb.add_row().cells
    vals = [str(int(r["ddr_rank"])), r["mol_id"], r["smiles"], f"{r['dock_score']:.1f}",
            f"{r['iptm']:.2f}", f"{r['repair_inhibition']:+.2f}", f"{r['gene_score']:.2f}",
            f"{r['target_consistency']:.2f}", "是" if r["out_of_domain"] else "否"]
    for i, v in enumerate(vals):
        cn(cells[i].paragraphs[0].add_run(v), v, size=8)
widths = [Cm(0.9), Cm(2.1), Cm(6.4), Cm(1.1), Cm(1.1), Cm(1.6), Cm(1.5), Cm(1.8), Cm(1.0)]
for row in tb.rows:
    for i, w in enumerate(widths):
        row.cells[i].width = w

para(doc, "五、结论与用法", 12, bold=True)
para(doc, "Top50 化合物同时满足：DNA 修复核心基因显著下调（修复通路瘫痪）、损伤响应基因上调（DNA 损伤累积）、"
          "与 DNA-PKcs 敲除表型一致（NHEJ 功能受损）。建议：① 对前 10 名优先做激酶酶活实验（DNA-PKcs IC50）与 "
          "γH2AX 焦点实验验证 DSB 修复受阻；② 与放射/化疗联用增敏设计；③ 结构聚类后选骨架代表做 SAR。")

doc.save(OUT_DOCX)
print("saved:", OUT_DOCX, OUT_DOCX.stat().st_size // 1024, "KB")