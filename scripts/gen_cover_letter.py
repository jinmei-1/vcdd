#!/usr/bin/env python3
"""Cover letter (Word) for JCIM submission."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path("/home/geng/vcdd-0.3.0")
OUT = ROOT / "paper/Cover_Letter_JCIM.docx"

doc = Document()
for s in doc.sections:
    s.left_margin = Cm(2.2); s.right_margin = Cm(2.2)

def cn(run, text=None, bold=False, italic=False, size=None):
    if text is None:
        text = run.text
    run.text = text
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.bold = bold; run.italic = italic
    if size:
        run.font.size = Pt(size)

def para(text="", size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space=8):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.space_after = Pt(space)
    if text:
        cn(p.add_run(text), text, bold=bold, italic=italic, size=size)
    return p

para("", size=6)
para("Date: <date>", size=11)
para("")
para("The Editor-in-Chief", size=11)
para("Journal of Chemical Information and Modeling", size=11)
para("American Chemical Society", size=11)
para("")
para("Dear Editor,", size=11, space=10)

para("We are pleased to submit our manuscript entitled:", size=11)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(p.add_run("\u201cVirtual-Cell-Guided Drug Discovery: An End-to-End Platform and Its Application to "
             "DNA-PKcs Inhibitors Targeting DNA Damage Repair\u201d"), bold=True, italic=True, size=12)
para("for consideration as a research article in the Journal of Chemical Information and Modeling.",
     size=11)

para("Computational drug discovery has long been limited by the gap between docking scores and "
     "cellular outcomes. In this work we present vcdd, an open, resumable, engine-agnostic platform "
     "that orchestrates four stages of a virtual-cell-guided campaign: library-scale docking "
     "(Vina-GPU 2.1), co-folding refinement (Boltz-2, locally deployed on two GPUs with offline MSAs), "
     "perturbation scoring in a virtual cell built on the LINCS L1000 transcriptional model XPert, "
     "and generative analog optimization (constrained GA + R-group) with Vina re-docking. A SQLite "
     "ledger makes every stage resumable at single-molecule granularity.", size=11)

para("We validate the platform end-to-end on DNA-PKcs, the master kinase of non-homologous end "
     "joining, with the objective of identifying compounds that inhibit DNA damage repair. From a "
     "200,000-molecule generative library, 167,342 molecules were docked, 500 were re-scored with "
     "Boltz-2, and 200 were subjected to transcriptomic perturbation prediction. We introduce a "
     "dedicated DNA-repair inhibition signature across 16 core repair genes and recommend five "
     "molecules that combine strong docking, high structure confidence, DDR activation, and PRKDC "
     "consistency. Generative optimization of the top ten parents produced 457 guardrail-passing "
     "analogs, 47% with improved cell objective and 80% with improved docking, together with 204 "
     "extractable SAR rules.", size=11)

para("We believe these results are of broad interest to the JCIM readership because (i) they provide "
     "a transferable, reproducible template for phenotype-aware computational drug discovery, (ii) "
     "they demonstrate robust local deployment of Boltz-2 with fully offline MSAs, and (iii) they "
     "connect transcriptomic signatures directly to chemical optimization for an oncology-relevant "
     "target.", size=11)

para("This manuscript is original, has not been published previously, and is not under consideration "
     "for publication elsewhere. All authors have approved the submission and declare no competing "
     "financial interests. All pipeline artifacts, code, and configuration are available (Data and "
     "Code Availability statement in the manuscript).", size=11)

para("Suggested reviewers (optional):", bold=True, size=11)
para("• <Reviewer 1 name>, <affiliation> — virtual cell / transcriptomics-driven drug discovery", size=10)
para("• <Reviewer 2 name>, <affiliation> — structure prediction / Boltz / AlphaFold applications", size=10)
para("• <Reviewer 3 name>, <affiliation> — DNA damage response inhibitors / kinase drug design", size=10)
para("")

para("Thank you for your consideration. We look forward to your response.", size=11)
para("")
para("Sincerely,", size=11)
para("")
para("<Corresponding author name>", size=11, bold=True)
para("<Department / Institution>", size=11)
para("<Email>  |  <Phone>", size=11)

doc.save(OUT)
print("saved:", OUT, OUT.stat().st_size // 1024, "KB")