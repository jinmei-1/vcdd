#!/usr/bin/env python3
"""Bundle a clean data attachment (zip) for JCIM submission:
tables (parquet/csv), all 500 stage-2 structures (CIF + confidence JSON),
config, and a data README."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/home/geng/vcdd-0.3.0")
RUN = ROOT / "dnapkcs_run"
ATT = ROOT / "paper" / "data_attachment"
STR = ATT / "structures"
ZIP = ROOT / "paper" / "Data_Attachment_JCIM.zip"

for d in (ATT, STR):
    d.mkdir(parents=True, exist_ok=True)

# ---- tables ----
tables = [
    RUN / "stage1/ranking.parquet",
    RUN / "stage2/consensus_ranking.parquet",
    RUN / "stage2/scores.parquet",
    RUN / "stage2/pose_qc.parquet",
    RUN / "stage3/objective_scores.parquet",
    RUN / "stage3/objective_by_backend.parquet",
    RUN / "stage3/signatures/xpert.parquet",
    RUN / "stage3/signatures/xpert_meta.parquet",
    RUN / "stage3/top50_ddr_inhibitors.csv",
    RUN / "stage3/stage3_summary.json",
    RUN / "stage4/analogs.parquet",
    RUN / "stage4/analogs_raw.parquet",
    RUN / "stage4/cell_scores.parquet",
    RUN / "stage4/pareto.parquet",
    RUN / "stage4/redock.parquet",
    RUN / "stage4/top_designs.sdf",
    RUN / "stage4/stage4_summary.json",
]
for t in tables:
    if t.exists():
        dst = ATT / "tables" / t.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(t, dst)

# ---- structures: flatten 500 CIF + confidence JSON ----
# The molecule id is the run directory under local/, i.e.
# local/<mol_id>/out/boltz_results_input/predictions/input/input_model_0.cif.
# Using c.parent.parent.name gave "predictions" for every file, so all 500
# structures were copied onto the same name and the bundle shipped one complex.
cifs = list(RUN.glob("stage2/boltz/local/*/out/boltz_results_input/predictions/*/*_model_0.cif"))
confs = list(RUN.glob("stage2/boltz/local/*/out/boltz_results_input/predictions/*/confidence_*_model_0.json"))
n_cif = n_conf = 0


def _mol_id(path):
    """Molecule id from a path inside a local Boltz run directory."""
    for parent in path.parents:
        if parent.parent.name == "local":
            return parent.name
    raise ValueError(f"cannot derive a molecule id from {path}")


for c in cifs:
    shutil.copy(c, STR / f"{_mol_id(c)}.cif")
    n_cif += 1
for c in confs:
    shutil.copy(c, STR / f"{_mol_id(c)}_confidence.json")
    n_conf += 1
n_mol = len({_mol_id(c) for c in cifs})
if cifs and n_mol != n_cif:
    raise RuntimeError(f"{n_cif} structure files collapsed onto {n_mol} molecule ids")

# ---- config + MSA ----
shutil.copy(ROOT / "run_dnapkcs.yaml", ATT / "run_dnapkcs.yaml")
if (ROOT / "msa_out/receptor_chain_A_vcdd.a3m").exists():
    shutil.copy(ROOT / "msa_out/receptor_chain_A_vcdd.a3m", ATT / "receptor_chain_A_vcdd.a3m")

# ---- data README ----
readme = f"""# Data attachment — Virtual-Cell-Guided Drug Discovery (DNA-PKcs)

This bundle accompanies the manuscript and contains the full computational output.

## tables/
  stage1/ranking.parquet          Vina-GPU scores of 167,342 molecules (columns: mol_id, smiles, score, rank)
  stage2/consensus_ranking.parquet 500 compounds: stage1_score, ipTM, ptm, ligand_iptm, consensus_score/rank
  stage2/scores.parquet           per-stage-2 raw scores
  stage2/pose_qc.parquet          pose QC (NOTE: coordinate-frame check requires receptor superposition)
  stage3/objective_scores.parquet 200 compounds: gene_score, target_consistency, objective, objective_rank
  stage3/signatures/xpert.parquet 200 x 978 genes log2FC (long format: mol_id, backend, gene, value)
  stage3/top50_ddr_inhibitors.csv top-50 DDR-inhibitory candidates (ddr_rank, smiles, scores)
  stage4/analogs.parquet          388 analogs: analog_smiles, transformation, delta_* scores, design_score
  stage4/pareto.parquet           Pareto-optimal analogs
  stage4/top_designs.sdf          100 top analog structures (SDF)

## structures/
  <mol_id>.cif                    Boltz-2 predicted complex (chain A = DNA-PKcs pocket fragment,
                                  chain LIG = ligand) — 500 compounds
  <mol_id>_confidence.json        confidence metrics (ipTM, ptm, ligand_iptm, pLDDT ...)

## other
  run_dnapkcs.yaml                full pipeline configuration (all stage parameters)
  receptor_chain_A_vcdd.a3m       offline MSA for the receptor (sequence-consistent)

Generated {json.dumps(str(Path.cwd()))}
Structures: {n_cif} CIF, {n_conf} confidence JSON.
"""
(ATT / "README.txt").write_text(readme)

# ---- zip ----
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(ATT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(ATT))
print("zip:", ZIP, ZIP.stat().st_size // (1024 * 1024), "MB")
print("files: tables", len(list((ATT / "tables").rglob("*"))),
      "| cifs", n_cif, "| unique molecules", n_mol, "| conf", n_conf)