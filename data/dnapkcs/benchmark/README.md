# DNA-PKcs retrospective recovery benchmark (v1)

Virtual-screening benchmark assembled for the vcdd manuscript, used to quantify the
retrospective enrichment performance of the stage-1 docking protocol on human DNA-PKcs.

## Contents

| File | Content |
|---|---|
| `dnapkcs_actives_clustered200.smi` / `.csv` | 200 actives (SMILES + ChEMBL ID): ChEMBL3142 (human DNA-PKcs) IC50/Ki ≤ 1 µM, MW < 600, rotatable bonds < 20; Bemis–Murcko scaffold clustering, most potent representative per scaffold (potency range 0.08–19 nM) |
| `dnapkcs_decoys.smi` / `.csv` | 4,524 unique decoys, property-matched per active following the DUD-E protocol across 6 descriptors (MW, cLogP, HBD, HBA, rotatable bonds, formal charge); ECFP4 Tanimoto < 0.35 to the matched active, inter-decoy Tc < 0.9 |
| `dnapkcs_benchmark.csv` / `.smi` | merged, shuffled evaluation table: `smiles, set_id, label` (1 = active, 0 = decoy), 4,724 rows |
| `dnapkcs_actives.csv` | all 1,190 ≤1 µM actives (unclustered), for alternative sampling |
| `dnapkcs_all_tested_best_nM.csv` | best potency of all 1,838 tested molecules; 40 with best potency > 30 µM are experimentally confirmed inactives (usable as hard decoys) |
| `dnapkcs_crop.pdb` | receptor: PDB 7OTW chain A, residues 3701–4123 (423-aa crop) |
| `azd7648_xtal.pdb` | co-crystallized AZD7648 ligand (docking-box definition and redocking control) |
| `dnapkcs_box.json` | docking box: center (154.925, 174.279, 209.114) Å, 17×17×17 Å |
| `run_docking_benchmark.py`, `run_docking_shard.py` | docking drivers (AutoDock Vina 1.2.5, exhaustiveness 8, num_modes 9) |
| `enrichment_analysis.py` | ROC-AUC, LogAUC, EF1%/5%/10%, BEDROC (α = 20) with 1,000-bootstrap 95% CIs |
| `plot_docking_results.py` | score-distribution and enrichment-curve figures |
| `results/` | Vina scores (`dnapkcs_benchmark_vina_scores.csv`), metrics (`dnapkcs_benchmark_vina_metrics.csv`), figures |

## Construction (reproducible)

1. **Actives**: ChEMBL API query for CHEMBL3142 IC50/Ki/Kd (nM) records → 3,085 records →
   1,837 unique molecules → most potent value per molecule → filters (≤1 µM, MW < 600,
   rotB < 20, parseable SMILES) → InChIKey deduplication → 1,190 → Murcko scaffold
   clustering (502 scaffolds) → 200 most potent representatives.
   Verified to contain AZD7648 (CHEMBL4439259, 0.6 nM), NU7441 (CHEMBL188678),
   NU7026 (CHEMBL104468), CC-115 (CHEMBL3586573, 13 nM) and
   nedisertib/peposertib (CHEMBL4297629, 0.36 nM).
2. **Decoys**: DUD-E decoy pool (ZINC-derived, property-matched, topologically
   dissimilar; obtained via the HuggingFace mirror `jglaser/binding_affinity`). Per active:
   property-window prefilter (MW ±15%, cLogP ±1.5, HBD ±1, HBA ±2, rotB ±3, same charge)
   → top 400 by standardized 6-D distance → ECFP4 Tc < 0.35 to the active and inter-decoy
   Tc < 0.9 → 50 per active → pooled, deduplicated, InChIKey conflicts with actives
   removed → 4,524.
3. **Property-match QC** (actives vs decoys, mean ± SD): MW 448.8 ± 66.3 vs
   443.4 ± 65.5; cLogP 3.66 ± 1.22 vs 3.78 ± 1.26; HBD 1.06 vs 1.15; HBA 7.82 vs 7.29;
   rotB 4.26 vs 4.65; formal charge 0 for all.

## Docking protocol (aligned with manuscript stage 1)

- Receptor: PDB 7OTW chain A residues 3701–4123, Gasteiger charges, hydrogens added (Open Babel).
- Box: AZD7648 co-crystal ligand centroid (154.925, 174.279, 209.114) Å, 17×17×17 Å.
- Engine: AutoDock Vina 1.2.5 (CPU baseline; the manuscript used quickvina2_gpu — same
  scoring function), exhaustiveness = 8, num_modes = 9, seed = 42.
- **Positive control**: AZD7648 redocking score −10.06 kcal/mol, best-pose RMSD 0.85 Å
  to the co-crystal pose (< 2.0 Å, passing the vcdd redock-validation criterion).

## Important usage notes

- **Seed leakage**: the manuscript's TransPharmer generation used 25 known DNA-PKcs
  inhibitors as seeds (Figure S1). If the evaluation goal is "can the workflow recover
  known actives", remove these 25 seeds and their near neighbors (ECFP4 Tc ≥ 0.4) from
  the active set first, or report with-seeds and seed-removed settings separately to
  avoid overestimation.
- **Decoy source**: decoys come from ZINC (DUD-E pool), not from the ChEMBL background,
  which is the more conservative direction for the "decoys may contain unlabeled
  actives" bias.
- The 40 experimentally confirmed inactive molecules can be reported separately as a
  hard-decoy subset.
