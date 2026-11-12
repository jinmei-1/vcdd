# Changelog

All notable changes to vcdd are recorded here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-23

First release prepared for public distribution alongside the accompanying
manuscript. The staged pipeline, configuration contract and artifact schemas are
frozen for the DNA-PKcs case study reported in the paper.

### Added

- Stage 2 Boltz-2 binding-affinity prediction as a first-class artifact: the
  affinity head (`affinity_pred_value`, `affinity_probability_binary`) is
  recorded per complex, so Stage 2 now yields an independent affinity estimate
  rather than complex confidence alone. Configured under
  `stage2.local_boltz.affinity`; off by default because it costs extra diffusion
  sampling.
- `scripts/select_stage2_shortlist.py`, which applies the structure-then-affinity
  selection used for the reported campaign: a hard gate on `ligand_in_box_frac >=
  0.80` **and** `pocket_ca_rmsd_A < 2.0`, then ascending `affinity_pred_value`
  among the survivors. **This is a post-hoc script, not part of the pipeline** —
  `vcdd stage3` consumes `stage2/consensus_ranking.parquet` and applies no such
  gate. See `docs/SELECTION.md`.
- `restore_residue_names` and `check_msa_agreement` in the receptor preparation
  path, guarding against residue-name misperception in receptor PDBQT files.
- Cache reuse on Stage 4 re-entry: a failed unit resumes from cached analog
  proposals instead of recomputing generation.

### Changed

- Composite Stage 4 objective is now reported as `Dscore`, an explicit weighted
  geometric mean with weights `{docking: 1.0, cell: 2.0, qed: 0.5, sa: 0.5}`:
  `Dscore = (S_dock * S_cell^2 * S_QED^(1/2) * S_SA^(1/2))^(1/4)`.
- Docking-engine failure is reported explicitly rather than degrading silently;
  the engine variant is a configuration-level setting while the artifact schema
  is fixed by the stage contract.

### Fixed

- Residue-name misperception that could fold an entire co-folding batch against
  a mislabelled receptor.
- Stage 4 guardrail feedback now records the specific rule that rejected each
  proposal, so a parent that already violates a guardrail is visible as such.

## [0.2.0]

- Four-stage pipeline split into independent, individually runnable stages
  (`stage1` docking, `stage2` co-folding, `stage3` virtual cell, `stage4`
  analog optimization) with a SQLite provenance ledger and per-stage JSON
  summaries.

## [0.1.0]

- Initial internal prototype.
