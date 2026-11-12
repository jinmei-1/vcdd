# Stage contract

Each stage is an independent unit: it reads its inputs from disk, writes its outputs to disk, and
can be run on its own. This file records what each stage reads and writes, so a stage can be
re-implemented or replaced without reading the source.

Every run directory also contains:

- `config.resolved.yaml` — the configuration actually used. A run is only reproducible if this file
  is available, which is why it is written by the pipeline rather than left to the caller.
- `manifest.json` — input hashes, package version, and a `stage1_deterministic: false` flag (see
  below).
- `run.db` — the SQLite ledger. Every unit of work is recorded here, which is what makes `resume`
  possible without recomputing finished work.

## `run.db` and resume

The ledger is keyed by a hash of the inputs *and the parameters* of each unit of work. Changing a
parameter that affects a numerical result therefore invalidates exactly the affected units and no
others. `vcdd resume` replays the ledger, skips everything already complete, and continues.

A consequence worth knowing: **stage-1 scores are not bit-reproducible.** Vina-GPU exposes no
`--seed`, so the manifest records `stage1_deterministic: false` rather than implying otherwise. The
`seed` in the configuration is used for conformer generation only.

## Stage outputs

### `prep-library`
Reads the raw library in any of `.sdf/.sdf.gz/.smi/.ism/.txt/.csv/.tsv/.parquet`, standardizes,
filters, deduplicates and shards it.

Writes `library/` — parquet shards plus an ingestion report. The library is streamed in chunks and
never fully loaded.

### `prep-receptor`
Reads the receptor structure, cleans it, builds the docking box, and runs the native-ligand redock
gate.

Writes `receptor/` — the prepared PDBQT, the box definition, and the redock RMSD. A failing gate is
recorded loudly; whether it aborts is controlled by `receptor.redock_fail_hard` (default: warn).

### `stage1`
Docks the whole library on GPU. Vina-GPU is batch-native: one process docks a whole directory of
ligands, so `gpu_batch_size` matters more than any per-ligand knob.

Writes `stage1/ranking.parquet` with one row per molecule: `mol_id`, `smiles`, `score`, `engine`,
`seconds`, `n_rotatable`, `error`, `shard`, `rank`.

### `stage2`
Re-predicts each shortlisted complex with Boltz from receptor sequence + ligand SMILES. It does
**not** re-dock, and `dock_score` in the output is the stage-1 score carried forward unchanged. That
is the point: a second docking program agreeing with the first mostly measures that they share a
scoring-function lineage, whereas Boltz is orthogonal evidence.

Writes:
- `scores.parquet` — per-molecule confidence, ipTM, pLDDT, pose QC, and (if enabled)
  `affinity_pred_value` / `affinity_probability_binary`
- `pose_qc.parquet` — ligand-in-box fraction, centroid offset, receptor C-alpha RMSD
- `consensus_ranking.parquet` — the ordering consumed by stage 3
- `stage2_summary.json` — headline numbers, including wall-clock

### `stage3`
Scores each shortlisted compound against a virtual-cell model, then aggregates the components into
`S_cell`.

Reads `stage2/consensus_ranking.parquet` and takes the top `cell.top_n`.

Writes:
- `signatures/<backend>.parquet` — the predicted transcriptomic signature per compound
- `activity/<backend>_tf.parquet`, `activity/<backend>_pathway.parquet` — inferred TF and pathway
  activity
- `objective_scores.parquet` — per-compound score and every component
- `stage3_summary.json`

Caching: if the signatures are already on disk with an unchanged input and parameter hash, the
backend is not called again.

### `stage4`
Proposes analogs of the top `stage4.top_parents` stage-3 compounds, applies the chemistry
guardrails, and re-docks and re-scores every survivor.

Writes:
- `analogs_raw.parquet` — every proposal, including the rejected ones and the rule that rejected
  each
- `analogs.parquet` — the guardrail-passing set with `delta_*` columns and `design_score`
- `pareto.parquet` — the Pareto front over the objective terms
- `redock.parquet` — re-docking results for the survivors
- `sar_rules.csv` — structure-activity rules mined from the analog set
- `top_designs.sdf` — the best designs as structures
- `generator_report.json`, `stage4_summary.json`

### `report`
Builds a standalone HTML report from a run directory. It reads the summary JSON files rather than
recomputing anything, so a report cannot disagree with the run it describes.

## Rules for adding or changing a stage

1. Keep the stage runnable on its own.
2. Keep the artifact schema stable, or bump it explicitly and record the change in `CHANGELOG.md`.
3. Record every run in the ledger so results stay traceable.
4. Never degrade silently. If a required engine or model is missing, raise with an actionable
   message. A different method is never substituted without an explicit configuration change.
