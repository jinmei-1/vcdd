<!--
Historical: this is the vcdd README as written for the 0.2.0 release. The name is
misleading - "upstream" here refers to the upstream Vina-GPU project and to the
previous stage's output in the data flow, NOT to a parent project. vcdd is not a
fork. The current README is at the repository root and supersedes this file.
-->

# vcdd — virtual-cell-guided drug design

A four-stage, resumable pipeline that goes from a large compound library to
concrete, ranked **medicinal-chemistry edits**:

| Stage | What it does | Engine |
|---|---|---|
| **1** | Fast docking of the whole library, one ranking | **Vina-GPU 2.1** (batch-native, one GPU process per batch) |
| **2** | Independent co-folding opinion on the top *N* | **Boltz API** — `binding_confidence` per molecule |
| **3** | Predict the cellular perturbation of each shortlisted compound with a **virtual cell model** and score a multi-objective cell readout | LINCS L1000 read-across (+ optional chemCPA / TranSiGen / PRnet / PerturbNet / STATE) |
| **4** | Propose and rank **how to modify the top 20 compounds** | CReM / R-group / constrained genetic algorithm, re-docked on Vina-GPU |

Every stage is resumable at unit granularity and writes a machine-readable
artifact plus a provenance manifest.

> **Upgrading from 0.1.0?** Read `MIGRATION-0.2.0.md`. This is a breaking
> release: stage 1 is Vina-GPU only, stage 2 no longer docks or rescores, and a
> 0.1.0 config will not load. `vcdd` names every removed key and its replacement
> rather than failing generically.

## Install

```bash
uv pip install -e .          # core: rdkit, numpy, pandas, pyarrow, scipy, sklearn, typer, pydantic
uv pip install -e '.[boltz]' # + the Boltz SDK for stage 2
vcdd doctor                  # prints exactly which engines/backends/generators are usable here
```

`vcdd doctor` never fails because something optional is missing — it tells you
what is available, what is not, and the one command that would fix it.

### Stage 1 needs a compiled Vina-GPU

There is no package for Vina-GPU 2.1; it has to be built against Boost 1.77 and
the CUDA Toolkit (which supplies OpenCL). The bundled script drives the upstream
Makefile and handles the two settings that otherwise cost an afternoon —
`-DOPENCL_3_0` and keeping the pre-built kernel binaries next to the executable:

```bash
bash scripts/build_vina_gpu.sh --variant vina --prefix ~/.local   # binaries land in $PREFIX/bin
```

`--variant` is `vina` (AutoDock-Vina-GPU-2-1, box < 100 Å), `quickvina2`
(faster, box < 70 Å) or `quickvinaw` (wide-box search, < 70 Å). Select at run
time with `stage1.variant`.

### Stage 2: hosted Boltz API, or the same model on your own GPU

The default route is the hosted API:

```bash
export BOLTZ_API_KEY=sk_bc_..._test_...    # start with a test-mode key
```

A test-mode key runs the complete lifecycle — submit, poll, paginate, fetch,
download — against synthetic results with no GPU and no billing, so you can
validate the whole pipeline for free. Runs made with one are labelled
`livemode: false` everywhere they surface. The key is read from the environment,
never from the config, and only its last four characters reach a log or report.

If you have a CUDA GPU (≈14-20 GB VRAM for a typical target), you can instead
run the same boltz-2 model **locally** — no key, no billing, no spend gate:

```bash
pip install boltz                    # the model + CLI; weights download on first run
vcdd doctor                          # should now report a usable "local boltz"
vcdd run -c run.yaml -s stage2.backend=local -s stage2.local_boltz.devices='[0,1]'
```

Two honest caveats for the local route: `boltz predict`'s input YAML has no
pocket-residue or reference-ligand fields, so the stage-1 docking box is not
sent to the model — the `ligand_in_box` pose QC becomes the only site check —
and local numbers are comparable to the `structure_and_binding` backend, not
to `library_screen` (BoltzMol-1 is hosted-only).

## Quick start

```bash
vcdd config -o run.yaml                       # commented template
vcdd boltz estimate -c run.yaml               # what stage 2 will cost, before spending
vcdd run -c run.yaml -o runs/my_screen        # stages 1-4 + HTML report
vcdd resume -c run.yaml -o runs/my_screen     # picks up exactly where it stopped
```

Run a single stage, or override anything from the shell:

```bash
vcdd stage1 -c run.yaml -s stage1.variant=quickvina2_gpu -s stage1.thread=5000
vcdd stage2 -c run.yaml --yes-spend
vcdd stage3 -c run.yaml -s cell.cell_line=A375 -s cell.objective.genes_up='["CDKN1A","GDF15"]'
```

## Minimal config

```yaml
name: my_screen
n_workers: 16
library:
  path: library.sdf            # .sdf/.sdf.gz, .smi, .csv, .parquet
receptor:
  path: target.pdb
  reference_ligand: LIG        # box is derived from the co-crystal ligand
stage1:
  variant: autodock_vina_gpu
  thread: 8000                 # parallel Monte-Carlo threads, not CPU threads
  gpu_batch_size: 1000
stage2:
  backend: library_screen      # or structure_and_binding
  top_n: 500                   # ~$12.50 at $0.025/molecule
  fusion: boltz_only           # boltz_only | rank_fusion | gate
  max_spend_usd: 25.0
cell:
  backends: [lincs_readacross]
  cell_line: MCF7
  objective:
    genes_up:   [CDKN1A]       # compounds that drive these genes UP score higher
    genes_down: [MYC]
stage4: {top_parents: 20, analogs_per_parent: 50}
```

## What you get

```
runs/my_screen/
├── config.resolved.yaml  manifest.json  run.db  logs/
├── library/   shards/*.parquet  index.parquet  rejected.parquet
├── receptor/  receptor.pdbqt  box.json  redock_validation.json
├── stage1/    ranking.parquet  throughput.json
├── stage2/    consensus_ranking.parquet  pose_qc.parquet  scores.parquet
│              boltz/{results.jsonl,runs.json,rejected.parquet}  poses/*.cif
├── stage3/    objective_scores.parquet  objective_by_backend.parquet  backend_report.json
├── stage4/    analog_recommendations.csv  sar_rules.csv  top_designs.sdf  pareto.parquet
└── report/    report.html + figures/*.svg,*.png
```

`analog_recommendations.csv` is the deliverable: one row per proposed analog with
its parent, the named transformation applied, Δdocking, Δcell-objective, QED, SA,
Pareto rank and every guardrail it passed. `sar_rules.csv` aggregates those edits
into transformation-level rules with an exact sign test across parents.

## Scale and cost

Stage 1 shards the library and hands batches of prepared ligands to one
long-lived GPU process per batch; `throughput.json` projects total wall hours
from a timed pilot before you commit. For million-compound libraries use
`--shard i --nshards k` to fan out across nodes, or enable the opt-in surrogate
active-learning loop (`stage1.surrogate_al: true`) — off by default, because it
trades recall for speed.

Stage 2 is the only stage that costs money per molecule. It prices every pending
chunk through the free `estimate-cost` endpoint before submitting anything and
refuses to spend past `stage2.max_spend_usd`; `--yes-spend` authorises. An
estimate the API cannot produce blocks the run rather than defaulting to zero.

## Validation

The receptor gate re-docks the co-crystal ligand before any screening compound
is touched; if the RMSD exceeds `receptor.redock_rmsd_threshold` the run warns
(or aborts with `redock_fail_hard: true`).

```bash
pytest -q -m "not slow"     # unit + contract tests, no GPU and no API key needed
pytest -q -m slow           # full end-to-end run on the bundled 200-molecule fixture
```

The slow suite needs a compiled Vina-GPU, a GPU, and a **test-mode** Boltz key;
it refuses to run against a live key rather than quietly billing you.

## Honest limitations

Read `DESIGN.md` § *What this pipeline does not tell you* before acting on any
output. In short:

- **Docking ranks, it does not measure** (~2–3 kcal/mol error), on one rigid receptor.
- **Stage 1 is no longer bit-reproducible.** Vina-GPU 2.1 exposes no `--seed`, so
  scores and near-tie orderings shift between identical runs. `manifest.json`
  records `stage1_deterministic: false`.
- **Boltz does not rescore your docked pose.** It re-predicts the complex from
  sequence and SMILES, so it is *orthogonal evidence*, not a better pose energy —
  and nothing constrains its ligand to your box except the pocket residues vcdd
  passes. `pose_qc.parquet` reports where each predicted ligand actually landed.
- **The virtual cell is a prediction.** The default read-across backend only
  interpolates LINCS, is cell-line agnostic, and compresses
  `delta_cell_objective` for small analog edits.
- **Protein readouts are inferred regulon activities**, not measured abundance;
  only 361/1,201 CollecTRI TFs have ≥5 measured targets in the 978-gene landmark space.
- **Nothing here has been tested prospectively.** No Tier-2/3 virtual-cell
  checkpoint has been validated against real weights.

Licences of the bundled resources (LINCS, CollecTRI, PROGENy, MSigDB Hallmark)
are listed in `LICENSE-NOTICES.md`.
