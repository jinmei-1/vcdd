# vcdd

**Virtual-cell-guided drug design.** vcdd is a configuration-driven workflow framework that links
GPU-accelerated docking, Boltz-2 complex-confidence and binding-affinity prediction, virtual-cell
transcriptomic scoring, and constrained parent-to-analog generation with joint structural and
cellular rescoring on a single executable, restartable path.

The design loop it implements is *parent -> analog -> rescoring*: a compound that survives
structure-based filtering is scored on a cellular phenotype, its best representatives are used as
seeds for constrained analog generation, and the resulting analogs are returned to both the docking
and the cellular objective. Prioritization can therefore follow a cellular phenotype rather than
docking geometry alone.

## The four stages

| Stage | Method | Input | Output |
|---|---|---|---|
| **1** Docking | Vina-GPU 2.1 (`quickvina2_gpu`) | full compound library | ranked docking scores, sharded |
| **2** Co-folding | Boltz-2 (local), IPTM + affinity head | stage-1 shortlist + controls | per-complex confidence, affinity, pose QC |
| **3** Virtual cell | XPert / ChemCPA / LINCS backends | structure-gated, affinity-ranked hits | per-compound transcriptomic scores and `S_cell` |
| **4** Analog design | constrained GA + R-group enumeration | stage-3 winners | analogs under chemistry guardrails, rescored |

Each stage reads its inputs from disk and writes its outputs to disk. Stages can be run
individually, and an interrupted run resumes from the ledger rather than recomputing finished work.

## Installation

The GPU docking engine and the chemistry stack are best installed with conda; the rest is
pip-installable.

```bash
cd vcdd                      # from a checkout of this repository
conda env create -f environment.yml
conda activate vcdd
pip install -e ".[dev]"
```

A pip-only install works for everything except the docking engine:

```bash
pip install "vcdd[chem,prep,optim,report]"
```

Check what a machine can actually do before starting a campaign:

```bash
vcdd doctor
```

`doctor` reports what is installed and what each missing piece would buy, so a partial environment
is visible before it wastes GPU time.

## Quick start

```bash
# 1. write an annotated configuration template, then edit it
vcdd config > run.yaml

# 2. prepare inputs
vcdd prep-library  -c run.yaml
vcdd prep-receptor -c run.yaml

# 3. run the stages
vcdd stage1 -c run.yaml
vcdd stage2 -c run.yaml
vcdd stage3 -c run.yaml
vcdd stage4 -c run.yaml

# or all of them, with the report
vcdd run -c run.yaml

# 4. render the standalone HTML report
vcdd report -c run.yaml
```

Any configuration key can be overridden without editing the file:

```bash
vcdd stage1 -c run.yaml -s docking.variant=quickvina2_gpu -s docking.thread=5000
```

If a run is interrupted, continue it. Completed work is reused, not repeated:

```bash
vcdd resume -c run.yaml
```

## The single-file campaign driver

The four stages above are the reusable framework. The campaign reported in the accompanying paper
was driven end to end by one self-contained script, `run_all.py`, which generates every stage
configuration, enforces the pre-registered gates, and writes the provenance record. Run it from the
repository root:

```bash
python run_all.py --list          # show the plan, gates and cost; run nothing
python run_all.py --dry-run       # print the exact commands; run nothing
python run_all.py                 # run everything not already done
python run_all.py --only <stage>  # one stage
python run_all.py --from <step>   # resume from a step
python run_all.py --report        # rebuild summary.md / INDEX.md from existing outputs
```

Everything lands under one root (`workflow_v2/`), so the artifacts the manuscript cites are in a
predictable place:

```
workflow_v2/
  configs/        every generated stage config (these are the Methods settings)
  logs/           one log per stage, plus workflow.log
  provenance.json hashes of receptor, MSA, configs and library, plus tool versions
  gates.json      every pre-registered gate and its measured value
  summary.md      the numbers, each with the artifact it came from
  INDEX.md        artifact index: file -> what it is -> which claim it backs
```

`--list` and `--dry-run` are worth running first: they show what would happen and what it would
cost without touching a GPU.

## Configuration

All behaviour is driven by one YAML file. The annotated template documents every key:

```bash
vcdd config
```

The resolved configuration is written to `config.resolved.yaml` inside every run directory. **A run
is only reproducible if that file is available**, which is why it is attached to the run by design
rather than being left to the caller.

Weights that define the composite objectives are configuration, not hard-coded:

```yaml
cell:
  objective:
    target_gene: PRKDC
    genes_up: [CDKN1A, GADD45A]
    weights: {gene: 2.0, pathway: 1.0, target_consistency: 1.0, specificity: 1.0}

score_weights: {docking: 1.0, cell: 2.0, qed: 0.5, sa: 0.5}
```

The stage-3 objective is the weighted geometric mean of its components:

```
S_cell = (gene_score^2 * target_consistency * specificity)^(1/4)
```

and the stage-4 composite design score is

```
Dscore = (S_dock * S_cell^2 * S_QED^(1/2) * S_SA^(1/2))^(1/4)
```

Both exponents follow directly from the weights summing to their root.

## What makes the results traceable

- **A SQLite ledger** (`run.db`) records every unit of work, so a stage can be resumed without
  recomputing what already finished.
- **Per-stage JSON summaries** carry the headline numbers, so reports and figures can read them from
  the run instead of hard-coding them and going stale after a re-run.
- **Explicit failure instead of silent degradation.** If a required engine or model is missing, the
  stage raises with an actionable message. A different method is never substituted silently: the
  docking variant and the virtual-cell backend are configuration-level settings, while the artifact
  schema is fixed by the stage contract.
- **Guardrail verdicts are recorded per proposal**, including which rule rejected it, so a parent
  that already violates a guardrail is visible as such rather than looking like a generation
  failure.

## Repository layout

```
vcdd/                 the package
  cell/               virtual-cell backends, objective scoring, reference resources
  docking/            engine abstraction (Vina-GPU, Vina, smina)
  io/                 library and structure I/O
  optimize/           stage-4 generators, guardrails, Pareto ranking
  receptor/           receptor preparation, box construction, redock gate
  refine/             pose refinement and structural QC
  report/             HTML report generation
  runtime/            ledger, resume, logging
  screening/          shortlist selection
tests/                pytest suite
scripts/              analysis and figure scripts used for the manuscript
docs/                 supplementary documentation
examples/             example configurations
```

## Documentation

- [`docs/STAGE_CONTRACT.md`](docs/STAGE_CONTRACT.md) — what each stage reads and writes, the
  ledger protocol, and the rules for changing a stage.
- [`docs/SELECTION.md`](docs/SELECTION.md) — **read this if you are reproducing a published
  result.** The stage 2 -> stage 3 structural gate and affinity ranking are applied by a post-hoc
  script, not by the pipeline.
- [`docs/GPU_DOCKING.md`](docs/GPU_DOCKING.md) — installing and troubleshooting the docking engine.
- [`docs/REPRODUCING_DNAPKCS.zh.md`](docs/REPRODUCING_DNAPKCS.zh.md) — full command-by-command
  reproduction of the DNA-PKcs campaign, including the environment quirks that matter (Chinese).
- [`DESIGN.md`](DESIGN.md) — why the code looks the way it does, where it deviates from the
  original plan, and what it deliberately refuses to claim.
- [`MIGRATION-0.2.0.md`](MIGRATION-0.2.0.md) — what changed when the pipeline narrowed to one
  engine per stage.
- [`CHANGELOG.md`](CHANGELOG.md) — release history.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development setup and the stage contract.

## Scope and limitations

vcdd is a *triage* framework. It ranks and prioritizes; it does not predict calibrated binding free
energies, and a docking score, an ipTM value, or a transcriptional response in this pipeline is not
potency. The composite scores compress multiobjective conflict into a single number, so the
per-objective components are always reported alongside them. Experimental validation is out of
scope, and no claim about target engagement or cellular efficacy should be read out of a vcdd run
alone.

## Citing

If you use vcdd, please cite the software (see [`CITATION.cff`](CITATION.cff)) and the accompanying
manuscript.

## License

MIT. See [`LICENSE`](LICENSE). Third-party components and bundled reference resources are listed in
[`LICENSE-NOTICES.md`](LICENSE-NOTICES.md).
