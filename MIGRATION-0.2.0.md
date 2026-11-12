# Migrating from vcdd 0.1.0 to 0.2.0

0.2.0 narrows the pipeline to one engine per stage:

| Stage | 0.1.0 | 0.2.0 |
|---|---|---|
| **1** | AutoDock Vina / smina / gnina / QuickVina, CPU, one process per ligand | **Vina-GPU 2.1 only**, batch-native |
| **2** | Re-dock top *N* + consensus rescoring (Vina, Vinardo, RFScore-VS, MM-GBSA…) | **Boltz API only** — no docking, no rescoring |
| **3** | unchanged | unchanged |
| **4** | unchanged | unchanged (re-docks with the stage-1 engine) |

Stages 3 and 4 were not redesigned. Where they appear below it is because a
stage-2 output they consume changed shape.

This is a **breaking** release. A 0.1.0 config will not load. `vcdd` detects the
removed keys and tells you what replaced each one instead of failing with
pydantic's generic "extra inputs are not permitted":

```
$ vcdd stage1 -c old.yaml
ValueError: this config uses keys that were removed in vcdd 0.2.0
(stage 1 is Vina-GPU only; stage 2 is Boltz only):
  stage1.exhaustiveness: Vina-GPU 2.1 has no --exhaustiveness. Search effort is
    set with stage1.thread (parallel Monte-Carlo threads, default 8000) and
    stage1.search_depth (depth per thread; null = adaptive).
  stage2.rescorers: the rescoring plugin subsystem was removed. Stage 2 is Boltz only.
...
See MIGRATION-0.2.0.md, or start from `vcdd config`.
```

Every problem is listed at once, so you fix the file in one pass.

---

## 1. Why stage 1 changed

**The 0.1.0 GPU engines could never have returned a score.** `VinaGPUEngine` and
`QVina2GPUEngine` subclassed `VinaLikeEngine`, so they emitted
`--ligand --out --exhaustiveness --num_modes --energy_range --cpu --seed`.
Vina-GPU 2.1 accepts none of those flags. It takes a *directory* of ligands and
writes a *directory* of poses. Anyone who set `engine: vina_gpu` in 0.1.0 got a
failed run, not a fast one.

Fixing that meant changing the engine contract, not the flag list. Vina-GPU
amortises OpenCL kernel compilation and the receptor grid over a whole batch;
calling it once per ligand is slower than CPU Vina. So `DockingEngine` grew
`dock_batch()`, `EngineCapabilities` grew `batch_native` / `preferred_batch_size`,
and `stage1.execute_jobs` dispatches on the capability — never on the engine name.

Keeping CPU Vina as a fallback was rejected deliberately. A silent fallback
changes the scoring function mid-run: the first half of a ranking would come from
one function and the second half from another, with nothing in the output saying
so. `vcdd doctor` tells you the GPU is missing; it does not paper over it.

### The three variants

All three upstream binaries are supported through one `variant` key, because
they share a CLI and differ only in search strategy and box limit.

| `stage1.variant` | Binary | Build with | Max box edge | Default `thread` |
|---|---|---|---|---|
| `autodock_vina_gpu` (default) | `AutoDock-Vina-GPU-2-1` | `--variant vina` | < 100 Å | 8000 |
| `quickvina2_gpu` | `QuickVina2-GPU-2-1` | `--variant quickvina2` | < 70 Å | 5000 |
| `quickvina_w_gpu` | `QuickVina-W-GPU-2-1` | `--variant quickvinaw` | < 70 Å | 8000 |

The config value and the build-script flag are spelled differently: the config
names the engine variant, the script names the upstream source directory.

The box is validated against the variant's limit **once, before the run starts**,
rather than a million ligands in.

### You must compile it

There is no conda or pip package for Vina-GPU 2.1. `scripts/build_vina_gpu.sh`
drives the upstream Makefile (Boost 1.77 + CUDA Toolkit for OpenCL + gcc), and
`vcdd doctor` reports which variants it found on `PATH`:

```bash
bash scripts/build_vina_gpu.sh --variant vina --prefix ~/.local   # vina | quickvina2 | quickvinaw
vcdd doctor
```

Two build details cost hours if missed, so the script handles both: build with
`OPENCL_VERSION=-DOPENCL_3_0` (not `-DOPENCL_1_2`), and keep `Kernel1_Opt.bin` /
`Kernel2_Opt.bin` next to the binary or point `stage1.opencl_binary_path` at them
— otherwise every invocation recompiles the kernels. The engine also raises the
stack soft limit to 8 MB per call, which upstream requires.

### Stage 1 is no longer bit-reproducible

**Vina-GPU 2.1 exposes no `--seed`.** Re-running the same library against the same
receptor gives slightly different scores and, near ties, a slightly different
rank order. This is a genuine regression against 0.1.0 and it is reported rather
than hidden:

- `EngineCapabilities.deterministic` is `False`
- `manifest.json` records `stage1_deterministic: "false"`
- `stage1/summary.json` carries the same flag

`seed` still exists in the config: it seeds RDKit conformer generation and the
active-learning split, which *are* deterministic. It does not reach the docking
search.

---

## 2. Why stage 2 changed

0.1.0's stage 2 re-docked the top *N* more carefully and combined several
scoring functions. Consensus of correlated empirical scoring functions mostly
buys precision on a quantity that was already biased; it does not add
independent evidence.

0.2.0 replaces it with a single **orthogonal** measurement: Boltz re-predicts the
protein-ligand complex from sequence and SMILES and reports a
`binding_confidence`. Benchmarking has found Boltz-2 discriminates true from
false docking hits better than the scoring functions it replaces, and is
relatively insensitive to the quality of the input pose.

### Read this before interpreting the output

**Boltz does not rescore your docked pose.** It never sees it. It predicts a
complex independently, from sequence and SMILES. Consequences:

- Stage 2 gives you a **second, independent opinion**, not a better pose energy.
- The predicted ligand can land somewhere other than your box. Only the pocket
  residues vcdd passes bias it, and nothing constrains it. This is why pose QC
  exists: `pose_qc.parquet` reports `ligand_in_box` and `centroid_offset_A` per
  molecule, and the run warns when under half the poses land in the box.
- `dock_score` in every downstream artifact is **stage 1's Vina-GPU score,
  carried forward unchanged**. It is not a stage-2 quantity.
- `binding_confidence` is documented as the primary triage metric, with 0.7+ the
  high-confidence range. `optimization_score` is only comparable *within one run*.

### Two endpoints

| `stage2.backend` | Endpoint | Cost | Use it when |
|---|---|---|---|
| `library_screen` (default) | BoltzMol-1 | $0.025/molecule | Triaging hundreds of molecules |
| `structure_and_binding` | boltz-2.1 | $0.025–$0.40/sample by crop size | You want the predicted complex for a handful |

`library_screen` on a top-500 shortlist is about **$12.50**.
`structure_and_binding` at one sample over a ~300-residue target is about **$25**
for the same 500. Note the backends run **different models** — do not pool their
confidences into one ranking.

### Spending is gated

Stage 2 calls the Boltz `estimate-cost` endpoint (free, no GPU, creates nothing)
for every pending chunk before submitting anything, and refuses to spend past
`stage2.max_spend_usd`:

```
SpendGateExceeded: estimated cost $31.25 exceeds stage2.max_spend_usd=$25.00.
Either raise the ceiling, lower stage2.top_n, or re-run with --yes-spend.
```

Pass `--yes-spend` to `vcdd stage2`, `vcdd run` or `vcdd resume` to authorise. An
estimate the API cannot produce blocks the run rather than defaulting to zero.

### API key

Set `BOLTZ_API_KEY` in the environment. Never put it in the config; the key name
is configurable (`stage2.api_key_env`) but the value is read from the
environment, and only the last four characters ever reach a log, the summary or
the doctor output.

**Use a test-mode key first.** A key whose prefix marks it test-mode
(`sk_bc_..._test_...`) runs the entire lifecycle — submit, poll, paginate, fetch,
download — against synthetic results, with no GPU and no billing. Every artifact
you get is real in shape and fake in value, so the run is labelled: `livemode:
false` in `stage2_summary.json`, a warning in the log, and a banner in the HTML
report. The bundled smoke test refuses to run against a live key.

### Fusion policies

`stage2.fusion` decides how stage 1 and Boltz combine into `consensus_score`
(always lower-is-better, always accompanied by a dense 1-based `consensus_rank`):

| Policy | Ranks by | When |
|---|---|---|
| `boltz_only` (default) | `-binding_confidence` | You trust Boltz over the docking score |
| `rank_fusion` | Borda count of both ranks | Neither alone; you want agreement to win |
| `gate` | stage-1 score among molecules passing `binding_confidence >= gate_threshold` | Boltz as a filter, docking as the order |

Molecules with no Boltz result — filtered by the SMARTS catalogue, invalid, or
failed — **keep their row and sort last** with a populated `error`. Dropping them
would silently shrink the shortlist stage 3 receives and hide a systematically
rejected chemotype. `stage2_summary.json` records both `fusion` (requested) and
`policy` (applied).

---

## 3. Config key migration

Delete or replace each of these. `vcdd` names every one of them for you.

### Stage 1

| Removed | Replacement |
|---|---|
| `stage1.exhaustiveness` | `stage1.thread` (parallel MC threads, default 8000) and `stage1.search_depth` (null = adaptive) |
| `stage1.cpu_per_ligand` | none — CPU workers only prepare ligands; size with top-level `n_workers` |
| `stage1.num_modes` | none — Vina-GPU writes every pose it keeps |
| `stage1.energy_range` | none |

New: `variant`, `gpu_batch_size` (default 1000), `rilc_bfgs`, `opencl_binary_path`,
`batch_timeout_s`.

**Search effort does not map one-to-one.** `thread` is the number of parallel
Monte-Carlo threads, not CPU threads, and it is not `exhaustiveness × something`.
Keep it under 10000; the schema warns above that. Start from the variant default
and calibrate on your own redock gate rather than translating an old number.

### Stage 2

| Removed | Replacement |
|---|---|
| `stage2.engine` | `stage2.backend`: `library_screen` \| `structure_and_binding` |
| `stage2.exhaustiveness`, `.num_modes`, `.energy_range` | none — stage 2 does not dock |
| `stage2.cpu_per_ligand` | none — concurrency is bounded by the client |
| `stage2.seed` | none — re-attaching to a run uses the idempotency key vcdd derives from the molecule set |
| `stage2.n_conformers` | none — Boltz predicts from SMILES |
| `stage2.enumerate_tautomers`, `.max_tautomers` | none — one SMILES per molecule |
| `stage2.timeout_s` | `stage2.request_timeout_s` + `stage2.poll_interval_s` |
| `stage2.rescorers` | none — stage 2 is Boltz only |
| `stage2.consensus` | `stage2.fusion` |
| `stage2.min_scorer_agreement` | `stage2.gate_threshold` on `binding_confidence` |
| `stage2.pose_qc` | always on |
| `stage2.use_ensemble`, `.ensemble_reduce` | none |

New: `fusion`, `gate_threshold`, `max_spend_usd`, `api_key_env`, `workspace_id`,
`pocket_from_box`, `pocket_margin_a`, `use_reference_ligand`,
`molecule_filter_level`, `keep_structures_top_k`, `chunk_size`, `num_samples`,
`use_receptor_template`.

### Stage 4

`stage4.redock_mode: stage2` is rejected — stage 2 is a prediction, not a docking
run, so there are no docking parameters to reuse. Use `stage1` or `none`.
Stage 4's re-dock now runs on Vina-GPU with the stage-1 parameters.

---

## 4. Artifacts that changed

| Path | Change |
|---|---|
| `stage2/poses/best_poses.sdf` | **gone** — replaced by `stage2/poses/*.cif`, the Boltz-predicted complexes |
| `stage2/boltz/results.jsonl` | **new** — one raw API result per line, append-safe and de-duplicated |
| `stage2/boltz/runs.json` | **new** — run IDs, idempotency keys, status, cost estimates |
| `stage2/boltz/rejected.parquet` | **new** — molecules the API filtered or rejected, with reasons |
| `stage2/pose_qc.parquet` | now Boltz pose-in-box QC, not docked-pose QC |
| `stage2/scores.parquet` | Boltz metrics + ADME; `dock_score` is stage 1's score carried forward |
| `stage2/consensus_ranking.parquet` | same column contract (`dock_score`, `consensus_score`, `consensus_rank`, `stage1_score`) so stages 3 and 4 are unchanged |
| `stage1/throughput.json` | `seconds_per_ligand_core` and `projected_core_hours` **gone** — with a batch-native GPU engine the CPU pool only prepares ligands, so a per-core figure measured the wrong resource. `batch_size` added, plus a warning when the pilot is smaller than one batch |
| `report/figures/` | `stage1_vs_2` → `stage1_vs_boltz`; `boltz_confidence_distribution` added; rescorer figures gone |

Stage-3 and stage-4 artifacts are unchanged.

---

## 5. Worked example

**0.1.0**

```yaml
stage1:
  engine: vina
  exhaustiveness: 4
  cpu_per_ligand: 1
  num_modes: 9
stage2:
  engine: vina
  top_n: 500
  exhaustiveness: 32
  n_conformers: 5
  max_tautomers: 4
  rescorers: [vina, vinardo]
  consensus: zscore
  min_scorer_agreement: 0.5
stage4:
  redock_mode: stage2
```

**0.2.0**

```yaml
stage1:
  engine: vina_gpu
  variant: autodock_vina_gpu
  thread: 8000
  gpu_batch_size: 1000
stage2:
  backend: library_screen
  top_n: 500
  chunk_size: 500
  fusion: boltz_only
  max_spend_usd: 25.0
  keep_structures_top_k: 50
stage4:
  redock_mode: stage1
```

---

## 6. Checklist

1. `bash scripts/build_vina_gpu.sh --variant vina` and confirm with `vcdd doctor`.
2. `export BOLTZ_API_KEY=<test-mode key>` and confirm `vcdd doctor` reports `mode: test`.
3. `vcdd config -o run.yaml` and port your settings using the tables above.
4. `vcdd boltz estimate -c run.yaml` — see the cost before spending anything.
5. Run the pipeline in test mode end to end. Every artifact will be correctly
   shaped and numerically synthetic.
6. Swap in the live key, re-check the estimate, then run with `--yes-spend`.

Do not skip step 5. Stage 2 is the only stage that costs money per molecule, and
a test-mode run is free.
