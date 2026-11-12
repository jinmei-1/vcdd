# vcdd design notes

Why the code looks the way it does, where it deviates from the original plan,
and what it deliberately refuses to claim.

---

## 1. Pipeline shape

```
library ─► [1] Vina-GPU dock ─► top 500 ─► [2] Boltz co-fold (independent opinion)
                                                     │
                                            top N (default 200)
                                                     ▼
                                        [3] virtual cell perturbation
                                            multi-objective scoring
                                                     │
                                              top 20 parents
                                                     ▼
                                        [4] analog generation ─► re-dock ─► re-score cell
                                                     ▼
                                   ranked edits + transformation-level SAR rules
```

0.2.0 collapses each of the first two stages to a single engine: **Vina-GPU 2.1**
for stage 1, the **Boltz API** for stage 2. Stages 3 and 4 are unchanged.

Each stage is a pure function of `(config, upstream artifact)` writing to its own
directory, with a SQLite ledger (`run.db`) recording per-unit status. `resume`
re-executes only units that are not `done`, and `RunLedger.is_done` deliberately
returns `False` when the recorded artifact has disappeared from disk — a deleted
file must invalidate the ledger entry, not be trusted because a row exists.

## 2. Stage 1 — throughput first

One conformer, GPU search. The point is *ranking*, not pose fidelity. A timed
pilot (`stage1.pilot_size`, default 50) writes `throughput.json` with measured
seconds/ligand and a projection to the full library, so the cost of a
million-compound run is known before it starts. The note field states the
projection is hardware-specific and must not be reused across machines.

**The engine contract is batch-native, and that is the whole reason 0.2.0
exists.** 0.1.0's `VinaGPUEngine` subclassed `VinaLikeEngine` and emitted
`--ligand --out --exhaustiveness --num_modes --energy_range --cpu --seed`.
Vina-GPU 2.1 accepts none of them: it takes a *directory* of ligands and writes a
*directory* of poses. Those engines could never have returned a score. The fix
was not a flag list but a contract — `DockingEngine.dock_batch()` plus
`EngineCapabilities.batch_native` / `preferred_batch_size`. `execute_jobs`
branches on the capability and never on the engine name, so the CPU pool
prepares ligands while one long-lived GPU process handles each batch of
`stage1.gpu_batch_size` (default 1000). Calling a GPU docking binary once per
ligand pays OpenCL kernel compilation and receptor-grid setup per molecule and is
*slower than CPU Vina*; that is the trap the batch contract exists to avoid.

The pilot warns when it is smaller than one batch, because in that regime the
setup cost is amortised over too few molecules and `projected_hours` is an upper
bound rather than an estimate. `throughput.json` no longer reports
`seconds_per_ligand_core`: with a batch-native engine the CPU pool only prepares
ligands, so a per-core figure measured the wrong resource entirely.

**No CPU fallback, by choice.** A silent fallback to CPU Vina when no GPU is
found would change the scoring function midway through a ranking, with nothing
in the output saying so. `vcdd doctor` reports the missing GPU instead. The
registry holds exactly one production engine and a test mock.

**Stage 1 is not bit-reproducible in 0.2.0.** Vina-GPU 2.1 exposes no `--seed`.
`EngineCapabilities.deterministic` is `False`, `manifest.json` records
`stage1_deterministic: "false"`, and the stage summary repeats it. `config.seed`
still governs RDKit conformer generation and the active-learning split, which
remain deterministic; it does not reach the docking search. The box is validated
against the variant's compiled limit (100 Å for AutoDock-Vina-GPU, 70 Å for the
QuickVina variants) once at startup rather than a million ligands in.

Sharding (`--shard i --nshards k`) partitions by molecule index, so N nodes can
write disjoint `stage1/scores/shard_*.parquet` files that `collect_ranking`
merges. The deterministic sort is `["score","mol_id"]` with `kind="mergesort"`,
which makes the ranking bit-for-bit reproducible across shard orders.

**Active learning is opt-in, not the default.** `stage1.surrogate_al: true`
docks a seed fraction, fits a random forest on Morgan fingerprints, and iterates
select→dock→refit. It measures its own quality on a held-out docked subset and
reports the gap. It is off by default because a surrogate trades recall for
speed, and a screen that silently drops actives is worse than a slow one.

## 3. Stage 2 — an independent opinion, not a better score

0.1.0 re-docked the shortlist at exhaustiveness 32 and combined Vina with
Vinardo. The problem with that design is not that it was slow: consensus over
*correlated* empirical scoring functions buys precision on a quantity that is
already biased. Vina and Vinardo share a functional form and a training regime,
so their agreement is largely self-agreement, and the `rank_spread` it reported
measured something narrower than it appeared to.

0.2.0 replaces the whole subsystem with one **orthogonal** measurement: Boltz
re-predicts the protein-ligand complex from sequence and SMILES and reports a
`binding_confidence`. Published benchmarking finds Boltz-2 separates true from
false docking hits better than the scoring functions it replaces, and is
relatively insensitive to input pose quality — which matters here precisely
because it never sees our pose.

**Boltz does not rescore the docked pose. It never receives it.** That single
fact drives most of the design:

- `dock_score` in every downstream artifact is **stage 1's Vina-GPU score,
  carried forward unchanged**. Stage 2 does not produce a docking score, and
  pretending otherwise would let a Boltz confidence masquerade as an energy in
  stage 3 and stage 4, which both read `dock_score`.
- Nothing constrains the predicted ligand to our box except the pocket residues
  we pass (0-indexed, keyed by chain, derived from the docking box). So pose QC
  is mandatory and re-scoped: `pose_qc.parquet` now records `ligand_in_box`,
  `centroid_offset_A`, `frac_atoms_in_box` and `max_axis_excess_A` per molecule,
  and the run warns when under half the predicted poses land in the box. That
  warning is the signal that the two stages are not talking about the same site.
- The three backends are **different models**; their confidences are not pooled.
  `library_screen` → BoltzMol-1 (hosted-only screening pipeline),
  `structure_and_binding` → boltz-2.1 (hosted API), `local` → the same boltz-2.1
  model run on this machine through a `boltz predict` subprocess. The local
  backend skips the spend gate entirely (no API, no billing) and cannot pass
  pocket residues or reference ligands - the input YAML has no such fields - so
  the ligand-in-box pose QC above is its *only* site check, and it reports
  `engine: boltz-2 (local)` so a local run can never be mistaken for
  BoltzMol-1. Its confidences are comparable to `structure_and_binding`'s, not
  `library_screen`'s, and `optimization_score` (a BoltzMol-1-only ADME-style
  metric) is always `None` locally.

**Molecules with no result keep their row.** Filtered, invalid and failed
molecules sort last with a populated `error` rather than being dropped. Dropping
them would silently shrink the shortlist stage 3 receives and hide a
systematically rejected chemotype — exactly the failure a SMARTS catalogue
produces when it dislikes one scaffold.

**Fusion is explicit and audited.** `boltz_only` ranks on `-binding_confidence`;
`rank_fusion` Borda-combines the two rankings; `gate` uses Boltz as a filter and
stage 1 as the order. All three emit lower-is-better `consensus_score` plus a
dense `consensus_rank`, so stages 3 and 4 need no changes. The summary records
both `fusion` (requested) and `policy` (applied) so an unrecognised name that
fell back to the default is visible rather than silent.

**Spending is gated before submission, not after** (API backends only; `local`
has nothing to gate). Every pending chunk is priced
through the free `estimate-cost` endpoint and checked against
`stage2.max_spend_usd`; `--yes-spend` authorises. An estimate the API declines to
give **blocks** the run rather than defaulting to zero, because the failure mode
of a cost-estimate bug is an unbounded bill. Chunks are idempotent: the key is
derived from the target and the molecule set, so a resumed run re-attaches to the
existing run instead of paying twice.

**Test mode is a first-class path.** A test-mode API key exercises the entire
lifecycle — submit, poll, paginate, fetch, download, QC — against synthetic
results with no GPU and no billing. Runs made with one are labelled
`livemode: false` in the summary, the log and the HTML report, because an
artifact that is correctly shaped and numerically fake is more dangerous than one
that is obviously broken. The bundled smoke test *refuses to run against a live
key*, which is why `requires_boltz_test_key` checks `livemode is False` rather
than merely checking that a key exists.

## 4. Stage 3 — the virtual cell

### 4.1 Backends are adapters, never reimplementations

The user asked to *integrate* existing compound-perturbation models, so every
backend is a thin adapter behind one interface
(`PerturbationRequest → Signature`), declaring capabilities honestly:
tier, gene space, value scale, whether it accepts novel SMILES, whether it is
cell-line/dose aware, and what it needs to become available.

| Tier | Backend | Status here |
|---|---|---|
| 1 | `lincs_readacross` | works offline from the bundled LINCS consensus signatures |
| 1 | `target_knockdown` | CRISPR-KO signature of the target gene — **reference only** |
| 1 | `mock` | deterministic, for tests and dry runs |
| 2 | `chemcpa`, `transigen` | adapter + contract tests only; needs `VCDD_*_DIR` weights |
| 3 | `prnet`, `perturbnet`, `state_tahoe` | adapter + contract tests only |

**Tier-2/3 adapters have never been executed against real checkpoints.** They
are validated by contract tests (capabilities are honest, unavailability is
explained, the install hint is actionable) and nothing more. Saying so is part
of the deliverable.

`reference_only` backends are a deliberate concept: a CRISPR knockout of the
target is a useful provenance anchor ("what *would* complete inhibition look
like?") but it is not a prediction about the compound. Stage 3 materialises
their signatures for inspection and **excludes them from the objective**,
raising if every usable backend is reference-only.

### 4.2 `protein_activity` is a transform, not a backend (deviation from plan)

The plan called protein-activity inference a "Tier-1 backend". It is implemented
instead as a *transform module* applied to any backend's signature. A backend
answers "what does the transcriptome look like?"; footprint inference answers
"what does that imply about TF and pathway activity?". Making it a backend would
have forced every future model to reimplement it, and would have made the
combination "chemCPA + TF activity" impossible to express.

Implementation: ULM / wmean / wsum over CollecTRI (TFs) and PROGENy (pathways),
matrices shaped `(n_genes, n_regulators)`, signatures `(n_genes, n_samples)`,
scores returned as `(n_regulators, n_samples)`.

**Coverage limit, stated up front:** only **361 of 1,201** CollecTRI TFs have
≥5 measured targets inside the 978-gene L1000 landmark space (median regulon
size 3); in BING space it is 693. Regulons below `MIN_TARGETS = 5` are dropped
and counted in `dropped_small`. A TF requested in `genes_up`/`genes_down` that is
not itself measured is routed to inferred activity and labelled
`route="inferred_tf_activity"` in `gene_detail` — the user can always see whether
a number came from a measurement or a model.

### 4.3 The objective is absolute, never batch-relative

Every component maps to `[0,1]` through `squash(x, scale) = 0.5(1+tanh(x/scale))`
with **fixed** scales per value type (`zscore` 2.0, `logfc` 1.0, `membership`
0.5). No z-scoring against the current batch. Two runs on different libraries
therefore produce comparable objective values, and adding one compound cannot
change another compound's score.

Four components, weighted geometric mean (default weights 2/1/1/1):

- `gene_score` — requested genes moved in the requested direction (`gene_mode:
  all` = min, `any` = weighted mean)
- `pathway_score` — Hallmark / PROGENy directional agreement
- `target_consistency` — agreement with the target-knockdown reference
- `specificity` — penalises "moves everything" and generic stress
  (HSPA5, DDIT3, ATF3, HMOX1, GDF15, CDKN1A, …)

Specificity was uninformative at `MOVED_THRESHOLD["membership"] = 0.5`; the
default is now 0.1 and the component spans ≈0.88–0.945 across a shortlist.
Components with no evidence are dropped and the geometric mean is renormalised
over the terms actually used (`n_terms_used`), so a missing pathway annotation
degrades to "no opinion" rather than to zero.

### 4.4 Applicability domain

Every signature carries `max_tanimoto` to its nearest LINCS reference and an
`out_of_domain` flag (`cell.objective.ad_tanimoto_warn`, default 0.4). This
metadata is written to a **separate** `signatures/<backend>_meta.parquet`,
because a silent resume bug once dropped it: on the second run `max_tanimoto`
came back NaN and `out_of_domain` flipped `True → False`. Cached signatures now
round-trip their metadata, and a regression test asserts run-1 and run-2 produce
identical applicability-domain columns.

## 5. Stage 4 — how to modify the top compounds

Four generators behind one interface: CReM (fragment replacement, needs the
public DB), R-group enumeration, MMP transforms (`mmpdb`), and a **constrained
genetic algorithm**.

**27 mapped reaction SMARTS, not `ReplaceSubstructs`.** The first implementation
used `ReplaceSubstructs`, which has substitution semantics that silently produce
wrong molecules for anything but the simplest pattern. Every transform is now a
named, atom-mapped reaction (`[cH1:1]>>[c:1]F`, …) compiled once and cached; the
primary-amine rule excludes amide nitrogens explicitly
(`[NX3H2;$(N[#6;!$([C]=[O,N,S])]):1]>>[N:1]C`). Named transformations are what
makes Stage 4 output *actionable* — and what SAR mining aggregates over.

The GA keeps chemistry sane by construction: mutation operators weighted
substituent 0.40 / atom-type 0.25 / grow 0.25 / trim 0.10, crossover only across
acyclic single bonds with both ends of degree > 1, tournament size 3, elitist
truncation, lineage truncated to the last 3 operations. Guardrails run on every
individual (similarity floor 0.5 GA / 0.6 enumerative, heavy-atom delta ≤ 8, ring
size ≤ 7, ≤2 bridgeheads, ≤1 spiro, SA ≤ 4.5, MW ≤ 650, cLogP ≤ 6, PAINS,
parent-identity rejection).

**Docking is deliberately excluded from the GA inner loop.** A docking call is
~10 s; a read-across cell evaluation is ~1 ms. Docking inside the loop would cut
the number of generations by three orders of magnitude for a signal the GA
cannot exploit anyway. Analogs are docked *after* generation, in bulk.

**Parents are re-docked under the identical protocol.** `delta_dock_score` is
otherwise confounded: comparing an analog docked under one protocol against a
parent scored under another measures the protocol, not the edit. In 0.2.0 there
is only one docking protocol left, so `stage4.redock_mode` is `stage1` (re-dock
both with the stage-1 Vina-GPU parameters) or `none` (skip). `stage2` is rejected
by the config with a message saying why: stage 2 is a prediction, not a docking
run, so it has no docking parameters to reuse.

Because Vina-GPU has no seed, `delta_dock_score` now carries run-to-run search
noise on both terms. Re-docking the parent in the same run is what keeps the
comparison honest — the difference is between two draws from the same protocol
rather than between a fresh draw and a stored number.

Ranking is a weighted geometric mean over normalised docking (anchors −4 → −12
kcal/mol), cell objective, QED and SA, reported alongside a true NSGA-II
`pareto_rank` so a user who disagrees with the weights can read the front.

SAR mining collapses **per parent first**, then votes across parents: 20 analogs
of one parent carrying the same edit count as one vote, not twenty. The verdict
is `consistent` (≥0.8 of parents improved and ≥3 parents), `suggestive` (≥0.6),
else `inconsistent`, with an exact two-sided sign test (ties count as failures).

## 6. Two bugs the tests found, and what they changed

**Active learning produced an empty ranking.** `run_active_learning` checkpoints
slices as `al_*.parquet`, but `collect_ranking` globbed only `shard_*.parquet`.
The collector now takes an explicit `pattern` argument. Design lesson: a shared
collector must not hard-code a producer's naming convention. The same fix
populated `surrogate_predicted_score` for never-docked molecules, which was
previously all-NaN.

**`score_only` was silently broken on AutoDock Vina 1.2.x.** With `--local_only`,
Vina 1.2.5 writes the minimised pose *without* a `REMARK VINA RESULT` line and
prints the affinity to stdout only — so every rescoring call returned
`no_score_in_output`. The engine now parses the pose file first (smina/gnina do
write the REMARK) and falls back to a stdout regex. Verified: co-crystal biotin
in streptavidin scores −6.64 kcal/mol. Design lesson: an engine adapter must
treat "no score parsed" as a hard error, never as a missing value — this failure
was invisible because the column merely went empty. *(0.2.0 has no `score_only`
path; Vina-GPU 2.1 has no such mode and raises `NotImplementedError` rather than
re-docking a pose and calling the result a score-only energy.)*

### 6.1 What the 0.2.0 tests found

**The 0.1.0 GPU engines could never have scored anything.** Covered in §2. It is
the reason this release exists, and the reason `tests/test_boltz.py` drives the
real Boltz SDK over a mock HTTP transport instead of patching the client with a
`Mock()`: a mock that accepts any call signature would have accepted the
impossible Vina-GPU flags too. Testing through the real client is what makes a
wrong CLI or a wrong payload shape fail.

**`scores.parquet` lost `dock_score`, breaking a stage-3 path that was supposed
to be untouched.** `stage3.load_shortlist` falls back to sorting
`stage2/scores.parquet` by `dock_score` when the ranking is missing. That column
existed in 0.1.0 only as a by-product of stage 2 re-docking; the Boltz rewrite
dropped it, so the fallback raised a bare `KeyError` on an otherwise successful
run. Fixed in stage 2 by carrying the column forward, not in stage 3. Design
lesson: "this stage is unchanged" is a claim about *its inputs*, and only a
cross-stage contract test can check it — every stage-2 unit test passed.

**`build_target` and `box_check` each assumed a shape their callers did not
provide.** `build_target` emitted `chains` as a dict keyed by chain ID while
`stage2.py` and `html.py` both iterated it expecting a list of records;
`box_check` annotated `coords: np.ndarray` and called `.shape` on it, so any
caller passing the list of tuples that the CIF parser actually returns got an
`AttributeError`. Both are the same failure: an internal boundary documented by
a type annotation nobody executed. They now normalise their inputs.

One more surfaced while writing the Stage-4 tests: `_write_sdf` treated
`embed_conformers`' return value as a list of Mols instead of one Mol carrying N
conformers, which crashed `top_designs.sdf` whenever `redock_mode: none` left no
docked pose to reuse.

**The console silently deleted chemistry.** `rich` interprets `[tag]` in any
printed string as markup, and its tag pattern matches anything starting with a
lowercase letter. `[nH]` qualifies. Every unescaped console path therefore
rewrote pyrrole `c1cc[nH]c1` as `c1ccc1` — not a garbled string a reader would
question, but a different, still copy-pasteable molecule. The same defect turned
the doctor's `pip install 'vcdd[boltz]'` into `pip install 'vcdd'`, an
instruction that installs the package *without* the Boltz SDK and leaves stage 2
broken in exactly the way the reader was trying to fix. Data-bearing cells now
render as `rich.text.Text` and interpolated strings go through
`rich.markup.escape`; the doctor also wraps long install hints instead of
truncating them at 96 characters, which had been cutting the kernel-cache
instruction mid-sentence. The lesson is narrow but general: a presentation layer
that accepts markup is an injection surface for your own data, and chemistry
notation collides with it.

**The doctor looked for the kernel cache in the wrong place.** `build_vina_gpu.sh`
installs `Kernel*_Opt.bin` to `$PREFIX/share/vcdd/vina-gpu-kernels`, a sibling of
`$PREFIX/bin`, but `_gpu_report` searched only the binary's own directory and the
CWD. A correct build therefore still reported `cached kernels: no`, whose remedy
is a slow `make source` the user had already done. The search now follows the
layout the script creates and the detail line prints the exact
`stage1.opencl_binary_path` value to copy. The README's `--prefix ~/.local/bin`
was wrong for the same reason — the script appends `bin` itself, so that spelling
installs to `~/.local/bin/bin`.

## 7. What this pipeline does not tell you

1. **Docking ranks, it does not measure.** Typical error is 2–3 kcal/mol and
   rank correlation with measured affinity is modest. Stage 1 produces a *triage
   order*, not predicted potencies. In 0.2.0 it is also not reproducible:
   Vina-GPU 2.1 has no seed, so scores and near-tie orderings move between
   identical runs.
2. **Stage 2 is a second opinion, not a verdict.** Boltz predicts the complex
   independently from sequence and SMILES; it does not evaluate the docked pose,
   and its ligand is not constrained to the docking box. A high
   `binding_confidence` on a molecule whose predicted pose sits outside the box
   (`ligand_in_box: false`) is evidence about a *different* binding event than
   the one stage 1 ranked. `optimization_score` is comparable only within one
   run, and the two backends are different models whose confidences do not pool.
   Published evaluations also show Boltz-2 is challenged by target mutations and
   by receptor shuffling, and that physics-based free-energy methods still beat
   it on some series — so treat it as a strong orthogonal filter, not ground truth.
3. **One rigid receptor.** No induced fit, no explicit water, no covalent
   chemistry, no metalloprotein parameterisation, no PROTAC ternary complexes.
   `receptor.ensemble_paths` mitigates but does not solve this.
4. **The virtual cell is a prediction.** `lincs_readacross` is a k-NN
   interpolation over LINCS consensus signatures (k=10, Tanimoto floor 0.2,
   power 3): it can only "discover" what LINCS already contains, it is
   cell-line agnostic (`cell_line` is provenance only), and small analog edits
   barely move the neighbour set — so `delta_cell_objective` is compressed and
   is sometimes exactly zero. That compression is a property of the backend,
   not evidence that the edit does nothing.
5. **Protein readouts are inferred activities**, computed from mRNA through a
   regulon footprint. They are not measured protein abundance, phosphorylation
   or localisation, and they inherit the 361/1,201 TF coverage limit above.
6. **Knockout is a proxy for complete inhibition**, not for a partial,
   reversible, dose-dependent small molecule.
7. **A test-mode stage-2 run produces fully-shaped, entirely synthetic numbers.**
   Check `livemode` in `stage2_summary.json` before reading any confidence.
8. **Nothing here is validated prospectively.** No compound in any output has
   been tested. The correct next step is a small, cheap, physical assay of the
   top edits — not a larger virtual screen.

## 8. Standing assumptions

1. No real screen was run: the deliverable is the codebase plus a passing
   end-to-end smoke test on a 200-molecule fixture (streptavidin/biotin). In
   0.2.0 that smoke test additionally requires a compiled Vina-GPU, a GPU, and a
   test-mode Boltz key, so the cheap `test_pipeline_contract.py` covers the same
   stage-1 → stage-4 wiring on the mock engine and a scripted Boltz transport.
2. Tier-2/3 virtual-cell backends are validated by contract tests only.
3. Protein-level readouts are inferred, never measured.
4. Stage 1 defaults to brute force; active learning is opt-in.
5. Docking ranks rather than measures; single rigid receptor; and in 0.2.0 the
   docking search is not seeded.
6. Vina-GPU and a GPU are **required**, not optional — stage 1 has no CPU
   fallback by design. A Boltz API key is required for stage 2, which is the only
   stage that costs money per molecule.
7. The CReM DB, mmpdb, AiZynthFinder, Meeko/ADFR and all Tier-2/3 weights remain
   optional — their absence degrades the run and is reported by `vcdd doctor`,
   but never breaks it.
