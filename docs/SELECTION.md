# Stage 2 -> stage 3 selection

**The structural gate and affinity ranking are not part of the `vcdd` pipeline.** They are applied
afterwards, by `scripts/select_stage2_shortlist.py`, which reads a finished stage-2 run and writes
the stage-3 shortlist. This separation is deliberate and is documented here because it is the one
place where reproducing the published numbers requires a step outside the package.

## Why the pipeline itself does not gate

`vcdd/refine/stage2.py` builds `consensus_ranking.parquet` from `build_ranking()`, which orders
complexes by the configured fusion rule (default `boltz_only`) and optionally filters on
`gate_threshold` over `binding_confidence`. Over the DNA-PKcs set that confidence score spans
0.820-0.937, and the default 0.70 threshold passes 100% of molecules, so ranking on it orders
noise. It cannot express "the pose is trustworthy *and* the affinity is good", which is what the
selection actually needs.

## What the post-hoc script does

In the order it applies:

1. **Structural validity gate (hard filter).** A complex whose predicted ligand sits outside the ATP
   box, or whose pocket is not reproduced, cannot carry a meaningful affinity value: the affinity
   head would be scoring a pose the model never found. Two thresholds, both fixed *before* the
   affinity numbers are looked at:

   | criterion | field | threshold |
   |---|---|---|
   | ligand placement | `ligand_in_box_frac` | `>= 0.80` |
   | pocket reproduction | `pocket_ca_rmsd_A` vs 7OTW | `< 2.0` |

   The yield is reported either way.

2. **Affinity ranking (soft order) among the survivors.** Ascending `affinity_pred_value`,
   i.e. `log10(IC50 / uM)`, so more negative is stronger.

## Why `affinity_pred_value` and not `binding_confidence`

Boltz's own guidance is that `affinity_probability_binary` separates binders from decoys while
`affinity_pred_value` compares binders with each other. Ranking analogs of a known binder is the
second question, so the value head is the right signal. The confidence head answers whether the
complex is plausible at all, which on this set it answers identically for everything.

## The controls the script also computes

Two controls decide whether an affinity ranking may be claimed at all, and both are reported by the
script rather than buried:

- **the positive control's percentile** under the same rule (a known inhibitor is co-folded inside
  the set), and
- **the molecular-weight confound** plus the **overlap** between the affinity top-N and the
  confidence top-N, which together say how much the choice of criterion actually matters.

## Enabling the affinity head

The affinity head must be switched on in stage 2, or `affinity_pred_value` is null for every
complex and the ranking has nothing to order by:

```yaml
stage2:
  backend: local
  local_boltz:
    affinity: true
    affinity_samples: 5
    affinity_sampling_steps: 200
    affinity_mw_correction: false
```

This costs extra diffusion sampling and is off by default.

## Reproducing the selection

```bash
# 1. run stage 2 with the affinity head enabled
vcdd stage2 -c run.yaml

# 2. apply the gate and the affinity ranking
python scripts/select_stage2_shortlist.py --stage2-dir runs/<name>/stage2 --top-n 200

# 3. score the selected compounds with the virtual cell
vcdd stage3 -c run.yaml
```
