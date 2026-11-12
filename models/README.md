# Model weights

vcdd stage 2 and stage 3 require the following third-party / internal model weights.
Weights are **not** redistributed in this repository; obtain them as described below.

| Component | Weight | Source |
|---|---|---|
| Stage 2 co-folding + affinity | Boltz-2 | Public; follow the official repository: https://github.com/jwohlwend/boltz |
| Stage 3 molecular representation | Uni-Mol | Public; follow the official repository: https://github.com/dptech-corp/Uni-Mol |
| Stage 3 virtual-cell scoring | XPert checkpoint `l1000_mdmt_warm_split` | **Not yet publicly released.** Available from the corresponding authors upon request; a public download link will be added here upon release. |

After downloading, place the weights in the paths expected by your run configuration
(`*.yaml`; see `docs/` and `examples/minimal.yaml`).

## Notes for reproducing the DNA-PKcs case study

- Stages 1, 2 and 4 are fully reproducible with the public weights above.
- Stage 3 requires the XPert checkpoint; without it, stage-3 scoring cannot be rerun.
  The stage-3 inputs and outputs used in the manuscript are provided as Supplementary
  Data so that downstream analyses (e.g. the weight-sensitivity analysis) remain
  reproducible.
