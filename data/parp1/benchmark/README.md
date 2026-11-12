# PARP1 second-target benchmark (v1)

Benchmark for the vcdd second-target case, demonstrating generality beyond DNA-PKcs.
PARP1 belongs to the same DNA-damage-repair pathway, so the manuscript's 16-gene repair
signature (BRCA1, RAD51C, CHEK1/2, MSH6, LIG1, PCNA, RFC2/5, RPA1/2/3, DDB2, TP53BP1,
TP53, TOP2A) can be reused directly and the S_cell objective needs no redesign.

## Contents (DUD-E benchmark, obtained via HuggingFace mirror)

| File | Content |
|---|---|
| `actives.smi` | 615 unique PARP1 actives (DUD-E, ChEMBL-derived, potency better than 1 µM) |
| `decoys.smi` | 29,065 unique property-matched decoys (DUD-E, ZINC-derived, 50:1 matching) |
| `benchmark_labels.csv` | `set_id, label` (1 = active, 0 = decoy) |

## Receptor

- DUD-E designated structure: **PDB 3L3M** (PARP1 catalytic domain, chain A residues
  2–349; co-crystal ligand residue name A92, chain B).
- The official DUD-E preprocessed receptor was unreachable from our environment; the
  native 3L3M structure was obtained from RCSB instead. Running vcdd `prep-receptor`
  with A92 as `reference_ligand` defines the docking box automatically.
- For the official DUD-E preprocessed receptor and crystal ligand, download
  `parp1.tar.gz` from https://dude.docking.org/targets/parp1.

## Scale and expected cost (extrapolated from the manuscript's DNA-PKcs run)

- Full stage-1 docking of 29,680 molecules: ~9 h with Vina-GPU (scaled from 200k
  molecules / 61 h); a quick variant can dock the 615 actives plus an equal sample of
  decoys first.
- Stage 2 (top 500) Boltz-2: ~9 h; stage 3 (top 200) XPert: ~15 min; stage 4: ~2.3 h.

## Notes

- The active count in this mirror (615) lies between the DUD-E paper's raw (1,031) and
  clustered (508) sets; for strict alignment with the literature, obtain the clustered
  set from the DUD-E website.
- PARP1 inhibitors (e.g. olaparib) are approved drugs; the case should be described in
  the Discussion as a methodological validation only.
