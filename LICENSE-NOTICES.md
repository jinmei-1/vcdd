# Third-party notices

`vcdd` itself is released under the MIT licence (see `LICENSE`). It bundles or
optionally invokes the third-party resources below. **You are responsible for
complying with each licence for your own use, especially commercial use.**
`vcdd doctor` prints what is actually present in your environment, and every run
writes the same list into `manifest.json` under `licenses`.

## Bundled data (shipped inside `vcdd/cell/resources/`, ~11 MB)

| Resource | File | Content | Source | Licence / terms |
|---|---|---|---|---|
| LINCS L1000 gene info | `l1000_genes.tsv` | 10,174 BING genes, 978 landmark | GEO GSE92742 (Broad Institute) | CC0 / public domain (NIH LINCS data release) |
| LINCS L1000 chemical consensus signatures | `lincs_chem_consensus.parquet` | 5,425 compounds, up/down gene lists | Enrichr `LINCS_L1000_Chem_Pert_Consensus_Sigs` (Ma'ayan Lab) | Free for academic and non-commercial use; cite Enrichr / SigCom LINCS |
| LINCS L1000 CRISPR-KO consensus signatures | `lincs_crispr_ko_consensus.parquet` | 5,212 knocked-out genes | Enrichr `LINCS_L1000_CRISPR_KO_Consensus_Sigs` | Same as above |
| CollecTRI | `collectri.tsv` | 62,411 signed TF-target edges, 1,201 TFs | OmniPath (`datasets=collectri`) | CollecTRI is CC-BY 4.0, but it aggregates sources with mixed terms; some contributing resources are **academic-use only**. Check `sources` before commercial use. |
| PROGENy | `progeny.tsv` | 14 pathways x top-100 responsive genes | OmniPath (`resources=PROGENy`) | Apache 2.0 (model weights); cite Schubert et al. 2018 |
| MSigDB Hallmark | `hallmark.tsv` | 50 gene sets | Enrichr `MSigDB_Hallmark_2020` | MSigDB terms: free for academic use; **commercial use requires a licence from the Broad Institute**. KEGG-derived sets are deliberately excluded from this bundle for the same reason. |

Each file's exact source URL, row count and build timestamp are recorded in
`vcdd/cell/resources/MANIFEST.json`, together with **two distinct digests**:

- `source_sha256` — the digest of the payload as downloaded. It does **not**
  match the shipped file, because the payload is re-serialised (parsed to a
  DataFrame and written back as TSV or zstd parquet). Use it to confirm you got
  the same upstream release.
- `file_sha256` / `file_bytes` — the digest and size of the file actually shipped
  in this package. Use these to verify the bundle itself.

The compound structures joined to the LINCS consensus terms come from the Broad
Drug Repurposing Hub (<https://repo-hub.broadinstitute.org/repurposing>). Rebuild
the bundle from source with `python scripts/build_cell_resources.py`; set
`VCDD_REPURPOSING_HUB` to a local copy of the hub if you want the SMILES re-joined
(otherwise the join is skipped, which is reported but not fatal).

## Required external software

| Software | Used for | Licence |
|---|---|---|
| **Vina-GPU 2.1** (`AutoDock-Vina-GPU-2-1`, `QuickVina2-GPU-2-1`, `QuickVina-W-GPU-2-1`) | Stage 1 and stage-4 re-docking — the only docking engine in 0.2.0 | Apache 2.0. Built from source (`scripts/build_vina_gpu.sh`) from `DeltaGroupNJUPT/Vina-GPU-2.1`; not redistributed with `vcdd`. |
| Boost ≥ 1.77 | build-time dependency of Vina-GPU | Boost Software License 1.0 |
| OpenCL runtime (via the CUDA Toolkit or an AMD equivalent) | Vina-GPU kernels | vendor terms — **NVIDIA CUDA is redistributable only under the NVIDIA EULA**; install it yourself |
| RDKit | standardisation, conformers, fingerprints, reaction SMARTS | BSD-3-Clause |
| NumPy / pandas / PyArrow / SciPy | data layer | BSD-3-Clause / Apache 2.0 |
| scikit-learn | surrogate model for opt-in active learning | BSD-3-Clause |
| Typer / Pydantic / Rich | CLI and config | MIT |
| Matplotlib / seaborn | report figures | PSF-based / BSD-3-Clause |

## Hosted services (stage 2)

Stage 2 sends **your target sequence and your ligand SMILES** to a third-party
API. That is a data-disclosure decision, not just a licence one — review it
against your own confidentiality obligations before running stage 2 on a
proprietary target or an unpublished chemical series.

| Service | Used for | Terms |
|---|---|---|
| Boltz API (`boltz-api` SDK, `api.boltz.bio`) | Stage-2 co-folding and binding confidence: `small_molecule.library_screen` (BoltzMol-1) and `predictions.structure_and_binding` (boltz-2.1) | Commercial hosted service, **billed per molecule/sample**. Subject to the provider's terms of service and privacy policy. The SDK is MIT; the service is not. `vcdd` never transmits an API key anywhere except to the configured `base_url`. |

The open-source Boltz-2 model weights are **not** used or bundled by this
release — stage 2 is the hosted API only. If you run Boltz-2 locally instead,
its own licence applies.

## Optional software (absence degrades, never breaks)

| Software | Used for | Licence note |
|---|---|---|
| Meeko / ADFR `prepare_receptor` | receptor and ligand PDBQT preparation | LGPL-2.1 / Mozilla-style academic terms (ADFR is free for academic use) |
| Open Babel | fallback ligand/receptor conversion | GPL-2.0 — invoked only as a subprocess, never linked |
| CReM + `replacements02_sc2.db` | Stage 4 fragment replacement | BSD-3-Clause (code); DB free for research use — download from `http://www.qsar4u.com/files/cremdb/` |
| mmpdb | matched-molecular-pair transforms | Apache 2.0 |
| AutoDock Vina 1.2 (CPU) | **not used by 0.2.0.** Listed only because it is commonly already installed; `vcdd` does not register it as an engine | Apache 2.0 |

Removed in 0.2.0 along with the rescoring subsystem: smina, gnina, ODDT /
RFScore-VS, OpenMM-based MM-GBSA and gemmi. If a 0.1.0 environment still has
them installed they are simply ignored.

## Optional virtual-cell model checkpoints (never bundled)

None of these weights ship with `vcdd`; each adapter is activated by pointing an
environment variable at a checkpoint you obtained yourself, under that project's
own terms. No adapter in this release has been executed against real weights.

| Backend | Env var | Terms to check |
|---|---|---|
| chemCPA | `VCDD_CHEMCPA_DIR` | MIT code; released checkpoints trained on LINCS/sciPlex — inherit LINCS terms |
| TranSiGen | `VCDD_TRANSIGEN_DIR` | academic use; see the authors' repository |
| PRnet | `VCDD_PRNET_DIR` | academic use; see the authors' repository |
| PerturbNet | `VCDD_PERTURBNET_DIR` | academic use; see the authors' repository |
| Arc Institute STATE (`ST-Tahoe`, `ST-Parse`, `SE-600M`) | `VCDD_STATE_DIR` | **Arc Institute Acceptable Use Policy** — restricted; review before any use, and note it conditions on drug identity within its training panel |

## Test fixtures

`tests/fixtures/receptor.pdb` and `reference_ligand.sdf` derive from **PDB 1STP**
(streptavidin + biotin), redistributed under the RCSB PDB terms of use (public
domain). The 200-molecule fixture library is synthetic and carries no
third-party rights.
