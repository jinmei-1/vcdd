"""Pydantic configuration schema for a vcdd run.

One YAML file describes the whole four-stage cascade. Everything that changes a
numerical result is here (and therefore hashed into the resume ledger and the
manifest); nothing that changes a result is hidden in code defaults that a user
cannot see. ``vcdd config`` prints an annotated version of this schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------- library
class FilterConfig(_Base):
    """Cheap property filters applied during library preparation."""

    enabled: bool = True
    mw_min: float = 150.0
    mw_max: float = 600.0
    logp_min: float = -3.0
    logp_max: float = 6.0
    hbd_max: int = 6
    hba_max: int = 12
    tpsa_max: float = 180.0
    rotatable_max: int = 12
    heavy_atoms_min: int = 10
    heavy_atoms_max: int = 60
    #: Elements permitted anywhere in the molecule. Anything else is rejected.
    allowed_elements: list[str] = Field(
        default_factory=lambda: ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si"]
    )
    pains: bool = True
    reos: bool = True


class LibraryConfig(_Base):
    """How the input library is read, cleaned and sharded."""

    path: str | None = None
    #: Explicit column names; ``None`` means auto-detect (CSV/Parquet only).
    smiles_column: str | None = None
    id_column: str | None = None
    #: Molecules per output shard. 25k keeps a shard's parquet under ~10 MB.
    shard_size: int = 25_000
    #: Rows read per chunk while streaming a large file.
    chunk_size: int = 5_000
    standardize: bool = True
    #: Largest-fragment selection, neutralisation, canonical tautomer.
    desalt: bool = True
    neutralize: bool = True
    canonical_tautomer: bool = True
    dedup_by: Literal["inchikey", "inchikey14", "smiles", "none"] = "inchikey"
    max_molecules: int | None = None
    filters: FilterConfig = Field(default_factory=FilterConfig)


# -------------------------------------------------------------------- receptor
class BoxConfig(_Base):
    """Docking box. Either derived from a reference ligand or given explicitly."""

    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    #: Padding added around the reference ligand's bounding box, in Angstrom.
    padding: float = 4.0
    #: Clamp the derived cube to this range; boxes outside it are suspicious.
    min_edge: float = 16.0
    max_edge: float = 22.0


class ReceptorConfig(_Base):
    path: str | None = None
    #: Residue name of the co-crystal ligand used for the box and the redock gate.
    reference_ligand: str | None = None
    reference_ligand_path: str | None = None
    #: Chain to keep; ``None`` keeps all protein chains.
    chain: str | None = None
    keep_waters: bool = False
    keep_metals: bool = True
    ph: float = 7.4
    box: BoxConfig = Field(default_factory=BoxConfig)
    #: Native-ligand redock validation gate.
    redock_validation: bool = True
    redock_rmsd_threshold: float = 2.0
    #: Warn-not-fail: a failing gate is recorded loudly but does not abort.
    redock_fail_hard: bool = False
    #: Extra receptor structures for ensemble docking in stage 2.
    ensemble_paths: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- docking
class Stage1Config(_Base):
    """Fast, deliberately low-accuracy GPU docking over the whole library.

    0.2.0 is Vina-GPU 2.1 only. The search-effort knobs are therefore the ones
    that binary actually accepts (``thread``, ``search_depth``); the CPU-Vina
    knobs ``exhaustiveness``, ``cpu_per_ligand``, ``num_modes`` and
    ``energy_range`` are gone, because Vina-GPU 2.1 has no flags for them and
    silently accepting them would produce a config that means nothing.
    """

    engine: str = "vina_gpu"
    #: ``auto`` picks the fastest available engine reported by the registry.
    engine_selection: Literal["fixed", "auto"] = "fixed"
    #: Which Vina-GPU 2.1 binary to drive.
    variant: Literal["autodock_vina_gpu", "quickvina2_gpu", "quickvina_w_gpu"] = "autodock_vina_gpu"
    #: Ligands handed to the GPU in one invocation. This is the single biggest
    #: throughput lever: the receptor grid and the OpenCL kernels are uploaded
    #: once per batch, so small batches waste the device.
    gpu_batch_size: int = 1000
    #: Parallel Monte-Carlo threads on the device. Upstream default 8000; use
    #: 5000 for ``quickvina2_gpu``. Upstream advises staying below 10000.
    thread: int = 8000
    #: Monte-Carlo depth per thread. ``None`` = let the binary adapt per ligand,
    #: which is what upstream recommends for a heterogeneous library.
    search_depth: int | None = None
    #: Reduced-Iteration Local-search BFGS (upstream default on).
    rilc_bfgs: bool = True
    #: Directory containing the compiled ``Kernel1_Opt.bin``/``Kernel2_Opt.bin``.
    #: Leaving this unset makes the binary recompile the OpenCL kernels on every
    #: batch, which can dominate the run.
    opencl_binary_path: str | None = None
    #: Seeds ligand preparation (ETKDG) and the active-learning RNG. It does
    #: **not** seed the docking search: Vina-GPU 2.1 exposes no ``--seed``, so
    #: stage-1 scores are reproducible in protocol but not bit-for-bit.
    seed: int = 42
    n_conformers: int = 1
    minimize_conformer: bool = True
    #: Per-ligand ceiling on the single-ligand path (stage 4 re-dock).
    timeout_s: int = 300
    #: Per-batch ceiling on the batch path. One batch is one process, so this
    #: must be large enough for ``gpu_batch_size`` ligands.
    batch_timeout_s: int = 3600
    #: Pilot slice used to measure real throughput and print an ETA.
    pilot_size: int = 50
    #: Optional Deep-Docking-style surrogate active learning. Off by default:
    #: it trades measured recall for wall-clock and must be an explicit choice.
    surrogate_al: bool = False
    al_iterations: int = 3
    al_initial_fraction: float = 0.02
    al_select_fraction: float = 0.05
    al_holdout_fraction: float = 0.01
    al_model: Literal["rf", "gbt"] = "rf"


class LocalBoltzConfig(_Base):
    """Local Boltz-2 inference settings (``stage2.backend: local``).

    Runs ``boltz predict`` on this machine instead of calling the hosted API:
    no API key, no billing, and the GPU is yours. The model and weights are the
    same boltz-2.1 that ``structure_and_binding`` serves, so results are
    comparable to that backend's semantics - but *not* to ``library_screen``
    (BoltzMol-1 is a hosted-only screening pipeline).
    """

    #: ``boltz`` executable, or an absolute path to one.
    binary: str = "boltz"
    #: GPU ids handed out round-robin to the concurrent predict processes.
    #: ``[0, 1]`` with ``max_concurrency=2`` runs one process per GPU.
    devices: list[int] = Field(default_factory=lambda: [0])
    #: Concurrent ``boltz predict`` processes. Each process gets exactly one
    #: device, so keep this <= len(devices) to avoid GPU oversubscription.
    max_concurrency: int = 2
    #: Model-weights directory; ``None`` = Boltz's default cache location.
    cache_dir: str | None = None
    #: boltz 2.2 refuses to run without per-chain MSA. Pass ``--use_msa_server``
    #: so it fetches MSAs from ``msa_server_url`` (default: the public ColabFold
    #: server). Setting this false with no ``protein_msa_files`` makes every
    #: predict fail - the flag exists so a fully offline setup is possible.
    use_msa_server: bool = True
    #: MMSeqs2 server URL for ``--use_msa_server``; ``None`` = boltz's default
    #: (``https://api.colabfold.com``). Point this at a self-hosted server when
    #: the target sequence must not leave the building - the sequence IS sent to
    #: whatever server is configured.
    msa_server_url: str | None = None
    #: Optional basic-auth username for the MSA server (--msa_server_username).
    msa_server_username: str | None = None
    #: Optional basic-auth password for the MSA server (--msa_server_password).
    msa_server_password: str | None = None
    #: Precomputed per-chain MSAs, ``{chain_id: path/to/file.a3m}``, injected as
    #: the ``msa:`` field of each protein entity. Use this for offline runs or
    #: to query the MSA server *once* per receptor instead of once per molecule
    #: (a 500-molecule shortlist against one protein = 500 identical queries).
    #: Every protein chain either needs an entry here or ``use_msa_server=True``.
    protein_msa_files: dict[str, str] = Field(default_factory=dict)
    #: Pass ``--half`` (fp16) to ``boltz predict``. ``None`` = don't pass the
    #: flag and let Boltz use its own default precision.
    half_precision: bool | None = None
    #: Extra CLI arguments passed through verbatim to ``boltz predict``.
    extra_args: tuple[str, ...] = ()
    #: Per-molecule wall-clock ceiling, in seconds. A single predict is one
    #: process, so this needs to be a generous budget.
    timeout_s: int = 3600
    #: Keep the raw Boltz outputs (input YAML, predicted CIFs, confidence JSON)
    #: under ``<run>/stage2/boltz_local/``. Off saves disk, at the cost of not
    #: being able to re-inspect a specific prediction.
    keep_raw_outputs: bool = True
    #: Ask Boltz-2 for its *affinity* head in addition to the structure and
    #: confidence heads: emits ``properties: [{affinity: {binder: LIG}}]`` in the
    #: input document and reads ``affinity_*_model_*.json`` back into
    #: ``affinity_pred_value`` (log10 IC50, IC50 in uM) and
    #: ``affinity_probability_binary`` (binder probability).
    #:
    #: Off by default because it answers a different question: the confidence
    #: head says "is this complex plausible", the affinity head says "how
    #: strongly does this ligand bind", and the two are trained on different
    #: data. Boltz's own guidance is that ``affinity_probability_binary``
    #: separates binders from decoys while ``affinity_pred_value`` compares
    #: binders with each other, which makes the value head the right signal for
    #: ranking analogs of a binder - not for validating a docking pose. It also
    #: costs extra sampling (``--diffusion_samples_affinity``, default 5).
    affinity: bool = False
    #: ``--diffusion_samples_affinity``: samples averaged by the affinity head.
    affinity_samples: int = 5
    #: ``--sampling_steps_affinity``: diffusion steps for the affinity head.
    affinity_sampling_steps: int = 200
    #: ``--affinity_mw_correction``: add the molecular-weight correction to the
    #: affinity value head (Boltz's default for the released model).
    affinity_mw_correction: bool = False


class Stage2Config(_Base):
    """Boltz co-folding + binding prediction over the stage-1 shortlist.

    0.2.0 replaces re-docking and consensus rescoring with a call to the Boltz
    API. Read the caveat before setting this up: Boltz does **not** rescore the
    stage-1 pose. It re-predicts the complex from receptor sequence + ligand
    SMILES and reports its own confidence. That is *orthogonal* evidence, which
    is the point - a second docking program agreeing with the first mostly
    measures that they share a scoring-function lineage - but it means the
    stage-2 pose is a Boltz prediction, not a refined stage-1 pose.

    Three backends:

    ``library_screen``
        BoltzMol-1 (``engine="boltzmol"``), the endpoint built for screening.
        ~$0.025/molecule. Default.
    ``structure_and_binding``
        Boltz-2.1, one prediction per molecule, priced by receptor crop size
        (~$0.025/sample for a target under 256 residues). Slower and dearer,
        but gives a full structure and the boltz-2.1 binding head.
    ``local``
        The same boltz-2.1 model run on this machine via ``boltz predict``.
        Free and uncapped; the spend gate is skipped. Needs a CUDA GPU with
        enough VRAM (~14-20 GB for a typical target) and the ``boltz`` package
        installed (``pip install boltz``, see :mod:`vcdd.refine.boltz.local`).
    """

    backend: Literal["library_screen", "structure_and_binding", "local"] = "library_screen"
    top_n: int = 500
    #: Molecules per API submission. Keeps a single failure from costing the
    #: whole shortlist and keeps ``results.jsonl`` appendable.
    chunk_size: int = 500

    # ---- ranking -----------------------------------------------------------
    #: How the Boltz result is combined with the stage-1 docking score.
    #: ``boltz_only``  - rank by ``binding_confidence`` (default).
    #: ``rank_fusion`` - mean of the stage-1 rank and the Boltz rank.
    #: ``gate``        - keep only ``binding_confidence >= gate_threshold``,
    #:                   then rank the survivors by stage-1 score.
    fusion: Literal["boltz_only", "rank_fusion", "gate"] = "boltz_only"
    #: Boltz's own documentation calls 0.7+ the high-confidence triage range.
    gate_threshold: float = 0.7

    # ---- spend -------------------------------------------------------------
    #: Hard ceiling, in USD. ``estimate_cost`` is called first (it is free and
    #: allocates no GPU) and the run aborts above this unless --yes-spend.
    max_spend_usd: float = 25.0

    # ---- credentials and routing ------------------------------------------
    #: Name of the environment variable holding the API key. The key itself is
    #: never read from the config, never logged and never written to the
    #: manifest - only whether it was present, and live vs test mode.
    api_key_env: str = "BOLTZ_API_KEY"
    workspace_id: str | None = None

    # ---- target definition -------------------------------------------------
    #: Derive ``pocket_residues`` from the stage-1 docking box, so Boltz is
    #: pointed at the same site that was docked rather than folding blind.
    pocket_from_box: bool = True
    #: Angstrom added around the box when selecting pocket residues.
    pocket_margin_a: float = 0.0
    #: Pass the co-crystal ligand as a ``reference_ligand`` hint when available.
    use_reference_ligand: bool = True

    # ---- filtering ---------------------------------------------------------
    #: Boltz's SMARTS catalogue filter. The API default is ``recommended``,
    #: which *silently drops* molecules from the results. Defaulting to
    #: ``disabled`` here keeps stage 2's input and output sets comparable; the
    #: property filters in ``library.filters`` already ran in stage 0.
    molecule_filter_level: Literal["recommended", "extra", "aggressive", "disabled"] = "disabled"

    # ---- outputs -----------------------------------------------------------
    #: Predicted complex structures downloaded, best-first. Archives are large.
    keep_structures_top_k: int = 50
    poll_interval_s: int = 20
    request_timeout_s: int = 120
    #: ``structure_and_binding`` only: samples per molecule (cost scales).
    num_samples: int = 1
    #: ``structure_and_binding`` only: pass the receptor as a structural template.
    use_receptor_template: bool = False
    #: ``local`` only: how to run ``boltz predict`` on this machine.
    local_boltz: LocalBoltzConfig = Field(default_factory=LocalBoltzConfig)


# ------------------------------------------------------------------------ cell
class ObjectiveConfig(_Base):
    """The stage-3 multi-objective definition: genes + pathways + mechanism."""

    #: Genes the compound should drive UP (the user's primary ask).
    genes_up: list[str] = Field(default_factory=list)
    #: Genes the compound should drive DOWN.
    genes_down: list[str] = Field(default_factory=list)
    #: Optional per-gene weights; missing genes default to 1.0.
    gene_weights: dict[str, float] = Field(default_factory=dict)
    #: Pathways (PROGENy names or Hallmark set names) with desired direction.
    pathways_up: list[str] = Field(default_factory=list)
    pathways_down: list[str] = Field(default_factory=list)
    #: "any" scores the weighted mean movement; "all" requires every requested
    #: gene to move in the requested direction (strict variant).
    gene_mode: Literal["any", "all"] = "any"
    #: Component weights used as geometric-mean exponents. A component with
    #: weight 0 is reported but excluded from the aggregate.
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "gene_score": 2.0,
            "pathway_score": 1.0,
            "target_consistency": 1.0,
            "specificity": 1.0,
        }
    )
    #: Floor applied to each [0,1] component before the geometric mean, so a
    #: single zero does not make every compound indistinguishable at 0.
    component_floor: float = 0.0
    #: Genes whose induction marks generic stress rather than on-target effect.
    stress_genes: list[str] = Field(
        default_factory=lambda: [
            "HSPA5", "DDIT3", "ATF3", "HMOX1", "GDF15", "CDKN1A", "TP53I3",
            "JUN", "FOS", "EGR1", "HSPA1A", "DNAJB1", "SQSTM1", "TXNRD1",
        ]
    )
    #: Applicability domain is reported as a confidence flag, never folded into
    #: the score (a model is not more right because the compound is familiar).
    ad_tanimoto_warn: float = 0.4
    #: |value| at which a gene counts as "moved" for the specificity breadth
    #: term, keyed by the backend's value scale. Read-across consensus values
    #: are diluted by neighbour disagreement and so need a lower bar than a
    #: model that emits genuine z-scores.
    moved_threshold: dict[str, float] = Field(
        default_factory=lambda: {"membership": 0.1, "zscore": 2.0, "logfc": 1.0}
    )


class CellConfig(_Base):
    """Stage-3 virtual-cell context and backend selection."""

    #: Backends to run, in priority order. Unavailable ones skip with a message.
    backends: list[str] = Field(default_factory=lambda: ["lincs_readacross"])
    cell_line: str = "MCF7"
    dose_um: float = 10.0
    time_h: float = 24.0
    dose_sweep: list[float] = Field(default_factory=list)
    top_n: int = 200
    #: Gene universe restriction; "landmark" = the 978 L1000 landmark genes.
    gene_space: Literal["landmark", "bing", "all"] = "landmark"
    #: The docking target's gene symbol. Enables the target_knockdown mechanism
    #: reference used by the target_consistency objective component.
    target_gene: str | None = None
    protein_activity: bool = True
    #: decoupleR-style inference method for TF / pathway activity.
    activity_method: Literal["ulm", "wsum", "wmean"] = "ulm"
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    cache_dir: str | None = None
    request_timeout_s: int = 60
    offline: bool = False


# -------------------------------------------------------------------- optimize
class GuardrailConfig(_Base):
    """Constraints that stop a generator from exploiting the surrogate scores."""

    similarity_floor_ga: float = 0.5
    similarity_floor_enum: float = 0.6
    heavy_atom_delta_max: int = 8
    max_ring_size: int = 7
    max_bridgehead_atoms: int = 2
    max_spiro_atoms: int = 1
    sa_score_max: float = 4.5
    pains: bool = True
    mw_max: float = 650.0
    logp_max: float = 6.0
    reject_parent_identity: bool = True


class Stage4Config(_Base):
    """Virtual-cell-guided analog design over the stage-3 winners."""

    top_parents: int = 20
    generators: list[str] = Field(default_factory=lambda: ["crem", "rgroup", "constrained_ga"])
    analogs_per_parent: int = 50
    #: Constrained GA settings.
    ga_generations: int = 5
    ga_population: int = 40
    ga_mutation_rate: float = 0.6
    seed: int = 42
    #: Re-dock analogs with stage-2 settings (slow) or stage-1 settings (fast).
    redock_mode: Literal["stage1", "stage2", "none"] = "stage1"
    #: Weights for the geometric-mean design score.
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {"docking": 1.0, "cell": 2.0, "qed": 0.5, "sa": 0.5}
    )
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    #: Minimum number of parents a matched-molecular-pair transformation must
    #: appear in before it is reported as an SAR rule.
    sar_min_parents: int = 2
    retrosynthesis: bool = False
    crem_db: str | None = None
    mmpdb_reference: str | None = None


# ------------------------------------------------------------------------- run
class RunConfig(_Base):
    """Top-level configuration object."""

    name: str = "vcdd-run"
    out_dir: str = "./vcdd_run"
    seed: int = 42
    n_workers: int | None = None
    shard: int = 0
    nshards: int = 1
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    library: LibraryConfig = Field(default_factory=LibraryConfig)
    receptor: ReceptorConfig = Field(default_factory=ReceptorConfig)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    cell: CellConfig = Field(default_factory=CellConfig)
    stage4: Stage4Config = Field(default_factory=Stage4Config)

    @field_validator("nshards")
    @classmethod
    def _check_nshards(cls, v: int) -> int:
        if v < 1:
            raise ValueError("nshards must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_shard(self) -> "RunConfig":
        if not (0 <= self.shard < self.nshards):
            raise ValueError(f"shard must be in [0, {self.nshards}), got {self.shard}")
        if self.stage2.top_n < 1:
            raise ValueError("stage2.top_n must be >= 1")
        if self.stage2.chunk_size < 1:
            raise ValueError("stage2.chunk_size must be >= 1")
        if self.stage2.local_boltz.max_concurrency < 1:
            raise ValueError("stage2.local_boltz.max_concurrency must be >= 1")
        if not self.stage2.local_boltz.devices:
            raise ValueError("stage2.local_boltz.devices must list at least one GPU id")
        if self.stage1.gpu_batch_size < 1:
            raise ValueError("stage1.gpu_batch_size must be >= 1")
        if self.stage4.redock_mode == "stage2":
            # Stage 2 no longer docks anything, so there are no stage-2 docking
            # settings to re-dock under.
            raise ValueError(
                "stage4.redock_mode='stage2' is not valid in 0.2.0: stage 2 is a Boltz "
                "prediction, not a docking run, so it has no docking parameters to reuse. "
                "Use 'stage1' (re-dock analogs with the Vina-GPU settings) or 'none'."
            )
        if self.stage1.thread > 10_000:
            import warnings

            warnings.warn(
                f"stage1.thread={self.stage1.thread}: Vina-GPU 2.1 upstream advises staying "
                f"below 10000; higher values can exhaust device memory.",
                stacklevel=2,
            )
        return self

    # -------------------------------------------------------------- accessors
    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir)

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
