"""Tier-2 cell backends: full adapters that need third-party weights.

Both models here accept an arbitrary SMILES string, which is what stage 4
requires -- a read-across backend cannot score a molecule that does not exist
yet, so analog design needs at least one of these (or ``mock``) to be live.

Each adapter checks for its package and checkpoint, prints the exact fetch
commands when they are absent, and is exercised in CI by contract tests.
``chemcpa`` has been run against the real released checkpoints
(``chemCPA_models.zip``, 27b401-… = RDKit encoder, Setting-1 shared genes) and
its loading/scoring path is verified; ``transigen`` ships no released weights,
so its adapter remains unvalidated.

chemCPA
    Compositional perturbation autoencoder extended to chemistry: a molecular
    encoder supplies a drug embedding that is added to a basal cell state, with
    dose and covariate adjustment [14, 16, 17].  Novel molecules are embedded
    with the authors' RDKit2D-descriptor pretraining recipe.
TranSiGen
    Self-supervised representation learning that maps a compound's structure
    directly to its L1000 differential signature [13].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..base import (
    CellModel,
    CellModelCapabilities,
    PerturbationRequest,
    Signature,
    register_backend,
)

__all__ = ["ChemCPAModel", "TranSiGenModel", "XpertModel"]


def _first_existing(*candidates: str | None) -> Path | None:
    for c in candidates:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser()
    return None


def _canonicalize_smiles(smiles: str) -> str | None:
    """chemCPA's canonicalization (chemCPA/helper.py), so lookups match the
    pretrained embedding tables."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,
        isomericSmiles=True,
        canonical=True,
        doRandom=False,
        allBondsExplicit=False,
        allHsExplicit=False,
        kekuleSmiles=False,
    )


@register_backend
class ChemCPAModel(CellModel):
    """chemCPA adapter, verified against the released ``chemCPA_models.zip``.

    Install (one-time)::

        pip install descriptastorus
        git clone https://github.com/theislab/chemCPA && cd chemCPA && pip install -e .
        mkdir -p ~/cell_models/chemcpa && cd ~/cell_models/chemcpa
        curl -LO https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip
        unzip chemCPA_models.zip
        mv chemCPA_models/27b401db1845eea26c102fb614df9c33.pt .   # keep ONLY this one
        rm -rf chemCPA_models __MACOSX chemCPA_models.zip
        export VCDD_CHEMCPA_DIR=$PWD

    Checkpoints
        The zip's hash-named ``.pt`` files are **not** pytorch-lightning
        checkpoints: each is a 5-tuple ``(state_dict, adversary_states,
        covariate_states, init_args, history)`` (``experiments/dom_experiments/
        utils.py`` in the chemCPA repo).  Only the RDKit-encoder, Setting-1
        (genes shared between L1000 and sci-Plex, 977 genes) checkpoint
        ``27b401d…`` is loadable here; the GROVER / JT-VAE checkpoints need
        their own external molecular encoders.  Loading needs the ``chemCPA``
        package (``chemCPA.model.ComPert``), and the released pickles reference
        ``sacred.config.custom_containers``, which the adapter stubs at load
        time so no ``sacred`` install is needed.

    Gene names, control cells and drug space
        The released checkpoints carry no gene names, and the model's 977 genes
        are not exactly vcdd's 978 landmarks (they are the L1000/sci-Plex
        intersection).  chemCPA predicts the response of a *cell state* to a
        drug, so scoring needs the target cell line's control (DMSO) expression
        plus the cell-type covariate.  Drop the official SciPlex3 data (the
        biolord split h5ad is dead; the published shared-gene file works) into
        ``$VCDD_CHEMCPA_DIR``::

            wget -O "$VCDD_CHEMCPA_DIR/sciplex_complete_middle_subset_lincs_genes.h5ad" \\
              https://f003.backblazeb2.com/file/chemCPA-datasets/sciplex_complete_middle_subset_lincs_genes.h5ad
            gdown 1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9   # rdkit2D_embedding_biolord.parquet
            pip install anndata

        The adapter then derives the 977 gene names from the h5ad ``var_names``
        (checkpoint order), builds the per-cell-line control basal from the
        ``control`` (DMSO) cells grouped by ``cell_type``, conditions on the
        matching covariate embedding, and resolves molecules that belong to the
        dataset's drug space through the official ``use_drugs_idx`` table
        (SMILES in first-appearance order -> ``drug_embeddings`` row, see
        ``_table_smiles``; doses are converted from vcdd's µM to the model's
        nM scale (config ``dose_key: dose``).  ``genes.txt`` remains available
        as an override.
        Without the h5ad, scoring fails fast with instructions instead of
        silently using a zero basal.

        Validation note: running the released checkpoints against the
        official SciPlex shared-gene data with the official protocol
        (all-control-cells basal, drugs_idx, nM doses, trained covariates,
        ``evaluate_logfold_r2`` metric) reproduces the per-gene reconstruction
        correlation (~0.85, driven by the control-state shape) but the drug
        component collapses:
        - the backblaze-zip 27b401 checkpoint: drug-embedding encoder outputs
          are almost identical across drugs (pairwise corr ~0.97) and the
          amortized doser saturates near zero (scale ~1e-4..7e-3);
        - the paper-figure models (``chemCPA_models-paperfigure.zip``, e.g.
          475e26950b / dde01c1c58 / c824e42f7c) are structurally healthy
          (encoder pairwise corr 0.19-0.39, doser scale 0.15-0.64) but the
          effective drug delta at the output is ~0.02 per gene (real effects
          are 0.5+), so held-out logfold R² stays large and negative
          (ood ~-11, train ~-50; PCC ~-0.25; sign accuracy ~0.56), unchanged
          across row-order hypotheses, direction and dose scaling.
        The checkpoints are therefore faithful to load and run, but none of
        the released weights is a working drug-response model - fine-tune on
        the target assay to get one, the adapter loads the result unchanged.

    Novel molecules (the authors' protocol)
        chemCPA embeds each drug by exact canonical-SMILES lookup in a
        pretrained embedding table (``chemCPA.embedding.get_chemical_
        representation``); a molecule missing from the table falls back to a
        **zero vector**, i.e. is scored as "no effect".  This adapter mirrors
        that lookup against the RDKit2D embedding parquet
        (``rdkit2D_embedding_biolord.parquet``, Drive id ``1oV2o5dVEVE3OwBVZzuu
        JTXuaamZJeFL9``) dropped into ``$VCDD_CHEMCPA_DIR``, and **extends it**:
        the parquet fixes the descriptor-to-embedding affine map (the authors'
        script drops the sentinel column, removes low-variance columns and
        z-scores), so a molecule not in the table still receives its true
        RDKit2D embedding via that fitted transform instead of a zero vector.
        Without the parquet, novel-molecule scoring raises an explicit error.
    """

    #: Released tuple layout: (state_dict, adversary, covariates, init_args, history).
    #: ``init_args`` holds exactly the ComPert constructor kwargs.
    _RELEASED_LAYOUT = (0, 1, 2, 3, 4)

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="chemcpa",
            tier=2,
            description=(
                "chemCPA compositional perturbation autoencoder; predicts dose- "
                "and covariate-conditioned expression from molecular structure."
            ),
            accepts_novel_smiles=True,
            gene_space="checkpoint",
            value_scale="logfc",
            python_requires=("torch", "rdkit", "descriptastorus", "anndata"),
            env_requires=("VCDD_CHEMCPA_DIR",),
            install_hint=(
                "pip install torch rdkit descriptastorus anndata; "
                "git clone https://github.com/theislab/chemCPA && pip install -e chemCPA; "
                "keep the RDKit Setting-1 checkpoint (27b401d…) in VCDD_CHEMCPA_DIR; "
                "drop sciplex_complete_middle_subset_lincs_genes.h5ad "
                "(f003.backblazeb2.com/file/chemCPA-datasets/, see the class docstring) "
                "and rdkit2D_embedding_biolord.parquet (Drive id "
                "1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9) there too - the model scores "
                "the drug response of a cell state, so the control expression "
                "and drug embedding tables are required"
            ),
            citation="Hetzel et al., NeurIPS 2022 (chemCPA)",
            cell_line_aware=True,
            dose_aware=True,
        )

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._model = None
        self._gene_names: list[str] = []
        self._embedding_df = None
        self._embedding_tf: tuple | None = None
        self._descriptor_gen = None
        self._cld_scale: float | None = None
        self._anndata = None
        self._control_vecs: dict[str, np.ndarray] | None = None
        self._cov_categories: list[str] | None = None
        self._drug_table: dict[str, int] | None = None

    # ------------------------------------------------------------ discovery
    def _checkpoint(self) -> Path | None:
        root = os.environ.get("VCDD_CHEMCPA_DIR")
        if not root:
            return None
        root_path = Path(root).expanduser()
        if root_path.is_file():
            return root_path
        for pattern in ("*27b401*.pt", "*rdkit*.pt", "*lincs*.ckpt", "*.ckpt", "*.pt"):
            hits = sorted(sorted(root_path.rglob(pattern)))
            if hits:
                return hits[0]
        return None

    def _extra_availability(self) -> tuple[bool, str]:
        ckpt = self._checkpoint()
        if ckpt is None:
            return False, "no checkpoint in VCDD_CHEMCPA_DIR; " + self.capabilities().install_hint
        try:
            import chemCPA  # noqa: F401,PLC0415
        except ImportError:
            return False, (
                "the chemCPA package is not importable; "
                "git clone https://github.com/theislab/chemCPA && pip install -e chemCPA"
            )
        try:
            import descriptastorus  # noqa: F401,PLC0415
        except ImportError:
            return False, "pip install descriptastorus (novel-molecule drug embeddings)"
        try:
            import anndata  # noqa: F401,PLC0415
        except ImportError:
            return False, "pip install anndata (control-cell expression matrix)"
        notes = []
        if not self._companion_genes() and not self._anndata_path() and self._checkpoint() is not None:
            notes.append("no h5ad (control cells) in VCDD_CHEMCPA_DIR")
        if self._embedding_parquet() is None:
            notes.append("no rdkit2D embedding parquet")
        return True, (
            f"checkpoint {ckpt.name}"
            + ("; " + "; ".join(notes) if notes else "")
        )

    # -------------------------------------------------------------- loading
    @staticmethod
    def _patch_sacred() -> None:
        """The released pickles reference ``sacred.config.custom_containers``
        (a training-only dependency); stub it so torch.load works without it.

        ``ReadOnlyDict``/``ConfigAdded`` must unpickle as real dicts (the model
        calls ``.update``/``.get``/``.items`` on them); the remaining sacred
        value wrappers only need to be inert and callable-safe.
        """
        import sys  # noqa: PLC0415
        import types  # noqa: PLC0415

        if "sacred.config.custom_containers" in sys.modules:
            return

        class _DictLike(dict):
            def __setitem__(self, k, v):  # ReadOnlyDict pickles must refill freely
                dict.__setitem__(self, k, v)

        class _Inert:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, k):
                return _Inert()

            def __call__(self, *a, **k):
                return _Inert()

            def __str__(self):
                return "<sacred-stub>"

            def __repr__(self):
                return "<sacred-stub>"

            def __eq__(self, other):
                return isinstance(other, _Inert)

            def __hash__(self):
                return 0

            def __bool__(self):
                return False

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        def _get_class(name: str):
            if name in ("ReadOnlyDict", "ConfigAdded"):
                return _DictLike
            return _Inert

        _stubs = {}
        for mod_name in ("sacred", "sacred.config", "sacred.config.custom_containers"):
            module = types.ModuleType(mod_name)
            module.__dict__["_get_class"] = _get_class
            module.__getattr__ = _get_class
            _stubs[mod_name] = module
        sys.modules.update(_stubs)
        # make submodule imports ("from sacred.config.custom_containers import X")
        # resolve children on the parent modules
        sys.modules["sacred"].config = sys.modules["sacred.config"]
        sys.modules["sacred.config"].custom_containers = sys.modules[
            "sacred.config.custom_containers"
        ]

    def _load(self):
        """Rebuild the ComPert module from the released 5-tuple checkpoint."""
        if self._model is not None:
            return self._model
        import torch  # noqa: PLC0415
        from chemCPA.model import ComPert  # noqa: PLC0415

        ckpt_path = self._checkpoint()
        self._patch_sacred()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict):  # pragma: no cover - legacy path kept for dict checkpoints
            state = ck.get("state_dict", ck)
            cfg = ck.get("hyper_parameters") or ck.get("hparams") or {}
            cfg = cfg.get("hparams", cfg) if hasattr(cfg, "get") else cfg
        elif isinstance(ck, (tuple, list)) and len(ck) >= 4:
            state, cfg = ck[0], ck[3]
            cov_states = ck[2] if len(ck) > 2 else None
        else:  # pragma: no cover - defensive
            raise RuntimeError(
                "unrecognised checkpoint format under VCDD_CHEMCPA_DIR "
                f"({type(ck).__name__}); use the released *.pt files"
            )
        state = {k.split("model.", 1)[-1]: v for k, v in state.items()}
        if not any(k.startswith("drug_embedding_encoder") for k in state):
            raise RuntimeError(
                "checkpoint has no drug_embedding_encoder; the GROVER / JT-VAE "
                "checkpoints need their own encoders - use the RDKit Setting-1 "
                "checkpoint (27b401d…), see the adapter docstring"
            )
        if "drug_embeddings.weight" not in state:
            raise RuntimeError("checkpoint lacks drug_embeddings.weight (not an RDKit model?)")

        allowed = set(ComPert.__init__.__code__.co_varnames) - {"self"}
        kwargs = {
            k: v for k, v in cfg.items() if k in allowed and not callable(v)
        } if hasattr(cfg, "items") else {}
        kwargs.setdefault("device", "cpu")
        # The released drug-embedding table is a pretrained RDKit2D table whose
        # width differs from hparams["dim"]; construct the embedding module with
        # the saved width so load_state_dict can fill it.
        kwargs.setdefault(
            "drug_embeddings",
            torch.nn.Embedding(
                int(state["drug_embeddings.weight"].shape[0]),
                int(state["drug_embeddings.weight"].shape[1]),
            ),
        )
        model = ComPert(**kwargs)
        model.load_state_dict(state, strict=False)
        # the released tuple also carries the trained covariate embeddings
        # (element [2]); load them so cell-type conditioning actually works
        if isinstance(cov_states, (tuple, list)) and hasattr(
            model, "covariates_embeddings"
        ):
            for i, s in enumerate(cov_states):
                if not isinstance(s, dict) or i >= len(model.covariates_embeddings):
                    continue
                try:
                    cs = {k.split("model.", 1)[-1]: v for k, v in s.items()}
                    model.covariates_embeddings[i].load_state_dict(cs, strict=False)
                except Exception:  # noqa: BLE001 - cosmetic; conditioning degrades
                    pass
        model.eval()
        self._model = model
        self._gene_names = self._companion_genes() or []
        self._cld_scale = self._probe_cld_scale(model)
        return model

    @staticmethod
    def _probe_cld_scale(model) -> float:
        """Mean CLD scaling at 10 µM for the first drug, a diagnostic for the
        released checkpoints: several published snapshots pin the amortized
        doser at either ~0 (saturated sigmoid) or exactly 0.5, making the drug
        pathway inert; surface that in signature metadata instead of hiding it."""
        import torch  # noqa: PLC0415

        try:
            with torch.no_grad():
                emb0 = model.drug_embeddings.weight[:1]
                dose = torch.tensor([[10.0]])
                scaled = model.dosers(torch.cat([emb0, dose], dim=1)).sigmoid()
                return float(scaled.mean())
        except Exception:  # noqa: BLE001
            return float("nan")

    # ------------------------------------------------------------ gene names
    def _anndata_path(self) -> Path | None:
        root = os.environ.get("VCDD_CHEMCPA_DIR")
        if not root:
            return None
        root_path = Path(root).expanduser()
        if root_path.is_file():
            root_path = root_path.parent
        for pattern in ("*.h5ad",):
            hits = sorted(sorted(root_path.rglob(pattern)))
            if hits:
                return hits[0]
        return None

    def _load_anndata(self):
        """The paper's single-cell data (adata_biolord_split_30.h5ad), which
        the model was trained on: shared-gene expression matrix, per-cell
        ``control`` flag, and the ``cell_type`` covariate."""
        if self._anndata is not None:
            return self._anndata
        p = self._anndata_path()
        if p is None:
            raise RuntimeError(
                "no *.h5ad in $VCDD_CHEMCPA_DIR; chemCPA predicts the response of "
                "a *cell state* to a drug, so it needs the control (DMSO) gene "
                "expression of the target cell line - wget "
                "https://f003.backblazeb2.com/file/chemCPA-datasets/"
                "sciplex_complete_middle_subset_lincs_genes.h5ad plus "
                "rdkit2D_embedding_biolord.parquet (Drive id "
                "1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9) into $VCDD_CHEMCPA_DIR"
            )
        try:
            import anndata  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "reading control cells from *.h5ad needs the anndata package; "
                "pip install anndata"
            ) from None
        self._anndata = anndata.read_h5ad(p)
        return self._anndata

    def _cell_type_key(self, adata) -> str:
        key = os.environ.get("VCDD_CHEMCPA_CELL_TYPE", "cell_type")
        if key in adata.obs.columns:
            return key
        candidates = [c for c in adata.obs.columns if adata.obs[c].dtype.name == "category"]
        if candidates:
            return candidates[0]
        raise RuntimeError(
            f"no cell-type column (looked for {key!r}) in the h5ad obs; "
            f"columns: {list(adata.obs.columns)[:12]}"
        )

    def _control_vectors(self) -> dict[str, np.ndarray]:
        """{cell-line label: mean control (DMSO) expression in gene order}.

        Also records the covariate category order (``self._cov_categories``) in
        the h5ad's own category order, which is how the model's trained
        covariate embeddings are indexed.
        """
        if self._control_vecs is not None:
            return self._control_vecs
        adata = self._load_anndata()
        control_key = os.environ.get("VCDD_CHEMCPA_CONTROL_KEY", "control")
        if control_key not in adata.obs.columns:
            raise RuntimeError(
                f"no control flag column (looked for {control_key!r}) in the h5ad "
                "obs; columns: " + str(list(adata.obs.columns)[:12])
            )
        ckey = self._cell_type_key(adata)
        flags = adata.obs[control_key]
        mask = np.asarray(
            [str(f) in ("True", "1", "true", "yes") for f in flags]
        )
        lab = adata.obs[ckey].astype("category")
        cats = list(lab.cat.categories)
        vectors: dict[str, np.ndarray] = {}
        for cat in cats:
            sel = np.asarray(mask) & (lab == cat).to_numpy()
            if sel.sum() < 2:
                continue
            X = adata.X[sel]
            if hasattr(X, "toarray"):
                X = X.toarray()
            vectors[str(cat)] = np.asarray(X, dtype=np.float32).mean(0)
        if not vectors:
            raise RuntimeError(
                "no control (DMSO) cells found in the h5ad; the adapter needs the "
                "control-state expression to predict drug responses"
            )
        self._control_vecs = vectors
        self._cov_categories = cats
        return vectors

    def _control_basal(self, cell_line: str) -> np.ndarray | None:
        vectors = self._control_vectors()
        return vectors.get(cell_line)

    def _covariate_index(self, cell_line: str) -> int | None:
        try:
            return self._cov_categories.index(cell_line)
        except (AttributeError, ValueError):
            return None

    def _companion_genes(self) -> list[str]:
        root = os.environ.get("VCDD_CHEMCPA_DIR")
        if not root:
            return []
        root_path = Path(root).expanduser()
        if root_path.is_file():  # VCDD_CHEMCPA_DIR may point at a file? keep dir-only
            root_path = root_path.parent
        for name in ("genes.txt", "var_names.txt", "gene_names.txt"):
            p = root_path / name
            if not p.exists():
                continue
            genes = [
                line.strip()
                for line in p.read_text().splitlines()
                if line.strip()
            ]
            if genes:
                return genes
        # otherwise derive the model's gene order from the h5ad var_names
        try:
            adata = self._load_anndata()
        except Exception:  # noqa: BLE001
            return []
        return [str(v) for v in adata.var_names]

    def genes(self) -> list[str]:
        if not self._gene_names:
            self._gene_names = self._companion_genes()
        if self._gene_names:
            return self._gene_names
        # try the checkpoint itself (never present in the released zip)
        if self._model is not None:
            hparams = getattr(self._model, "hparams", {}) or {}
            names = hparams.get("var_names") or hparams.get("gene_names") or []
            if names:
                self._gene_names = list(names)
                return self._gene_names
        raise RuntimeError(
            "the chemCPA checkpoint records no gene names and no genes.txt companion "
            "was found in $VCDD_CHEMCPA_DIR; export the model's 977 shared genes from "
            "the paper's biolord split into $VCDD_CHEMCPA_DIR/genes.txt "
            "(one symbol per line, checkpoint order) - see the adapter docstring"
        )

    # --------------------------------------------------------- drug embedding
    def _descriptor_generator(self):
        if self._descriptor_gen is None:
            from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator  # noqa: PLC0415

            self._descriptor_gen = MakeGenerator(("RDKit2D",))
        return self._descriptor_gen

    def _embedding_parquet(self) -> Path | None:
        root = os.environ.get("VCDD_CHEMCPA_DIR")
        if not root:
            return None
        root_path = Path(root).expanduser()
        if root_path.is_file():
            root_path = root_path.parent
        for pattern in ("*rdkit*.parquet", "*embedding*.parquet", "*.parquet"):
            hits = sorted(sorted(root_path.rglob(pattern)))
            if hits:
                return hits[0]
        return None

    def _load_embedding_df(self):
        """(canonical-smiles indexed, d columns) pretrained RDKit2D parquet."""
        if self._embedding_df is None:
            import pandas as pd  # noqa: PLC0415

            p = self._embedding_parquet()
            if p is None:
                raise RuntimeError(
                    "no RDKit2D embedding parquet in $VCDD_CHEMCPA_DIR; novel molecules "
                    "need rdkit2D_embedding_biolord.parquet (Drive id "
                    "1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9) dropped into "
                    "$VCDD_CHEMCPA_DIR - see the adapter docstring"
                )
            df = pd.read_parquet(p)
            if callable(getattr(df.index, "str", None)):
                pass
            if not isinstance(df.index, pd.Index):
                df = df.set_index(df.columns[0])
            df.index = df.index.astype(str)
            self._embedding_df = df.astype(np.float32)
        return self._embedding_df

    def _embedding_transform(self):
        """Fitted raw-RDKit2D -> embedding affine.

        The authors' pretraining recipe (embeddings/rdkit/embedding_rdkit.py):
        RDKit2D descriptors, drop the sentinel column, drop low-variance
        columns, z-score against the reference drug set.  The parquet is the
        result of that pipeline, so per embedding column a single raw column
        (and per-column slope/intercept) recovers the transform exactly;
        novel molecules then embed through it instead of falling back to the
        authors' zero vector.
        """
        if self._embedding_tf is not None:
            return self._embedding_tf

        df = self._load_embedding_df()
        gen = self._descriptor_generator()
        # Canonical-smiles -> raw descriptors (199 cols after the sentinel).
        raw_by_canon: dict[str, np.ndarray] = {}
        for s in df.index:
            if len(raw_by_canon) >= 400:
                break
            c = _canonicalize_smiles(str(s)) if isinstance(s, str) else None
            if c is None or c in raw_by_canon:
                continue
            r = gen.process(c)
            if r is None or len(r) != 201:
                continue
            x = np.asarray(r[:-1], dtype=np.float64)[1:]
            x[~np.isfinite(x)] = 0.0  # authors zero NaN/Inf descriptors
            raw_by_canon[c] = x
        if len(raw_by_canon) < 20:
            raise RuntimeError(
                "could not compute RDKit2D descriptors for the parquet's reference "
                "molecules; is descriptastorus installed?"
            )
        # (n, 199) raw descriptors and (n, d) parquet rows, aligned by df order.
        common = [c for c in raw_by_canon]
        X = np.asarray([raw_by_canon[c] for c in common])
        canon_of_index = {
            i: _canonicalize_smiles(str(s)) for i, s in enumerate(df.index)
        }
        Y = np.asarray(
            [
                df.iloc[i].to_numpy(np.float64)
                for i in range(len(df))
                if canon_of_index[i] in raw_by_canon
            ],
            dtype=np.float32,
        )[: len(common)]
        slope = np.zeros(Y.shape[1], dtype=np.float64)
        intercept = np.zeros(Y.shape[1], dtype=np.float64)
        src_col = np.zeros(Y.shape[1], dtype=int)
        for k in range(Y.shape[1]):
            yk = Y[:, k]
            if yk.std() < 1e-9:
                slope[k], intercept[k] = 0.0, float(yk[0])
                src_col[k] = -1
                continue
            best_c, best_j = 0.0, 0
            for j in range(X.shape[1]):
                c = abs(np.corrcoef(X[:, j], yk)[0, 1])
                if c > best_c:
                    best_c, best_j = c, j
            if best_c < 0.9:
                raise RuntimeError(
                    f"parquet column {k} does not match any RDKit2D descriptor "
                    f"(best corr {best_c:.3f}); is the parquet the authors' "
                    "rdkit2D embedding output?"
                )
            a_coef, b_coef = np.polyfit(X[:, best_j], yk, 1)  # y = a*x + b
            slope[k], intercept[k], src_col[k] = a_coef, b_coef, best_j
        self._embedding_tf = (slope, intercept, src_col)
        return self._embedding_tf

    def _table_smiles(self) -> dict[str, int] | None:
        """canonical SMILES -> index in the model's drug_embeddings table.

        The official datasets (sciplex_complete_middle_subset* / the biolord
        split) share exactly ``num_drugs`` unique drugs.  We assume the
        released table rows are indexed in *first-appearance order* of the
        canonical SMILES in the h5ad: the authors' ``embedding_rdkit.py``
        rebuilds the embedding matrix by iterating the input SMILES list
        (``for smile in smiles_list``), so the row order follows the h5ad's
        unique-SMILES occurrence order, not a sorted order.

        Caveat: the true row ordering of the released 194-column tables is not
        independently verifiable (the downloadable rdkit2D parquet is a
        different 177-column run), and held-out validation shows the released
        checkpoints have ~no drug response under *any* row ordering, so this
        mapping only affects the ``drug_idx`` annotation and which row is fed
        to the model.  Enabled only when the count matches the checkpoint.
        """
        if self._drug_table is not None:
            return self._drug_table
        try:
            adata = self._load_anndata()
        except Exception:  # noqa: BLE001 - drug-space map is an optional extra
            self._drug_table = {}
            return self._drug_table
        if "SMILES" not in adata.obs.columns:
            self._drug_table = {}
            return self._drug_table
        canons: dict[str, int] = {}
        for s in adata.obs["SMILES"].unique():
            c = _canonicalize_smiles(str(s))
            if c is not None and c not in canons:
                canons[c] = len(canons)
        if len(canons) != int(self._model.drug_embeddings.weight.shape[0]):
            self._drug_table = {}
        else:
            self._drug_table = canons
        return self._drug_table

    def _dose_to_model(self, dose_um: float) -> float:
        """The released checkpoints were trained with obs['dose'] in nM
        (config dose_key: dose; sci-Plex doses 10-10000), while vcdd's
        requests carry µM - re-scale so the amortized doser sees its observed
        input domain (note: the dosing pathway is ~saturated in these
        checkpoints anyway)."""
        return float(dose_um) * 1000.0

    def _drug_embedding(self, smiles: str) -> tuple[np.ndarray, int | None]:
        """(embedding, table-index-or-None) for a drug.

        Molecules that are part of the dataset's drug space are resolved via
        the official ``use_drugs_idx`` table lookup (index into
        ``drug_embeddings.weight``); anything else is embedded with the
        pretrained RDKit2D table (exact canonical lookup, else per-column
        affine fit against the RDKit2D parquet).
        """
        c = _canonicalize_smiles(smiles)
        if c is None:
            raise ValueError(f"unparsable SMILES: {smiles}")
        table = self._table_smiles()
        if self._model is not None and table and c in table:
            idx = table[c]
            row = self._model.drug_embeddings.weight[idx].detach().cpu().numpy()
            return np.asarray(row, dtype=np.float64), int(idx)
        df = self._load_embedding_df()
        canon_of_index = {
            i: _canonicalize_smiles(str(s)) for i, s in enumerate(df.index)
        }
        for i, cc in canon_of_index.items():
            if cc == c:
                return np.asarray(df.iloc[i].to_numpy(np.float32), dtype=np.float64), None
        slope, intercept, src_col = self._embedding_transform()
        gen = self._descriptor_generator()
        r = gen.process(c)
        if r is None or len(r) != 201:
            raise ValueError(f"RDKit2D descriptors failed for {smiles}")
        x = np.asarray(r[:-1], dtype=np.float64)[1:]
        x[~np.isfinite(x)] = 0.0
        out = np.zeros(len(slope), dtype=np.float64)
        for k, j in enumerate(src_col):
            if j < 0:  # constant parquet column (zero-variance reference data)
                out[k] = intercept[k]
            else:
                out[k] = x[j] * slope[k] + intercept[k]
        return out, None

    # -------------------------------------------------------------- scoring
    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        import torch  # noqa: PLC0415

        model = self._load()
        cfg = getattr(model, "hparams", {}) or {}
        doser_type = str(cfg.get("doser_type", "amortized"))
        emb_dim = int(model.drug_embeddings.weight.shape[1])
        has_cov_emb = bool(
            getattr(model, "covariates_embeddings", None)
        ) and len(model.covariates_embeddings) > 0
        genes: list[str] = []
        n_genes = 0
        out: list[Signature] = []
        with torch.no_grad():
            for req in requests:
                try:
                    # chemCPA maps a *cell state* through the drug; the basal
                    # input is the cell line's control expression, not zeros -
                    # checked first so the most specific diagnosis wins
                    control = self._control_basal(req.cell_line)
                    if control is None:
                        raise RuntimeError(
                            f"no control (DMSO) expression for cell line {req.cell_line!r}; "
                            "put the paper's adata_biolord_split_30.h5ad into "
                            "$VCDD_CHEMCPA_DIR (see the adapter docstring) "
                            f"or request one of: {sorted(self._control_vectors())}"
                        )
                    genes = self.genes()  # cached afterwards
                    n_genes = len(genes)
                    if len(control) != n_genes:
                        raise RuntimeError(
                            f"control vector has {len(control)} genes but the model "
                            f"expects {n_genes}; the h5ad var order must match the checkpoint"
                        )
                    basal = torch.from_numpy(control.astype(np.float32)).unsqueeze(0)
                    latent_basal = model.encoder(basal)
                    cov_index = (
                        self._covariate_index(req.cell_line)
                        if has_cov_emb and self._cov_categories is not None
                        else None
                    )
                    cov_meta = "none"
                    if cov_index is not None:
                        n_classes = model.covariates_embeddings[0].weight.shape[0]
                        if cov_index < n_classes:
                            onehot = torch.zeros(1, n_classes)
                            onehot[0, cov_index] = 1.0
                            latent_basal = latent_basal + model.covariates_embeddings[0](
                                onehot.argmax(1)
                            )
                            cov_meta = f"cell_type:{req.cell_line}"
                    dose_scale = self._dose_to_model(req.dose_um)
                    emb, drug_idx = self._drug_embedding(req.smiles)
                    emb_t = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0)
                    if doser_type != "amortized":
                        raise ValueError(
                            f"doser_type {doser_type!r} is not supported for novel "
                            "molecules; the released RDKit checkpoint uses 'amortized'"
                        )
                    dose = torch.tensor([[float(dose_scale)]], dtype=torch.float32)
                    scaled = model.dosers(torch.cat([emb_t, dose], dim=1)).sigmoid()
                    delta = model.drug_embedding_encoder(emb_t) * scaled
                    rec = model.decoder(latent_basal + delta)
                    d = rec.shape[1] // 2
                    vals = np.asarray(rec[0, :d].detach().cpu(), dtype=np.float32)[:n_genes]
                    cld = getattr(self, "_cld_scale", float("nan"))
                    meta = {
                        "checkpoint": str(self._checkpoint()),
                        "doser_type": doser_type,
                        "emb_dim": emb_dim,
                        "basal": f"control:{req.cell_line}",
                        "covariate": cov_meta,
                        "drug_idx": drug_idx if drug_idx is not None else "novel",
                        "dose_scale": f"{dose_scale:g}",
                        "cld_scale_at_10uM": cld,
                    }
                    if cld < 0.01:
                        meta["note"] = (
                            "released checkpoints pin the amortized dose scaling near "
                            "zero, so predictions are ~drug-invariant; "
                            "fine-tune chemCPA on the target assay to unlock dosing"
                        )
                    out.append(
                        Signature(
                            mol_id=req.mol_id,
                            smiles=req.smiles,
                            genes=genes,
                            values=vals,
                            backend=self.name,
                            cell_line=req.cell_line,
                            dose_um=req.dose_um,
                            time_h=req.time_h,
                            value_scale="logfc",
                            meta=meta,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    out.append(Signature.failed(req, self.name, f"{type(exc).__name__}: {exc}"))
        return out


@register_backend
class TranSiGenModel(CellModel):
    """TranSiGen adapter.

    Set ``VCDD_TRANSIGEN_DIR`` to a directory holding the released weights::

        git clone https://github.com/myzhengSIMM/TranSiGen
        # place the released .pkl/.pt checkpoint in $VCDD_TRANSIGEN_DIR

    TranSiGen maps a compound fingerprint (ECFP4 by default, optionally a
    pretrained molecular representation) to the L1000 differential signature,
    which matches this pipeline's landmark gene space directly.
    """

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="transigen",
            tier=2,
            description=(
                "TranSiGen self-supervised structure-to-L1000 signature model."
            ),
            accepts_novel_smiles=True,
            gene_space="checkpoint",
            value_scale="zscore",
            python_requires=("torch",),
            env_requires=("VCDD_TRANSIGEN_DIR",),
            install_hint=(
                "pip install torch; git clone https://github.com/myzhengSIMM/TranSiGen "
                "and set VCDD_TRANSIGEN_DIR to the checkpoint directory"
            ),
            citation="Xu et al., Nature Communications 2024 (TranSiGen)",
            cell_line_aware=True,
            dose_aware=False,
        )

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._model = None
        self._gene_names: list[str] = []

    def _checkpoint(self) -> Path | None:
        root = os.environ.get("VCDD_TRANSIGEN_DIR")
        if not root:
            return None
        p = Path(root).expanduser()
        if p.is_file():
            return p
        for pattern in ("*.pt", "*.pth", "*.pkl"):
            hits = sorted(p.glob(pattern))
            if hits:
                return hits[0]
        return None

    def _extra_availability(self) -> tuple[bool, str]:
        ckpt = self._checkpoint()
        if ckpt is None:
            return False, (
                "no checkpoint under VCDD_TRANSIGEN_DIR; "
                + self.capabilities().install_hint
            )
        return True, f"checkpoint {ckpt.name}"

    def _load(self):
        if self._model is not None:
            return self._model
        import torch  # noqa: PLC0415

        obj = torch.load(self._checkpoint(), map_location="cpu", weights_only=False)
        model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        if hasattr(model, "eval"):
            model.eval()
        if isinstance(obj, dict):
            self._gene_names = list(obj.get("gene_names") or obj.get("landmark_genes") or [])
        self._model = model
        return model

    def genes(self) -> list[str]:
        if not self._gene_names:
            self._load()
        if self._gene_names:
            return self._gene_names
        # TranSiGen is trained on the 978 landmark genes; fall back to that
        # ordering when the checkpoint does not carry names.
        from .. import assets  # noqa: PLC0415

        return assets.gene_universe("landmark")

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        import torch  # noqa: PLC0415
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

        model = self._load()
        genes = self.genes()
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        out: list[Signature] = []
        with torch.no_grad():
            for req in requests:
                try:
                    mol = Chem.MolFromSmiles(req.smiles)
                    if mol is None:
                        raise ValueError(f"unparsable SMILES: {req.smiles}")
                    fp = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
                    x = torch.from_numpy(fp).unsqueeze(0)
                    pred = model(x)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[-1]
                    vals = np.asarray(pred.detach().cpu()).ravel()[: len(genes)]
                    out.append(
                        Signature(
                            mol_id=req.mol_id,
                            smiles=req.smiles,
                            genes=genes[: len(vals)],
                            values=vals,
                            backend=self.name,
                            cell_line=req.cell_line,
                            dose_um=req.dose_um,
                            time_h=req.time_h,
                            value_scale="zscore",
                            meta={"checkpoint": str(self._checkpoint())},
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    out.append(Signature.failed(req, self.name, f"{type(exc).__name__}: {exc}"))
        return out

# --------------------------------------------------------------------------- #
# XPert (Yue et al., Nature Machine Intelligence 2026, doi:10.1038/s42256-025-01165-w)
# knowledge-informed dual-branch transformer.  Predicts compound-induced
# log2 fold-change on the 978 L1000 landmark genes given a cell line's
# control expression: trt/ctl branches cross-attend a UniMol atom sequence
# (with the HG molecule embedding injected as the anchor token), dose(10) and
# time(3) embeddings are prepended, and the deg head reads the difference
# trt-ctl, i.e. the output is a log-fold-change vector directly.
# --------------------------------------------------------------------------- #

_XPERT_ATOM2ID = {
    "hg_molecule": 0, "molecule": 1, "C": 2, "N": 3, "O": 4, "S": 5, "F": 6,
    "Cl": 7, "Br": 8, "P": 9, "I": 10, "B": 11, "Si": 12, "Na": 13, "Ca": 14,
    "Se": 15, "Li": 16, "Mg": 17, "Zn": 18, "Al": 19, "Fe": 20, "Cu": 21,
    "As": 22, "Ag": 23, "Au": 24, "H": 25,
}


def _xpert_config(root: str) -> dict:
    """Config for the released l1000_mdmt warm checkpoint (config_l1000.yaml
    with data paths resolved against ``root``)."""
    r = str(Path(root))
    return {
        "model": {
            "ATTN": {
                "n_heads": 8,
                "attention_probs_dropout_prob": 0.1,
                "hidden_dropout_prob": 0.1,
                "cell_input_hidden_dropout_prob": 0.1,
                "drug_input_hidden_dropout_prob": 0.1,
                "topk_cell": 128,
                "topk_drug": 32,
                "sparse_flag": False,
                "hidden_size": 256,
                "trt_structure": "CA+SA+SA+CA",
                "ctl_structure": "SA+SA+SA+SA",
                "ppi_gene_vector_path": f"{r}/processed_data/PPI_gene_vector_128d.npy",
            },
            "HG": {
                "node_input_dim": 512,
                "conv_layers": 2,
                "hg_path": f"{r}/HG_data/",
                "drug_hg_pretrained_embed_path": f"{r}/HG_data/saved_embedding/HG_drug_embeddings.npy",
            },
        },
        "dataset": {
            "n_bins": 128,
            "gene_num": 978,
            "atom_num": 24,
            "max_atom_size": 122,
            "max_value": 15.002,
            "min_value": 0.0,
            "l1000": {"num_pert_dose": 10, "num_pert_time": 3,
                      "num_cell_id": 240, "num_tissue_id": 21},
            "l1000_mdmt_data_root": f"{r}/processed_data/l1000_mdmt_68830_subset.h5ad",
        },
    }


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


@register_backend
class XpertModel(CellModel):
    """XPert dual-branch transformer for drug-induced perturbation response.

    Layout of ``$VCDD_XPERT_DIR``::

        saved_model/l1000_mdmt_warm_split.pth   released checkpoint
        models/model_XPert.py, model_utils.py   from the official repo (XPert.zip)
        processed_data/PPI_gene_vector_128d.npy
        processed_data/all_drugs_unimol_arr.npy  (4.5 GB, Zenodo 17182939)
        processed_data/all_drugs_idx2smi_8981.npy
        processed_data/l1000_gene_info_978.csv
        processed_data/l1000_mdmt_68830_subset.h5ad   (ctl baselines)
        HG_data/saved_embedding/HG_drug_embeddings.npy

    Inference runs through the pure-PyTorch attention path
    (``output_attention=True``), so no CUDA flash-attn build is required; a
    torch SDPA-backed stand-in is injected for the module's unconditional
    ``import flash_attn``.  Output is the deg head, a per-landmark-gene log2
    fold-change (``value_scale="logfc"``) aligned to the checkpoint's gene
    order (``l1000_gene_info_978.csv``).  Novel molecules get a freshly
    computed UniMol representation with the zero-vector HG placeholder row.
    """

    _ENV = "VCDD_XPERT_DIR"

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="xpert",
            tier=2,
            description=(
                "XPert knowledge-informed dual-branch transformer; predicts "
                "compound-induced log2 fold-change across the 978 L1000 "
                "landmark genes from a cell line's control expression."
            ),
            accepts_novel_smiles=True,
            gene_space="checkpoint",
            value_scale="logfc",
            python_requires=("torch", "torch_geometric", "unimol_tools",
                             "anndata", "h5py"),
            env_requires=("VCDD_XPERT_DIR",),
            install_hint=(
                "pip install torch torch-geometric 'unimol-tools==0.1.4.post1' "
                "anndata h5py; git clone https://github.com/GSanShui/XPert and "
                "put its models/ into $VCDD_XPERT_DIR; download from Zenodo "
                "record 17182939: saved_model.zip -> saved_model/, "
                "processed_data.zip -> processed_data/, all_drugs_unimol_arr.npy "
                "-> processed_data/ (see the class docstring for the layout)"
            ),
            citation="Yue et al., Nature Machine Intelligence 8:96-112 (2026); "
                     "doi:10.1038/s42256-025-01165-w",
            cell_line_aware=True,
            dose_aware=True,
        )

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._model = None
        self._symbols: list[str] = []
        self._basal_cache: dict[str, np.ndarray] | None = None
        self._smiles_cache: dict[str, int] | None = None
        self._unimol_cache = None  # mmap of (8981, 122, 514)
        self._hg_cache: np.ndarray | None = None

    # ------------------------------------------------------------ discovery
    def _checkpoint(self) -> Path | None:
        root = os.environ.get(self._ENV)
        if not root:
            return None
        root_path = Path(root).expanduser()
        if root_path.is_file():
            return root_path
        for pattern in ("*l1000_mdmt*warm*.pth", "*mdmt*warm*.pth",
                        "*warm*.pth", "*.pth"):
            hits = sorted(sorted(root_path.rglob(pattern)))
            if hits:
                return hits[0]
        return None

    def _extra_availability(self) -> tuple[bool, str]:
        ckpt = self._checkpoint()
        if ckpt is None:
            return False, "no checkpoint in VCDD_XPERT_DIR; " + self.capabilities().install_hint
        notes = []
        root = Path(os.environ[self._ENV]).expanduser()
        for req, what in (
            (root / "models" / "model_XPert.py", "XPert model code"),
            (root / "processed_data" / "l1000_gene_info_978.csv", "gene table"),
            (root / "processed_data" / "PPI_gene_vector_128d.npy", "PPI gene vectors"),
            (root / "processed_data" / "all_drugs_unimol_arr.npy", "UniMol drug table"),
            (root / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy", "HG embeddings"),
            (root / "processed_data" / "l1000_mdmt_68830_subset.h5ad", "control-expression h5ad"),
        ):
            if not req.exists():
                notes.append(f"missing {req.relative_to(root)} ({what})")
        try:
            import torch_geometric  # noqa: F401,PLC0415
        except ImportError:
            notes.append("pip install torch-geometric")
        return True, f"checkpoint {ckpt.name}" + ("; " + "; ".join(notes) if notes else "")

    # -------------------------------------------------------------- loading
    @staticmethod
    def _install_flash_stub() -> None:
        """``models/model_utils.py`` imports ``flash_attn.flash_attn_interface``
        unconditionally.  We run inference through the PyTorch attention path
        (``output_attention=True``) so a CUDA flash-attn build is never needed;
        this stand-in only has to make the import succeed."""
        import sys  # noqa: PLC0415
        import types  # noqa: PLC0415
        if "flash_attn" in sys.modules:
            return
        import torch.nn.functional as _F  # noqa: PLC0415

        def _fa(query, key, value, dropout_p=0.0, softmax_scale=None,
                causal=False, *a, **k):
            return _F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2),
                value.transpose(1, 2),
            ).transpose(1, 2)

        iface = types.ModuleType("flash_attn.flash_attn_interface")
        iface.flash_attn_func = _fa
        fake = types.ModuleType("flash_attn")
        fake.flash_attn_interface = iface
        sys.modules["flash_attn"] = fake
        sys.modules["flash_attn.flash_attn_interface"] = iface

    def _load(self):
        if self._model is not None:
            return self._model
        import importlib  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import types  # noqa: PLC0415
        import torch  # noqa: PLC0415

        root = Path(os.environ[self._ENV]).expanduser()
        self._install_flash_stub()
        sys.path.insert(0, str(root))
        xpert = importlib.import_module("models.model_XPert")
        args = types.SimpleNamespace(
            model="XPert",
            dataset="l1000_mdmt",
            drug_feat="unimol",
            pretrained_mode="global",
            include_cell_idx=True,
            mode="infer",
            wo_ppi=False,
            use_gene_pos_emed=False,
            wo_HG=False,
            wo_atom=False,
            wo_atom_HG=False,
            wo_unimol=False,
            output_attention=True,
        )
        model = xpert.XPertNet(args, _xpert_config(str(root)), "cpu", _NullLogger())
        ck = torch.load(self._checkpoint(), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self._model = model

        # gene symbols in checkpoint order (column 0 = XPert input order)
        import csv  # noqa: PLC0415
        with open(root / "processed_data" / "l1000_gene_info_978.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        syms = []
        for r in rows:
            sym = (r.get("corrected_gene_name") or r.get("gene_symbol") or "").strip()
            syms.append(sym or r.get("gene_symbol", ""))
        self._symbols = syms or [r.get("gene_symbol", "") for r in rows]
        return model

    # ---------------------------------------------------------------- data
    def _basal_tables(self) -> dict[str, np.ndarray]:
        if self._basal_cache is not None:
            return self._basal_cache
        import anndata  # noqa: PLC0415
        root = Path(os.environ[self._ENV]).expanduser()
        adata = anndata.read_h5ad(root / "processed_data" / "l1000_mdmt_68830_subset.h5ad")
        xctl = np.asarray(adata.obsm["X_ctl"])
        names = np.asarray(adata.obs["cell_iname"]).astype(str)
        tabs: dict[str, np.ndarray] = {}
        for ci in np.unique(names):
            tabs[ci] = xctl[names == ci].mean(axis=0).astype(np.float64)
        self._basal_cache = tabs
        return tabs

    def _smiles_idx(self) -> dict[str, int]:
        if self._smiles_cache is not None:
            return self._smiles_cache
        root = Path(os.environ[self._ENV]).expanduser()
        tbl = np.load(root / "processed_data" / "all_drugs_idx2smi_8981.npy",
                      allow_pickle=True).item()
        m: dict[str, int] = {}
        for idx, smi in tbl.items():
            key = _canonicalize_smiles(smi) if smi else None
            m[smi] = int(idx)
            if key and key != smi:
                m[key] = int(idx)
        self._smiles_cache = m
        return m

    def _unimol_arr(self):
        if self._unimol_cache is None:
            root = Path(os.environ[self._ENV]).expanduser()
            self._unimol_cache = np.load(
                root / "processed_data" / "all_drugs_unimol_arr.npy", mmap_mode="r")
        return self._unimol_cache

    def _hg_arr(self) -> np.ndarray:
        if self._hg_cache is None:
            root = Path(os.environ[self._ENV]).expanduser()
            self._hg_cache = np.load(root / "HG_data" / "saved_embedding"
                                   / "HG_drug_embeddings.npy", allow_pickle=True)
        return self._hg_cache

    def _ctl_basal(self, cell_line: str) -> np.ndarray | None:
        tabs = self._basal_tables()
        keys = [cell_line, cell_line.upper(), cell_line.lower()]
        for k in keys:
            if k in tabs:
                return tabs[k]
        return None

    @staticmethod
    def _dose_idx(dose_um: float) -> int:
        """assign_pert_dose_corrected (µM -> 10 bins, [0, 9])."""
        if dose_um < 0.21:
            return 0
        if dose_um < 0.41:
            return 1
        if dose_um < 0.71:
            return 2
        if dose_um < 1.01:
            return 3
        if dose_um < 1.51:
            return 4
        if dose_um < 3.1:
            return 5
        if dose_um < 4.1:
            return 6
        if dose_um < 7.1:
            return 7
        if dose_um < 12.1:
            return 8
        return 9

    @staticmethod
    def _time_idx(time_h: float) -> int:
        return {3: 0, 6: 1, 24: 2}.get(int(round(float(time_h))), 2)

    def _novel_feature(self, smiles: str) -> np.ndarray:
        """Fresh UniMol representation in the (122, 514) table format:
        col0 = mask, col1 = atom-symbol id, col2: += 512-d atom repr."""
        from unimol_tools import UniMolRepr  # noqa: PLC0415

        clf = UniMolRepr(data_type="molecule", remove_hs=False)
        rep = clf.get_repr([smiles], return_atomic_reprs=True)
        # unimol_tools 0.1.4.post1 returns a dict whose values are lists
        # with one entry per input molecule
        cls_repr = np.asarray(rep["cls_repr"][0], dtype=np.float32)
        atoms = [np.asarray(a, dtype=np.float32) for a in rep["atomic_reprs"][0]]
        syms = [s for s in rep["atomic_symbol"][0]]
        feat = np.vstack([cls_repr, cls_repr] + atoms)          # (2+n, 512)
        ids = np.asarray([0, 1] + [_XPERT_ATOM2ID.get(s, 1) for s in syms], dtype=np.int64)
        n = len(ids)
        # pad (and if ever larger, truncate) to max_atom_size=122
        n = min(n, 122)
        feat, ids = feat[:n], ids[:n]
        mask = np.ones(n, dtype=np.float32)
        feat_p = np.zeros((122, 512), dtype=np.float32)
        feat_p[:n] = feat
        ids_p = np.zeros(122, dtype=np.int64)
        ids_p[:n] = ids
        mask_p = np.zeros(122, dtype=np.float32)
        mask_p[:n] = mask
        return np.hstack([mask_p[:, None], ids_p[:, None], feat_p])

    def _drug_inputs(self, smiles: str) -> tuple[np.ndarray, np.ndarray, int | None]:
        """(feat(122,514), hg(256,), pert_idx|None)"""
        m = self._smiles_idx()
        pidx = m.get(smiles) or m.get(_canonicalize_smiles(smiles))
        if pidx is not None:
            feat = np.asarray(self._unimol_arr()[pidx], dtype=np.float32)
            hg = np.asarray(self._hg_arr()[pidx], dtype=np.float32)
            return feat, hg, int(pidx)
        # novel molecule: fresh UniMol features, zero-vector HG placeholder
        feat = self._novel_feature(smiles)
        hg = np.asarray(self._hg_arr()[0], dtype=np.float32)  # row 0 is all zeros
        return feat, hg, None

    # -------------------------------------------------------------- scoring
    def genes(self) -> list[str]:
        if not self._symbols:
            self._load()
        return list(self._symbols)

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        import torch  # noqa: PLC0415
        model = self._load()
        bins = np.quantile([0.0, 15.002], np.linspace(0.0, 1.0, 127))
        out: list[Signature] = []
        with torch.no_grad():
            for req in requests:
                try:
                    ctl = self._ctl_basal(req.cell_line)
                    if ctl is None:
                        raise RuntimeError(
                            f"no control expression for cell line {req.cell_line!r}; "
                            f"available (l1000_mdmt subset): {sorted(self._basal_tables())}"
                        )
                    if ctl.shape[0] != len(self._symbols):
                        raise RuntimeError(
                            f"control vector has {ctl.shape[0]} genes; checkpoint expects "
                            f"{len(self._symbols)}"
                        )
                    ctl_bin = np.digitize(ctl, bins).astype(np.int64)
                    feat, hg, pidx = self._drug_inputs(req.smiles)
                    dose_i = self._dose_idx(req.dose_um)
                    time_i = self._time_idx(req.time_h)
                    ctl_f = torch.from_numpy(ctl.astype(np.float32)).unsqueeze(0)
                    ctl_b = torch.from_numpy(ctl_bin).unsqueeze(0)
                    df = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
                    # HG is resolved inside the model via drug_idx
                    # (unimol_Embeddings replaces token 0 with the HG row),
                    # so `hg` here is only needed for its novel-molecule
                    # zero-placeholder semantics.
                    result = model(
                        (
                            ctl_f, ctl_f, ctl_b, ctl_b, df,
                            torch.tensor([dose_i], dtype=torch.long),
                            torch.tensor([time_i], dtype=torch.long),
                            torch.tensor([pidx if pidx is not None else 0], dtype=torch.long),
                            torch.tensor([0], dtype=torch.long),    # cell_idx (unused)
                            torch.tensor([0], dtype=torch.long),    # tissue_idx (unused)
                        )
                    )
                    deg = np.asarray(result[2][0].detach().cpu(), dtype=np.float32)
                    if deg.shape[0] != len(self._symbols):
                        raise RuntimeError(
                            f"model returned {deg.shape[0]} genes; expected {len(self._symbols)}"
                        )
                    out.append(
                        Signature(
                            mol_id=req.mol_id,
                            smiles=req.smiles,
                            genes=list(self._symbols),
                            values=deg,
                            backend=self.name,
                            cell_line=req.cell_line,
                            dose_um=req.dose_um,
                            time_h=req.time_h,
                            value_scale="logfc",
                            meta={
                                "checkpoint": str(self._checkpoint()),
                                "drug": "table" if pidx is not None else "novel",
                                "pert_idx": pidx if pidx is not None else "novel",
                                "dose_idx": dose_i,
                                "time_idx": time_i,
                                "basal": f"ctl:{req.cell_line}",
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    out.append(Signature.failed(req, self.name, f"{type(exc).__name__}: {exc}"))
        return out
