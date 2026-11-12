"""Virtual-cell backend contract.

A *cell model* answers one question: given a compound structure and a cellular
context, what transcriptional change does it produce?  Everything downstream --
the gene objective, pathway scoring, target-consistency and stage-4 design --
consumes only :class:`Signature`, so backends of very different provenance
(read-across, generative models, HPC services) stay interchangeable.

Backends are graded by tier:

``1``
    Ships working out of the box from the bundled reference data.
``2``
    Full adapter implemented; needs third-party weights the user must fetch.
``3``
    Interface + graceful skip; implemented against the published API but not
    exercisable here.

Unavailable backends never raise at resolve time: they report
``is_available() -> (False, hint)`` and stage 3 records the skip.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config.schema import CellConfig

__all__ = [
    "PerturbationRequest",
    "Signature",
    "CellModelCapabilities",
    "CellModel",
    "register_backend",
    "get_backend_class",
    "list_backends",
    "resolve_backends",
    "signatures_to_frame",
]


@dataclass(frozen=True)
class PerturbationRequest:
    """One compound in one cellular context."""

    mol_id: str
    smiles: str
    cell_line: str = "MCF7"
    dose_um: float = 10.0
    time_h: float = 24.0

    def key(self) -> str:
        return f"{self.mol_id}|{self.cell_line}|{self.dose_um:g}|{self.time_h:g}"


@dataclass
class Signature:
    """A predicted transcriptional response.

    ``values`` are signed, roughly z-scaled differential expression estimates
    aligned to ``genes``.  Backends that only produce ranked up/down sets encode
    membership as +1 / -1; ``value_scale`` records which of the two it is so the
    objective never over-interprets a set-membership indicator as an effect
    size.
    """

    mol_id: str
    smiles: str
    genes: list[str]
    values: np.ndarray
    backend: str
    cell_line: str = "MCF7"
    dose_um: float = 10.0
    time_h: float = 24.0
    value_scale: str = "zscore"  # "zscore" | "membership" | "logfc"
    meta: dict = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32).ravel()
        if len(self.values) != len(self.genes):
            raise ValueError(
                f"signature length mismatch for {self.mol_id}: "
                f"{len(self.values)} values vs {len(self.genes)} genes"
            )

    @property
    def ok(self) -> bool:
        return self.error is None and self.values.size > 0

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.genes, name=self.mol_id)

    def get(self, gene: str, default: float = np.nan) -> float:
        try:
            return float(self.values[self.genes.index(gene)])
        except ValueError:
            return default

    def top(self, n: int = 20, direction: str = "up") -> list[tuple[str, float]]:
        if self.values.size == 0:
            return []
        order = np.argsort(self.values)
        idx = order[::-1][:n] if direction == "up" else order[:n]
        return [(self.genes[i], float(self.values[i])) for i in idx]

    @classmethod
    def failed(cls, req: PerturbationRequest, backend: str, error: str) -> "Signature":
        return cls(
            mol_id=req.mol_id,
            smiles=req.smiles,
            genes=[],
            values=np.zeros(0, dtype=np.float32),
            backend=backend,
            cell_line=req.cell_line,
            dose_um=req.dose_um,
            time_h=req.time_h,
            error=error,
        )


@dataclass(frozen=True)
class CellModelCapabilities:
    """Static description of a backend, shown by ``vcdd doctor``."""

    name: str
    tier: int
    description: str
    #: Can it score a structure it has never seen? False means read-across or
    #: lookup only -- such a backend cannot drive stage 4.
    accepts_novel_smiles: bool
    gene_space: str = "landmark"
    value_scale: str = "zscore"
    python_requires: tuple[str, ...] = ()
    resources_requires: tuple[str, ...] = ()
    env_requires: tuple[str, ...] = ()
    install_hint: str = ""
    citation: str = ""
    cell_line_aware: bool = False
    dose_aware: bool = False
    #: True when the backend returns the same signature regardless of compound
    #: (a mechanistic reference). Such a backend is materialised for provenance
    #: but excluded from the objective, where it would contribute a constant
    #: and score target_consistency against itself.
    reference_only: bool = False


class CellModel(ABC):
    """Base class for compound-perturbation predictors."""

    def __init__(self, cfg: CellConfig) -> None:
        self.cfg = cfg
        self._genes: list[str] | None = None

    # ------------------------------------------------------------ metadata
    @classmethod
    @abstractmethod
    def capabilities(cls) -> CellModelCapabilities: ...

    @property
    def name(self) -> str:
        return self.capabilities().name

    # --------------------------------------------------------- availability
    def is_available(self) -> tuple[bool, str]:
        """Check imports, bundled resources and environment variables."""
        from . import assets

        caps = self.capabilities()
        missing_py = [
            m for m in caps.python_requires if importlib.util.find_spec(m) is None
        ]
        if missing_py:
            return False, f"missing python packages: {', '.join(missing_py)}"
        missing_res = [r for r in caps.resources_requires if not assets.available(r)]
        if missing_res:
            return False, (
                f"missing cell resources: {', '.join(missing_res)}; {assets.BUILD_HINT}"
            )
        import os

        missing_env = [v for v in caps.env_requires if not os.environ.get(v)]
        if missing_env:
            return False, f"set environment variable(s): {', '.join(missing_env)}"
        return self._extra_availability()

    def _extra_availability(self) -> tuple[bool, str]:
        """Backend-specific checks (weights on disk, service reachable, ...)."""
        return True, "ready"

    # -------------------------------------------------------------- scoring
    def genes(self) -> list[str]:
        """The gene ordering this backend emits."""
        from . import assets

        if self._genes is None:
            space = self.capabilities().gene_space
            if space == "config":
                space = self.cfg.gene_space
            self._genes = assets.gene_universe(space)
        return self._genes

    @abstractmethod
    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        """Predict one signature per request, in the same order.

        Implementations must return a :class:`Signature` for every request --
        use :meth:`Signature.failed` for per-compound errors rather than
        raising, so one bad structure cannot abort a batch.
        """

    def close(self) -> None:
        """Release models/sessions. Called once when stage 3 finishes."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[CellModel]] = {}


def register_backend(cls: type[CellModel]) -> type[CellModel]:
    _REGISTRY[cls.capabilities().name] = cls
    return cls


def get_backend_class(name: str) -> type[CellModel]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown cell backend '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_backends() -> dict[str, type[CellModel]]:
    return dict(_REGISTRY)


def resolve_backends(
    names: Iterable[str], cfg: CellConfig
) -> tuple[list[CellModel], list[dict]]:
    """Instantiate the requested backends, skipping unavailable ones.

    Returns ``(usable, report)``; ``report`` has one record per requested
    backend so the run manifest documents what was skipped and why.
    """
    usable: list[CellModel] = []
    report: list[dict] = []
    for name in names:
        try:
            cls = get_backend_class(name)
        except KeyError as exc:
            report.append(
                {"backend": name, "available": False, "reason": str(exc), "tier": None}
            )
            continue
        caps = cls.capabilities()
        record = {
            "backend": name,
            "tier": caps.tier,
            "accepts_novel_smiles": caps.accepts_novel_smiles,
            "description": caps.description,
            "citation": caps.citation,
        }
        try:
            model = cls(cfg)
            ok, reason = model.is_available()
        except Exception as exc:  # noqa: BLE001 - a broken backend must not kill the run
            ok, reason, model = False, f"{type(exc).__name__}: {exc}", None
        record["available"] = ok
        record["reason"] = reason
        if not ok and caps.install_hint:
            record["install_hint"] = caps.install_hint
        if ok and model is not None:
            usable.append(model)
        report.append(record)
    return usable, report


def signature_meta_frame(signatures: Sequence[Signature]) -> pd.DataFrame:
    """Per-compound metadata that the long value frame cannot carry.

    Written next to the values so a resumed run keeps the applicability-domain
    flags, value scale and per-compound errors. Without it a cached rerun would
    silently report every compound as in-domain.
    """
    import json

    return pd.DataFrame(
        [
            {
                "mol_id": s.mol_id,
                "smiles": s.smiles,
                "backend": s.backend,
                "cell_line": s.cell_line,
                "dose_um": s.dose_um,
                "time_h": s.time_h,
                "value_scale": s.value_scale,
                "n_genes": len(s.genes),
                "error": s.error,
                "meta_json": json.dumps(s.meta, default=str),
            }
            for s in signatures
        ]
    )


def signatures_to_frame(signatures: Sequence[Signature]) -> pd.DataFrame:
    """Long-format table: one row per (compound, gene)."""
    frames = []
    for sig in signatures:
        if not sig.ok:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "mol_id": sig.mol_id,
                    "backend": sig.backend,
                    "gene": sig.genes,
                    "value": sig.values,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["mol_id", "backend", "gene", "value"])
    return pd.concat(frames, ignore_index=True)
