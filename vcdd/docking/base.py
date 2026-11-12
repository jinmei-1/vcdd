"""Docking engine plugin contract.

Every engine implements the same small interface, and *nothing outside this
module hardcodes an engine name*. Drivers ask the registry for an engine, ask
the engine what it can do via :meth:`DockingEngine.capabilities`, and branch on
that.

0.2.0 changes two things in this contract.

**Batch is a first-class operation.** GPU docking programs are batch-native:
they take a *directory* of ligands, upload the receptor grid and the OpenCL
kernels once, and then dock everything resident on the device. Driving them one
ligand per process - which is what the 0.1.0 contract forced - pays the whole
setup cost per molecule and throws away the speedup that is the entire reason to
use a GPU. :meth:`DockingEngine.dock_batch` is therefore the primary entry
point, with a default implementation that loops :meth:`dock` so CPU-shaped
engines need no changes.

**Search parameters are engine-shaped, not Vina-shaped.** ``exhaustiveness`` and
``cpu`` were removed from :class:`DockParams`: Vina-GPU 2.1 accepts neither.
Its search effort knobs are ``--thread`` (how many parallel Monte-Carlo threads
run on the device) and ``--search_depth`` (how deep each one goes). Carrying the
old names would have produced a config that silently means nothing.
"""

from __future__ import annotations

import abc
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from vcdd.receptor.box import Box


@dataclass
class DockParams:
    """Everything that changes a docking result. Hashed into the resume key.

    The defaults are Vina-GPU 2.1's own defaults, not invented ones.
    """

    #: Which Vina-GPU 2.1 binary to drive. See ``vcdd.docking.engines.vina_gpu``.
    variant: str = "autodock_vina_gpu"
    #: Parallel Monte-Carlo threads on the device. Upstream default 8000
    #: (5000 for QuickVina2-GPU-2.1). Upstream advises staying below 10000.
    thread: int = 8000
    #: Monte-Carlo depth per thread. ``None`` lets the binary choose adaptively,
    #: which is what upstream recommends for a heterogeneous library.
    search_depth: int | None = None
    #: Reduced-Iteration Local-search BFGS. On by default upstream; turning it
    #: off is slower and only useful when comparing against classic Vina.
    rilc_bfgs: bool = True
    #: Directory holding the compiled ``Kernel1_Opt.bin`` / ``Kernel2_Opt.bin``.
    #: Without it the binary recompiles the OpenCL kernels on every invocation.
    opencl_binary_path: str | None = None
    num_modes: int = 9
    energy_range: float = 3.0
    #: Seeds ligand preparation (ETKDG) and the mock engine. It does **not**
    #: seed the GPU search: Vina-GPU 2.1 exposes no ``--seed`` flag, so stage 1
    #: is reproducible in protocol but not bit-for-bit in score. See
    #: ``EngineCapabilities.deterministic``.
    seed: int = 42
    #: Per-ligand wall clock ceiling, used by the single-ligand ``dock()`` path.
    timeout_s: int = 600
    #: Per-batch wall clock ceiling, used by ``dock_batch()``. A batch of 1000
    #: ligands is one process, so it needs its own much larger budget.
    batch_timeout_s: int = 3600
    # ``scoring`` (vina | vinardo | ad4) is gone in 0.2.0. It was a smina/gnina
    # flag; Vina-GPU 2.1 has no --scoring and always uses the Vina function. A
    # field no engine reads is worse than absent here, because ``as_dict`` feeds
    # the resume param hash: setting it would have invalidated the ledger and
    # re-docked the library without changing a single score.
    extra_args: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "thread": self.thread,
            "search_depth": self.search_depth,
            "rilc_bfgs": self.rilc_bfgs,
            "num_modes": self.num_modes,
            "energy_range": self.energy_range,
            "seed": self.seed,
            "extra_args": list(self.extra_args),
        }


@dataclass
class DockResult:
    """Outcome of docking one prepared ligand."""

    mol_id: str
    score: float | None = None            # best (most negative) affinity, kcal/mol
    all_scores: list[float] = field(default_factory=list)
    pose_pdbqt: str | None = None         # best pose, PDBQT text
    all_poses_pdbqt: str | None = None    # full multi-MODEL output
    engine: str = ""
    seconds: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None


@dataclass
class EngineCapabilities:
    """What an engine can do. Drivers branch on this, never on the name."""

    name: str
    needs_gpu: bool = False
    supports_flexible_residues: bool = False
    supports_scoring_functions: tuple[str, ...] = ("vina",)
    supports_local_only: bool = False
    speed_class: str = "standard"  # fast | standard | slow
    #: Relative throughput hint used by ``engine_selection: auto`` (higher is faster).
    relative_speed: float = 1.0
    #: True when the engine docks a whole directory in one process. Drivers must
    #: route through :meth:`DockingEngine.dock_batch` and must not fan the
    #: engine out over a process pool.
    batch_native: bool = False
    #: Ligands per ``dock_batch`` call that the engine would like to receive.
    preferred_batch_size: int = 1
    #: False when repeated runs with identical inputs can return different
    #: scores. Stage 1 warns once when this is False so nobody reports a
    #: single-decimal-place ranking as reproducible.
    deterministic: bool = True
    #: Largest box edge, in Angstrom, the compiled binary accepts. ``None``
    #: means unconstrained.
    max_box_edge: float | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "needs_gpu": self.needs_gpu,
            "supports_flexible_residues": self.supports_flexible_residues,
            "supports_scoring_functions": list(self.supports_scoring_functions),
            "supports_local_only": self.supports_local_only,
            "speed_class": self.speed_class,
            "relative_speed": self.relative_speed,
            "batch_native": self.batch_native,
            "preferred_batch_size": self.preferred_batch_size,
            "deterministic": self.deterministic,
            "max_box_edge": self.max_box_edge,
            "notes": self.notes,
        }


class DockingEngine(abc.ABC):
    """Abstract docking engine."""

    name: str = "abstract"

    @abc.abstractmethod
    def available(self) -> bool:
        """True when this engine can actually run on this machine right now."""

    @abc.abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Static declaration of what the engine supports."""

    @abc.abstractmethod
    def dock(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        mol_id: str = "",
    ) -> DockResult:
        """Dock one prepared ligand and return the best pose + score."""

    def dock_batch(
        self,
        items: Sequence[tuple[str, str]],
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
    ) -> list[DockResult]:
        """Dock many prepared ligands.

        ``items`` is a sequence of ``(mol_id, ligand_pdbqt_text)`` pairs.

        Contract, which the driver relies on and a contract test asserts:

        * the returned list has the same length as ``items``;
        * results are in **input order**;
        * a ligand that failed comes back as a :class:`DockResult` carrying an
          ``error``. Nothing is ever silently dropped, because a dropped row
          turns a docking failure into a missing molecule.

        The default implementation loops :meth:`dock`, which is correct for
        every CPU-shaped engine. Batch-native engines override it and must set
        ``batch_native=True`` in their capabilities.
        """
        return [
            self.dock(pdbqt, receptor_pdbqt_path, box, params, mol_id=mol_id)
            for mol_id, pdbqt in items
        ]

    # Optional: engines that can score/minimise an existing pose in place.
    def score_only(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        mol_id: str = "",
    ) -> DockResult:
        raise NotImplementedError(f"{self.name} does not support score_only")

    def version(self) -> str | None:
        return None


# ------------------------------------------------------------------- registry
_ENGINES: dict[str, Callable[[], DockingEngine]] = {}


def register_engine(name: str, factory: Callable[[], DockingEngine]) -> None:
    _ENGINES[name] = factory


def engine_names() -> list[str]:
    return sorted(_ENGINES)


def get_engine(name: str) -> DockingEngine:
    if name not in _ENGINES:
        raise KeyError(f"unknown docking engine {name!r}; registered: {engine_names()}")
    return _ENGINES[name]()


def available_engines() -> dict[str, EngineCapabilities]:
    """Engines that can run here, mapped to their capabilities."""
    out: dict[str, EngineCapabilities] = {}
    for name in engine_names():
        try:
            eng = get_engine(name)
            if eng.available():
                out[name] = eng.capabilities()
        except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
            continue
    return out


def select_engine(requested: str, selection: str = "fixed", prefer_fast: bool = False) -> DockingEngine:
    """Resolve the engine to use.

    ``fixed``: use *requested*, raising a clear error when it is unavailable.
    ``auto`` : use the fastest available engine, falling back to *requested*.

    0.2.0 registers only ``vina_gpu`` and the ``mock`` test double, so ``auto``
    has little left to choose between. It is kept because the registry is the
    extension point: adding an engine must never require editing a driver.
    """
    avail = available_engines()
    if selection == "auto" and avail:
        candidates = {k: v for k, v in avail.items() if k != "mock"} or avail
        if prefer_fast:
            best = max(candidates.items(), key=lambda kv: kv[1].relative_speed)[0]
        else:
            best = requested if requested in candidates else sorted(candidates)[0]
        return get_engine(best)

    engine = get_engine(requested)
    if not engine.available():
        raise RuntimeError(
            f"docking engine {requested!r} is not available on this machine. "
            f"Available: {sorted(avail)}. Run `vcdd doctor` for details."
        )
    return engine


def binary_available(name: str) -> bool:
    return shutil.which(name) is not None


def gpu_available() -> bool:
    """True when an NVIDIA GPU is visible.

    Deliberately shells out to ``nvidia-smi`` rather than importing a CUDA
    binding: the OpenCL runtime that Vina-GPU actually uses ships with the
    driver, and a Python binding being installed says nothing about whether a
    device is present. ``VCDD_ASSUME_GPU=1`` overrides the probe for CI and for
    schedulers that hide ``nvidia-smi`` from the submit node.
    """
    import os

    if os.environ.get("VCDD_ASSUME_GPU") == "1":
        return True
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20, check=False
        )
    except Exception:  # noqa: BLE001
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout
