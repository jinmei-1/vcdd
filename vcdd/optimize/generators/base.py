"""Analog-generator contract.

A generator proposes structural neighbours of a parent molecule.  It does not
score them and it does not enforce guardrails -- stage 4 owns both, so that
every generator is judged on the same terms and a generator cannot quietly
widen the constraints to make itself look productive.

Each proposal carries a human-readable ``transformation`` string.  That string
is the unit the SAR miner aggregates over, and it is what makes the final
recommendation actionable ("replace aryl-Cl with aryl-CF3") rather than an
opaque SMILES diff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["Proposal", "AnalogGenerator", "register_generator", "get_generator_class", "list_generators", "resolve_generators"]


@dataclass(frozen=True)
class Proposal:
    """One generated analog."""

    smiles: str
    parent_id: str
    parent_smiles: str
    generator: str
    transformation: str = ""
    meta: dict = field(default_factory=dict, hash=False, compare=False)


class AnalogGenerator(ABC):
    """Base class for structural analog generators."""

    def __init__(self, cfg, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @classmethod
    def description(cls) -> str:
        return cls.__doc__ or ""

    def is_available(self) -> tuple[bool, str]:
        return True, "ready"

    @abstractmethod
    def generate(self, parent_id: str, parent_smiles: str, n: int) -> list[Proposal]:
        """Propose up to ``n`` analogs. Must not raise on a bad parent."""


_REGISTRY: dict[str, type[AnalogGenerator]] = {}


def register_generator(cls: type[AnalogGenerator]) -> type[AnalogGenerator]:
    _REGISTRY[cls.name()] = cls
    return cls


def get_generator_class(name: str) -> type[AnalogGenerator]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown generator '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_generators() -> dict[str, type[AnalogGenerator]]:
    return dict(_REGISTRY)


def resolve_generators(names, cfg, seed: int = 42):
    """Instantiate generators, skipping unavailable ones with a reason."""
    usable, report = [], []
    for name in names:
        try:
            cls = get_generator_class(name)
        except KeyError as exc:
            report.append({"generator": name, "available": False, "reason": str(exc)})
            continue
        try:
            gen = cls(cfg, seed=seed)
            ok, reason = gen.is_available()
        except Exception as exc:  # noqa: BLE001
            ok, reason, gen = False, f"{type(exc).__name__}: {exc}", None
        report.append({"generator": name, "available": ok, "reason": reason})
        if ok and gen is not None:
            usable.append(gen)
    return usable, report
