"""Stage-3 virtual-cell layer: perturbation prediction and objective scoring."""

from . import backends as _backends  # noqa: F401 - import registers the backends
from .base import (
    CellModel,
    CellModelCapabilities,
    PerturbationRequest,
    Signature,
    list_backends,
    resolve_backends,
)
from .objective import ComponentScores, ObjectiveContext, score_signatures
from .stage3 import Stage3Result, run_stage3

__all__ = [
    "CellModel",
    "CellModelCapabilities",
    "PerturbationRequest",
    "Signature",
    "list_backends",
    "resolve_backends",
    "ObjectiveContext",
    "ComponentScores",
    "score_signatures",
    "run_stage3",
    "Stage3Result",
]
