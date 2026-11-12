"""Cell-model backend registry.

Importing this package registers every backend, so ``vcdd doctor`` can report
the full catalogue -- including the ones that are unavailable here and why.
"""

from .tier1 import LincsReadAcrossModel, MockCellModel, TargetKnockdownModel
from .tier2 import ChemCPAModel, TranSiGenModel
from .tier3 import PerturbNetModel, PRnetModel, StateTahoeModel

__all__ = [
    "MockCellModel",
    "LincsReadAcrossModel",
    "TargetKnockdownModel",
    "ChemCPAModel",
    "TranSiGenModel",
    "PRnetModel",
    "PerturbNetModel",
    "StateTahoeModel",
]
