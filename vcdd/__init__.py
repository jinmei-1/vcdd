"""vcdd - Virtual-Cell-guided Drug Design.

A four-stage pipeline:

1. ``stage1``  fast high-throughput docking over a whole library (scales to 1e6).
2. ``stage2``  accurate docking + multi-scoring-function consensus over the top N.
3. ``stage3``  virtual-cell perturbation prediction + multi-objective gene/pathway scoring.
4. ``stage4``  virtual-cell-guided analog generation and modification recommendations.

The package is target-agnostic and engine-agnostic: docking engines, rescoring
functions, virtual-cell backends and analog generators are all plugins behind
abstract base classes, discovered through registries and gated by an
``available()`` / ``capabilities()`` contract so that a missing optional
dependency degrades a feature instead of breaking the pipeline.
"""

from vcdd.version import __version__

__all__ = ["__version__"]
