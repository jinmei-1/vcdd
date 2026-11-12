"""Docking engine implementations.

Importing this package registers every engine in the global registry. Engines
whose binary is absent still register - they simply report ``available() ==
False``, which is what ``vcdd doctor`` shows the user.

0.2.0 registers exactly two: ``vina_gpu`` (the Vina-GPU 2.1 family) and
``mock`` (a test double, not a docking program). The CPU ``vina`` / ``smina`` /
``gnina`` engines were removed.
"""

from vcdd.docking.engines import mock_engine, vina_gpu  # noqa: F401

__all__ = ["mock_engine", "vina_gpu"]
