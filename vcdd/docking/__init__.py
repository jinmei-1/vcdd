"""Docking: engine plugins and ligand preparation.

0.2.0 removed the rescoring plugin subsystem: stage 2 no longer rescores a
docked pose, it calls the Boltz API (see :mod:`vcdd.refine.boltz`).
"""

from vcdd.docking.base import (  # noqa: F401
    DockParams,
    DockResult,
    DockingEngine,
    EngineCapabilities,
    available_engines,
    engine_names,
    get_engine,
    register_engine,
    select_engine,
)
from vcdd.docking import engines  # noqa: F401  (registers built-in engines)

__all__ = [
    "DockParams",
    "DockResult",
    "DockingEngine",
    "EngineCapabilities",
    "available_engines",
    "engine_names",
    "get_engine",
    "register_engine",
    "select_engine",
]
