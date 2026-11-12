"""Analog generators.

Importing this package registers every generator. Generators whose optional
dependency or database is missing still register -- they report
``is_available() == False`` with an install hint, which is what ``vcdd doctor``
shows and what ``stage4/generator_report.json`` records.
"""

from .base import (
    AnalogGenerator,
    Proposal,
    get_generator_class,
    list_generators,
    register_generator,
    resolve_generators,
)
from .constrained_ga import ConstrainedGA
from .enumerative import CRemGenerator, MMPDBGenerator, RGroupGenerator

__all__ = [
    "AnalogGenerator",
    "Proposal",
    "register_generator",
    "get_generator_class",
    "list_generators",
    "resolve_generators",
    "RGroupGenerator",
    "CRemGenerator",
    "MMPDBGenerator",
    "ConstrainedGA",
]
