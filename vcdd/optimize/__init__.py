"""Stage 4: analog design, guardrails, multi-objective ranking, SAR mining.

Public surface::

    from vcdd.optimize import run_stage4, Stage4Result

The generator registry lives in :mod:`vcdd.optimize.generators`; importing this
package imports it, which is what registers the built-in generators.
"""

from __future__ import annotations

from vcdd.optimize.generators import (
    AnalogGenerator,
    Proposal,
    get_generator_class,
    list_generators,
    register_generator,
    resolve_generators,
)
from vcdd.optimize.guardrails import (
    GuardrailReport,
    check_analog,
    filter_analogs,
    morgan_fp,
    pains_alerts,
    sa_score,
    tanimoto,
)
from vcdd.optimize.pareto import (
    DESIGN_OBJECTIVES,
    crowding_distance,
    design_score,
    normalise_objectives,
    pareto_fronts,
)
from vcdd.optimize.sar import SARRule, format_rule, mine_sar, sign_test
from vcdd.optimize.stage4 import (
    RECOMMENDATION_COLUMNS,
    STAGE,
    Stage4Result,
    load_parents,
    run_stage4,
)

__all__ = [
    # driver
    "run_stage4",
    "Stage4Result",
    "load_parents",
    "RECOMMENDATION_COLUMNS",
    "STAGE",
    # generators
    "AnalogGenerator",
    "Proposal",
    "register_generator",
    "get_generator_class",
    "list_generators",
    "resolve_generators",
    # guardrails
    "GuardrailReport",
    "check_analog",
    "filter_analogs",
    "sa_score",
    "pains_alerts",
    "tanimoto",
    "morgan_fp",
    # ranking
    "DESIGN_OBJECTIVES",
    "design_score",
    "normalise_objectives",
    "pareto_fronts",
    "crowding_distance",
    # SAR
    "SARRule",
    "mine_sar",
    "sign_test",
    "format_rule",
]
