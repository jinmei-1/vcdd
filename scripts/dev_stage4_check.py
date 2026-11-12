"""Dev probe: run stage 4 end-to-end on the dev run directory.

Reduced budgets so the probe finishes in a couple of minutes. Usage::

    python scripts/dev_stage4_check.py [--force] [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vcdd.config.schema import RunConfig
from vcdd.receptor.box import Box
from vcdd.runtime.ledger import RunLedger
from vcdd.runtime.logging import setup_logging
from vcdd.optimize import run_stage4

RUN = Path("/workspace/dev_run")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def build_cfg() -> RunConfig:
    cfg = RunConfig()
    cfg.out_dir = str(RUN)
    cfg.n_workers = 8
    # cell settings identical to the stage-3 dev check
    cfg.cell.backends = ["mock", "lincs_readacross", "target_knockdown"]
    cfg.cell.target_gene = "EGFR"
    cfg.cell.objective.genes_up = ["CDKN1A", "TP53", "GDF15", "MYC"]
    cfg.cell.objective.genes_down = ["CCND1"]
    cfg.cell.objective.pathways_up = ["p53"]
    cfg.cell.objective.pathways_down = ["EGFR", "MAPK"]
    # small stage-4 budgets
    cfg.stage4.top_parents = 6
    cfg.stage4.generators = ["rgroup", "constrained_ga"]
    cfg.stage4.analogs_per_parent = 12
    cfg.stage4.ga_population = 12
    cfg.stage4.ga_generations = 2
    cfg.stage4.redock_mode = "stage1"
    cfg.stage4.sar_min_parents = 2
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", help="second pass, compare outputs")
    args = ap.parse_args()

    setup_logging(run_dir=RUN, level="INFO")
    cfg = build_cfg()
    box = Box.load(RUN / "receptor" / "box.json")
    receptor = RUN / "receptor" / "receptor.pdbqt"
    out_dir = RUN / "stage4"
    ledger = RunLedger(RUN / "run.db")

    res = run_stage4(
        cfg, receptor, box, RUN / "stage3", out_dir, ledger, force=args.force
    )
    ledger.sync()

    print("\n=== summary ===")
    print(json.dumps(res.summary, indent=2, default=str)[:4000])

    print("\n=== analogs (head) ===")
    cols = [
        c
        for c in [
            "rank", "analog_id", "parent_id", "generator", "transformation",
            "dock_score", "delta_dock_score", "cell_objective",
            "delta_cell_objective", "qed", "sa_score", "design_score",
            "delta_design_score", "pareto_rank", "similarity_to_parent",
        ]
        if c in res.analogs.columns
    ]
    print(res.analogs[cols].head(12).to_string(index=False))

    print("\n=== sar rules ===")
    if res.sar_rules.empty:
        print("(none)")
    else:
        print(res.sar_rules.head(15).to_string(index=False))

    print("\n=== artifacts ===")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name:34s} {p.stat().st_size:>10,d} B")

    if args.resume:
        snap = res.analogs[["analog_id", "analog_smiles", "dock_score",
                            "cell_objective", "design_score", "pareto_rank"]].copy()
        snap.to_parquet("/tmp/stage4_snapshot.parquet", index=False)
        print("\nsnapshot written to /tmp/stage4_snapshot.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
