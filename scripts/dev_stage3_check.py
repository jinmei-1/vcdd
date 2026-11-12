"""Developer probe: exercise the stage-3 cell layer on the dev run directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from vcdd.cell import list_backends, resolve_backends, run_stage3
from vcdd.cell import assets
from vcdd.config.schema import RunConfig
from vcdd.runtime.ledger import RunLedger
from vcdd.runtime.logging import setup_logging

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/dev_run")


def main() -> int:
    setup_logging(RUN, level="INFO")
    print("=== resource bundle ===")
    for name, meta in assets.bundle_status().items():
        print(f"  {name:26s} present={meta['present']} rows={meta.get('rows', '-')}")

    cfg = RunConfig(name="dev", out_dir=str(RUN))
    cfg.cell.backends = ["mock", "lincs_readacross", "target_knockdown", "chemcpa", "state_tahoe"]
    cfg.cell.target_gene = "EGFR"
    cfg.cell.top_n = 8
    cfg.cell.objective.genes_up = ["CDKN1A", "TP53", "GDF15", "MYC"]
    cfg.cell.objective.genes_down = ["CCND1"]
    cfg.cell.objective.pathways_up = ["p53"]
    cfg.cell.objective.pathways_down = ["EGFR", "MAPK"]

    print("\n=== registered backends ===")
    for name, cls in sorted(list_backends().items()):
        caps = cls.capabilities()
        print(f"  {name:20s} tier={caps.tier} novel_smiles={caps.accepts_novel_smiles}")

    print("\n=== availability ===")
    _, report = resolve_backends(sorted(list_backends()), cfg.cell)
    for rec in report:
        mark = "ok " if rec["available"] else "SKIP"
        print(f"  [{mark}] {rec['backend']:20s} {rec['reason'][:96]}")

    print("\n=== running stage 3 ===")
    with RunLedger(RUN / "run.db") as ledger:
        res = run_stage3(cfg, RUN / "stage2", RUN / "stage3", ledger, force=True)

    print(json.dumps(res.summary, indent=2, default=str)[:2600])
    pd.set_option("display.width", 200)
    cols = [
        c
        for c in [
            "mol_id", "objective_rank", "objective", "gene_score", "pathway_score",
            "target_consistency", "specificity", "max_tanimoto", "out_of_domain",
            "backend_spread", "n_backends",
        ]
        if c in res.scores.columns
    ]
    print("\n=== objective_scores.parquet ===")
    print(res.scores[cols].round(4).to_string(index=False))

    byb = pd.read_parquet(RUN / "stage3" / "objective_by_backend.parquet")
    print("\n=== per-backend objective ===")
    print(
        byb.groupby("backend")[["gene_score", "pathway_score", "target_consistency", "specificity", "objective"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n=== gene routing detail (first row) ===")
    print(json.dumps(json.loads(byb.iloc[0]["gene_detail"]), indent=2)[:1200])

    print("\n=== artifacts ===")
    for p in sorted((RUN / "stage3").rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(RUN)}  {p.stat().st_size:>9,d} B")

    print("\n=== resume check ===")
    with RunLedger(RUN / "run.db") as ledger:
        res2 = run_stage3(cfg, RUN / "stage2", RUN / "stage3", ledger, force=False)
    a = res.scores[["mol_id", "objective"]].round(6).to_dict("records")
    b = res2.scores[["mol_id", "objective"]].round(6).to_dict("records")
    print(f"  identical after resume: {a == b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
