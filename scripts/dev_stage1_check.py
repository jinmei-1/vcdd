"""Developer probe: build a small run dir and exercise stage 1 end to end.

Not part of the package API; used while building to catch integration errors
before the full smoke test exists.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# No GPU on a dev box, so the probe runs the mock engine, which is gated.
os.environ.setdefault("VCDD_ALLOW_MOCK", "1")

from vcdd.config import RunConfig
from vcdd.io.library import prepare_library
from vcdd.receptor.box import resolve_box
from vcdd.receptor.prepare import prepare_receptor
from vcdd.runtime.ledger import RunLedger
from vcdd.runtime.logging import setup_logging
from vcdd.screening.stage1 import run_stage1

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main(run_dir: str = "/workspace/dev_run", n_mols: int = 24, fresh: bool = True) -> None:
    run = Path(run_dir)
    if fresh and run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    setup_logging(run, level="INFO")

    cfg = RunConfig(
        name="dev",
        out_dir=str(run),
        n_workers=8,
        library={"path": str(FIX / "library.sdf"), "shard_size": 12, "max_molecules": n_mols},
        receptor={
            "path": str(FIX / "receptor.pdb"),
            "reference_ligand_path": str(FIX / "reference_ligand.sdf"),
            "redock_validation": False,
        },
        stage1={"engine": "mock", "pilot_size": 8, "gpu_batch_size": 8},
    )

    t0 = time.time()
    stats = prepare_library(cfg.library.path, run / "library", cfg.library)
    print(f"[library] {stats.as_dict()}  ({time.time() - t0:.1f}s)")

    rec = prepare_receptor(
        cfg.receptor.path, run / "receptor",
        chain=cfg.receptor.chain, keep_waters=cfg.receptor.keep_waters,
        keep_metals=cfg.receptor.keep_metals,
    )
    print(f"[receptor] method={rec.method} atoms={rec.n_atoms} warnings={rec.warnings}")
    box = resolve_box(cfg.receptor, cfg.receptor.reference_ligand_path)
    box.save(run / "receptor" / "box.json")
    print(f"[box] center={box.center} size={box.size}")

    with RunLedger(run / "run.db") as ledger:
        t0 = time.time()
        summary = run_stage1(cfg, rec.pdbqt_path, box, run / "library", run / "stage1", ledger)
        print(f"[stage1] {summary}  ({time.time() - t0:.1f}s)")

        # Resume must be a no-op. With the mock engine the ranking is also
        # reproducible; real Vina-GPU has no seed, so only the resume no-op
        # property holds there.
        import pandas as pd

        before = pd.read_parquet(run / "stage1" / "ranking.parquet")
        t0 = time.time()
        run_stage1(cfg, rec.pdbqt_path, box, run / "library", run / "stage1", ledger)
        after = pd.read_parquet(run / "stage1" / "ranking.parquet")
        same = before.equals(after)
        print(f"[resume] identical={same} in {time.time() - t0:.1f}s")
        if not same:
            sys.exit("resume did not reproduce the ranking")

    print(before.head(8).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
