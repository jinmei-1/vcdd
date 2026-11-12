"""Developer probe: run stage 2 on the dev run dir produced by dev_stage1_check.

Stage 2 calls the hosted Boltz API, so this probe needs a key in
``BOLTZ_API_KEY``. Use a ``_test_`` key: the full submit/poll/fetch lifecycle
runs, the results are synthetic, and nothing is billed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

from vcdd.config import RunConfig
from vcdd.receptor.box import Box
from vcdd.runtime.ledger import RunLedger
from vcdd.runtime.logging import setup_logging
from vcdd.refine.stage2 import run_stage2

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main(run_dir: str = "/workspace/dev_run", top_n: int = 8) -> None:
    run = Path(run_dir)
    setup_logging(run, level="INFO")
    cfg = RunConfig(
        name="dev",
        out_dir=str(run),
        n_workers=8,
        library={"path": str(FIX / "library.sdf")},
        receptor={"path": str(FIX / "receptor.pdb")},
        stage2={
            "top_n": int(top_n), "chunk_size": 4, "backend": "library_screen",
            "fusion": "boltz_only", "max_spend_usd": 1.0, "keep_structures_top_k": 4,
            "poll_interval_s": 5,
        },
    )
    if not os.environ.get("BOLTZ_API_KEY"):
        sys.exit("BOLTZ_API_KEY is not set; export a _test_ key to run this probe")
    box = Box.load(run / "receptor" / "box.json")
    rec = run / "receptor" / "receptor.pdbqt"

    with RunLedger(run / "run.db") as ledger:
        t0 = time.time()
        summary = run_stage2(cfg, rec, box, run / "stage1", run / "stage2", ledger)
        print(f"[stage2] {time.time() - t0:.1f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    cons = pd.read_parquet(run / "stage2" / "consensus_ranking.parquet")
    cols = [c for c in ["consensus_rank", "mol_id", "stage1_score", "dock_score",
                        "binding_confidence", "optimization_score", "consensus_score",
                        "ligand_in_box"]
            if c in cons.columns]
    print(cons[cols].to_string(index=False))

    qc_path = run / "stage2" / "pose_qc.parquet"
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)
        print("\n[pose QC]")
        cols = [c for c in ["mol_id", "ligand_in_box", "centroid_offset_A",
                            "frac_atoms_in_box", "max_axis_excess_A", "pose_n_atoms",
                            "pose_error"] if c in qc.columns]
        print(qc[cols].to_string(index=False))

    poses = run / "stage2" / "poses"
    n_pose = len(list(poses.glob("*.cif"))) if poses.exists() else 0
    print(f"\n[poses] {poses} -> {n_pose} structures")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
