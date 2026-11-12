"""End-to-end smoke test: `vcdd run` over the bundled 200-molecule fixture.

This is the only test that exercises the *real* external dependencies, so it
needs all three of them and skips otherwise:

* a GPU plus a compiled Vina-GPU 2.1 binary (stage 1),
* a **test-mode** Boltz API key (stage 2) - a live key would bill the user for
  a fixture run, so the gate deliberately refuses to run against one,
* the cell resource bundle (stage 3).

Marked ``slow`` (~15 min). Run it with::

    pytest -m slow tests/test_smoke_pipeline.py

The cheap counterpart that runs everywhere is ``test_pipeline_contract.py``,
which drives the same stage boundaries in-process with a scripted Boltz API.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from _helpers import requires_boltz_test_key, requires_cell_resources, requires_vina_gpu

pytestmark = [pytest.mark.slow, requires_vina_gpu, requires_boltz_test_key,
              requires_cell_resources]

# every path the run-directory contract promises
CONTRACT = [
    "config.resolved.yaml",
    "manifest.json",
    "run.db",
    "library/index.parquet",
    "receptor/receptor.pdbqt",
    "receptor/receptor_clean.pdb",
    "receptor/box.json",
    "receptor/redock_validation.json",
    "stage1/ranking.parquet",
    "stage2/scores.parquet",
    "stage2/consensus_ranking.parquet",
    "stage2/pose_qc.parquet",
    "stage2/boltz/results.jsonl",
    "stage2/boltz/runs.json",
    "stage2/stage2_summary.json",
    "stage3/objective_scores.parquet",
    "stage3/objective_by_backend.parquet",
    "stage3/backend_report.json",
    "stage3/stage3_summary.json",
    "stage4/analogs.parquet",
    "stage4/analog_recommendations.csv",
    "stage4/sar_rules.csv",
    "stage4/pareto.parquet",
    "stage4/top_designs.sdf",
    "stage4/stage4_summary.json",
    "stage4/generator_report.json",
    "report/report.html",
]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, fixtures_dir):
    out = tmp_path_factory.mktemp("smoke") / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "vcdd.cli", "run",
         "-c", str(fixtures_dir / "smoke.yaml"), "-o", str(out), "--quiet"],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return out


def test_run_directory_contract_is_complete(smoke_run):
    missing = [p for p in CONTRACT if not (smoke_run / p).exists()]
    assert not missing, f"missing artifacts: {missing}"
    empty = [p for p in CONTRACT if (smoke_run / p).stat().st_size == 0]
    assert not empty, f"zero-byte artifacts: {empty}"


def test_every_stage_ranking_is_monotonic(smoke_run):
    s1 = pd.read_parquet(smoke_run / "stage1" / "ranking.parquet")
    assert len(s1) > 0
    assert s1["score"].is_monotonic_increasing  # more negative is better
    assert list(s1["rank"]) == list(range(1, len(s1) + 1))

    s2 = pd.read_parquet(smoke_run / "stage2" / "consensus_ranking.parquet")
    assert len(s2) <= len(s1)
    assert list(s2["consensus_rank"]) == list(range(1, len(s2) + 1))

    s3 = pd.read_parquet(smoke_run / "stage3" / "objective_scores.parquet")
    assert s3["objective"].is_monotonic_decreasing  # higher objective is better
    assert s3["objective"].between(0, 1).all()

    s4 = pd.read_csv(smoke_run / "stage4" / "analog_recommendations.csv")
    assert len(s4) > 0
    assert s4["rank"].is_monotonic_increasing


def test_stage2_is_a_boltz_run_that_never_touched_the_dock_score(smoke_run):
    """Stage 2 adds orthogonal evidence; it must not silently rewrite stage 1."""
    summary = json.loads((smoke_run / "stage2" / "stage2_summary.json").read_text())
    assert summary["backend"] in ("library_screen", "structure_and_binding")
    assert summary["livemode"] is False, "the suite must never bill a live key"
    assert summary["n_scored"] > 0

    s2 = pd.read_parquet(smoke_run / "stage2" / "consensus_ranking.parquet")
    assert "binding_confidence" in s2.columns
    pd.testing.assert_series_equal(
        s2["dock_score"], s2["stage1_score"], check_names=False,
    )


def test_redock_gate_ran_and_is_recorded(smoke_run):
    gate = json.loads((smoke_run / "receptor" / "redock_validation.json").read_text())
    assert "rmsd" in json.dumps(gate).lower()


def test_manifest_records_inputs_stages_and_versions(smoke_run):
    man = json.loads((smoke_run / "manifest.json").read_text())
    assert man["vcdd_version"]
    assert set(man["stages"]) >= {"library", "receptor", "stage1", "stage2",
                                  "stage3", "stage4", "report"}
    for rec in man["inputs"].values():
        assert rec["exists"] and rec["sha256_16"]
    assert man["tool_versions"]


def test_report_embeds_figures(smoke_run):
    html = (smoke_run / "report" / "report.html").read_text()
    assert len(html) > 50_000
    figs = list((smoke_run / "report" / "figures").glob("*.svg"))
    assert len(figs) >= 5
    assert all(f.with_suffix(".png").exists() for f in figs)


def test_resume_is_a_no_op_on_a_finished_run(smoke_run, fixtures_dir):
    before = pd.read_parquet(smoke_run / "stage1" / "ranking.parquet")
    proc = subprocess.run(
        [sys.executable, "-m", "vcdd.cli", "resume",
         "-c", str(fixtures_dir / "smoke.yaml"), "-o", str(smoke_run), "--quiet"],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    after = pd.read_parquet(smoke_run / "stage1" / "ranking.parquet")
    cols = ["mol_id", "smiles", "score", "rank"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
