"""Contract tests for the ``local`` Boltz backend (``stage2.backend: local``).

The stage-2 driver is exercised end to end against :mod:`fake_boltz_cli`, a
scripted ``boltz`` executable: vcdd builds the real argv, launches the real
``subprocess.run`` path, parses the files a real ``boltz predict`` would write,
and feeds them through the unmodified ranking / pose-QC / ledger machinery.

The fake CLI is a subprocess, not a mock: a wrong flag name, wrong YAML shape
or wrong output-file convention fails here exactly as it would against the
real binary - which is the same contract discipline ``fake_boltz`` applies to
the API backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vcdd.receptor.box import Box
from vcdd.refine.boltz import local as bloc
from vcdd.refine.stage2 import run_stage2

FAKE_CLI = str(Path(__file__).parent / "fake_boltz_cli.py")


def _local_cfg(base_cfg, **stage2_kwargs):
    """A config wired to the fake boltz CLI, with any stage-2 overrides."""
    cfg = base_cfg
    cfg.stage2.backend = "local"
    cfg.stage2.local_boltz.binary = FAKE_CLI
    cfg.stage2.local_boltz.max_concurrency = 2
    cfg.stage2.local_boltz.devices = [0, 1]
    cfg.stage2.keep_structures_top_k = 4
    # A ceiling of $0 would abort any API run; local must ignore it entirely.
    cfg.stage2.max_spend_usd = 0.0
    for key, value in stage2_kwargs.items():
        setattr(cfg.stage2, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# input assembly
# --------------------------------------------------------------------------- #

def test_build_yaml_puts_ligand_first_and_each_chain_once():
    # The exact document boltz's schema parser wants (verified against 2.2.1):
    # entity type = sequence key, id nested inside - a sibling id is rejected
    # with "Invalid entity type: id".
    entities = [
        {"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"},
        {"chain_ids": ["B"], "type": "protein", "value": "MKV"},
    ]
    doc = bloc.build_yaml(entities, "CCO")
    assert doc["version"] == 1
    seqs = doc["sequences"]
    assert seqs[0] == {"ligand": {"id": "LIG", "smiles": "CCO"}}
    assert {"protein": {"id": "A", "sequence": "MKVLAAGIVGAWQ"}} in seqs
    assert {"protein": {"id": "B", "sequence": "MKV"}} in seqs


def test_build_yaml_avoids_colliding_ligand_id():
    # ... and the ligand id must stay short: boltz's Chain dtype is <U5
    entities = [{"chain_ids": ["LIG"], "type": "protein", "value": "MKV"}]
    doc = bloc.build_yaml(entities, "CCO")
    ids = [next(iter(s.values()))["id"] for s in doc["sequences"]]
    assert ids == ["LIG", "LIG_2"]
    assert len(set(ids)) == len(ids)
    assert all(len(i) <= 5 for i in ids), "boltz Chain.name is a 5-char field"


def test_build_yaml_text_quotes_smiles_safely():
    txt = bloc.build_yaml_text([{"chain_ids": ["A"], "type": "protein", "value": "MKV"}],
                               "CCO[C@@H](N)C")
    import yaml

    doc = yaml.safe_load(txt)
    assert doc["sequences"][0]["ligand"]["smiles"] == "CCO[C@@H](N)C"


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #

def test_predict_command_matches_the_real_boltz_cli(base_cfg):
    """Locked to `boltz predict --help` on boltz 2.2.x (verified against it).

    --devices is a *count* (1, the process is CUDA_VISIBLE_DEVICES-pinned),
    --diffusion_samples replaces the nonexistent --num_samples, and there is
    no --half/--no-half flag, so half_precision must never reach the argv.
    """
    cfg = base_cfg.stage2.local_boltz
    cfg.devices = [0, 1]
    cfg.cache_dir = "/tmp/w"
    cfg.half_precision = True
    cfg.extra_args = ("--low_mem",)
    cmd = bloc.predict_command("boltz", "in.yaml", "/tmp/out", device=1, cfg=cfg,
                               num_samples=3)
    assert cmd[:4] == ["boltz", "predict", "in.yaml", "--out_dir"]
    assert "/tmp/out" in cmd
    assert "--devices" in cmd and cmd[cmd.index("--devices") + 1] == "1"
    assert "--diffusion_samples" in cmd
    assert cmd[cmd.index("--diffusion_samples") + 1] == "3"
    assert "--num_samples" not in cmd
    assert "--cache" in cmd and cmd[cmd.index("--cache") + 1] == "/tmp/w"
    assert "--low_mem" in cmd
    assert "--half" not in cmd and "--no-half" not in cmd, \
        "these builds have no half-precision flag; half_precision is ignored"


def test_predict_command_leaves_half_precision_out_when_null(base_cfg):
    cfg = base_cfg.stage2.local_boltz
    cmd = bloc.predict_command("boltz", "in.yaml", "/tmp/out", device=0, cfg=cfg)
    assert "--half" not in cmd and "--no-half" not in cmd


def test_predict_command_emits_msa_server_flags(base_cfg):
    """boltz 2.2 refuses MSA-less inputs; --use_msa_server is the default."""
    cfg = base_cfg.stage2.local_boltz
    cfg.use_msa_server = True
    cfg.msa_server_url = "https://msa.internal:8080"
    cfg.msa_server_username = "user"
    cfg.msa_server_password = "pw"
    cmd = bloc.predict_command("boltz", "in.yaml", "/tmp/out", device=0, cfg=cfg)
    assert "--use_msa_server" in cmd
    assert cmd[cmd.index("--msa_server_url") + 1] == "https://msa.internal:8080"
    assert cmd[cmd.index("--msa_server_username") + 1] == "user"
    assert cmd[cmd.index("--msa_server_password") + 1] == "pw"


def test_predict_command_skips_msa_flags_fully_offline(base_cfg):
    cfg = base_cfg.stage2.local_boltz
    cfg.use_msa_server = False
    cmd = bloc.predict_command("boltz", "in.yaml", "/tmp/out", device=0, cfg=cfg)
    assert "--use_msa_server" not in cmd
    assert "--msa_server_url" not in cmd


def test_build_yaml_injects_precomputed_msa_files():
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}]
    doc = bloc.build_yaml(entities, "CCO", msa_files={"A": "/data/a.a3m"})
    protein = doc["sequences"][1]["protein"]
    assert protein["id"] == "A"
    assert protein["msa"] == "/data/a.a3m"
    # a chain without an entry gets no msa field (the server covers it)
    doc = bloc.build_yaml(entities, "CCO")
    assert "msa" not in doc["sequences"][1]["protein"]


def test_run_predict_pins_the_subprocess_to_the_requested_gpu(tmp_path, base_cfg):
    """Round-robin honesty: each process sees only its GPU (CUDA_VISIBLE_DEVICES)."""
    cfg = base_cfg.stage2.local_boltz
    cfg.binary = FAKE_CLI
    wd = tmp_path / "m1"
    row, meta = bloc.run_predict(
        cfg.binary,
        [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}],
        "CCO", wd, device=1, cfg=cfg, mol_id="M1",
    )
    assert meta["status"] == "succeeded"
    assert (wd / "out" / "cuda_visible.txt").read_text() == "1"


# --------------------------------------------------------------------------- #
# output parsing
# --------------------------------------------------------------------------- #

def test_parse_confidence_flattens_boltz_metrics(tmp_path):
    # The exact schema boltz-2.2's writer emits (verified against its source).
    conf = {
        "confidence_score": 0.808,
        "ptm": 0.72,
        "iptm": 0.71,
        "ligand_iptm": 0.8,
        "protein_iptm": 0.79,
        "complex_plddt": 0.7,
        "complex_iplddt": 0.69,
        "complex_pde": 0.4,
        "complex_ipde": 0.5,
    }
    p = tmp_path / "confidence_x_model_0.json"
    p.write_text(json.dumps(conf))
    row = bloc.parse_confidence(p, "M1", "CCO")
    assert row["mol_id"] == "M1" and row["boltz_smiles"] == "CCO"
    assert row["binding_confidence"] == 0.808           # <- confidence_score
    assert row["iptm"] == 0.71
    assert row["ligand_iptm"] == 0.8
    assert row["optimization_score"] is None           # BoltzMol-1-only metric
    assert row["adme_lipophilicity"] is None           # API-only
    assert row["url_structure"] is None
    assert row["boltz_run_id"] == "local_M1" == row["boltz_result_id"]


def test_parse_confidence_reconstructs_binding_confidence_from_interface_metrics(tmp_path):
    """A writer that omits confidence_score must not silently lose the column."""
    conf = {"ligand_iptm": 0.8, "protein_iptm": 0.74}
    p = tmp_path / "confidence_y_model_0.json"
    p.write_text(json.dumps(conf))
    row = bloc.parse_confidence(p, "M2", "CCO")
    assert row["binding_confidence"] == pytest.approx(0.77)


# --------------------------------------------------------------------------- #
# binary discovery
# --------------------------------------------------------------------------- #

def test_find_binary_and_version_against_the_fake_cli():
    assert bloc.find_binary(FAKE_CLI) == FAKE_CLI
    assert bloc.version(FAKE_CLI) == "2.1.1"


def test_find_binary_reports_absent_paths():
    assert bloc.find_binary("/definitely/not/here/boltz") is None
    assert bloc.version("/definitely/not/here/boltz") is None


@pytest.mark.skipif(bloc.find_binary("boltz") is None,
                    reason="real boltz binary not on PATH")
def test_version_falls_back_to_importlib_when_cli_probe_fails():
    """boltz 2.2.x has no --version flag; the version still comes back via
    importlib.metadata when boltz is installed in this interpreter."""
    v = bloc.version("boltz")
    assert v is not None
    assert bloc._SEMVER_RE.fullmatch(v)


# --------------------------------------------------------------------------- #
# end to end through run_stage2
# --------------------------------------------------------------------------- #

@pytest.fixture()
def local_run(base_cfg, stage1_done, tmp_path, ledger, small_box, monkeypatch):
    cfg = _local_cfg(base_cfg)
    monkeypatch.setenv("FAKE_BOLTZ_CENTER",
                       ",".join(str(v) for v in small_box.center))
    receptor, stage1_dir = stage1_done
    # The fake complex carries the receptor's C-alpha trace in a rotated and
    # translated frame, exactly like a co-folding model does.
    monkeypatch.setenv("FAKE_BOLTZ_RECEPTOR", str(receptor))
    out = tmp_path / "stage2"
    summary = run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)
    return summary, out


def test_local_run_writes_every_artifact_the_run_contract_promises(local_run):
    summary, out = local_run
    for rel in ("scores.parquet", "consensus_ranking.parquet", "pose_qc.parquet",
                "stage2_summary.json", "boltz/results.jsonl", "boltz/runs.json"):
        path = out / rel
        assert path.exists(), f"missing {rel}"
        assert path.stat().st_size > 0, f"zero-byte {rel}"
    assert summary["n_scored"] == 6
    assert summary["backend"] == "local"
    assert "boltz-2 (local)" == summary["engine"]
    assert summary["engine_version"] == "2.1.1"
    assert summary["livemode"] is True
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["local_boltz"]["devices"] == [0, 1]
    assert "backend=local" in summary["note"]


def test_local_run_ranks_by_binding_confidence(local_run):
    _, out = local_run
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    assert list(cons["consensus_rank"]) == list(range(1, 7))
    assert cons["binding_confidence"].notna().all()
    scored = cons.dropna(subset=["binding_confidence"])
    assert scored["binding_confidence"].is_monotonic_decreasing
    pd.testing.assert_series_equal(cons["dock_score"], cons["stage1_score"],
                                   check_names=False)


def test_local_run_pose_qc_measures_the_predicted_ligand_against_the_box(local_run):
    """The predicted complex arrives in the model's frame; QC must superpose it.

    The fake CLI writes the receptor trace rotated and translated by ~1000 A, so
    a QC that skipped the superposition would report every ligand hundreds of
    Angstroms away, and one that measured protein atoms would be dragged off the
    box centre by the whole chain.
    """
    summary, out = local_run
    qc = pd.read_parquet(out / "pose_qc.parquet")
    assert not qc.empty
    assert qc["pose_frame_aligned"].all(), "receptor trace should be superposable"
    assert qc["ligand_in_box"].all(), "fake CLI placed every ligand in the box"
    assert qc["centroid_offset_A"].max() < 2.0
    assert qc["receptor_ca_rmsd_A"].max() < 0.01, \
        "the transform is rigid, so the only residual is CIF coordinate rounding"
    assert summary["pose_qc"]["fraction_in_box"] == 1.0
    assert summary["pose_qc"]["n_frame_aligned"] == len(qc)


def test_pose_qc_reports_an_unalignable_complex_instead_of_a_box_distance(
        base_cfg, stage1_done, tmp_path, ledger, small_box, monkeypatch):
    """A chain that cannot be matched to the receptor makes the box test meaningless."""
    cfg = _local_cfg(base_cfg)
    monkeypatch.setenv("FAKE_BOLTZ_CENTER", ",".join(str(v) for v in small_box.center))
    monkeypatch.delenv("FAKE_BOLTZ_RECEPTOR", raising=False)  # two dummy CAs only
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    summary = run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)
    qc = pd.read_parquet(out / "pose_qc.parquet")
    assert not qc.empty
    assert not qc["pose_frame_aligned"].any()
    assert set(qc["pose_error"]) == {"frame_not_aligned"}
    assert qc["ligand_in_box"].isna().all()
    assert qc["centroid_offset_A"].isna().all()
    assert summary["pose_qc"]["n_checked"] == 0
    assert summary["pose_qc"]["fraction_in_box"] is None


# --------------------------------------------------------------------------- #
# affinity head (opt-in)
# --------------------------------------------------------------------------- #


def test_affinity_is_off_by_default_and_the_schema_is_unchanged(local_run):
    """The confidence-only run must not silently grow an affinity column of values."""
    _, out = local_run
    scores = pd.read_parquet(out / "scores.parquet")
    assert "affinity_pred_value" in scores.columns
    assert scores["affinity_pred_value"].isna().all()
    assert scores["affinity_probability_binary"].isna().all()


def test_affinity_head_is_requested_in_the_yaml_and_read_back(
        base_cfg, stage1_done, tmp_path, ledger, small_box, monkeypatch):
    """``affinity: true`` must reach boltz as a properties block, not just as flags."""
    cfg = _local_cfg(base_cfg)
    cfg.stage2.local_boltz.affinity = True
    monkeypatch.setenv("FAKE_BOLTZ_CENTER", ",".join(str(v) for v in small_box.center))
    monkeypatch.setenv("FAKE_BOLTZ_RECEPTOR", str(stage1_done[0]))
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)

    scores = pd.read_parquet(out / "scores.parquet")
    assert scores["affinity_pred_value"].notna().all()
    assert scores["affinity_probability_binary"].notna().all()
    # the fake CLI ties both to the SMILES hash, so the values must differ across
    # molecules rather than being a constant that any parser would produce
    assert scores["affinity_pred_value"].nunique() > 1

    yaml_text = next((out / "boltz" / "local").glob("*/input.yaml")).read_text()
    assert "properties:" in yaml_text and "binder: LIG" in yaml_text


def test_affinity_flags_are_only_passed_when_the_head_is_on():
    class _Cfg:
        cache_dir = None
        use_msa_server = False
        extra_args = ()
        affinity = True
        affinity_samples = 5
        affinity_sampling_steps = 200
        affinity_mw_correction = True

    cmd = bloc.predict_command("boltz", "in.yaml", "out", 0, _Cfg())
    assert "--diffusion_samples_affinity" in cmd and "5" in cmd
    assert "--sampling_steps_affinity" in cmd and "200" in cmd
    assert "--affinity_mw_correction" in cmd

    _Cfg.affinity = False
    off = bloc.predict_command("boltz", "in.yaml", "out", 0, _Cfg())
    assert not any("affinity" in tok for tok in off)


def test_local_run_skips_the_spend_gate(local_run):
    """A ceiling of $0.00 must not stop a local run (it would stop any API run)."""
    summary, _ = local_run
    assert summary["estimated_cost_usd"] == 0.0


def test_local_run_records_one_run_per_molecule_in_runs_json(local_run):
    _, out = local_run
    runs = json.loads((out / "boltz" / "runs.json").read_text())
    assert len(runs) == 6
    assert all(r["status"] == "succeeded" for r in runs)
    assert all(r["engine"] == "boltz-2 (local)" for r in runs)


def test_local_predict_failure_lands_in_rejected_and_does_not_abort(base_cfg, stage1_done,
                                                                    tmp_path, ledger,
                                                                    small_box, monkeypatch):
    cfg = _local_cfg(base_cfg)
    monkeypatch.setenv("FAKE_BOLTZ_FAIL", "1")
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    summary = run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)
    assert summary["n_scored"] == 0
    assert summary["n_failed"] == 6
    scores = pd.read_parquet(out / "scores.parquet")
    assert (scores["error"] == "boltz_predict_nonzero").all()
    rej = pd.read_parquet(out / "boltz" / "rejected.parquet")
    assert set(rej["reason"]) == {"boltz_predict_nonzero"}
    # a failed chunk must still be marked done in the ledger so a resume does
    # not re-run predictions that are known to fail
    assert len(ledger.completed_units("stage2")) == 2


def test_local_predict_timeout_is_recorded_not_hung(base_cfg, stage1_done, tmp_path,
                                                    ledger, small_box, monkeypatch):
    cfg = _local_cfg(base_cfg)
    cfg.stage2.local_boltz.timeout_s = 1
    monkeypatch.setenv("FAKE_BOLTZ_TIMEOUT", "1")
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    summary = run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)
    assert summary["n_scored"] == 0
    scores = pd.read_parquet(out / "scores.parquet")
    assert (scores["error"] == "vcdd_predict_timeout").all()


def test_stale_boltz_outputs_are_not_reused_when_the_input_changes(base_cfg):
    """boltz skips predictions that already exist, so vcdd must invalidate them.

    The per-molecule output directory is keyed by molecule id alone. Re-running
    a molecule with a different SMILES (or MSA, or affinity setting) without
    wiping ``out/`` would let boltz skip the work and vcdd report the previous
    run's numbers as if they were new - a silent, invisible wrong answer.
    """
    cfg = base_cfg.stage2.local_boltz
    cfg.binary = FAKE_CLI
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKV"}]
    out_dir = base_cfg.run_dir / "local"

    def _run(smiles, force=False):
        rows, runs, _ = bloc.screen_molecules(
            cfg, entities, ["M1"], [smiles], out_dir=out_dir, force=force,
        )
        assert len(rows) == 1
        return rows[0]

    work = out_dir / "M1"
    first = _run("CCO")
    assert len((work / "invocations.log").read_text().splitlines()) == 1

    # same input, no force: boltz is allowed to reuse what it already wrote
    second = _run("CCO")
    assert second["binding_confidence"] == first["binding_confidence"]
    assert len((work / "invocations.log").read_text().splitlines()) == 2

    # changed input: the stale directory must be wiped, so the numbers move
    third = _run("CCN")
    assert third["binding_confidence"] != first["binding_confidence"], \
        "a changed SMILES must not return the previous molecule's numbers"

    # force: a genuine re-prediction even when the input is byte-identical
    n_before = len((work / "invocations.log").read_text().splitlines())
    _run("CCN", force=True)
    assert len((work / "invocations.log").read_text().splitlines()) == n_before + 1


def test_invalid_smiles_are_rejected_before_any_predict(base_cfg):
    cfg = base_cfg.stage2.local_boltz
    cfg.binary = FAKE_CLI
    rows, runs, rejected = bloc.screen_molecules(
        cfg,
        [{"chain_ids": ["A"], "type": "protein", "value": "MKV"}],
        ["M1", "M2"], ["CCO", "not a smiles"], out_dir=base_cfg.run_dir / "local",
    )
    assert rows and len(rows) == 1
    assert rejected == [{"mol_id": "M2", "smiles": "not a smiles",
                         "boltz_run_id": None, "reason": "invalid_smiles",
                         "run_status": "not_submitted"}]


def test_keep_raw_outputs_false_skips_pose_qc_and_removes_dirs(base_cfg, stage1_done,
                                                               tmp_path, ledger,
                                                               small_box, monkeypatch):
    cfg = _local_cfg(base_cfg)
    cfg.stage2.local_boltz.keep_raw_outputs = False
    monkeypatch.setenv("FAKE_BOLTZ_CENTER",
                       ",".join(str(v) for v in small_box.center))
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    summary = run_stage2(cfg, receptor, small_box, stage1_dir, out, ledger)
    assert summary["pose_qc"]["n_checked"] == 0
    qc = pd.read_parquet(out / "pose_qc.parquet")
    assert qc.empty
    local_root = out / "boltz" / "local"
    assert local_root.exists()
    assert not any(p.is_dir() for p in local_root.iterdir()), \
        "per-molecule dirs should have been deleted"
