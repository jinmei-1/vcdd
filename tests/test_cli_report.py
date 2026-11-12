"""CLI surface and report-generation tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from _helpers import SMILES
from vcdd.cli import app
from vcdd.version import __version__

runner = CliRunner()


def _run(*args):
    return runner.invoke(app, list(args))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_version_matches_the_package():
    res = _run("version")
    assert res.exit_code == 0
    assert __version__ in res.stdout


def test_help_lists_every_pipeline_stage():
    res = _run("--help")
    assert res.exit_code == 0
    for cmd in ("doctor", "config", "prep-library", "prep-receptor",
                "stage1", "stage2", "stage3", "stage4", "report", "run", "resume", "boltz"):
        assert cmd in res.stdout


def test_boltz_subcommands_are_reachable():
    res = _run("boltz", "--help")
    assert res.exit_code == 0
    for cmd in ("estimate", "status", "fetch"):
        assert cmd in res.stdout


def test_doctor_json_reports_engines_backends_and_generators():
    res = _run("doctor", "--json")
    assert res.exit_code == 0
    payload = json.loads(res.stdout[res.stdout.index("{"):])
    for section in ("docking_engines", "gpu", "boltz", "cell_backends", "analog_generators"):
        assert section in payload, payload.keys()
    # stage 1 is Vina-GPU only; the rescorer plugin subsystem is gone
    assert "rescorers" not in payload
    assert set(payload["docking_engines"]) <= {"vina_gpu", "mock"}
    # every unavailable component must explain itself
    for section in ("docking_engines", "cell_backends", "analog_generators"):
        for name, info in payload[section].items():
            if isinstance(info, dict) and info.get("available") is False:
                assert info.get("detail"), f"{section}.{name} unavailable with no reason"


def test_doctor_prints_the_pip_extra_instead_of_letting_rich_eat_it(monkeypatch):
    """`vcdd[boltz]` is a rich style tag unless it is escaped.

    Unescaped, the doctor printed `pip install 'vcdd'` - an instruction that
    installs the package without the Boltz SDK and leaves stage 2 broken in
    exactly the way the user was trying to fix.
    """
    import vcdd.refine.boltz.client as bclient

    monkeypatch.setattr(bclient, "boltz_available", lambda: False)
    monkeypatch.setenv("COLUMNS", "200")  # keep the table cell off a wrap point
    res = _run("doctor")
    assert res.exit_code == 0
    assert "vcdd[boltz]" in res.stdout, res.stdout


def test_console_output_never_eats_an_aromatic_smiles():
    """`[nH]` is a valid rich style tag, so rich deletes it from an unescaped string.

    `c1cc[nH]c1` (pyrrole) would print as `c1ccc1` - a different, still
    copy-pasteable molecule. Any console path that can carry data must render
    it literally.
    """
    from rich.text import Text

    from vcdd.cli import _print_kv, console

    smi = "O=C(O)c1cc[nH]n1"
    with console.capture() as cap:
        _print_kv("t", {"best_smiles": smi})
    assert smi in cap.get(), cap.get()

    with console.capture() as cap:
        console.print(Text(f"transformation  {smi}>>C{smi}"))
    assert smi in cap.get(), cap.get()


def test_doctor_does_not_truncate_an_install_hint(monkeypatch):
    """Details wrap; they are never cut mid-instruction."""
    import vcdd.cli as cli

    long_hint = "make source in the Vina-GPU-2.1 checkout, then copy Kernel1_Opt.bin next to the binary"
    monkeypatch.setattr(
        cli, "_gpu_report",
        lambda: {"cached kernels": {"available": False, "detail": "no Kernel*_Opt.bin found",
                                    "install_hint": long_hint}},
    )
    monkeypatch.setenv("COLUMNS", "220")
    res = _run("doctor")
    assert res.exit_code == 0
    assert long_hint in res.stdout, res.stdout


def test_doctor_finds_the_kernel_cache_where_the_build_script_installs_it(monkeypatch, tmp_path):
    """`build_vina_gpu.sh` puts kernels in $PREFIX/share, not next to the binary.

    Searching only the binary's directory reported "no cached kernels" after a
    correct build and sent the user back to a slow `make source` for nothing.
    """
    import vcdd.cli as cli
    from vcdd.docking.engines.vina_gpu import VinaGPUEngine

    binary = tmp_path / "bin" / "AutoDock-Vina-GPU-2-1"
    binary.parent.mkdir(parents=True)
    binary.touch()
    kdir = tmp_path / "share" / "vcdd" / "vina-gpu-kernels"
    kdir.mkdir(parents=True)
    for k in ("Kernel1_Opt.bin", "Kernel2_Opt.bin"):
        (kdir / k).write_bytes(b"\x00")

    monkeypatch.setattr(VinaGPUEngine, "variant_binaries",
                        staticmethod(lambda: {"autodock_vina_gpu": str(binary)}))
    report = cli._gpu_report()["cached kernels"]
    assert report["available"] is True, report
    assert str(kdir) in report["detail"], report["detail"]
    assert "opencl_binary_path" in report["detail"], report["detail"]


def test_doctor_reports_boltz_key_state_without_leaking_it(monkeypatch):
    monkeypatch.setenv("BOLTZ_API_KEY", "sk_bc_notarealkey_test_0123456789abcdef")
    res = _run("doctor", "--json")
    assert res.exit_code == 0
    payload = json.loads(res.stdout[res.stdout.index("{"):])
    blob = json.dumps(payload)
    assert "notarealkey" not in blob, "doctor must never echo the API key"
    key = payload["boltz"]["api key"]
    assert key["available"] is True
    assert "test" in key["detail"], key["detail"]


def test_config_writes_a_loadable_template(tmp_path):
    out = tmp_path / "cfg.yaml"
    res = _run("config", "-o", str(out))
    assert res.exit_code == 0
    assert out.exists()
    from vcdd.config.loader import load_config

    cfg = load_config(out)
    assert cfg.stage1.engine == "vina_gpu"
    assert cfg.stage2.backend in ("library_screen", "structure_and_binding")


def test_unknown_command_fails_cleanly():
    res = _run("not-a-command")
    assert res.exit_code != 0


def test_stage_command_without_inputs_fails_with_a_hint(tmp_path):
    res = _run("stage1", "-o", str(tmp_path / "run"))
    assert res.exit_code != 0
    assert res.stdout or res.stderr


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_run(tmp_path):
    """A minimal run directory carrying one artifact per stage."""
    run = tmp_path / "run"
    names = ["aspirin", "caffeine", "ibuprofen", "phenol", "toluene", "aniline"]
    smis = [SMILES[n] for n in names]
    n = len(names)
    ids = [f"M{i:03d}" for i in range(n)]

    (run / "stage1").mkdir(parents=True)
    pd.DataFrame({"mol_id": ids, "smiles": smis,
                  "score": [-9.0 + 0.3 * i for i in range(n)],
                  "rank": list(range(1, n + 1))}
                 ).to_parquet(run / "stage1" / "ranking.parquet", index=False)

    # Stage 2 is a Boltz co-folding run under rank_fusion: dock_score is stage 1
    # carried forward unchanged, and binding_confidence is the new evidence.
    (run / "stage2").mkdir(parents=True)
    s1 = [-9.0 + 0.3 * i for i in range(n)]
    bc = [0.91, 0.84, 0.72, 0.68, 0.55, 0.41]
    pd.DataFrame({"mol_id": ids, "smiles": smis,
                  "stage1_score": s1,
                  "stage1_rank": list(range(1, n + 1)),
                  "dock_score": s1,
                  "binding_confidence": bc,
                  "optimization_score": [0.80 - 0.06 * i for i in range(n)],
                  "structure_confidence": [0.86] * n,
                  "ligand_in_box": [True, True, True, False, True, False],
                  "centroid_offset_A": [1.1, 1.6, 2.0, 12.4, 1.9, 21.7],
                  "norm_stage1": [1 - i / n for i in range(n)],
                  "norm_boltz": [1 - 0.13 * i for i in range(n)],
                  "consensus_score": [1.0 + i for i in range(n)],
                  "consensus_rank": list(range(1, n + 1))}
                 ).to_parquet(run / "stage2" / "consensus_ranking.parquet", index=False)
    pd.DataFrame({"mol_id": ids,
                  "ligand_in_box": [True, True, True, False, True, False],
                  "centroid_offset_A": [1.1, 1.6, 2.0, 12.4, 1.9, 21.7],
                  "frac_atoms_in_box": [1.0, 1.0, 0.95, 0.0, 1.0, 0.0],
                  "max_axis_excess_A": [0.0, 0.0, 0.4, 8.2, 0.0, 15.1],
                  "pose_n_atoms": [13, 14, 15, 7, 7, 6],
                  "pose_path": [f"poses/{i}.cif" for i in ids],
                  "pose_error": [None] * n}
                 ).to_parquet(run / "stage2" / "pose_qc.parquet", index=False)
    (run / "stage2" / "stage2_summary.json").write_text(json.dumps({
        "backend": "library_screen", "engine": "boltzmol", "engine_version": "1.0",
        "sdk_version": "0.46.0", "livemode": False, "api_key_env": "BOLTZ_API_KEY",
        "n_input": n, "n_scored": n, "n_failed": 0, "n_rejected": 0,
        "estimated_cost_usd": 0.15, "n_runs": 1, "fusion": "rank_fusion",
        "gate_threshold": 0.7, "best_binding_confidence": max(bc),
        "spearman_vs_stage1": 0.83,
        "pose_qc": {"n_checked": n, "n_in_box": 4, "fraction_in_box": 4 / n,
                    "median_centroid_offset_A": 1.8, "n_errors": 0},
        "target": {"chains": [{"chain_id": "A", "length": 121}],
                   "pocket_sizes": {"A": 17}},
        "seconds": 42.0,
        "note": "dock_score is the stage-1 Vina-GPU score carried forward; "
                "Boltz does not rescore the docked pose.",
    }))

    (run / "stage3").mkdir(parents=True)
    pd.DataFrame({"mol_id": ids, "smiles": smis,
                  "dock_score": [-9.5 + 0.25 * i for i in range(n)],
                  "consensus_score": [1 - 0.1 * i for i in range(n)],
                  "consensus_rank": list(range(1, n + 1)),
                  "gene_score": [0.8 - 0.05 * i for i in range(n)],
                  "pathway_score": [0.6] * n,
                  "target_consistency": [0.5] * n,
                  "specificity": [0.9] * n,
                  "objective": [0.75 - 0.05 * i for i in range(n)],
                  "objective_rank": list(range(1, n + 1)),
                  "max_tanimoto": [0.5] * n,
                  "out_of_domain": [False] * n,
                  "n_backends": [1] * n,
                  "backend_spread": [0.0] * n}
                 ).to_parquet(run / "stage3" / "objective_scores.parquet", index=False)
    (run / "stage3" / "stage3_summary.json").write_text(json.dumps(
        {"n_compounds": n, "backends_used": ["mock"], "backends_scored": ["mock"],
         "backends_reference_only": [], "cell_line": "MCF7", "dose_um": 10.0,
         "time_h": 24.0, "gene_space": "landmark", "genes_up": ["CDKN1A"],
         "genes_down": ["MYC"], "warnings": [], "n_out_of_domain": 0}))

    (run / "stage4").mkdir(parents=True)
    an = pd.DataFrame({
        "rank": [1, 2, 3, 4],
        "parent_id": ["M000", "M000", "M001", "M002"],
        "parent_smiles": [smis[0], smis[0], smis[1], smis[2]],
        "analog_id": [f"AN{i:06d}" for i in range(4)],
        "analog_smiles": [smis[3], smis[4], smis[5], smis[0]],
        "generator": ["rgroup"] * 4,
        "transformation": ["methylate", "fluorinate", "methylate", "fluorinate"],
        "delta_dock_score": [-0.4, 0.1, -0.2, 0.3],
        "delta_cell_objective": [0.05, -0.01, 0.03, -0.02],
        "delta_gene_of_interest": [0.0] * 4,
        "gene_of_interest": ["CDKN1A"] * 4,
        "design_score": [0.7, 0.6, 0.55, 0.4],
        "pareto_rank": [1, 1, 2, 3],
        "dock_score": [-9.9, -9.4, -9.2, -8.9],
        "cell_objective": [0.80, 0.74, 0.72, 0.70],
        "qed": [0.7] * 4, "sa_score": [2.5] * 4,
        "similarity_to_parent": [0.7, 0.65, 0.72, 0.61],
        "guardrail_flags": [""] * 4, "out_of_domain": [False] * 4,
    })
    an.to_csv(run / "stage4" / "analog_recommendations.csv", index=False)
    an.to_parquet(run / "stage4" / "analogs.parquet", index=False)
    pd.DataFrame({
        "transformation": ["methylate", "fluorinate"],
        "metric": ["delta_cell_objective"] * 2,
        "n_analogs": [2, 2], "n_parents": [2, 2], "n_parents_improved": [2, 0],
        "frac_parents_improved": [1.0, 0.0], "median_delta": [0.04, -0.015],
        "mean_delta": [0.04, -0.015], "iqr_delta": [0.01, 0.005],
        "p_sign_test": [0.5, 0.5], "verdict": ["suggestive", "inconsistent"],
        "mean_lineage_length": [1.0, 1.0],
    }).to_csv(run / "stage4" / "sar_rules.csv", index=False)

    (run / "manifest.json").write_text(json.dumps({
        "vcdd_version": __version__, "git_hash": "test", "created_at_iso": "now",
        "tool_versions": {"rdkit": "test"}, "config": {}, "inputs": {},
        "stages": {}, "licenses": [], "warnings": [],
    }))
    return run


def test_figures_render_svg_and_png(fake_run):
    from vcdd.report.figures import build_all

    made = build_all(fake_run)
    assert made, "no figures were produced"
    fig_dir = fake_run / "report" / "figures"
    svgs = sorted(p.name for p in fig_dir.glob("*.svg"))
    assert len(svgs) >= 4
    # the stage-2 figures are Boltz figures now, not rescorer figures
    assert "stage1_vs_boltz.svg" in svgs
    assert "boltz_confidence_distribution.svg" in svgs
    assert not [s for s in svgs if "rescor" in s or "vinardo" in s]
    for svg in fig_dir.glob("*.svg"):
        assert svg.stat().st_size > 1000
        assert svg.with_suffix(".png").exists()
        # SVG text must stay selectable, not converted to paths
        assert "font-family" in svg.read_text(errors="ignore")


def test_report_html_is_self_describing(fake_run):
    from vcdd.report.html import build_report

    path = build_report(fake_run)
    assert path.exists()
    html = path.read_text()
    assert len(html) > 20_000
    for token in ("Stage 1", "Stage 2", "Stage 3", "Stage 4",
                  "does not tell you", "Methods"):
        assert token in html
    # the caveats section must survive templating
    assert "docking" in html.lower()


def test_report_states_that_boltz_does_not_rescore_the_pose(fake_run):
    """The single most misreadable number in the run deserves a caveat in the HTML.

    ``binding_confidence`` comes from a *re-predicted* complex, not from the
    stage-1 pose, so a reader must not treat it as a better docking score.
    """
    from vcdd.report.html import build_report

    html = build_report(fake_run).read_text().lower()
    assert "boltz" in html
    assert "does not rescore" in html or "not a rescor" in html
    assert "carried forward" in html
    # a test-mode run must be flagged as synthetic, loudly
    assert "test mode" in html or "test-mode" in html
