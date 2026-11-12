"""Docking engines, ligand prep and stage 1 (brute force + active learning).

Everything here runs on the ``mock`` engine unless explicitly marked, so the
whole module finishes in seconds and needs no docking binary. Vina-GPU argv
construction lives in ``test_vina_gpu.py``; stage 2 lives in
``test_boltz_stage2.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from vcdd.docking.base import (
    DockParams,
    DockResult,
    engine_names,
    get_engine,
    select_engine,
)
from vcdd.docking.ligprep import prepare_ligand
from vcdd.receptor.box import Box
from vcdd.refine.consensus import build_consensus, drop_duplicate_scores
from vcdd.screening.stage1 import collect_ranking, dock_molecules, run_stage1

from _helpers import SMILES, requires_vina


# ------------------------------------------------------------------ registry
def test_mock_engine_is_always_available():
    engine = get_engine("mock")
    assert engine.available() is True
    assert "mock" in engine_names()


def test_unknown_engine_raises_with_a_useful_message():
    with pytest.raises(KeyError):
        get_engine("definitely_not_an_engine")


def test_only_vina_gpu_and_mock_are_registered():
    """0.2.0 removed the CPU engines; nothing should silently reintroduce one."""
    assert set(engine_names()) == {"vina_gpu", "mock"}


def test_select_engine_falls_back_when_requested_is_missing():
    """A missing engine must degrade, not crash the run.

    There is no GPU in CI, so ``vina_gpu`` is genuinely unavailable here and the
    fallback is the mock engine.
    """
    engine = select_engine("vina_gpu", selection="auto", prefer_fast=True)
    assert engine.available()


def test_select_engine_fixed_raises_when_unavailable(monkeypatch):
    monkeypatch.delenv("VCDD_ASSUME_GPU", raising=False)
    with pytest.raises(RuntimeError):
        select_engine("vina_gpu", selection="fixed")


def test_mock_engine_is_deterministic(small_box, tmp_path):
    engine = get_engine("mock")
    lig = prepare_ligand("X", SMILES["aspirin"], n_conformers=1, seed=42)
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    params = DockParams(num_modes=3)
    a = engine.dock(lig.pdbqt, str(receptor), small_box, params, mol_id="X")
    b = engine.dock(lig.pdbqt, str(receptor), small_box, params, mol_id="X")
    assert a.score == b.score
    assert len(a.all_scores) == 3
    assert a.all_scores == sorted(a.all_scores)   # best pose first


def test_mock_engine_labels_itself_as_not_a_docking_program():
    caps = get_engine("mock").capabilities()
    assert "NOT A DOCKING PROGRAM" in caps.notes


def test_mock_engine_reports_empty_ligand_as_error(small_box, tmp_path):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = get_engine("mock").dock("", str(receptor), small_box, DockParams(), mol_id="Y")
    assert out.error == "empty_ligand" and out.score is None


# ------------------------------------------------------------------ ligprep
def test_prepare_ligand_returns_pdbqt_with_coordinates():
    lig = prepare_ligand("L1", SMILES["ibuprofen"], n_conformers=1, seed=42)
    assert lig.ok and lig.pdbqt
    assert any(l.startswith(("ATOM", "HETATM")) for l in lig.pdbqt.splitlines())
    assert lig.n_rotatable >= 3


def test_prepare_ligand_variants_expand_tautomers_and_conformers():
    from vcdd.docking.ligprep import prepare_ligand_variants

    variants = prepare_ligand_variants("L2", SMILES["imatinib_core"], n_conformers=2,
                                       max_tautomers=2, seed=42)
    assert len(variants) >= 2
    assert all(v.ok for v in variants)
    assert len({v.pdbqt for v in variants}) == len(variants)


def test_prepare_ligand_is_seed_reproducible():
    a = prepare_ligand("L3", SMILES["caffeine"], n_conformers=2, seed=7)
    b = prepare_ligand("L3", SMILES["caffeine"], n_conformers=2, seed=7)
    assert a.pdbqt == b.pdbqt


def test_prepare_ligand_reports_bad_smiles_without_raising():
    lig = prepare_ligand("L4", "not a molecule", n_conformers=1)
    assert not lig.ok and lig.error


def test_pdbqt_roundtrips_back_to_a_molecule():
    """Stage 2 pose QC depends on being able to read poses back."""
    from rdkit import Chem
    from vcdd.docking.ligprep import pdbqt_to_mol

    lig = prepare_ligand("L5", SMILES["aspirin"], n_conformers=1, seed=42)
    mol = pdbqt_to_mol(lig.pdbqt, template_smiles=SMILES["aspirin"])
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == Chem.MolFromSmiles(SMILES["aspirin"]).GetNumHeavyAtoms()


# --------------------------------------------------------------------- box
def test_box_roundtrips_through_json(tmp_path: Path, small_box: Box):
    path = tmp_path / "box.json"
    small_box.save(path)
    again = Box.load(path)
    assert tuple(again.center) == pytest.approx(tuple(small_box.center))
    assert tuple(again.size) == pytest.approx(tuple(small_box.size))


def test_box_contains_point(small_box: Box):
    assert small_box.contains(small_box.center)
    far = tuple(c + 100 for c in small_box.center)
    assert not small_box.contains(far)


# ------------------------------------------------------------------ stage 1
def test_dock_molecules_returns_one_row_per_input(tmp_path: Path, small_box):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    ids = ["A", "B", "C"]
    smis = [SMILES["aspirin"], SMILES["benzene"], SMILES["caffeine"]]
    df = dock_molecules(ids, smis, receptor, small_box, "mock", {}, n_workers=1)
    assert list(df["mol_id"]) == ids
    assert df["score"].notna().all()


def test_dock_molecules_empty_input_returns_empty_frame(tmp_path: Path, small_box):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    df = dock_molecules([], [], receptor, small_box, "mock", {}, n_workers=1)
    assert df.empty


def test_collect_ranking_sorts_ascending_and_breaks_ties_on_mol_id(tmp_path: Path):
    scores = tmp_path / "scores"
    scores.mkdir()
    pd.DataFrame({
        "mol_id": ["m2", "m1", "m3"], "smiles": ["C", "C", "C"],
        "score": [-7.0, -7.0, None], "engine": "mock", "seconds": 0.1,
        "n_rotatable": 0, "error": None, "shard": 0,
    }).to_parquet(scores / "shard_00000.parquet", index=False)
    out = collect_ranking(scores, tmp_path / "ranking.parquet")
    assert list(out["mol_id"]) == ["m1", "m2"]      # failures dropped, tie broken on id
    assert list(out["rank"]) == [1, 2]


def test_collect_ranking_pattern_selects_active_learning_files(tmp_path: Path):
    """Regression: the AL path checkpoints ``al_*.parquet``, not ``shard_*``."""
    scores = tmp_path / "scores"
    scores.mkdir()
    frame = pd.DataFrame({
        "mol_id": ["m1"], "smiles": ["C"], "score": [-8.0], "engine": "mock",
        "seconds": 0.1, "n_rotatable": 0, "error": None, "shard": -1,
    })
    frame.to_parquet(scores / "al_iter00.parquet", index=False)
    assert collect_ranking(scores, tmp_path / "a.parquet").empty
    assert len(collect_ranking(scores, tmp_path / "b.parquet", pattern="al_*.parquet")) == 1


def _stage1(cfg, receptor, box, library, out, ledger, force=False):
    return run_stage1(cfg, receptor, box, library, out, ledger, force=force)


def test_stage1_scores_every_molecule(base_cfg, prepared_library, tmp_path, small_box, ledger):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    summary = _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    ranking = pd.read_parquet(out / "ranking.parquet")
    assert summary["n_scored"] == len(ranking) > 0
    assert ranking["score"].is_monotonic_increasing
    assert ranking["mol_id"].is_unique
    assert summary["best_score"] == pytest.approx(ranking["score"].min())


def test_stage1_pilot_measures_throughput_and_projects_an_eta(
    base_cfg, prepared_library, tmp_path, small_box, ledger
):
    """The 1M-compound ETA must be a measurement on this machine, not a prior."""
    base_cfg.stage1.pilot_size = 4
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    run_stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)

    tp = json.loads((out / "throughput.json").read_text())
    assert tp["pilot_size"] == 4
    assert tp["ligands_per_hour"] > 0
    assert tp["seconds_per_ligand_wall"] > 0
    assert tp["total_molecules"] == 16
    assert tp["projected_hours"] == pytest.approx(
        16 * tp["seconds_per_ligand_wall"] / 3600.0, abs=1e-3
    )
    assert "do not reuse across different hardware" in tp["note"]
    # 0.2.0: the CPU pool only prepares ligands for a batch-native GPU engine,
    # so a per-core figure would be measuring the wrong resource. It is gone,
    # and the batch size it was replaced by is recorded instead.
    assert "seconds_per_ligand_core" not in tp
    assert "projected_core_hours" not in tp
    assert "batch_size" in tp


def _pilot_jobs(receptor, box, n=4):
    from vcdd.screening.stage1 import _DockJob

    return [
        _DockJob(
            mol_id=f"P{i}",
            smiles=SMILES["benzene"],
            receptor_pdbqt=str(receptor),
            box_center=tuple(box.center),
            box_size=tuple(box.size),
            engine_name="mock",
            params={},
            n_conformers=1,
            minimize=True,
            seed=42,
        )
        for i in range(n)
    ]


def test_stage1_pilot_warns_when_it_is_smaller_than_one_gpu_batch(tmp_path, small_box):
    """A pilot below one batch amortises kernel setup over too few ligands.

    Without this warning a user reads ``projected_hours`` off a run that paid
    the whole OpenCL/grid setup for four molecules and concludes the GPU engine
    is slow. In that regime the projection is an upper bound and has to say so.
    """
    from vcdd.screening.stage1 import measure_throughput

    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    tp = measure_throughput(
        _pilot_jobs(receptor, small_box, 4),
        n_workers=1,
        total_molecules=16,
        out_path=tmp_path / "throughput.json",
        batch_size=64,
    )
    assert tp["batch_size"] == 64
    assert "upper bound" in tp["warning"]
    assert "64" in tp["warning"]


def test_stage1_pilot_is_silent_when_the_pilot_covers_a_full_batch(tmp_path, small_box):
    """The warning must not cry wolf on a properly sized pilot."""
    from vcdd.screening.stage1 import measure_throughput

    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    tp = measure_throughput(
        _pilot_jobs(receptor, small_box, 4),
        n_workers=1,
        total_molecules=16,
        out_path=tmp_path / "throughput.json",
        batch_size=2,
    )
    assert "warning" not in tp


def test_stage1_resume_reproduces_the_ranking_bit_for_bit(
    base_cfg, prepared_library, tmp_path, small_box, ledger
):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    first = pd.read_parquet(out / "ranking.parquet")

    # Delete one shard's scores and resume: only that shard should be re-docked.
    shard0 = out / "scores" / "shard_00000.parquet"
    shard0.unlink()
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    second = pd.read_parquet(out / "ranking.parquet")
    # Wall-clock columns legitimately differ; the *ranking* must not.
    cols = ["mol_id", "smiles", "score", "rank"]
    pd.testing.assert_frame_equal(first[cols], second[cols])


def test_stage1_force_reruns_everything(base_cfg, prepared_library, tmp_path, small_box, ledger):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    mtime = (out / "scores" / "shard_00000.parquet").stat().st_mtime_ns
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger, force=True)
    assert (out / "scores" / "shard_00000.parquet").stat().st_mtime_ns != mtime


def test_stage1_protocol_change_invalidates_the_cache(
    base_cfg, prepared_library, tmp_path, small_box, ledger
):
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    mtime = (out / "scores" / "shard_00000.parquet").stat().st_mtime_ns

    base_cfg.stage1.search_depth = 16        # different protocol -> different param hash
    _stage1(base_cfg, receptor, small_box, prepared_library, out, ledger)
    assert (out / "scores" / "shard_00000.parquet").stat().st_mtime_ns != mtime


# ------------------------------------------------- stage 1 active learning
def test_active_learning_docks_a_subset_and_reports_the_gap(
    base_cfg, tmp_path, small_box, ledger, library_sdf
):
    """The AL path must (a) produce a real ranking, (b) name what it never docked."""
    from vcdd.config.loader import load_config
    from vcdd.io.library import load_shard_index, prepare_library
    from vcdd.screening.active_learning import run_active_learning

    cfg = load_config(None, overrides={
        "name": "al", "out_dir": str(tmp_path / "run"), "n_workers": 4,
        "library": {"path": str(library_sdf), "shard_size": 60, "max_molecules": 120},
        "stage1": {"engine": "mock", "surrogate_al": True,
                   "al_iterations": 2, "al_initial_fraction": 0.15,
                   "al_select_fraction": 0.10, "al_holdout_fraction": 0.10},
    })
    lib = tmp_path / "lib"
    prepare_library(cfg.library.path, lib, cfg.library)
    index = load_shard_index(lib)

    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    out = tmp_path / "stage1"
    common = {
        "receptor_pdbqt": str(receptor),
        "box_center": tuple(small_box.center),
        "box_size": tuple(small_box.size),
        "engine_name": "mock",
        "params": {"num_modes": 1, "seed": 42, "timeout_s": 60},
        "n_conformers": 1,
        "minimize": False,
        "seed": 42,
    }
    summary = run_active_learning(cfg, common, index, out, ledger, 4, "mock", "phash")

    ranking = pd.read_parquet(out / "ranking.parquet")
    assert len(ranking) > 0, "AL produced an empty ranking (collect_ranking glob mismatch)"
    assert ranking["score"].is_monotonic_increasing
    assert ranking["score"].notna().all(), "every reported score must be a real docking score"

    report = json.loads((out / "active_learning.json").read_text())
    assert report["enabled"] is True
    assert 0 < report["n_docked"] < report["n_total"], "AL should dock a strict subset"
    assert report["n_never_docked"] == report["n_total"] - report["n_docked"]
    assert report["speedup_vs_brute_force"] > 1.0
    assert "not_docked.parquet" in report["caveat"]

    never = pd.read_parquet(out / "not_docked.parquet")
    assert len(never) == report["n_never_docked"]
    assert never["surrogate_predicted_score"].notna().any(), \
        "molecules skipped by the surrogate should carry the prediction that skipped them"


def test_active_learning_measures_surrogate_quality_on_a_holdout(
    base_cfg, tmp_path, small_box, ledger, library_sdf
):
    from vcdd.config.loader import load_config
    from vcdd.io.library import load_shard_index, prepare_library
    from vcdd.screening.active_learning import run_active_learning

    cfg = load_config(None, overrides={
        "name": "al2", "out_dir": str(tmp_path / "run"), "n_workers": 4,
        "library": {"path": str(library_sdf), "shard_size": 60, "max_molecules": 120},
        "stage1": {"engine": "mock", "surrogate_al": True,
                   "al_iterations": 1, "al_initial_fraction": 0.2,
                   "al_select_fraction": 0.1, "al_holdout_fraction": 0.15},
    })
    lib = tmp_path / "lib"
    prepare_library(cfg.library.path, lib, cfg.library)
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    common = {
        "receptor_pdbqt": str(receptor), "box_center": tuple(small_box.center),
        "box_size": tuple(small_box.size), "engine_name": "mock",
        "params": {"num_modes": 1, "seed": 42, "timeout_s": 60},
        "n_conformers": 1, "minimize": False, "seed": 42,
    }
    summary = run_active_learning(cfg, common, load_shard_index(lib), tmp_path / "s1",
                                  ledger, 4, "mock", "phash")
    history = summary["active_learning"]["history"]
    guided = [h for h in history if h["kind"] == "surrogate_selected"]
    assert guided, "at least one surrogate-guided iteration should have run"
    assert "holdout_spearman" in guided[0]
    assert -1.0 <= guided[0]["holdout_spearman"] <= 1.0
    assert 0.0 <= guided[0]["holdout_recall_top1pct"] <= 1.0


# ----------------------------------------------------------------- consensus
def test_drop_duplicate_scores_removes_a_redundant_column():
    df = pd.DataFrame({
        "mol_id": list("abcde"),
        "norm_dock": [0.1, 0.2, 0.3, 0.4, 0.5],
        "norm_vina": [0.1, 0.2, 0.3, 0.4, 0.5],       # identical -> redundant
        "norm_vinardo": [0.5, 0.1, 0.9, 0.2, 0.3],
    })
    kept, dropped = drop_duplicate_scores(df, ["norm_dock", "norm_vina", "norm_vinardo"], tol=0.02)
    assert "norm_vina" in dropped and "norm_dock" in kept and "norm_vinardo" in kept


def test_build_consensus_ranks_and_flags_disagreement():
    df = pd.DataFrame({
        "mol_id": [f"m{i}" for i in range(6)],
        "smiles": ["C"] * 6,
        "dock_score": [-9.0, -8.5, -8.0, -7.5, -7.0, -6.5],
        "rescore_vinardo": [-6.0, -9.0, -5.0, -8.5, -4.0, -8.0],   # deliberately discordant
    })
    out, info = build_consensus(df, ["dock_score", "rescore_vinardo"], method="both")
    assert out["consensus_rank"].is_monotonic_increasing
    assert out["consensus_rank"].tolist() == list(range(1, 7))
    assert "rank_spread" in out.columns
    assert out["low_confidence_consensus"].any(), "discordant scorers should flag low confidence"


def test_build_consensus_with_a_single_scorer_is_the_identity_ranking():
    df = pd.DataFrame({
        "mol_id": [f"m{i}" for i in range(4)], "smiles": ["C"] * 4,
        "dock_score": [-9.0, -8.0, -7.0, -6.0],
    })
    out, info = build_consensus(df, ["dock_score"], method="zscore")
    assert out["mol_id"].tolist() == ["m0", "m1", "m2", "m3"]


# --------------------------------------------------------- the engine roster
def test_only_vina_gpu_and_the_test_mock_are_registered():
    """0.2.0 is a one-engine release. CPU Vina, smina, gnina and QVina-CPU are gone.

    0.1.0 registered a family of ``VinaLikeEngine`` subclasses that all emitted
    ``--ligand/--out/--exhaustiveness/--cpu/--seed``. Vina-GPU 2.1 accepts none
    of those flags, so the two 0.1.0 "GPU" engines could never have returned a
    score. Rather than keep a CPU fallback that silently changes the scoring
    function under the user, the roster is now exactly one production engine.
    """
    from vcdd.docking.base import engine_names

    assert set(engine_names()) == {"vina_gpu", "mock"}
    with pytest.raises(KeyError, match="unknown docking engine 'vina'"):
        get_engine("vina")


def test_vina_gpu_refuses_score_only_instead_of_faking_it():
    """Vina-GPU 2.1 has no ``--score_only``. Saying so beats returning a number.

    Every rescoring path in 0.1.0 went through ``score_only``. Those are gone in
    0.2.0 (Boltz is the only stage-2 evidence), so nothing in the package calls
    it - but an engine that silently re-docked a pose and called the result a
    score-only energy would be worse than one that raises.
    """
    engine = get_engine("vina_gpu")
    with pytest.raises(NotImplementedError, match="score_only"):
        engine.score_only("REMARK\n", "r.pdbqt", Box((0, 0, 0), (20, 20, 20)), DockParams())


def test_vina_gpu_capabilities_declare_the_batch_and_determinism_contract():
    """Stage 1 dispatches on capabilities, never on the engine name."""
    caps = get_engine("vina_gpu").capabilities()
    assert caps.batch_native is True
    assert caps.preferred_batch_size > 1
    assert caps.needs_gpu is True
    # Vina-GPU 2.1 exposes no --seed, so a run is not bit-reproducible. The
    # manifest reports this; the capability is where the claim originates.
    assert caps.deterministic is False
    assert caps.max_box_edge is not None


@requires_vina
def test_the_cpu_vina_binary_on_this_machine_is_not_wired_into_the_package():
    """A stray ``vina`` on PATH must not resurrect a second scoring function.

    Kept as a protocol-drift guard: the binary is still installed in this image
    and in most user environments, and the failure mode we are protecting
    against is a future patch quietly re-registering it as a fallback when no
    GPU is found. Mixing CPU-Vina and Vina-GPU scores in one ranking would be
    an invisible scoring-function change mid-run.
    """
    from vcdd.docking.base import engine_names

    assert shutil.which("vina")
    assert "vina" not in engine_names()
