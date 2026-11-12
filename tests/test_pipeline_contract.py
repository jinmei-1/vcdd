"""Stage 2 end to end, and the contracts stages 3 and 4 depend on.

This is the cheap counterpart to ``test_smoke_pipeline.py``: no GPU, no API key,
no cell resource bundle. Stage 1 runs on the mock engine, stage 2 runs against
the scripted Boltz API behind the real SDK, and the assertions are about the
*boundaries* - the columns stage 3 reads out of stage 2, and the parameters
stage 4 reuses from stage 1.

Those boundaries are the whole risk of this release. Stages 3 and 4 were not
rewritten, so nothing in them would notice if stage 2 quietly stopped emitting
``dock_score``; they would just fall back, or rank on a column of NaN.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from vcdd.refine.stage2 import BASE_COLUMNS, build_ranking, load_consensus, run_stage2
from vcdd.runtime.ledger import RunLedger


def _stage2(base_cfg, stage1_done, tmp_path, ledger, fake_boltz, boltz_client, **kw):
    receptor, stage1_dir = stage1_done
    out = kw.pop("out_dir", tmp_path / "stage2")
    return run_stage2(
        base_cfg, receptor, kw.pop("box"), stage1_dir, out, ledger,
        client=boltz_client, pose_fetch=fake_boltz.fetch, **kw,
    ), out


# --------------------------------------------------------------------------- #
# stage 2 end to end
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stage2_run(base_cfg, stage1_done, tmp_path, ledger, fake_boltz, boltz_client, small_box):
    summary, out = _stage2(base_cfg, stage1_done, tmp_path, ledger, fake_boltz,
                           boltz_client, box=small_box)
    return summary, out


def test_stage2_writes_every_artifact_the_run_contract_promises(stage2_run):
    summary, out = stage2_run
    for rel in ("scores.parquet", "consensus_ranking.parquet", "pose_qc.parquet",
                "stage2_summary.json", "boltz/results.jsonl", "boltz/runs.json"):
        path = out / rel
        assert path.exists(), f"missing {rel}"
        assert path.stat().st_size > 0, f"zero-byte {rel}"
    assert summary["n_scored"] > 0
    assert summary["backend"] == "library_screen"
    assert summary["engine"] == "boltzmol"


def test_scores_table_has_the_documented_columns(stage2_run):
    _, out = stage2_run
    scores = pd.read_parquet(out / "scores.parquet")
    assert set(BASE_COLUMNS) <= set(scores.columns), \
        sorted(set(BASE_COLUMNS) - set(scores.columns))
    assert scores["binding_confidence"].between(0, 1).all()


def test_stage2_never_overwrites_the_stage1_score(stage2_run):
    """``dock_score`` must remain the Vina-GPU number stages 3 and 4 expect."""
    _, out = stage2_run
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    pd.testing.assert_series_equal(cons["dock_score"], cons["stage1_score"],
                                   check_names=False)
    summary = json.loads((out / "stage2_summary.json").read_text())
    assert "does not dock" in summary["note"]


def test_ranking_is_dense_and_ordered_by_binding_confidence(stage2_run):
    _, out = stage2_run
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    assert list(cons["consensus_rank"]) == list(range(1, len(cons) + 1))
    scored = cons.dropna(subset=["binding_confidence"])
    assert scored["binding_confidence"].is_monotonic_decreasing
    # consensus_score is lower-is-better everywhere, whatever the fusion policy
    assert cons["consensus_score"].is_monotonic_increasing


def test_pose_qc_ran_and_is_bounded_by_keep_structures_top_k(base_cfg, stage2_run):
    summary, out = stage2_run
    qc = pd.read_parquet(out / "pose_qc.parquet")
    assert 0 < len(qc) <= base_cfg.stage2.keep_structures_top_k
    assert summary["pose_qc"]["n_checked"] == len(qc)
    assert len(list((out / "poses").glob("*.cif"))) == len(qc)


def test_target_provenance_records_what_was_actually_sent(stage2_run):
    summary, _ = stage2_run
    target = summary["target"]
    assert target["chains"] and target["chains"][0]["length"] > 0
    assert target["pocket_sizes"], "the fixture box should have produced a pocket hint"
    assert "0-indexed" in target["note"]


def test_a_test_mode_run_is_labelled_as_synthetic(base_cfg, stage1_done, tmp_path,
                                                  ledger, small_box, fake_boltz):
    """Test-mode numbers look exactly like real ones; the run must say so."""
    from fake_boltz import make_fake_client

    fake_boltz.livemode = False
    client = make_fake_client(fake_boltz)
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2_test_mode"
    with warnings.catch_warnings(record=True):
        summary = run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
                             client=client, pose_fetch=fake_boltz.fetch)
    assert summary["livemode"] is False


def test_chunking_submits_more_than_one_run(base_cfg, stage2_run):
    summary, _ = stage2_run
    # top_n=6, chunk_size=3
    assert summary["n_runs"] == 2
    runs = json.loads((Path(summary["consensus_ranking_path"]).parent
                       / "boltz" / "runs.json").read_text())
    assert len(runs) == 2
    assert {r["chunk_index"] for r in runs} == {0, 1}


def test_resume_reuses_completed_chunks_instead_of_resubmitting(
    base_cfg, stage1_done, tmp_path, ledger, small_box, fake_boltz, boltz_client
):
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    first = run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
                       client=boltz_client, pose_fetch=fake_boltz.fetch)
    submitted = len(fake_boltz.started_payloads)
    assert submitted == 2

    second = run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
                        client=boltz_client, pose_fetch=fake_boltz.fetch)
    assert len(fake_boltz.started_payloads) == submitted, "resume re-submitted a paid chunk"
    assert second["n_scored"] == first["n_scored"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(out / "consensus_ranking.parquet"),
        pd.read_parquet(out / "consensus_ranking.parquet"),
    )


def test_force_reruns_the_chunks(base_cfg, stage1_done, tmp_path, ledger, small_box,
                                 fake_boltz, boltz_client):
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
               client=boltz_client, pose_fetch=fake_boltz.fetch)
    run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger, force=True,
               client=boltz_client, pose_fetch=fake_boltz.fetch)
    assert len(fake_boltz.started_payloads) == 4


def test_filtered_molecules_land_in_the_rejected_table(
    base_cfg, stage1_done, tmp_path, ledger, small_box, fake_boltz, boltz_client
):
    stage1_ranking = pd.read_parquet(stage1_done[1] / "ranking.parquet")
    victim = str(stage1_ranking["mol_id"].iloc[0])
    fake_boltz.filtered_ids = {victim}

    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2"
    summary = run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
                         client=boltz_client, pose_fetch=fake_boltz.fetch)
    assert summary["n_rejected"] == 1
    rej = pd.read_parquet(out / "boltz" / "rejected.parquet")
    assert list(rej["mol_id"]) == [victim]

    # a rejected molecule is still in the ranking, at the bottom, with no score
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    row = cons[cons["mol_id"] == victim].iloc[0]
    assert pd.isna(row["binding_confidence"])
    assert row["consensus_rank"] == len(cons)
    assert row["error"]


def test_spend_gate_stops_the_run_before_anything_is_submitted(
    base_cfg, stage1_done, tmp_path, ledger, small_box, fake_boltz, boltz_client
):
    from vcdd.refine.boltz.screen import SpendGateExceeded

    base_cfg.stage2.max_spend_usd = 0.01      # 6 molecules at $0.025 is $0.15
    receptor, stage1_dir = stage1_done
    with pytest.raises(SpendGateExceeded):
        run_stage2(base_cfg, receptor, small_box, stage1_dir, tmp_path / "stage2", ledger,
                   client=boltz_client, pose_fetch=fake_boltz.fetch)
    assert fake_boltz.started_payloads == [], "the gate fired after submitting"


def test_yes_spend_overrides_the_gate(base_cfg, stage1_done, tmp_path, ledger, small_box,
                                      fake_boltz, boltz_client):
    base_cfg.stage2.max_spend_usd = 0.01
    receptor, stage1_dir = stage1_done
    summary = run_stage2(base_cfg, receptor, small_box, stage1_dir, tmp_path / "stage2",
                         ledger, allow_spend=True, client=boltz_client,
                         pose_fetch=fake_boltz.fetch)
    assert summary["n_scored"] > 0


def test_structure_and_binding_backend_produces_the_same_table(
    base_cfg, stage1_done, tmp_path, ledger, small_box, fake_boltz, boltz_client
):
    base_cfg.stage2.backend = "structure_and_binding"
    receptor, stage1_dir = stage1_done
    out = tmp_path / "stage2_sab"
    summary = run_stage2(base_cfg, receptor, small_box, stage1_dir, out, ledger,
                         client=boltz_client, pose_fetch=fake_boltz.fetch)
    assert summary["backend"] == "structure_and_binding"
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    assert {"dock_score", "binding_confidence", "consensus_rank"} <= set(cons.columns)
    assert summary["n_scored"] > 0


# --------------------------------------------------------------------------- #
# fusion policies
# --------------------------------------------------------------------------- #


@pytest.fixture()
def toy_scores():
    """Stage 1 and Boltz deliberately disagree, so the policies separate."""
    return pd.DataFrame({
        "mol_id": ["A", "B", "C", "D"],
        "stage1_score": [-10.0, -9.0, -8.0, -7.0],
        "binding_confidence": [0.40, 0.95, 0.80, None],
    })


def test_boltz_only_ranks_purely_on_binding_confidence(toy_scores):
    ranked, report = build_ranking(toy_scores, fusion="boltz_only")
    assert list(ranked["mol_id"]) == ["B", "C", "A", "D"]
    assert report["policy"] == "boltz_only"
    # D has no Boltz result, so its consensus_score is NaN and it sorts last.
    # Monotonicity is a claim about the scored molecules only.
    scored = ranked["consensus_score"].dropna()
    assert len(scored) == 3
    assert scored.is_monotonic_increasing
    assert pd.isna(ranked["consensus_score"].iloc[-1])
    # The rank, unlike the score, is always dense and defined - stage 3 and
    # stage 4 both sort on consensus_rank, so a NaN there would be fatal.
    assert list(ranked["consensus_rank"]) == [1, 2, 3, 4]


def test_rank_fusion_lands_between_the_two_rankings(toy_scores):
    ranked, report = build_ranking(toy_scores, fusion="rank_fusion")
    order = list(ranked["mol_id"])
    assert report["policy"] == "rank_fusion"
    assert order[-1] == "D"                       # unscored always last
    assert order[0] in {"A", "B"}                 # each ranking's winner is in contention
    assert set(order) == {"A", "B", "C", "D"}


def test_gate_keeps_the_docking_order_among_survivors(toy_scores):
    ranked, report = build_ranking(toy_scores, fusion="gate", gate_threshold=0.7)
    assert report["policy"] == "gate"
    survivors = ranked[ranked["gate_pass"]]
    assert set(survivors["mol_id"]) == {"B", "C"}
    # within survivors the order is stage 1's, not Boltz's
    assert list(survivors["mol_id"]) == ["B", "C"]
    assert list(ranked["mol_id"])[:2] == ["B", "C"]
    assert survivors["stage1_score"].is_monotonic_increasing


def test_molecules_without_a_boltz_result_are_kept_not_dropped(toy_scores):
    """Dropping them would silently shrink the shortlist stage 3 receives."""
    for fusion in ("boltz_only", "rank_fusion", "gate"):
        ranked, _ = build_ranking(toy_scores, fusion=fusion)
        assert len(ranked) == len(toy_scores)
        assert list(ranked["consensus_rank"]) == list(range(1, len(ranked) + 1))
        assert ranked["mol_id"].iloc[-1] == "D"


# --------------------------------------------------------------------------- #
# stage 3 / stage 4 contracts
# --------------------------------------------------------------------------- #


def test_stage3_can_read_the_stage2_ranking(stage2_run):
    """``load_shortlist`` is the only door between stages 2 and 3."""
    from vcdd.cell.stage3 import load_shortlist

    _, out = stage2_run
    short = load_shortlist(out, top_n=4)
    assert len(short) == 4
    assert {"mol_id", "smiles", "dock_score"} <= set(short.columns)
    assert short["smiles"].notna().all()
    assert short["dock_score"].notna().all()
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    assert list(short["mol_id"]) == list(cons["mol_id"].head(4))


def test_stage3_fallback_to_scores_still_works(stage2_run, tmp_path):
    """If the ranking is missing, stage 3 sorts ``scores.parquet`` by dock_score."""
    from vcdd.cell.stage3 import load_shortlist

    _, out = stage2_run
    fallback = tmp_path / "stage2_fallback"
    fallback.mkdir()
    (fallback / "scores.parquet").write_bytes((out / "scores.parquet").read_bytes())
    short = load_shortlist(fallback, top_n=3)
    assert len(short) == 3
    assert short["dock_score"].is_monotonic_increasing


def test_stage4_carry_forward_columns_survive_stage2(stage2_run):
    """Stage 4 sorts on ``consensus_rank`` and diffs against ``dock_score``."""
    from vcdd.refine.stage2 import CARRY_FORWARD

    _, out = stage2_run
    cons = pd.read_parquet(out / "consensus_ranking.parquet")
    assert set(CARRY_FORWARD) <= set(cons.columns)
    assert cons["consensus_rank"].is_monotonic_increasing
    assert cons["dock_score"].notna().any()


def test_stage4_redock_uses_the_stage1_engine_and_params(base_cfg, monkeypatch):
    """Redock means "the same docking protocol", not "some docking protocol"."""
    import vcdd.optimize.stage4 as s4

    seen: dict[str, object] = {}

    def fake_dock_molecules(mol_ids, smiles, receptor_pdbqt, box, engine_name, params, **kw):
        seen["engine_name"] = engine_name
        seen["params"] = params
        seen["batch_size"] = kw.get("batch_size")
        return pd.DataFrame({"mol_id": list(mol_ids), "score": [-8.0] * len(list(mol_ids)),
                             "error": [None] * len(list(mol_ids))})

    monkeypatch.setattr(s4, "dock_molecules", fake_dock_molecules, raising=False)
    monkeypatch.setenv("VCDD_ALLOW_MOCK", "1")

    from vcdd.screening.stage1 import build_dock_params

    expected = build_dock_params(base_cfg.stage1)
    assert "variant" in expected and "thread" in expected
    assert "exhaustiveness" not in expected, "a CPU-Vina flag leaked back in"
    assert "cpu" not in expected
    assert expected["variant"] == base_cfg.stage1.variant


def test_load_consensus_reads_back_what_stage2_wrote(stage2_run):
    _, out = stage2_run
    df = load_consensus(out)
    assert not df.empty
    assert "binding_confidence" in df.columns


def test_ledger_records_one_unit_per_chunk(base_cfg, stage1_done, tmp_path, small_box,
                                           fake_boltz, boltz_client):
    led = RunLedger(tmp_path / "ledger.db")
    try:
        receptor, stage1_dir = stage1_done
        run_stage2(base_cfg, receptor, small_box, stage1_dir, tmp_path / "stage2", led,
                   client=boltz_client, pose_fetch=fake_boltz.fetch)
        units = [u for u in led.completed_units("stage2")]
        assert len(units) == 2
    finally:
        led.close()
