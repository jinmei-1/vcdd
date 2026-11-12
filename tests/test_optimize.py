"""Stage 4 tests: transforms, guardrails, the constrained GA, Pareto and SAR."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from _helpers import SMILES
from vcdd.config.schema import CellConfig, GuardrailConfig, ObjectiveConfig, Stage4Config
from vcdd.optimize import guardrails as gr
from vcdd.optimize import pareto, sar
from vcdd.optimize.generators import base as genbase
from vcdd.optimize.generators.constrained_ga import ConstrainedGA
from vcdd.optimize.generators.enumerative import TRANSFORMS, RGroupGenerator

# --------------------------------------------------------------------------
# reaction SMARTS transforms
# --------------------------------------------------------------------------


def test_every_transform_compiles_and_is_named():
    from vcdd.optimize.generators.enumerative import compiled_transforms

    compiled = compiled_transforms()
    assert len(compiled) == len(TRANSFORMS) == 27
    for rxn, name in compiled:
        assert isinstance(name, str) and name
        assert rxn.GetNumReactantTemplates() == 1
        assert rxn.GetNumProductTemplates() >= 1


@pytest.mark.parametrize(
    "smiles", [SMILES["aspirin"], SMILES["ibuprofen"], SMILES["aniline"], SMILES["phenol"]]
)
def test_transforms_yield_sanitisable_products(smiles):
    from vcdd.optimize.generators.enumerative import apply_transform, compiled_transforms

    parent = Chem.MolFromSmiles(smiles)
    n_products = 0
    for rxn, _name in compiled_transforms():
        for prod in apply_transform(parent, rxn):
            assert Chem.MolFromSmiles(prod) is not None, prod
            n_products += 1
    assert n_products > 0, f"no transform fired on {smiles}"


def test_primary_amine_rule_skips_amide_nitrogens():
    """`primary amine -> N-methyl amine` must not fire on a primary amide N-H."""
    from vcdd.optimize.generators.enumerative import apply_transform, compiled_transforms

    rxn = next(r for r, n in compiled_transforms()
               if n == "primary amine -> N-methyl amine")
    # a genuine primary amine reacts
    assert apply_transform(Chem.MolFromSmiles(SMILES["aniline"]), rxn)
    # a primary amide (N bonded to C=O) must be invisible to this rule
    for amide in ("NC(=O)c1ccccc1", "CC(=O)N", "CC(=O)NCC"):
        assert apply_transform(Chem.MolFromSmiles(amide), rxn) == [], amide
    # in glycinamide only the amine N is methylated; the amide N-H2 is preserved
    prods = apply_transform(Chem.MolFromSmiles("NC(=O)CN"), rxn)
    assert prods
    primary_amide = Chem.MolFromSmarts("[NX3H2][CX3]=[OX1]")
    for smi in prods:
        assert Chem.MolFromSmiles(smi).HasSubstructMatch(primary_amide), smi
    # ...while the dedicated amide rule does fire on it
    amide_rxn = next(r for r, n in compiled_transforms() if n.startswith("amide N-H"))
    assert apply_transform(Chem.MolFromSmiles("CNC(=O)c1ccccc1"), amide_rxn)


def test_transforms_do_not_corrupt_an_amide():
    """Every transform must leave a sanitisable molecule, amides included."""
    from vcdd.optimize.generators.enumerative import apply_transform, compiled_transforms

    amide = Chem.MolFromSmiles(SMILES["piperidine_amide"])
    fired = 0
    for rxn, _name in compiled_transforms():
        for prod in apply_transform(amide, rxn):
            assert Chem.MolFromSmiles(prod) is not None, prod
            fired += 1
    assert fired > 0


# --------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------


def test_guardrails_accept_a_conservative_analog():
    parent = Chem.MolFromSmiles(SMILES["ibuprofen"])
    rep = gr.check_analog("CC(C)Cc1ccc(C(C)C(=O)OC)cc1", parent, GuardrailConfig())
    row = rep.to_row()
    assert row["guardrail_pass"] is True or row["guardrail_pass"] == True  # noqa: E712
    assert row["similarity_to_parent"] > 0.5
    assert row["heavy_atom_delta"] == 1
    assert 0 < row["qed"] <= 1


def test_guardrails_reject_parent_identity_and_distant_analogs():
    cfg = GuardrailConfig()
    parent = Chem.MolFromSmiles(SMILES["ibuprofen"])
    same = gr.check_analog(SMILES["ibuprofen"], parent, cfg)
    assert not same.to_row()["guardrail_pass"]
    assert "parent" in same.to_row()["guardrail_reasons"].lower()

    distant = gr.check_analog(SMILES["caffeine"], parent, cfg)
    assert not distant.to_row()["guardrail_pass"]


def test_guardrails_enforce_the_heavy_atom_budget():
    cfg = GuardrailConfig(heavy_atom_delta_max=2, similarity_floor_enum=0.1)
    parent = Chem.MolFromSmiles(SMILES["benzene"])
    grown = gr.check_analog("CCCCCCCCCCc1ccccc1", parent, cfg)
    row = grown.to_row()
    assert not row["guardrail_pass"]
    assert "heavy_atom" in row["guardrail_reasons"]


def test_filter_analogs_keeps_only_passing_structures():
    cands = ["CC(C)Cc1ccc(C(C)C(=O)OC)cc1", SMILES["ibuprofen"], "not a smiles"]
    reports = gr.filter_analogs(cands, SMILES["ibuprofen"], GuardrailConfig())
    rows = [r.to_row() for r in reports]
    # unparsable input is dropped, the parent itself is reported but rejected
    assert 0 < len(rows) <= len(cands)
    assert any(r["guardrail_pass"] for r in rows)
    assert any(not r["guardrail_pass"] for r in rows)
    # de-duplication is on canonical SMILES
    assert len({r["analog_smiles"] for r in rows}) == len(rows)
    with pytest.raises(ValueError):
        gr.filter_analogs(cands, "not a smiles", GuardrailConfig())


def test_tanimoto_is_one_for_identical_molecules():
    mol = Chem.MolFromSmiles(SMILES["aspirin"])
    assert gr.tanimoto(mol, mol) == pytest.approx(1.0)
    assert gr.morgan_fp(mol).GetNumBits() == 2048


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def test_resolve_generators_reports_availability():
    cfg = Stage4Config()
    usable, report = genbase.resolve_generators(
        ["rgroup", "constrained_ga", "crem", "nope"], cfg
    )
    names = [r["generator"] for r in report]
    assert names == ["rgroup", "constrained_ga", "crem", "nope"]
    assert {g.name() for g in usable} >= {"rgroup", "constrained_ga"}
    assert all(r["reason"] for r in report if not r["available"])


def test_rgroup_generator_returns_labelled_proposals():
    gen = RGroupGenerator(Stage4Config(), seed=7)
    props = gen.generate("P1", SMILES["ibuprofen"], 8)
    assert 0 < len(props) <= 8
    for p in props:
        assert isinstance(p, genbase.Proposal)
        assert p.parent_id == "P1"
        assert p.generator == "rgroup"
        assert p.transformation
        assert Chem.MolFromSmiles(p.smiles) is not None
        assert p.smiles != Chem.CanonSmiles(SMILES["ibuprofen"])


def test_constrained_ga_is_deterministic_under_a_seed():
    cfg = Stage4Config(ga_generations=2, ga_population=12)
    a = ConstrainedGA(cfg, seed=11).generate("P1", SMILES["ibuprofen"], 6)
    b = ConstrainedGA(cfg, seed=11).generate("P1", SMILES["ibuprofen"], 6)
    assert [p.smiles for p in a] == [p.smiles for p in b]
    assert len(a) > 0
    for p in a:
        assert p.meta["ga_generation"] >= 0
        assert p.meta["similarity_to_parent"] >= cfg.guardrails.similarity_floor_ga
        assert len(p.meta["transformations"]) <= 3


# --------------------------------------------------------------------------
# Pareto / design score
# --------------------------------------------------------------------------


def test_constrained_ga_uses_an_injected_batch_fitness():
    """Stage 4 steers the GA by injecting a cell-objective scorer."""
    cfg = Stage4Config(ga_generations=3, ga_population=16)
    calls: list[int] = []

    def fluorine_fitness(smiles, parent_smiles):
        calls.append(len(smiles))
        assert isinstance(parent_smiles, str)
        return [1.0 if "F" in s else 0.0 for s in smiles]

    ga = ConstrainedGA(cfg, seed=5)
    ga.set_fitness(fluorine_fitness)
    steered = ga.generate("P1", SMILES["ibuprofen"], 8)
    assert calls, "injected fitness was never called"
    assert all(n > 1 for n in calls), "fitness must be called on whole batches"

    baseline = ConstrainedGA(cfg, seed=5).generate("P1", SMILES["ibuprofen"], 8)
    frac_f = lambda props: sum("F" in p.smiles for p in props) / max(len(props), 1)
    assert frac_f(steered) >= frac_f(baseline), (
        [p.smiles for p in steered], [p.smiles for p in baseline]
    )
    assert frac_f(steered) > 0


def test_normalisers_map_to_the_unit_interval():
    assert pareto.normalise_docking(pareto.DOCK_ANCHORS[0]) == pytest.approx(0.0)
    assert pareto.normalise_docking(pareto.DOCK_ANCHORS[1]) == pytest.approx(1.0)
    assert 0 <= pareto.normalise_docking(0.0) <= 1
    assert pareto.normalise_sa(1.0) == pytest.approx(1.0)
    assert pareto.normalise_sa(10.0) == pytest.approx(0.0)


def test_pareto_fronts_rank_a_hand_built_frontier():
    df = pd.DataFrame({
        "dock_score": [-10.0, -9.0, -8.0, -7.0],
        "cell_objective": [0.9, 0.8, 0.4, 0.3],
        "qed": [0.8, 0.8, 0.8, 0.8],
        "sa_score": [2.0, 2.0, 2.0, 2.0],
    })
    ranks = pareto.pareto_fronts(df, pareto.DESIGN_OBJECTIVES)
    assert list(ranks)[0] == 1
    assert min(ranks) == 1
    # row 0 dominates every other row on both varying objectives
    assert ranks[0] == 1 and ranks[3] == max(ranks)


def test_design_score_renormalises_when_a_term_is_missing():
    df = pd.DataFrame({
        "dock_score": [-10.0, -10.0],
        "cell_objective": [0.9, np.nan],
        "qed": [0.8, 0.8],
        "sa_score": [2.0, 2.0],
    })
    weights = {"docking": 1.0, "cell": 2.0, "qed": 0.5, "sa": 0.5}
    score, terms = pareto.design_score(df, weights)
    assert len(score) == 2
    assert (score > 0).all()
    assert terms.loc[0, "n_terms_used"] == 4
    assert terms.loc[1, "n_terms_used"] == 3


# --------------------------------------------------------------------------
# SAR mining
# --------------------------------------------------------------------------


def test_sign_test_matches_the_exact_binomial():
    assert sar.sign_test(5, 5) == pytest.approx(2 * 0.5**5)
    assert sar.sign_test(1, 2) == pytest.approx(1.0)
    assert sar.sign_test(8, 8) < sar.sign_test(6, 8)


def test_explode_transformations_recovers_individual_edits():
    df = pd.DataFrame({
        "analog_id": ["A1", "A2", "A3"],
        "transformation": ["x|y", "solo", "fallback"],
        "transformations": [["x", "y"], ["solo"], None],
        "delta_cell_objective": [0.1, 0.2, 0.3],
    })
    out = sar.explode_transformations(df)
    assert set(out["op"]) == {"x", "y", "solo", "fallback"}
    assert len(out) == 4
    # a chained GA lineage is flagged as weaker evidence via lineage_length
    assert out.loc[out["op"] == "x", "lineage_length"].iloc[0] == 2
    assert out.loc[out["op"] == "solo", "lineage_length"].iloc[0] == 1
    # the fallback path uses the compound label when no list is present
    assert out.loc[out["op"] == "fallback", "lineage_length"].iloc[0] == 1


def test_mine_sar_collapses_per_parent_before_voting():
    """A rule seen 20x in one parent must not outvote two independent parents."""
    rows = []
    for i in range(20):  # one parent, many analogs, all improved
        rows.append({"parent_id": "P1", "transformation": "spam",
                     "delta_cell_objective": 0.2, "delta_dock_score": -0.1})
    for pid in ("P1", "P2", "P3"):
        rows.append({"parent_id": pid, "transformation": "real",
                     "delta_cell_objective": 0.3, "delta_dock_score": -0.2})
    rules = sar.mine_sar(pd.DataFrame(rows), metrics=["delta_cell_objective"],
                         min_parents=1)
    by_rule = rules.set_index("transformation")
    assert by_rule.loc["spam", "n_parents"] == 1
    assert by_rule.loc["spam", "n_analogs"] == 20
    assert by_rule.loc["real", "n_parents"] == 3
    assert "[consistent]" in str(by_rule.loc["real", "verdict"])
    assert "[consistent]" not in str(by_rule.loc["spam", "verdict"])


def test_mine_sar_respects_lower_is_better_metrics():
    rows = [
        {"parent_id": p, "transformation": "shrink", "delta_dock_score": -0.5}
        for p in ("P1", "P2", "P3")
    ]
    rules = sar.mine_sar(pd.DataFrame(rows), metrics=["delta_dock_score"], min_parents=1)
    row = rules.iloc[0]
    assert row["frac_parents_improved"] == pytest.approx(1.0), \
        "more negative docking scores are improvements"
    assert sar.format_rule(row, up_is_good=False)


def test_mine_sar_drops_rules_below_the_parent_floor():
    rows = [{"parent_id": "P1", "transformation": "rare", "delta_cell_objective": 0.1}]
    rules = sar.mine_sar(pd.DataFrame(rows), metrics=["delta_cell_objective"],
                         min_parents=2)
    assert rules.empty


# --------------------------------------------------------------------------
# Stage 4 end to end (no docking, mock cell backend)
# --------------------------------------------------------------------------


@pytest.fixture()
def stage3_dir(tmp_path):
    d = tmp_path / "stage3"
    d.mkdir(parents=True)
    names = ["ibuprofen", "aspirin", "phenol"]
    pd.DataFrame({
        "mol_id": [f"M{i}" for i in range(len(names))],
        "smiles": [SMILES[n] for n in names],
        "dock_score": [-9.0, -8.5, -7.0],
        "consensus_score": [1.0, 0.8, 0.6],
        "consensus_rank": [1, 2, 3],
        "objective": [0.7, 0.6, 0.5],
        "objective_rank": [1, 2, 3],
    }).to_parquet(d / "objective_scores.parquet", index=False)
    return d


def _stage4_cfg():
    from vcdd.config.schema import RunConfig

    cfg = RunConfig()
    cfg.stage4 = Stage4Config(
        top_parents=3, generators=["rgroup"], analogs_per_parent=6,
        redock_mode="none", ga_generations=2, ga_population=10,
    )
    cfg.cell = CellConfig(
        backends=["mock"], protein_activity=False, gene_space="landmark",
        objective=ObjectiveConfig(genes_up=["CDKN1A"], genes_down=["MYC"]),
    )
    return cfg


def test_stage4_generates_ranked_recommendations(stage3_dir, tmp_path, ledger, small_box):
    from vcdd.optimize.stage4 import RECOMMENDATION_COLUMNS, run_stage4

    out = tmp_path / "stage4"
    res = run_stage4(_stage4_cfg(), tmp_path / "receptor.pdbqt", small_box,
                     stage3_dir, out, ledger)
    recs = pd.read_csv(out / "analog_recommendations.csv")
    assert len(recs) > 0
    assert list(recs.columns) == list(RECOMMENDATION_COLUMNS)
    assert recs["rank"].is_monotonic_increasing
    assert recs["analog_smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None).all()
    assert (recs["similarity_to_parent"] > 0).all()
    assert res.out_dir == out

    summary = json.loads((out / "stage4_summary.json").read_text())
    assert summary["n_parents"] >= 1
    assert (out / "generator_report.json").exists()
    assert (out / "analogs.parquet").exists()

    analogs = pd.read_parquet(out / "analogs.parquet")
    assert "delta_cell_objective" in analogs.columns
    assert analogs["analog_id"].is_unique


def test_stage4_resumes_without_regenerating(stage3_dir, tmp_path, ledger, small_box):
    from vcdd.optimize.stage4 import run_stage4

    out = tmp_path / "stage4"
    cfg = _stage4_cfg()
    run_stage4(cfg, tmp_path / "receptor.pdbqt", small_box, stage3_dir, out, ledger)
    first = pd.read_parquet(out / "analogs.parquet")
    run_stage4(cfg, tmp_path / "receptor.pdbqt", small_box, stage3_dir, out, ledger)
    second = pd.read_parquet(out / "analogs.parquet")
    pd.testing.assert_frame_equal(first, second)
