"""Stage 3 (virtual cell) tests: resources, backends, activity inference, objective.

Everything that touches the bundled LINCS/CollecTRI/PROGENy resources is guarded
by ``requires_cell_resources`` so the suite still runs on a bare checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from _helpers import SMILES, requires_cell_resources
from vcdd.cell import assets
from vcdd.cell import base as cellbase
from vcdd.cell import objective as objmod
from vcdd.cell import protein_activity as pa
from vcdd.cell.stage3 import load_shortlist, run_stage3
from vcdd.config.schema import CellConfig, ObjectiveConfig

# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------


@requires_cell_resources
def test_manifest_describes_every_bundled_resource():
    man = assets.manifest()
    # schema 2 distinguishes the digest of the downloaded source (source_sha256)
    # from the digest of the file shipped here (file_sha256). Schema 1 carried a
    # bare "sha256" that was the source digest sitting next to the written file's
    # size, so nothing about the shipped file could actually be verified.
    assert man["schema"] >= 2
    for name in assets.FILES:
        entry = man["resources"][name]
        assert entry["file"] == assets.FILES[name]
        assert entry["rows"] > 0
        assert entry["file_bytes"] > 0
        assert len(entry["file_sha256"]) == 64
        assert len(entry["source_sha256"]) == 64
        assert entry["source"].startswith("http")


@requires_cell_resources
def test_bundle_status_reports_all_present():
    status = assets.bundle_status()
    assert set(status) == set(assets.FILES)
    assert all(v["present"] for v in status.values())
    assert all(assets.available(n) for n in assets.FILES)


@requires_cell_resources
def test_l1000_landmark_space_is_exactly_978_genes():
    genes = assets.load_l1000_genes()
    assert int(genes["is_landmark"].sum()) == 978
    assert len(assets.gene_universe("landmark")) == 978
    # "all" deliberately means BING, not the whole transcriptome.
    assert len(assets.gene_universe("all")) == int(genes["is_bing"].sum())
    assert set(assets.gene_universe("landmark")) <= set(assets.gene_universe("all"))


@requires_cell_resources
def test_collectri_is_signed_and_free_of_complex_regulators():
    net = assets.load_collectri()
    assert set(net["mor"].unique()) <= {1, -1}
    assert not net["tf"].astype(str).str.startswith("COMPLEX").any()
    assert net["tf"].nunique() == 1201
    # activation should dominate repression in CollecTRI
    assert (net["mor"] == 1).sum() > (net["mor"] == -1).sum()


@requires_cell_resources
def test_progeny_has_fourteen_pathways_of_one_hundred_genes():
    prog = assets.load_progeny()
    assert prog["pathway"].nunique() == 14
    assert prog.groupby("pathway").size().max() == 100
    assert prog["weight"].abs().sum() > 0


@requires_cell_resources
def test_hallmark_has_fifty_gene_sets():
    hm = assets.load_hallmark()
    assert hm["gene_set"].nunique() == 50
    assert hm["gene_set"].str.startswith("").all()


@requires_cell_resources
def test_lincs_reference_signatures_are_usable():
    chem = assets.load_lincs_chem()
    assert len(chem) == 5425
    assert chem["smiles"].notna().sum() > 3000
    # up/down lists deserialise as arrays, never as scalars
    row = chem.iloc[0]
    assert len(list(row["up_genes"])) > 100
    assert len(list(row["down_genes"])) > 100

    ko = assets.load_lincs_crispr()
    assert len(ko) == 5212
    assert ko["gene"].is_unique


@requires_cell_resources
def test_signature_from_updown_is_signed_and_sparse():
    genes = assets.gene_universe("landmark")
    vec = assets.signature_from_updown(genes[:5], genes[5:10], genes)
    vec = np.asarray(vec, dtype=float)
    assert vec.shape == (len(genes),)
    assert vec[:5].min() > 0
    assert vec[5:10].max() < 0
    assert np.count_nonzero(vec) == 10


# --------------------------------------------------------------------------
# backend registry / capabilities contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(cellbase.list_backends()))
def test_backend_declares_honest_capabilities(name):
    cls = cellbase.get_backend_class(name)
    caps = cls.capabilities()
    assert caps.name == name
    assert caps.tier in (1, 2, 3)
    assert caps.description
    assert caps.value_scale in ("zscore", "logfc", "membership")
    assert isinstance(caps.accepts_novel_smiles, bool)
    if caps.tier > 1:
        # external-weight backends must tell the user how to install them
        assert caps.install_hint
        assert caps.env_requires or caps.python_requires
    ok, reason = cls(CellConfig()).is_available()
    assert isinstance(ok, bool)
    if not ok:
        assert reason, f"{name} unavailable without an explanation"


def test_resolve_backends_reports_every_request():
    cfg = CellConfig(backends=["mock", "chemcpa", "not_a_backend"])
    usable, report = cellbase.resolve_backends(cfg.backends, cfg)
    assert [r["backend"] for r in report] == cfg.backends
    assert [m.capabilities().name for m in usable] == ["mock"]
    unknown = [r for r in report if r["backend"] == "not_a_backend"][0]
    assert unknown["available"] is False and unknown["reason"]
    chemcpa = [r for r in report if r["backend"] == "chemcpa"][0]
    assert chemcpa["available"] is False
    assert "VCDD_CHEMCPA_DIR" in (chemcpa["reason"] + chemcpa.get("install_hint", ""))


def test_signature_frames_carry_values_and_provenance():
    genes = ["A", "B", "C"]
    sigs = [
        cellbase.Signature(
            mol_id="M1",
            smiles=SMILES["aspirin"],
            genes=genes,
            values=np.array([1.0, -2.0, 0.5]),
            backend="mock",
            cell_line="MCF7",
            dose_um=10.0,
            time_h=24.0,
            value_scale="zscore",
            meta={"max_tanimoto": 0.42, "n_neighbors": 3},
        ),
        cellbase.Signature(
            mol_id="M2",
            smiles="not_a_smiles",
            genes=[],
            values=np.array([]),
            backend="mock",
            cell_line="MCF7",
            dose_um=10.0,
            time_h=24.0,
            value_scale="zscore",
            error="parse failure",
        ),
    ]
    assert sigs[0].ok and not sigs[1].ok
    assert sigs[0].get("B") == -2.0
    ser = sigs[0].to_series()
    assert list(ser.index) == genes

    frame = cellbase.signatures_to_frame(sigs)
    assert len(frame) >= 1
    meta = cellbase.signature_meta_frame(sigs)
    assert "mol_id" in meta.columns
    assert set(meta["mol_id"]) == {"M1", "M2"}
    # the applicability-domain metadata must survive the round trip to disk
    m1 = json.loads(meta.set_index("mol_id").loc["M1", "meta_json"])
    assert m1["max_tanimoto"] == 0.42 and m1["n_neighbors"] == 3


# --------------------------------------------------------------------------
# protein activity inference
# --------------------------------------------------------------------------


def test_ulm_recovers_a_planted_regulon_signal():
    genes = [f"G{i}" for i in range(40)]
    net = pd.DataFrame(
        [{"tf": "TFA", "target": g, "mor": 1} for g in genes[:20]]
        + [{"tf": "TFB", "target": g, "mor": 1} for g in genes[20:]]
    )
    mat, regs, n_dropped = pa.build_network_matrix(
        net, genes, source_col="tf", target_col="target", weight_col="mor"
    )
    assert regs == ["TFA", "TFB"]
    assert mat.shape == (len(genes), 2)
    assert n_dropped == 0

    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.5, size=(len(genes), 1))  # genes on rows, samples on columns
    y[:20, 0] += 3.0  # TFA targets up, TFB targets flat
    scores = pa.run_ulm(mat, y)  # -> (n_regulators, n_samples)
    assert scores.shape == (2, 1)
    assert scores[regs.index("TFA"), 0] > scores[regs.index("TFB"), 0]
    assert scores[regs.index("TFA"), 0] > 0

    # flipping the sign of the perturbation flips the inferred activity
    flipped = pa.run_ulm(mat, -y)
    assert flipped[regs.index("TFA"), 0] < 0


def test_regulons_below_the_coverage_floor_are_dropped():
    genes = [f"G{i}" for i in range(20)]
    net = pd.DataFrame(
        [{"tf": "BIG", "target": g, "mor": 1} for g in genes[:10]]
        + [{"tf": "TINY", "target": g, "mor": 1} for g in genes[:2]]
    )
    mat, regs, n_dropped = pa.build_network_matrix(
        net, genes, source_col="tf", target_col="target",
        weight_col="mor", min_targets=pa.MIN_TARGETS,
    )
    assert regs == ["BIG"], "regulons with < MIN_TARGETS measured genes must be dropped"
    assert mat.shape[1] == 1 and n_dropped == 1


def test_activity_methods_agree_on_direction():
    genes = [f"G{i}" for i in range(30)]
    net = pd.DataFrame([{"tf": "TFA", "target": g, "mor": 1} for g in genes[:15]])
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.5, size=(len(genes), 1))
    y[:15, 0] += 2.0
    out = {}
    for method in ("ulm", "wmean", "wsum"):
        res = pa.infer_activity(
            y, genes, net, source_col="tf", target_col="target", weight_col="mor",
            sample_ids=["S1"], method=method, network_name="test",
        )
        assert isinstance(res.scores, pd.DataFrame)
        assert res.method == method
        assert res.n_regulators == 1
        assert 0 < res.coverage <= 1
        out[method] = float(res.scores.loc["TFA", "S1"])
    assert all(v > 0 for v in out.values()), out


@requires_cell_resources
def test_tf_activity_uses_collectri_and_reports_dropped_regulons():
    genes = assets.gene_universe("landmark")
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.5, size=(len(genes), 2))
    y[:50, 0] += 2.0
    res = pa.tf_activity(y, genes, sample_ids=["A", "B"])
    assert res.network == "collectri"
    # only a minority of the 1201 CollecTRI TFs clear MIN_TARGETS in landmark space
    assert 300 < res.n_regulators < 500
    assert res.dropped_small > 500
    assert res.scores.shape == (res.n_regulators, 2)


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def test_squash_is_monotonic_and_centred():
    assert objmod.squash(0.0, 1.0) == pytest.approx(0.5)
    xs = np.linspace(-5, 5, 21)
    ys = objmod.squash(xs, 2.0)
    assert np.all(np.diff(ys) > 0)
    assert ys.min() > 0 and ys.max() < 1
    # scale controls saturation, so a bigger scale is closer to the centre
    assert objmod.squash(2.0, 5.0) < objmod.squash(2.0, 1.0)


def test_objective_defaults_are_absolute_not_batch_relative():
    assert set(objmod.COMPONENT_NAMES) == {
        "gene_score", "pathway_score", "target_consistency", "specificity"
    }
    # the "moved" threshold must be low enough for membership-scale backends
    assert objmod.MOVED_THRESHOLD["membership"] < objmod.SQUASH_SCALE["membership"]


def _sig(values: dict[str, float], genes: list[str], backend="mock", scale="zscore"):
    vec = np.array([values.get(g, 0.0) for g in genes], dtype=float)
    return cellbase.Signature(
        mol_id="M1", smiles=SMILES["aspirin"], genes=genes, values=vec,
        backend=backend, cell_line="MCF7", dose_um=10.0, time_h=24.0,
        value_scale=scale, meta={"max_tanimoto": 0.9},
    )


def test_objective_rewards_the_requested_direction():
    genes = [f"G{i}" for i in range(30)] + ["CDKN1A", "MYC"]
    cfg = ObjectiveConfig(genes_up=["CDKN1A"], genes_down=["MYC"])
    ctx = objmod.ObjectiveContext.build(cfg, genes)
    good = objmod.score_signature(_sig({"CDKN1A": 4.0, "MYC": -4.0}, genes), ctx)
    bad = objmod.score_signature(_sig({"CDKN1A": -4.0, "MYC": 4.0}, genes), ctx)
    flat = objmod.score_signature(_sig({}, genes), ctx)
    assert good.gene_score > flat.gene_score > bad.gene_score
    assert good.objective > bad.objective
    assert flat.gene_score == pytest.approx(0.5, abs=0.05)


def test_gene_mode_all_is_stricter_than_any():
    genes = ["CDKN1A", "MYC"] + [f"G{i}" for i in range(20)]
    sig = _sig({"CDKN1A": 4.0, "MYC": 4.0}, genes)  # one right, one wrong
    ctx_any = objmod.ObjectiveContext.build(
        ObjectiveConfig(genes_up=["CDKN1A"], genes_down=["MYC"], gene_mode="any"), genes
    )
    ctx_all = objmod.ObjectiveContext.build(
        ObjectiveConfig(genes_up=["CDKN1A"], genes_down=["MYC"], gene_mode="all"), genes
    )
    assert objmod.score_signature(sig, ctx_all).gene_score < \
        objmod.score_signature(sig, ctx_any).gene_score


def test_unmeasurable_gene_is_reported_in_context_warnings():
    genes = [f"G{i}" for i in range(20)]
    ctx = objmod.ObjectiveContext.build(
        ObjectiveConfig(genes_up=["TOTALLY_MADE_UP_GENE"]), genes
    )
    assert any("TOTALLY_MADE_UP_GENE" in w for w in ctx.warnings)


@requires_cell_resources
def test_non_landmark_transcription_factor_is_routed_to_inferred_activity():
    genes = assets.gene_universe("landmark")
    tf = next(
        t for t in assets.load_collectri()["tf"].unique()
        if t not in set(genes)
    )
    ctx = objmod.ObjectiveContext.build(ObjectiveConfig(genes_up=[tf]), genes)
    assert tf in ctx.tf_requests, "an unmeasured TF must fall back to regulon activity"


def test_aggregate_backends_averages_and_records_spread():
    rows = pd.DataFrame([
        {"mol_id": "M1", "backend": "a", "gene_score": 0.8, "pathway_score": 0.5,
         "target_consistency": 0.5, "specificity": 0.9, "objective": 0.70,
         "max_tanimoto": 0.5, "out_of_domain": False, "frac_genes_moved": 0.2,
         "stress_score": 0.1, "n_components_used": 4, "error": None},
        {"mol_id": "M1", "backend": "b", "gene_score": 0.4, "pathway_score": 0.5,
         "target_consistency": 0.5, "specificity": 0.9, "objective": 0.50,
         "max_tanimoto": 0.7, "out_of_domain": False, "frac_genes_moved": 0.3,
         "stress_score": 0.1, "n_components_used": 4, "error": None},
    ])
    agg = objmod.aggregate_backends(rows)
    assert len(agg) == 1
    r = agg.iloc[0]
    assert r["objective"] == pytest.approx(0.60)
    assert r["n_backends"] == 2
    assert r["backend_spread"] == pytest.approx(np.std([0.70, 0.50], ddof=1), abs=1e-9)


# --------------------------------------------------------------------------
# Stage 3 end to end
# --------------------------------------------------------------------------


class _Cfg:
    """Minimal duck type for run_stage3 (it only reads .cell / .seed / .n_workers)."""

    def __init__(self, cell: CellConfig):
        self.cell = cell
        self.seed = 42
        self.n_workers = 1


@pytest.fixture()
def stage2_dir(tmp_path):
    d = tmp_path / "stage2"
    d.mkdir(parents=True)
    names = ["aspirin", "caffeine", "ibuprofen", "phenol", "toluene", "aniline"]
    df = pd.DataFrame({
        "mol_id": [f"M{i:03d}" for i in range(len(names))],
        "smiles": [SMILES[n] for n in names],
        "dock_score": [-9.0 + i * 0.3 for i in range(len(names))],
        "stage1_score": [-8.5 + i * 0.3 for i in range(len(names))],
        "consensus_score": [1.0 - i * 0.1 for i in range(len(names))],
        "consensus_rank": list(range(1, len(names) + 1)),
    })
    df.to_parquet(d / "consensus_ranking.parquet", index=False)
    return d


def _cell_cfg(**kw) -> CellConfig:
    base = dict(
        backends=["mock"],
        top_n=6,
        gene_space="landmark",
        protein_activity=False,
        objective=ObjectiveConfig(genes_up=["CDKN1A"], genes_down=["MYC"]),
    )
    base.update(kw)
    return CellConfig(**base)


def test_load_shortlist_prefers_the_consensus_ranking(stage2_dir):
    short = load_shortlist(stage2_dir, top_n=3)
    assert len(short) == 3
    assert list(short["mol_id"]) == ["M000", "M001", "M002"]


@requires_cell_resources
def test_stage3_scores_and_ranks_the_shortlist(stage2_dir, tmp_path, ledger):
    out = tmp_path / "stage3"
    res = run_stage3(_Cfg(_cell_cfg()), stage2_dir, out, ledger)
    assert res.out_dir == out
    scores = pd.read_parquet(out / "objective_scores.parquet")
    assert len(scores) == 6
    for col in ("gene_score", "pathway_score", "target_consistency", "specificity",
                "objective", "objective_rank", "max_tanimoto", "out_of_domain"):
        assert col in scores.columns
    assert scores["objective"].is_monotonic_decreasing
    assert list(scores["objective_rank"]) == list(range(1, 7))
    assert scores["objective"].between(0, 1).all()

    summary = json.loads((out / "stage3_summary.json").read_text())
    assert summary["n_compounds"] == 6
    assert summary["backends_used"] == ["mock"]
    assert summary["gene_space"] == "landmark"
    assert (out / "backend_report.json").exists()


@requires_cell_resources
def test_stage3_resume_preserves_applicability_domain_metadata(
    stage2_dir, tmp_path, ledger
):
    out = tmp_path / "stage3"
    cfg = _Cfg(_cell_cfg(backends=["lincs_readacross"]))
    first = run_stage3(cfg, stage2_dir, out, ledger)
    a = pd.read_parquet(out / "objective_scores.parquet")
    assert first.out_dir == out
    # second call must reuse the cached signatures *and* their metadata
    run_stage3(cfg, stage2_dir, out, ledger)
    b = pd.read_parquet(out / "objective_scores.parquet")
    pd.testing.assert_series_equal(a["max_tanimoto"], b["max_tanimoto"])
    assert a["max_tanimoto"].notna().all()
    pd.testing.assert_series_equal(a["out_of_domain"], b["out_of_domain"])
    pd.testing.assert_series_equal(a["objective"], b["objective"])


@requires_cell_resources
def test_stage3_excludes_reference_only_backends_from_scoring(
    stage2_dir, tmp_path, ledger
):
    cfg = _Cfg(_cell_cfg(backends=["mock", "target_knockdown"], target_gene="EGFR"))
    out = tmp_path / "stage3"
    run_stage3(cfg, stage2_dir, out, ledger)
    summary = json.loads((out / "stage3_summary.json").read_text())
    assert "target_knockdown" in summary["backends_reference_only"]
    assert "target_knockdown" not in summary["backends_scored"]
    assert (out / "signatures" / "target_knockdown.parquet").exists(), \
        "reference-only backends are still materialised for provenance"
    by_backend = pd.read_parquet(out / "objective_by_backend.parquet")
    assert set(by_backend["backend"]) == {"mock"}


@requires_cell_resources
def test_stage3_refuses_to_score_with_only_reference_only_backends(
    stage2_dir, tmp_path, ledger
):
    cfg = _Cfg(_cell_cfg(backends=["target_knockdown"], target_gene="EGFR"))
    with pytest.raises(Exception):
        run_stage3(cfg, stage2_dir, tmp_path / "stage3", ledger)
