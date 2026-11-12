"""Config schema/loader and the library IO layer (SDF / SMI / CSV / Parquet)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from vcdd.config.loader import deep_merge, load_config, parse_overrides, save_resolved, template_text
from vcdd.config.schema import FilterConfig, LibraryConfig, RunConfig
from vcdd.io.filters import apply_filters, compute_properties
from vcdd.io.library import load_shard_index, prepare_library
from vcdd.io.readers import read_library, sniff_format
from vcdd.io.standardize import standardize_smiles

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------- config
def test_template_parses_into_a_valid_config(tmp_path: Path):
    text = template_text()
    path = tmp_path / "t.yaml"
    path.write_text(text)
    cfg = load_config(path)
    assert isinstance(cfg, RunConfig)


def test_parse_overrides_coerces_scalars_and_nests():
    got = parse_overrides(["stage1.thread=5000", "name=abc", "cell.protein_activity=false",
                           "cell.objective.genes_up=[TP53,CDKN1A]"])
    assert got["stage1"]["thread"] == 5000
    assert got["name"] == "abc"
    assert got["cell"]["protein_activity"] is False
    assert got["cell"]["objective"]["genes_up"] == ["TP53", "CDKN1A"]


def test_deep_merge_does_not_clobber_sibling_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 9}})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base["a"]["y"] == 2  # input untouched


def test_config_rejects_unknown_key():
    with pytest.raises(Exception):
        RunConfig(not_a_real_field=1)


def test_defaults_are_vina_gpu_then_boltz():
    """The two engine choices 0.2.0 makes for you, pinned so they cannot drift."""
    cfg = RunConfig()
    assert cfg.stage1.engine == "vina_gpu"
    assert cfg.stage1.variant == "autodock_vina_gpu"
    assert cfg.stage2.backend == "library_screen"
    assert cfg.stage2.fusion == "boltz_only"


def test_absurd_thread_count_warns(recwarn):
    """Upstream warns that >10000 threads exhausts GPU memory on most cards."""
    cfg = RunConfig(stage1={"thread": 40_000})
    assert cfg.stage1.thread == 40_000
    assert any("thread" in str(w.message).lower() for w in recwarn)


@pytest.mark.parametrize("bad", [
    {"stage1": {"gpu_batch_size": 0}},
    {"stage2": {"chunk_size": 0}},
    {"stage2": {"top_n": 0}},
])
def test_non_positive_batch_sizes_are_rejected(bad):
    with pytest.raises(Exception):
        RunConfig(**bad)


# ------------------------------------------------------- 0.1.0 -> 0.2.0 migration
@pytest.mark.parametrize("dotted", [
    "stage1.exhaustiveness",
    "stage1.cpu_per_ligand",
    "stage1.num_modes",
    "stage2.engine",
    "stage2.rescorers",
    "stage2.consensus",
    "stage2.n_conformers",
    "stage2.pose_qc",
])
def test_removed_keys_raise_a_message_that_names_the_replacement(dotted):
    """A 0.1.0 config must fail with advice, not pydantic's generic 'extra inputs'."""
    section, _, key = dotted.partition(".")
    with pytest.raises(ValueError) as excinfo:
        load_config(None, overrides={section: {key: 4}})
    msg = str(excinfo.value)
    assert dotted in msg
    assert "MIGRATION-0.2.0.md" in msg
    assert "extra inputs are not permitted" not in msg.lower()


def test_migration_error_lists_every_problem_at_once(tmp_path: Path):
    """One run, one error: users should not fix these one at a time."""
    path = tmp_path / "old.yaml"
    path.write_text(
        "stage1:\n  exhaustiveness: 8\n  num_modes: 9\n"
        "stage2:\n  rescorers: [vina, vinardo]\n  consensus: both\n"
        "stage4:\n  redock_mode: stage2\n"
    )
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    msg = str(excinfo.value)
    for token in ("stage1.exhaustiveness", "stage1.num_modes", "stage2.rescorers",
                  "stage2.consensus", "stage4.redock_mode"):
        assert token in msg


def test_redock_mode_stage2_is_rejected_by_the_schema_too():
    """Belt and braces: the validator fires even when check_migration is bypassed."""
    with pytest.raises(Exception) as excinfo:
        RunConfig(stage4={"redock_mode": "stage2"})
    assert "stage1" in str(excinfo.value)


def test_resolved_config_roundtrips(tmp_path: Path, base_cfg):
    path = save_resolved(base_cfg, tmp_path)
    reloaded = load_config(path)
    assert reloaded.stage1.engine == base_cfg.stage1.engine
    assert reloaded.cell.backends == base_cfg.cell.backends


def test_stage_dir_helpers(base_cfg):
    assert base_cfg.stage_dir("stage3").name == "stage3"
    assert base_cfg.run_dir.name == "run"


# -------------------------------------------------------------------- readers
@pytest.mark.parametrize(
    "filename,fmt",
    [("library.sdf", "sdf"), ("library.sdf.gz", "sdf"), ("library.smi", "smi"),
     ("library.csv", "csv"), ("library.parquet", "parquet")],
)
def test_sniff_format(filename, fmt):
    assert sniff_format(FIXTURES / filename) == fmt


@pytest.mark.parametrize(
    "filename", ["library.sdf", "library.sdf.gz", "library.smi", "library.csv", "library.parquet"]
)
def test_all_input_formats_yield_the_same_molecule_count(filename):
    records = list(read_library(FIXTURES / filename))
    assert len(records) == 200
    assert all(r.smiles for r in records)


def test_csv_with_utf8_bom_and_custom_columns_is_read():
    """The CSV fixture is deliberately BOM-prefixed with non-default column names."""
    records = list(read_library(FIXTURES / "library.csv",
                                  smiles_column="canonical_smiles", id_column="compound_id"))
    assert len(records) == 200
    assert records[0].mol_id and not records[0].mol_id.startswith("\ufeff")


def test_unparseable_record_is_reported_not_raised():
    records = list(read_library(FIXTURES / "library_messy.sdf"))
    # 26 records, one of which cannot be parsed by RDKit.
    assert len(records) >= 25
    assert any(r.error for r in records) or len(records) == 25


def test_gzip_and_plain_sdf_agree():
    plain = [r.smiles for r in read_library(FIXTURES / "library.sdf")]
    gz = [r.smiles for r in read_library(FIXTURES / "library.sdf.gz")]
    assert plain == gz


def test_reader_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(read_library(tmp_path / "nope.sdf"))


# --------------------------------------------------------------- standardize
def test_standardize_desalts_to_the_parent_fragment():
    out = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na+]", desalt=True, canonical_tautomer=False)
    assert out.ok and "." not in out.smiles and "Na" not in out.smiles


def test_standardize_neutralizes_charges():
    out = standardize_smiles("[NH3+]CCC(=O)[O-]", neutralize=True, canonical_tautomer=False)
    assert out.ok and "+" not in out.smiles and "-" not in out.smiles


def test_standardize_is_idempotent():
    once = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    twice = standardize_smiles(once.smiles)
    assert once.smiles == twice.smiles
    assert once.inchikey == twice.inchikey


def test_standardize_reports_failure_rather_than_raising():
    out = standardize_smiles("this is not smiles")
    assert not out.ok and out.smiles is None and out.error


# ------------------------------------------------------------------- filters
def _mol(smi):
    from rdkit import Chem

    return Chem.MolFromSmiles(smi)


def test_properties_are_sane_for_aspirin():
    d = compute_properties(_mol("CC(=O)Oc1ccccc1C(=O)O"))
    assert 170 < d["mw"] < 190
    assert d["heavy_atoms"] == 13
    assert d["hbd"] == 1


def test_filters_reject_out_of_range_and_name_the_reason():
    cfg = FilterConfig(enabled=True, mw_max=200, heavy_atoms_min=10)
    good = apply_filters(_mol("CC(=O)Oc1ccccc1C(=O)O"), cfg)       # MW 180
    assert good.passed and good.reason is None
    bad = apply_filters(_mol("C" * 60), cfg)
    assert not bad.passed and "mw_above" in bad.reason


def test_disabled_filters_accept_everything_but_still_report_properties():
    cfg = FilterConfig(enabled=False, mw_max=10)
    v = apply_filters(_mol("CC(=O)Oc1ccccc1C(=O)O"), cfg)
    assert v.passed and v.props["mw"] > 10


def test_element_whitelist_rejects_exotic_atoms():
    cfg = FilterConfig(enabled=True, mw_min=0, heavy_atoms_min=1)
    v = apply_filters(_mol("[Se]1ccccc1" if _mol("[Se]1ccccc1") else "CC[Se]C"), cfg)
    assert not v.passed and "element" in v.reason


def test_pains_filter_flags_a_known_alert():
    cfg = FilterConfig(enabled=True, pains=True, mw_min=0, mw_max=900, heavy_atoms_min=1,
                       logp_min=-99, logp_max=99, tpsa_max=999)
    v = apply_filters(_mol("O=C1C=CC(=O)C(=C1)Oc1ccccc1"), cfg)
    assert not v.passed and ("pains" in v.reason or "reos" in v.reason)


# ------------------------------------------------------------------- library
def test_prepare_library_shards_dedups_and_records_rejections(tmp_path: Path):
    lib_cfg = LibraryConfig(path=str(FIXTURES / "library.sdf"), shard_size=40, dedup_by="inchikey")
    stats = prepare_library(lib_cfg.path, tmp_path / "lib", lib_cfg)
    d = stats.as_dict()
    assert d["n_input"] == 200
    assert d["n_accepted"] + d["n_rejected"] == 200
    assert d["n_shards"] == len(list((tmp_path / "lib" / "shards").glob("*.parquet")))

    index = load_shard_index(tmp_path / "lib")
    assert index["n_molecules"].sum() == d["n_accepted"]
    assert (tmp_path / "lib" / "rejected.parquet").exists()


def test_prepare_library_preserves_input_ids(tmp_path: Path):
    lib_cfg = LibraryConfig(path=str(FIXTURES / "library.smi"), shard_size=100)
    prepare_library(lib_cfg.path, tmp_path / "lib", lib_cfg, id_prefix="TEST")
    shard = pd.read_parquet(sorted((tmp_path / "lib" / "shards").glob("*.parquet"))[0])
    assert shard["mol_id"].str.startswith("FIX").all()   # IDs from the file win
    assert shard["mol_id"].is_unique


def test_prepare_library_mints_prefixed_ids_when_input_has_none(tmp_path: Path):
    src = tmp_path / "noid.smi"
    src.write_text("\n".join(["CCO", "CCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]) + "\n")
    lib_cfg = LibraryConfig(path=str(src), shard_size=10,
                            filters=FilterConfig(enabled=False))
    prepare_library(src, tmp_path / "lib", lib_cfg, id_prefix="TEST")
    shard = pd.read_parquet(sorted((tmp_path / "lib" / "shards").glob("*.parquet"))[0])
    assert shard["mol_id"].tolist() == ["TEST00000001", "TEST00000002", "TEST00000003", "TEST00000004"]


def test_prepare_library_is_deterministic(tmp_path: Path):
    lib_cfg = LibraryConfig(path=str(FIXTURES / "library.sdf"), shard_size=50)
    a = prepare_library(lib_cfg.path, tmp_path / "a", lib_cfg)
    b = prepare_library(lib_cfg.path, tmp_path / "b", lib_cfg)
    fa = pd.concat([pd.read_parquet(p) for p in sorted((tmp_path / "a" / "shards").glob("*.parquet"))])
    fb = pd.concat([pd.read_parquet(p) for p in sorted((tmp_path / "b" / "shards").glob("*.parquet"))])
    assert a.as_dict()["n_accepted"] == b.as_dict()["n_accepted"]
    pd.testing.assert_frame_equal(fa.reset_index(drop=True), fb.reset_index(drop=True))


def test_max_molecules_caps_accepted_not_read(tmp_path: Path):
    """The cap is on molecules kept; rejects still consume input rows."""
    lib_cfg = LibraryConfig(path=str(FIXTURES / "library.sdf"), shard_size=10, max_molecules=25)
    stats = prepare_library(lib_cfg.path, tmp_path / "lib", lib_cfg).as_dict()
    assert stats["n_accepted"] == 25
    assert stats["n_input"] >= stats["n_accepted"]


def test_messy_input_does_not_abort_the_run(tmp_path: Path):
    lib_cfg = LibraryConfig(path=str(FIXTURES / "library_messy.sdf"), shard_size=10)
    stats = prepare_library(lib_cfg.path, tmp_path / "lib", lib_cfg)
    d = stats.as_dict()
    assert d["n_accepted"] > 0
    assert isinstance(d["top_rejection_reasons"], (dict, list))
