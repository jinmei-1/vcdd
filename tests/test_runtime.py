"""Runtime layer: hashing, atomic writes, sharding, the resume ledger, manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcdd.runtime.hashing import hash_obj
from vcdd.runtime.ledger import RunLedger, UnitKey
from vcdd.runtime.manifest import Manifest, tool_versions
from vcdd.runtime.parallel import ItemResult, map_items, resolve_workers, shard_indices
from vcdd.runtime.paths import atomic_write_text, ensure_dir


# ----------------------------------------------------------------- hashing
def test_hash_obj_is_order_insensitive_for_dicts():
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})


def test_hash_obj_is_order_sensitive_for_lists():
    assert hash_obj([1, 2]) != hash_obj([2, 1])


def test_hash_obj_distinguishes_float_precision():
    assert hash_obj({"e": 4.0}) != hash_obj({"e": 4.0001})


# ------------------------------------------------------------------- paths
def test_atomic_write_text_leaves_no_partial_file(tmp_path: Path):
    target = tmp_path / "nested" / "out.json"
    atomic_write_text(target, json.dumps({"ok": True}))
    assert json.loads(target.read_text())["ok"] is True
    assert not list(tmp_path.glob("**/*.tmp"))


def test_ensure_dir_is_idempotent(tmp_path: Path):
    a = ensure_dir(tmp_path / "x" / "y")
    b = ensure_dir(tmp_path / "x" / "y")
    assert a == b and a.is_dir()


# ---------------------------------------------------------------- parallel
def test_shard_indices_partition_exactly():
    n, k = 97, 5
    shards = [shard_indices(n, i, k) for i in range(k)]
    flat = sorted(x for s in shards for x in s)
    assert flat == list(range(n))
    assert len({len(s) for s in shards}) <= 2  # round-robin stays balanced


def test_shard_indices_rejects_out_of_range():
    with pytest.raises(ValueError):
        shard_indices(10, 5, 3)


def _double(x: int) -> int:
    return x * 2


def _boom(x: int) -> int:
    if x == 3:
        raise ValueError("bad item")
    return x


def test_map_items_preserves_order_and_isolates_failures():
    results = map_items(_boom, list(range(6)), n_workers=1)
    assert [r.index for r in results] == list(range(6))
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1 and bad[0].index == 3 and "ValueError" in bad[0].error


def test_map_items_parallel_matches_serial():
    items = list(range(20))
    serial = [r.value for r in map_items(_double, items, n_workers=1)]
    parallel = [r.value for r in map_items(_double, items, n_workers=4)]
    assert serial == parallel == [i * 2 for i in items]


def test_map_items_handles_empty_input():
    assert map_items(_double, [], n_workers=4) == []


def test_resolve_workers():
    assert resolve_workers(3) == 3
    assert resolve_workers(None) >= 1
    assert resolve_workers(0) >= 1


# ------------------------------------------------------------------ ledger
def _key(unit: str = "u0", inputs: str = "i", params: str = "p") -> UnitKey:
    return UnitKey("stage1", unit, inputs, params)


def test_ledger_marks_and_reports_done(tmp_path: Path):
    artifact = tmp_path / "a.parquet"
    artifact.write_bytes(b"x")
    with RunLedger(tmp_path / "run.db") as led:
        k = _key()
        assert not led.is_done(k)
        led.mark_running(k)
        assert not led.is_done(k)
        led.mark_done(k, artifact=artifact, n_items=7)
        assert led.is_done(k)
        assert led.status(k)[0] == "done"


def test_ledger_treats_deleted_artifact_as_not_done(tmp_path: Path):
    """Resume must not skip a unit whose output file has vanished."""
    artifact = tmp_path / "gone.parquet"
    artifact.write_bytes(b"x")
    with RunLedger(tmp_path / "run.db") as led:
        k = _key()
        led.mark_done(k, artifact=artifact)
        assert led.is_done(k)
        artifact.unlink()
        assert not led.is_done(k)


def test_ledger_invalidates_on_param_change(tmp_path: Path):
    """A protocol change must not be silently reused - this is the resume contract."""
    with RunLedger(tmp_path / "run.db") as led:
        led.mark_done(_key(params="thread=8000"))
        assert led.is_done(_key(params="thread=8000"))
        assert not led.is_done(_key(params="thread=5000"))


def test_ledger_invalidates_on_input_change(tmp_path: Path):
    with RunLedger(tmp_path / "run.db") as led:
        led.mark_done(_key(inputs="libA"))
        assert not led.is_done(_key(inputs="libB"))


def test_ledger_failed_units_are_not_done(tmp_path: Path):
    with RunLedger(tmp_path / "run.db") as led:
        k = _key()
        led.mark_failed(k, "engine crashed")
        assert not led.is_done(k)
        assert led.status(k)[0] == "failed"


def test_ledger_reset_stage_clears_only_that_stage(tmp_path: Path):
    with RunLedger(tmp_path / "run.db") as led:
        led.mark_done(UnitKey("stage1", "a", "i", "p"))
        led.mark_done(UnitKey("stage2", "b", "i", "p"))
        led.reset_stage("stage1")
        assert not led.is_done(UnitKey("stage1", "a", "i", "p"))
        assert led.is_done(UnitKey("stage2", "b", "i", "p"))


def test_ledger_survives_reopen(tmp_path: Path):
    db = tmp_path / "run.db"
    with RunLedger(db) as led:
        led.mark_done(_key())
    with RunLedger(db) as led2:
        assert led2.is_done(_key())


def test_ledger_summary_counts_units(tmp_path: Path):
    with RunLedger(tmp_path / "run.db") as led:
        led.mark_done(UnitKey("stage1", "a", "i", "p"))
        led.mark_done(UnitKey("stage1", "b", "i", "p"))
        led.mark_failed(UnitKey("stage1", "c", "i", "p"), "nope")
        summary = led.summary()
    assert summary["stage1"]["done"] == 2
    assert summary["stage1"]["failed"] == 1


# ---------------------------------------------------------------- manifest
def test_manifest_records_inputs_stages_and_warnings(tmp_path: Path, base_cfg):
    lib = tmp_path / "lib.smi"
    lib.write_text("CCO ethanol\n")
    man = Manifest(tmp_path, base_cfg)
    man.add_input("library", lib)
    man.add_stage("stage1", {"n_scored": 3})
    man.warn("docking deltas are inside noise")
    path = man.save()

    data = json.loads(Path(path).read_text())
    assert data["inputs"]["library"]["sha256_16"]
    assert data["inputs"]["library"]["bytes"] == lib.stat().st_size
    assert data["stages"]["stage1"]["n_scored"] == 3
    assert data["warnings"][0]["message"] == "docking deltas are inside noise"
    assert data["vcdd_version"]
    assert "python" in data["tool_versions"]


def test_tool_versions_reports_python_and_rdkit():
    versions = tool_versions()
    assert "python" in versions
    assert "rdkit" in versions
