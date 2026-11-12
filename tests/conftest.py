"""Shared pytest fixtures.

Design notes for anyone adding tests:

* The default engine everywhere in this suite is ``mock`` so the unit layer runs
  in seconds and never depends on a docking binary being installed. Tests that
  genuinely need a real binary or credential are gated behind the markers in
  ``_helpers.py``: :data:`requires_vina_gpu` (a compiled Vina-GPU plus a GPU),
  :data:`requires_boltz_test_key` (a *test-mode* Boltz key - it refuses live
  keys so the suite can never bill anyone) and :data:`requires_vina` (a stray
  CPU Vina, used only to assert it is NOT wired into the package).
* Nothing here writes into the package tree. Every run directory is a
  ``tmp_path`` so tests can run in parallel and leave no residue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def library_sdf() -> Path:
    return FIXTURES / "library.sdf"


@pytest.fixture(scope="session")
def receptor_pdb() -> Path:
    return FIXTURES / "receptor.pdb"


@pytest.fixture(scope="session")
def reference_ligand() -> Path:
    return FIXTURES / "reference_ligand.sdf"


@pytest.fixture
def base_cfg(tmp_path: Path, library_sdf: Path, receptor_pdb: Path, reference_ligand: Path):
    """A tiny end-to-end-capable config wired to the mock engine."""
    from vcdd.config.loader import load_config

    overrides = {
        "name": "unit",
        "out_dir": str(tmp_path / "run"),
        "n_workers": 1,
        "library": {"path": str(library_sdf), "shard_size": 8, "max_molecules": 16},
        "receptor": {
            "path": str(receptor_pdb),
            "reference_ligand_path": str(reference_ligand),
            "chain": "A",
            "redock_validation": False,
        },
        "stage1": {"engine": "mock", "pilot_size": 0, "gpu_batch_size": 8},
        "stage2": {"top_n": 6, "chunk_size": 3, "backend": "library_screen",
                   "fusion": "boltz_only", "keep_structures_top_k": 4,
                   "poll_interval_s": 0, "max_spend_usd": 5.0},
        "cell": {"backends": ["mock"], "top_n": 6},
        "stage4": {"top_parents": 2, "generators": ["rgroup"], "analogs_per_parent": 4,
                   "redock_mode": "stage1"},
    }
    return load_config(None, overrides=overrides)


@pytest.fixture
def ledger(tmp_path: Path):
    from vcdd.runtime.ledger import RunLedger

    led = RunLedger(tmp_path / "run.db")
    yield led
    led.close()


@pytest.fixture
def small_box():
    """Box around the biotin site of the streptavidin fixture."""
    from vcdd.receptor.box import Box

    return Box(center=(11.419, 2.371, -11.373), size=(16.0, 16.0, 16.0))


@pytest.fixture
def prepared_receptor(tmp_path: Path, receptor_pdb: Path) -> Path:
    from vcdd.receptor.prepare import prepare_receptor

    result = prepare_receptor(receptor_pdb, tmp_path / "receptor", chain="A")
    return Path(result.pdbqt_path)


@pytest.fixture
def prepared_library(tmp_path: Path, base_cfg) -> Path:
    from vcdd.io.library import prepare_library

    out = tmp_path / "run" / "library"
    prepare_library(base_cfg.library.path, out, base_cfg.library)
    return out


# ------------------------------------------------------------------- stage 2
@pytest.fixture
def fake_boltz(small_box):
    """A scripted Boltz API centred on the fixture docking box."""
    from fake_boltz import FakeBoltz

    return FakeBoltz(box_center=small_box.center, running_polls=1, page_size=2)


@pytest.fixture
def boltz_client(fake_boltz):
    """A real SDK client wired to the scripted API."""
    from fake_boltz import make_fake_client

    return make_fake_client(fake_boltz)


@pytest.fixture
def stage1_done(base_cfg, prepared_library, prepared_receptor, tmp_path, small_box, ledger):
    """A finished stage 1 on the real prepared receptor, ready for stage 2.

    Stage 2 needs a receptor it can read *sequences* out of, so this uses the
    prepared PDBQT rather than the one-line stub the stage-1 tests get away with.
    """
    from vcdd.screening.stage1 import run_stage1

    out = tmp_path / "stage1"
    run_stage1(base_cfg, prepared_receptor, small_box, prepared_library, out, ledger)
    return prepared_receptor, out


