"""Test-only constants and skip markers, importable from any test module."""

from __future__ import annotations

import shutil

import pytest

SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "benzene": "c1ccccc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "imatinib_core": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "phenol": "Oc1ccccc1",
    "toluene": "Cc1ccccc1",
    "aniline": "Nc1ccccc1",
    "pyridine": "c1ccncc1",
    "piperidine_amide": "O=C(N1CCCCC1)c1ccccc1",
}


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


requires_vina = pytest.mark.skipif(not _have("vina"), reason="AutoDock Vina binary not installed")
requires_obabel = pytest.mark.skipif(not _have("obabel"), reason="Open Babel not installed")


def _vina_gpu_present() -> bool:
    """A usable stage-1 engine means a compiled binary *and* a visible GPU."""
    try:
        from vcdd.docking.base import gpu_available
        from vcdd.docking.engines.vina_gpu import VinaGPUEngine

        return any(VinaGPUEngine.variant_binaries().values()) and gpu_available()
    except Exception:  # pragma: no cover - defensive
        return False


requires_vina_gpu = pytest.mark.skipif(
    not _vina_gpu_present(),
    reason="Vina-GPU 2.1 binary + GPU not available; run scripts/build_vina_gpu.sh",
)


def _boltz_test_key_present() -> bool:
    """Only a *test-mode* key may be used by the suite: live keys cost money."""
    try:
        from vcdd.refine.boltz.client import boltz_available, resolve_credentials

        if not boltz_available():
            return False
        creds = resolve_credentials("BOLTZ_API_KEY")
        return creds.present and creds.livemode is False
    except Exception:  # pragma: no cover - defensive
        return False


requires_boltz_test_key = pytest.mark.skipif(
    not _boltz_test_key_present(),
    reason="BOLTZ_API_KEY is unset or is a live key; the suite only runs against test-mode keys",
)


def _resources_present() -> bool:
    from vcdd.cell import assets

    try:
        return all(entry.get("present") for entry in assets.bundle_status().values())
    except Exception:  # pragma: no cover - defensive
        return False


requires_cell_resources = pytest.mark.skipif(
    not _resources_present(),
    reason="cell resource bundle missing; run scripts/build_cell_resources.py",
)
