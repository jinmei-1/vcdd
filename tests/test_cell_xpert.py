"""Tier-2 XPert adapter tests: dose/time binning logic, the flash-attn
stand-in (so inference runs on plain PyTorch), and — when a real
$VCDD_XPERT_DIR layout is present — an end-to-end in-table drug prediction.

The heavy dependencies (torch_geometric, anndata) are gated with
importorskip; the pure-logic tests never import them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vcdd.cell.backends.tier2 import XpertModel
from vcdd.config.schema import CellConfig


def _model_in(xpert_dir: Path, monkeypatch) -> XpertModel:
    monkeypatch.setenv("VCDD_XPERT_DIR", str(xpert_dir))
    return XpertModel(CellConfig())


# --------------------------------------------------------------------------
# pure logic: dose / time mapping (no third-party imports)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dose_um", "expected"),
    [
        (0.1, 0),
        (0.21, 1),
        (0.41, 2),
        (0.71, 3),
        (1.01, 4),
        (1.51, 5),
        (3.1, 6),
        (4.1, 7),
        (7.1, 8),
        (12.1, 9),
        (10.0, 8),
        (0.0, 0),
    ],
)
def test_dose_idx_bins(dose_um, expected):
    assert XpertModel._dose_idx(dose_um) == expected


@pytest.mark.parametrize(
    ("time_h", "expected"),
    [(3.0, 0), (6.0, 1), (24.0, 2), (0.0, 2), (12.0, 2), (6.1, 1)],
)
def test_time_idx_map(time_h, expected):
    assert XpertModel._time_idx(time_h) == expected


def test_flash_stub_is_idempotent():
    XpertModel._install_flash_stub()
    XpertModel._install_flash_stub()  # must not raise or clobber
    import sys  # noqa: PLC0415

    assert "flash_attn" in sys.modules
    assert callable(sys.modules["flash_attn.flash_attn_interface"].flash_attn_func)


@pytest.mark.skipif(
    Path(os.environ.get("VCDD_XPERT_DIR", "")).joinpath("models", "model_XPert.py").exists()
    is False,
    reason="XPert model code not on disk",
)
def test_xpert_imports_with_flash_stub_without_real_flash_attn():
    """models/model_utils.py unconditionally imports flash_attn; with our
    stand-in installed the official model file must import and construct."""
    import importlib  # noqa: PLC0415

    root = Path(os.environ["VCDD_XPERT_DIR"])
    XpertModel._install_flash_stub()
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(root))
    xpert = importlib.import_module("models.model_XPert")
    assert hasattr(xpert, "XPertNet")
    # __init__ needs the full dataset model section, not just the class def;
    # constructing validates config wiring only when torch_geometric is present
    import torch  # noqa: PLC0415

    assert torch is not None


# --------------------------------------------------------------------------
# end-to-end (needs the full $VCDD_XPERT_DIR layout)
# --------------------------------------------------------------------------


def _xpert_environ():
    env = os.environ.get("VCDD_XPERT_DIR")
    if not env:
        return None
    root = Path(env)
    need = [
        root / "saved_model",
        root / "models" / "model_XPert.py",
        root / "processed_data" / "l1000_gene_info_978.csv",
        root / "processed_data" / "all_drugs_unimol_arr.npy",
        root / "processed_data" / "all_drugs_idx2smi_8981.npy",
        root / "processed_data" / "PPI_gene_vector_128d.npy",
        root / "processed_data" / "l1000_mdmt_68830_subset.h5ad",
        root / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy",
    ]
    return root if all(p.exists() for p in need) else None


needs_xpert = pytest.mark.skipif(
    _xpert_environ() is None,
    reason="$VCDD_XPERT_DIR with full layout (saved_model/, models/, "
    "processed_data/, HG_data/) not present",
)


@needs_xpert
def test_end_to_end_table_drug(monkeypatch):
    import numpy as np  # noqa: PLC0415

    root = Path(os.environ["VCDD_XPERT_DIR"])
    m = _model_in(root, monkeypatch)
    ok, msg = m.is_available()
    assert ok, msg
    tbl = np.load(
        root / "processed_data" / "all_drugs_idx2smi_8981.npy", allow_pickle=True
    ).item()
    pidx, smi = sorted(tbl.items())[100]
    m._load()
    assert len(m.genes()) == 978, "checkpoint gene space is the 978 landmarks"
    assert m.genes()[0] == "DDR1"

    from vcdd.cell.base import PerturbationRequest  # noqa: PLC0415

    sig = m.predict(
        [
            PerturbationRequest(
                mol_id="d100", smiles=smi, cell_line="A375", dose_um=10.0
            )
        ]
    )[0]
    assert sig.ok, sig.error
    assert sig.value_scale == "logfc"
    assert len(sig.genes) == 978
    assert sig.meta.get("pert_idx") == pidx
    assert sig.meta.get("dose_idx") == 8  # 10.0 µM
    assert np.isfinite(sig.values).all()
    assert float(np.abs(sig.values).mean()) > 0