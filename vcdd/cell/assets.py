"""Access to the offline reference bundle in ``vcdd/cell/resources``.

The bundle is produced by ``scripts/build_cell_resources.py``.  Every loader
raises :class:`ResourceMissing` rather than returning a silently empty table, so
a backend can convert the failure into an "unavailable, here is how to fix it"
message instead of scoring compounds against nothing.

Set ``VCDD_RESOURCE_DIR`` to point at a bundle outside the installed package
(useful when the package lives on a read-only mount).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "ResourceMissing",
    "resource_dir",
    "manifest",
    "available",
    "bundle_status",
    "load_l1000_genes",
    "gene_universe",
    "load_progeny",
    "load_collectri",
    "load_hallmark",
    "load_lincs_chem",
    "load_lincs_crispr",
    "signature_from_updown",
]

BUILD_HINT = (
    "run `python scripts/build_cell_resources.py` (needs network) or set "
    "VCDD_RESOURCE_DIR to an existing bundle"
)

#: Logical resource name -> file name inside the bundle.
FILES = {
    "l1000_genes": "l1000_genes.tsv",
    "progeny": "progeny.tsv",
    "collectri": "collectri.tsv",
    "hallmark": "hallmark.tsv",
    "lincs_chem_consensus": "lincs_chem_consensus.parquet",
    "lincs_crispr_ko_consensus": "lincs_crispr_ko_consensus.parquet",
}


class ResourceMissing(FileNotFoundError):
    """A required reference table is not present in the bundle."""

    def __init__(self, name: str) -> None:
        self.resource = name
        super().__init__(
            f"cell resource '{name}' ({FILES.get(name, name)}) not found in "
            f"{resource_dir()}; {BUILD_HINT}"
        )


def resource_dir() -> Path:
    env = os.environ.get("VCDD_RESOURCE_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "resources"


def _path(name: str) -> Path:
    p = resource_dir() / FILES[name]
    if not p.exists():
        raise ResourceMissing(name)
    return p


@lru_cache(maxsize=1)
def manifest() -> dict:
    p = resource_dir() / "MANIFEST.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def available(name: str) -> bool:
    """True when ``name`` is present on disk."""
    return name in FILES and (resource_dir() / FILES[name]).exists()


def bundle_status() -> dict[str, dict]:
    """Per-resource presence, size and provenance, for ``vcdd doctor``."""
    meta = manifest().get("resources", {})
    out: dict[str, dict] = {}
    for name, fname in FILES.items():
        p = resource_dir() / fname
        entry = {"file": fname, "present": p.exists()}
        if p.exists():
            entry["bytes"] = p.stat().st_size
        for key in ("rows", "source", "built_utc", "n_landmark", "n_with_smiles"):
            if key in meta.get(name, {}):
                entry[key] = meta[name][key]
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_l1000_genes() -> pd.DataFrame:
    """L1000 gene annotation with ``is_landmark`` / ``is_bing`` flags."""
    return pd.read_csv(_path("l1000_genes"), sep="\t", dtype={"entrez_id": str})


def gene_universe(space: str = "landmark") -> list[str]:
    """Ordered gene-symbol universe for a :class:`CellConfig` ``gene_space``.

    ``all`` still returns the BING set: the bundle only annotates genes L1000
    actually measures or infers, and scoring against unmeasured genes would
    invent data.
    """
    df = load_l1000_genes()
    if space == "landmark":
        df = df[df["is_landmark"] == 1]
    syms = df["gene_symbol"].dropna().astype(str).unique().tolist()
    return sorted(syms)


@lru_cache(maxsize=1)
def load_progeny() -> pd.DataFrame:
    """PROGENy footprints: ``pathway, gene_symbol, weight, p_value``."""
    return pd.read_csv(_path("progeny"), sep="\t")


@lru_cache(maxsize=1)
def load_collectri() -> pd.DataFrame:
    """Signed TF regulon: ``tf, target, mor`` (mode of regulation, +1/-1)."""
    return pd.read_csv(_path("collectri"), sep="\t")


@lru_cache(maxsize=1)
def load_hallmark() -> pd.DataFrame:
    """MSigDB Hallmark membership: ``gene_set, gene_symbol``."""
    return pd.read_csv(_path("hallmark"), sep="\t")


@lru_cache(maxsize=1)
def load_lincs_chem() -> pd.DataFrame:
    """LINCS consensus compound signatures with Repurposing-Hub SMILES."""
    return pd.read_parquet(_path("lincs_chem_consensus"))


@lru_cache(maxsize=1)
def load_lincs_crispr() -> pd.DataFrame:
    """LINCS consensus CRISPR-knockout signatures, keyed by gene symbol."""
    return pd.read_parquet(_path("lincs_crispr_ko_consensus"))


def signature_from_updown(
    up_genes,
    down_genes,
    genes: list[str],
    up_value: float = 1.0,
    down_value: float = -1.0,
) -> np.ndarray:
    """Project an up/down gene-set pair onto a fixed gene ordering.

    Enrichr consensus sets are ranked but unsigned in magnitude, so membership
    is encoded as +1 / -1 rather than a fabricated effect size.
    """
    index = {g: i for i, g in enumerate(genes)}
    vec = np.zeros(len(genes), dtype=np.float32)
    # Parquet list columns deserialise as numpy arrays, whose truthiness is
    # ambiguous -- test for None explicitly rather than with `or`.
    for members, value in ((up_genes, up_value), (down_genes, down_value)):
        if members is None:
            continue
        for g in members:
            i = index.get(g)
            if i is not None:
                vec[i] = value
    return vec
