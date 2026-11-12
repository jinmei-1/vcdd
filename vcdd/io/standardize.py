"""Molecule standardisation.

Pipeline (each step optional, controlled by ``LibraryConfig``):

1. parse + sanitize
2. largest-fragment selection (desalt / demix)
3. charge neutralisation
4. canonical tautomer
5. canonical SMILES + InChIKey for deduplication

The function is deliberately **idempotent**: standardising an already
standardised molecule returns the identical SMILES. A unit test enforces this,
because a non-idempotent standardiser silently breaks deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# Tautomer canonicalisation is the expensive step; cap the enumeration so a
# pathological molecule cannot stall a shard.
_MAX_TAUTOMERS = 100
_MAX_TRANSFORMS = 200


@dataclass
class StandardResult:
    ok: bool
    smiles: str | None = None
    inchikey: str | None = None
    error: str | None = None


@lru_cache(maxsize=1)
def _tautomer_enumerator():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(_MAX_TAUTOMERS)
    enumerator.SetMaxTransforms(_MAX_TRANSFORMS)
    return enumerator


@lru_cache(maxsize=1)
def _uncharger():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _largest_fragment_chooser():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.LargestFragmentChooser()


def standardize_mol(
    mol: Any,
    desalt: bool = True,
    neutralize: bool = True,
    canonical_tautomer: bool = True,
) -> Any | None:
    """Standardise an RDKit ``Mol`` in place-safe fashion; ``None`` on failure."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        if desalt:
            mol = _largest_fragment_chooser().choose(mol)
        if neutralize:
            mol = _uncharger().uncharge(mol)
        if canonical_tautomer:
            mol = _tautomer_enumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:  # noqa: BLE001 - a single unparseable molecule must not abort a shard
        return None


def standardize_smiles(
    smiles: str,
    desalt: bool = True,
    neutralize: bool = True,
    canonical_tautomer: bool = True,
    compute_inchikey: bool = True,
) -> StandardResult:
    """Standardise a SMILES string and return canonical SMILES + InChIKey."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    if not smiles or not smiles.strip():
        return StandardResult(ok=False, error="empty_smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StandardResult(ok=False, error="unparseable_smiles")
    if mol.GetNumAtoms() == 0:
        return StandardResult(ok=False, error="no_atoms")

    std = standardize_mol(mol, desalt=desalt, neutralize=neutralize, canonical_tautomer=canonical_tautomer)
    if std is None or std.GetNumAtoms() == 0:
        return StandardResult(ok=False, error="standardization_failed")

    try:
        canonical = Chem.MolToSmiles(std)
    except Exception as exc:  # noqa: BLE001
        return StandardResult(ok=False, error=f"canonicalization_failed:{exc}")

    inchikey = None
    if compute_inchikey:
        try:
            inchikey = Chem.MolToInchiKey(std) or None
        except Exception:  # noqa: BLE001 - InChI can fail on exotic valences
            inchikey = None
    return StandardResult(ok=True, smiles=canonical, inchikey=inchikey)


def dedup_key(result: StandardResult, mode: str) -> str | None:
    """Key used for library deduplication."""
    if mode == "none":
        return None
    if mode == "smiles":
        return result.smiles
    if result.inchikey is None:
        return result.smiles  # fall back rather than dropping the molecule
    if mode == "inchikey14":
        return result.inchikey.split("-")[0]  # skeleton only: merges stereoisomers
    return result.inchikey
