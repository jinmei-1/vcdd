"""Structural guardrails for generated analogs.

A generator optimising a surrogate score will happily exploit it: fuse rings
until the docking box is filled, hang greasy tails off every vector, or return
the parent unchanged because that is a safe local optimum.  Guardrails are the
constraint set that makes the generated set worth a chemist's attention.

Every check returns a reason string rather than a bare boolean, so rejected
molecules stay auditable -- ``analogs.parquet`` records why each candidate was
dropped instead of silently shrinking the output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from ..config.schema import GuardrailConfig

__all__ = [
    "GuardrailReport",
    "sa_score",
    "pains_alerts",
    "check_analog",
    "filter_analogs",
    "tanimoto",
    "morgan_fp",
]


@dataclass
class GuardrailReport:
    """Outcome of guardrail evaluation for one candidate."""

    smiles: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    similarity: float = math.nan
    heavy_atom_delta: int = 0
    mw: float = math.nan
    clogp: float = math.nan
    qed: float = math.nan
    sa: float = math.nan
    max_ring_size: int = 0
    n_bridgehead: int = 0
    n_spiro: int = 0
    pains: tuple[str, ...] = ()

    def to_row(self) -> dict:
        return {
            "analog_smiles": self.smiles,
            "guardrail_pass": self.passed,
            "guardrail_reasons": ";".join(self.reasons),
            "similarity_to_parent": self.similarity,
            "heavy_atom_delta": self.heavy_atom_delta,
            "mw": self.mw,
            "clogp": self.clogp,
            "qed": self.qed,
            "sa_score": self.sa,
            "max_ring_size": self.max_ring_size,
            "n_bridgehead": self.n_bridgehead,
            "n_spiro": self.n_spiro,
            "pains_alerts": ";".join(self.pains),
        }


# --------------------------------------------------------------------------- #
# Chemistry helpers
# --------------------------------------------------------------------------- #


def morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def tanimoto(mol_a, mol_b) -> float:
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(morgan_fp(mol_a), morgan_fp(mol_b)))


@lru_cache(maxsize=1)
def _sa_module():
    """RDKit's synthetic-accessibility contribution, which is not a public API."""
    try:
        from rdkit.Chem import RDConfig  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None
    import os
    import sys

    path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if path not in sys.path:
        sys.path.append(path)
    try:
        import sascorer  # noqa: PLC0415

        return sascorer
    except ImportError:  # pragma: no cover - contrib dir missing
        return None


def sa_score(mol) -> float:
    """Ertl-Schuffenhauer synthetic accessibility, 1 (easy) to 10 (hard).

    Returns NaN when the RDKit contrib scorer is unavailable, in which case the
    SA guardrail is skipped rather than silently passing everything.
    """
    mod = _sa_module()
    if mod is None:
        return math.nan
    try:
        return float(mod.calculateScore(mol))
    except Exception:  # noqa: BLE001 - a scorer failure must not stop generation
        return math.nan


@lru_cache(maxsize=1)
def _pains_catalog():
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def pains_alerts(mol) -> tuple[str, ...]:
    try:
        cat = _pains_catalog()
    except Exception:  # noqa: BLE001
        return ()
    return tuple(m.GetDescription() for m in cat.GetMatches(mol))


def _ring_stats(mol) -> tuple[int, int, int]:
    from rdkit.Chem import rdMolDescriptors

    ri = mol.GetRingInfo()
    largest = max((len(r) for r in ri.AtomRings()), default=0)
    return (
        largest,
        int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
    )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_analog(
    smiles: str,
    parent_mol,
    cfg: GuardrailConfig,
    similarity_floor: float | None = None,
) -> GuardrailReport:
    """Evaluate one candidate against every guardrail.

    ``similarity_floor`` overrides the config value so the GA (which explores
    further) and enumerative generators can use different floors.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED

    rep = GuardrailReport(smiles=smiles, passed=False)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rep.reasons.append("unparsable")
        return rep
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        rep.reasons.append(f"sanitize_failed:{type(exc).__name__}")
        return rep

    canon = Chem.MolToSmiles(mol)
    rep.smiles = canon
    parent_canon = Chem.MolToSmiles(parent_mol)

    rep.similarity = tanimoto(mol, parent_mol)
    rep.heavy_atom_delta = mol.GetNumHeavyAtoms() - parent_mol.GetNumHeavyAtoms()
    rep.mw = float(Descriptors.MolWt(mol))
    rep.clogp = float(Crippen.MolLogP(mol))
    rep.qed = float(QED.qed(mol))
    rep.sa = sa_score(mol)
    rep.max_ring_size, rep.n_bridgehead, rep.n_spiro = _ring_stats(mol)
    rep.pains = pains_alerts(mol) if cfg.pains else ()

    floor = cfg.similarity_floor_enum if similarity_floor is None else similarity_floor

    if cfg.reject_parent_identity and canon == parent_canon:
        rep.reasons.append("identical_to_parent")
    if rep.similarity < floor:
        rep.reasons.append(f"similarity<{floor:g}")
    if abs(rep.heavy_atom_delta) > cfg.heavy_atom_delta_max:
        rep.reasons.append(f"heavy_atom_delta>{cfg.heavy_atom_delta_max}")
    if rep.max_ring_size > cfg.max_ring_size:
        rep.reasons.append(f"ring_size>{cfg.max_ring_size}")
    if rep.n_bridgehead > cfg.max_bridgehead_atoms:
        rep.reasons.append(f"bridgeheads>{cfg.max_bridgehead_atoms}")
    if rep.n_spiro > cfg.max_spiro_atoms:
        rep.reasons.append(f"spiro>{cfg.max_spiro_atoms}")
    if math.isfinite(rep.sa) and rep.sa > cfg.sa_score_max:
        rep.reasons.append(f"sa_score>{cfg.sa_score_max:g}")
    if rep.mw > cfg.mw_max:
        rep.reasons.append(f"mw>{cfg.mw_max:g}")
    if rep.clogp > cfg.logp_max:
        rep.reasons.append(f"clogp>{cfg.logp_max:g}")
    if rep.pains:
        rep.reasons.append(f"pains:{rep.pains[0]}")

    rep.passed = not rep.reasons
    return rep


def filter_analogs(
    smiles_iter: Iterable[str],
    parent_smiles: str,
    cfg: GuardrailConfig,
    similarity_floor: float | None = None,
) -> list[GuardrailReport]:
    """Evaluate many candidates, de-duplicating on canonical SMILES."""
    from rdkit import Chem

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparsable parent SMILES: {parent_smiles}")
    seen: set[str] = set()
    out: list[GuardrailReport] = []
    for smi in smiles_iter:
        if not smi:
            continue
        rep = check_analog(smi, parent, cfg, similarity_floor)
        if rep.smiles in seen:
            continue
        seen.add(rep.smiles)
        out.append(rep)
    return out
