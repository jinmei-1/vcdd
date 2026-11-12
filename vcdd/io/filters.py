"""Cheap physicochemical + structural-alert filters.

Applied once during library preparation. Every rejection carries a machine-
readable reason and is written to ``library/rejected.parquet`` - molecules are
never dropped silently, because "how many did you throw away and why" is the
first question anyone asks of a screening funnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

#: REOS-style hard structural alerts (SMARTS). A deliberately compact set: the
#: aim is to remove obviously non-drug-like reactive matter, not to reproduce a
#: full commercial alert package.
REOS_SMARTS: dict[str, str] = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "alkyl_halide_reactive": "[CX4;!$(C(F)(F)F)][Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "michael_acceptor_nitrile": "[C;!R]=[C;!R]C#N",
    "isocyanate_isothiocyanate": "[NX2]=[CX2]=[O,S]",
    "epoxide_aziridine": "[OX2,NX3]1[CX4][CX4]1",
    "peroxide": "[OX2][OX2]",
    "azide": "[NX1]~[NX2]~[NX2,NX1]",
    "diazo": "[#6]=[NX2+]=[NX1-]",
    "phosphorane": "[PX4](=[OX1])([OX2H1])[OX2H1]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "thiol": "[SX2H]",
    "anhydride": "[CX3](=O)[OX2][CX3]=O",
    "nitroso": "[NX2]=[OX1]",
    "triflate": "[#6][OX2][SX4](=O)(=O)C(F)(F)F",
    "crown_ether": "[OX2r12,OX2r15,OX2r18]",
    "polyene": "[#6]=[#6]-[#6]=[#6]-[#6]=[#6]-[#6]=[#6]",
}


@dataclass
class FilterVerdict:
    passed: bool
    reason: str | None = None
    props: dict[str, float] | None = None


@lru_cache(maxsize=1)
def _pains_catalog():
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


@lru_cache(maxsize=1)
def _reos_patterns() -> tuple[tuple[str, Any], ...]:
    from rdkit import Chem

    out = []
    for name, smarts in REOS_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            out.append((name, patt))
    return tuple(out)


def compute_properties(mol: Any) -> dict[str, float]:
    """Physicochemical descriptors used by the filters and by later reporting."""
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
    }


def has_pains(mol: Any) -> str | None:
    """Return the PAINS family name if the molecule matches, else ``None``."""
    try:
        entry = _pains_catalog().GetFirstMatch(mol)
    except Exception:  # noqa: BLE001
        return None
    if entry is None:
        return None
    return entry.GetDescription()


def has_reos_alert(mol: Any) -> str | None:
    """Return the first matching REOS alert name, else ``None``."""
    for name, patt in _reos_patterns():
        try:
            if mol.HasSubstructMatch(patt):
                return name
        except Exception:  # noqa: BLE001
            continue
    return None


def disallowed_elements(mol: Any, allowed: set[str]) -> str | None:
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in allowed:
            return symbol
    return None


def apply_filters(mol: Any, cfg: Any) -> FilterVerdict:
    """Evaluate *mol* against a :class:`~vcdd.config.schema.FilterConfig`.

    Returns the first failing reason (cheap checks first) plus the computed
    property dictionary, which callers keep for downstream reporting.
    """
    props = compute_properties(mol)
    if not getattr(cfg, "enabled", True):
        return FilterVerdict(True, None, props)

    checks: list[tuple[bool, str]] = [
        (props["mw"] < cfg.mw_min, f"mw_below_{cfg.mw_min}"),
        (props["mw"] > cfg.mw_max, f"mw_above_{cfg.mw_max}"),
        (props["logp"] < cfg.logp_min, f"logp_below_{cfg.logp_min}"),
        (props["logp"] > cfg.logp_max, f"logp_above_{cfg.logp_max}"),
        (props["hbd"] > cfg.hbd_max, f"hbd_above_{cfg.hbd_max}"),
        (props["hba"] > cfg.hba_max, f"hba_above_{cfg.hba_max}"),
        (props["tpsa"] > cfg.tpsa_max, f"tpsa_above_{cfg.tpsa_max}"),
        (props["rotatable"] > cfg.rotatable_max, f"rotatable_above_{cfg.rotatable_max}"),
        (props["heavy_atoms"] < cfg.heavy_atoms_min, f"heavy_atoms_below_{cfg.heavy_atoms_min}"),
        (props["heavy_atoms"] > cfg.heavy_atoms_max, f"heavy_atoms_above_{cfg.heavy_atoms_max}"),
    ]
    for failed, reason in checks:
        if failed:
            return FilterVerdict(False, reason, props)

    bad_element = disallowed_elements(mol, set(cfg.allowed_elements))
    if bad_element:
        return FilterVerdict(False, f"element_{bad_element}", props)

    if cfg.pains:
        pains = has_pains(mol)
        if pains:
            return FilterVerdict(False, f"pains:{pains}", props)

    if cfg.reos:
        alert = has_reos_alert(mol)
        if alert:
            return FilterVerdict(False, f"reos:{alert}", props)

    return FilterVerdict(True, None, props)
