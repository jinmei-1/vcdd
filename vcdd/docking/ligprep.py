"""Ligand preparation: SMILES -> 3D conformers -> PDBQT.

Two profiles, matching the two docking stages:

* **fast** (stage 1): one ETKDGv3 conformer + a short MMFF minimisation, single
  protonation state. Cheap enough for 10^6 molecules.
* **accurate** (stage 2): N diverse conformers, optional tautomer enumeration,
  full minimisation.

PDBQT conversion prefers Meeko (correct rotatable-bond detection and atom
typing) and falls back to Open Babel.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vcdd.runtime.logging import get_logger

LOG = get_logger()


@dataclass
class PreparedLigand:
    mol_id: str
    smiles: str
    pdbqt: str | None = None
    n_conformers: int = 0
    n_rotatable: int = 0
    tautomer_index: int = 0
    error: str | None = None
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.pdbqt)


def embed_conformers(
    smiles: str,
    n_conformers: int = 1,
    minimize: bool = True,
    seed: int = 42,
    prune_rms: float = 0.5,
    max_iters: int = 400,
) -> Any | None:
    """Return an RDKit ``Mol`` with hydrogens and >=1 embedded conformer."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms if n_conformers > 1 else -1.0
    params.useSmallRingTorsions = True
    params.enforceChirality = True

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_conformers), params=params)
    if len(cids) == 0:
        # Retry with random coordinates: helps macrocycles and cages.
        params.useRandomCoords = True
        params.maxIterations = 2000
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_conformers), params=params)
    if len(cids) == 0:
        return None

    if minimize:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        except Exception:  # noqa: BLE001 - MMFF has no parameters for some elements
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
            except Exception:  # noqa: BLE001
                pass
    return mol


def enumerate_tautomers(smiles: str, max_tautomers: int = 4) -> list[str]:
    """Return up to *max_tautomers* tautomer SMILES, canonical form first."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max(1, max_tautomers) * 4)
    try:
        canonical = Chem.MolToSmiles(enumerator.Canonicalize(mol))
        tauts = [Chem.MolToSmiles(t) for t in enumerator.Enumerate(mol)]
    except Exception:  # noqa: BLE001
        return [smiles]
    ordered = [canonical] + [t for t in tauts if t != canonical]
    return ordered[: max(1, max_tautomers)]


#: AutoDock atom types the Vina-GPU binary parses, in the exact case it expects
#: (Vina's type table is case-sensitive: ``Cl`` and ``Br`` are not ``CL``/``BR``).
#: Open Babel's PDBQT writer annotates types with the Gasteiger charge (``CG0``,
#: ``CG+``, ``OA-``), which the binary rejects with "not a valid AutoDock type" -
#: and because it docks a whole directory in one invocation, one such ligand
#: aborts the entire batch.
_AD_TYPE_MAP = {
    "C": "C", "A": "A", "N": "N", "NA": "NA", "NS": "NS", "OA": "OA", "OS": "OS",
    "S": "S", "SA": "SA", "H": "H", "HD": "HD", "HS": "HS", "F": "F", "P": "P",
    "CL": "Cl", "BR": "Br", "I": "I",
}
#: Metal / metalloid types: rejected rather than guessed. ``CA`` would otherwise
#: be read as carbon by the prefix scan below, silently mis-typing a metal.
_AD_REJECT = frozenset({
    "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "PT", "AU", "AG",
    "AL", "SI", "B", "SE", "AS", "MO", "SN", "PB", "LI", "K", "RB", "CS", "SR", "BA",
})
#: Meeko's glue pseudo-atoms and the atoms they are anchored to (``G0``-``G3``,
#: ``CG0``-``CG3``). They exist only to hold a bond open after meeko broke a
#: macrocycle; they are not atom types Vina has, and the ligand cannot be
#: docked while they are present - see :func:`mol_to_pdbqt`.
_AD_GLUE = frozenset({f"G{i}" for i in range(4)} | {f"CG{i}" for i in range(4)})
#: Longest first, so ``NA`` is not matched as ``N``.
_AD_FALLBACKS = ("NA", "NS", "OA", "OS", "SA", "HD", "HS", "CL", "BR", "A", "C",
                 "N", "O", "S", "H", "F", "P", "I")


def _ad_type(raw: str) -> str | None:
    """Map one PDBQT type field onto a type Vina accepts, or None if unknown.

    Metals and anything else outside the ligand-relevant set are rejected rather
    than guessed: dropping one ligand costs a molecule, mis-typing an atom costs
    a wrong score.
    """
    t = raw.strip()
    if not t:
        return None
    if t.upper() in _AD_GLUE:
        return None
    if t in _AD_TYPE_MAP.values():
        return t
    up = t.upper()
    if up in _AD_TYPE_MAP:
        return _AD_TYPE_MAP[up]
    base = up.rstrip("0123456789+-")
    if base in _AD_REJECT:
        return None
    if base in _AD_TYPE_MAP:
        return _AD_TYPE_MAP[base]
    for candidate in _AD_FALLBACKS:
        if base.startswith(candidate):
            return _AD_TYPE_MAP["OA" if candidate == "O" else candidate]
    return None


def sanitize_pdbqt_types(text: str) -> tuple[str | None, str | None]:
    """Rewrite non-AutoDock atom types, or report the first one that is unknown.

    Returns ``(text, error)``: the ligand is rejected rather than passed on with
    a type the engine will choke on, because a rejected ligand costs one
    molecule while an unparsable one costs the whole batch.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 79:
            # The type is the tail of the line, not a fixed 2-column field:
            # charge- and glue-annotated types are three characters wide, and
            # slicing [77:79] would read "CG0" as "CG".
            raw = line[77:].strip()
            mapped = _ad_type(raw)
            if mapped is None:
                kind = "glue_atom_type" if raw.upper() in _AD_GLUE else "unmappable_atom_type"
                return None, f"{kind}:{raw!r}"
            line = f"{line[:77]}{mapped:>2s}"
        out.append(line)
    return "\n".join(out) + "\n", None


def _meeko_pdbqt(mol: Any, conf_id: int, rigid_macrocycles: bool) -> tuple[str | None, str | None]:
    """One meeko pass with sanitised atom types. Returns ``(pdbqt, error)``."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    prep = MoleculePreparation(rigid_macrocycles=rigid_macrocycles)
    setups = prep.prepare(mol, conformer_id=conf_id if conf_id >= 0 else -1)
    if not setups:
        return None, "meeko_no_setup"
    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not (ok and text):
        return None, f"meeko:{err}"
    clean, type_err = sanitize_pdbqt_types(text)
    if type_err:
        return None, f"meeko:{type_err}"
    return clean, None


def mol_to_pdbqt(mol: Any, conf_id: int = -1) -> tuple[str | None, str | None]:
    """Convert one conformer to a PDBQT string. Returns ``(pdbqt, error)``.

    Meeko opens macrocycles to model their ring flexibility and marks the two
    atoms of every opened bond with glue pseudo-atoms (types ``G0``/``CG0``).
    Vina-GPU has no such types: it either aborts the invocation or silently
    skips the ligand, so those molecules could never be docked - 3.2% of the
    production library was lost this way. The second pass therefore keeps every
    ring rigid (classic AutoDock behaviour: ring bonds are not torsions anyway)
    which produces standard types and a dockable ligand, at the cost of the
    ring-opening torsions for those few molecules.
    """
    meeko_err = "meeko:unavailable"
    try:
        pdbqt, meeko_err = _meeko_pdbqt(mol, conf_id, rigid_macrocycles=False)
        if pdbqt:
            return pdbqt, None
        if "atom_type" in meeko_err:
            retry, retry_err = _meeko_pdbqt(mol, conf_id, rigid_macrocycles=True)
            if retry:
                return retry, None
            meeko_err = f"{meeko_err}|rigid_rings:{retry_err}"
    except Exception as exc:  # noqa: BLE001
        meeko_err = f"meeko:{type(exc).__name__}:{exc}"

    # Fallback: Open Babel via a temporary SDF.
    exe = shutil.which("obabel")
    if exe is None:
        return None, meeko_err
    try:
        from rdkit import Chem

        with tempfile.TemporaryDirectory() as td:
            sdf = Path(td) / "lig.sdf"
            out = Path(td) / "lig.pdbqt"
            w = Chem.SDWriter(str(sdf))
            w.write(mol, confId=conf_id)
            w.close()
            proc = subprocess.run(
                [exe, str(sdf), "-O", str(out), "--partialcharge", "gasteiger"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                # Open Babel annotates types with the Gasteiger charge too.
                clean, type_err = sanitize_pdbqt_types(out.read_text())
                if type_err:
                    return None, f"{meeko_err}|obabel:{type_err}"
                return clean, None
            return None, f"{meeko_err}|obabel_rc{proc.returncode}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{meeko_err}|obabel:{exc}"


def prepare_ligand(
    mol_id: str,
    smiles: str,
    n_conformers: int = 1,
    minimize: bool = True,
    seed: int = 42,
) -> PreparedLigand:
    """Fast path: one PDBQT string per ligand (best conformer only)."""
    from rdkit.Chem import rdMolDescriptors

    mol = embed_conformers(smiles, n_conformers=n_conformers, minimize=minimize, seed=seed)
    if mol is None:
        return PreparedLigand(mol_id=mol_id, smiles=smiles, error="embedding_failed")
    pdbqt, err = mol_to_pdbqt(mol, conf_id=0 if mol.GetNumConformers() else -1)
    if pdbqt is None:
        return PreparedLigand(mol_id=mol_id, smiles=smiles, error=f"pdbqt_failed:{err}")
    return PreparedLigand(
        mol_id=mol_id,
        smiles=smiles,
        pdbqt=pdbqt,
        n_conformers=mol.GetNumConformers(),
        n_rotatable=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    )


def prepare_ligand_variants(
    mol_id: str,
    smiles: str,
    n_conformers: int = 5,
    enumerate_tauts: bool = True,
    max_tautomers: int = 4,
    minimize: bool = True,
    seed: int = 42,
) -> list[PreparedLigand]:
    """Accurate path: one :class:`PreparedLigand` per (tautomer, conformer)."""
    from rdkit.Chem import rdMolDescriptors

    smiles_list = enumerate_tautomers(smiles, max_tautomers) if enumerate_tauts else [smiles]
    out: list[PreparedLigand] = []
    for t_idx, smi in enumerate(smiles_list):
        mol = embed_conformers(smi, n_conformers=n_conformers, minimize=minimize, seed=seed + t_idx)
        if mol is None:
            out.append(PreparedLigand(mol_id=mol_id, smiles=smi, tautomer_index=t_idx, error="embedding_failed"))
            continue
        n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        for conf in mol.GetConformers():
            pdbqt, err = mol_to_pdbqt(mol, conf_id=conf.GetId())
            if pdbqt is None:
                out.append(
                    PreparedLigand(mol_id=mol_id, smiles=smi, tautomer_index=t_idx, error=f"pdbqt_failed:{err}")
                )
                continue
            out.append(
                PreparedLigand(
                    mol_id=mol_id,
                    smiles=smi,
                    pdbqt=pdbqt,
                    n_conformers=1,
                    n_rotatable=n_rot,
                    tautomer_index=t_idx,
                    props={"conf_id": conf.GetId()},
                )
            )
    return out


#: AutoDock atom types -> element symbols.
AD_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O",
    "O": "O", "SA": "S", "S": "S", "P": "P", "F": "F", "Cl": "Cl", "CL": "Cl",
    "Br": "Br", "BR": "Br", "I": "I", "H": "H", "HD": "H", "HS": "H",
    "Mg": "Mg", "MG": "Mg", "Zn": "Zn", "ZN": "Zn", "Mn": "Mn", "MN": "Mn",
    "Ca": "Ca", "CA": "Ca", "Fe": "Fe", "FE": "Fe", "Si": "Si", "B": "B",
}


def pdbqt_heavy_coords(pdbqt_text: str) -> list[tuple[float, float, float]]:
    """Heavy-atom coordinates from a PDBQT block, in file order.

    PDBQT keeps polar hydrogens (``HD``); those are dropped so the result lines
    up with an RDKit mol that has had ``RemoveHs`` applied.
    """
    coords: list[tuple[float, float, float]] = []
    for line in pdbqt_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = line[77:79].strip() if len(line) > 77 else line[76:78].strip()
        element = AD_TYPE_TO_ELEMENT.get(ad_type, ad_type.capitalize())
        if element == "H":
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return coords


def _mol_from_pdbqt_meeko(pdbqt_text: str) -> Any | None:
    """Official Meeko round trip, using the ``REMARK SMILES`` records.

    Meeko embeds the input SMILES and an atom-index map in the ligand PDBQT, and
    Vina passes those REMARK lines through to its output, so this reconstructs
    full bond orders and stereochemistry rather than guessing from geometry.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except Exception:  # noqa: BLE001
        return None
    if "REMARK SMILES" not in pdbqt_text:
        return None
    try:
        pmol = PDBQTMolecule(pdbqt_text, skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        for m in mols:
            if m is not None and m.GetNumAtoms() > 0:
                return m
    except Exception:  # noqa: BLE001
        return None
    return None


def _mol_from_pdbqt_template(pdbqt_text: str, template_smiles: str) -> Any | None:
    """Transfer pose coordinates onto a template molecule by atom order.

    Meeko writes atoms in RDKit index order, so the k-th heavy atom of the pose
    corresponds to the k-th heavy atom of the template. Verified by atom-count
    and element agreement; returns ``None`` when they disagree rather than
    producing a silently scrambled pose.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        return None
    template = Chem.RemoveHs(template)
    coords = pdbqt_heavy_coords(pdbqt_text)
    if len(coords) != template.GetNumAtoms():
        return None
    mol = Chem.Mol(template)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    try:
        AllChem.AssignStereochemistryFrom3D(mol)
    except Exception:  # noqa: BLE001
        pass
    return mol


def _mol_from_pdbqt_pdbblock(pdbqt_text: str) -> Any | None:
    """Last resort: reinterpret the PDBQT as a PDB and let RDKit infer bonds."""
    from rdkit import Chem

    lines = []
    for line in pdbqt_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue  # ROOT/BRANCH/ENDROOT/ENDBRANCH/TORSDOF must be dropped,
            # otherwise "ENDROOT" is read as a PDB END record and truncates the
            # molecule at the first atom.
        ad_type = line[77:79].strip() if len(line) > 77 else line[76:78].strip()
        element = AD_TYPE_TO_ELEMENT.get(ad_type, ad_type.capitalize())
        if element == "H":
            continue
        padded = line.ljust(80)
        lines.append(padded[:76] + f"{element:>2s}" + "  ")
    if not lines:
        return None
    mol = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=True, sanitize=False, proximityBonding=True)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:  # noqa: BLE001
        pass
    return mol


def pdbqt_to_mol(pdbqt_text: str, template_smiles: str | None = None) -> Any | None:
    """Parse a docked PDBQT pose back into an RDKit ``Mol`` with 3D coordinates.

    Tries, in order: the Meeko round trip (full bond orders), coordinate
    transfer onto a template SMILES, then geometric bond perception. Returns
    ``None`` rather than raising - pose parsing must never abort a run.
    """
    mol = _mol_from_pdbqt_meeko(pdbqt_text)
    if mol is not None and mol.GetNumAtoms() > 1:
        return mol
    if template_smiles:
        mol = _mol_from_pdbqt_template(pdbqt_text, template_smiles)
        if mol is not None and mol.GetNumAtoms() > 1:
            return mol
    return _mol_from_pdbqt_pdbblock(pdbqt_text)


def split_pdbqt_models(pdbqt_text: str) -> list[str]:
    """Split a multi-MODEL Vina output PDBQT into individual pose blocks."""
    models: list[str] = []
    current: list[str] = []
    in_model = False
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                models.append("\n".join(current))
            in_model = False
            current = []
            continue
        if in_model:
            current.append(line)
    if not models and pdbqt_text.strip():
        models = [pdbqt_text]
    return models
