"""Native-ligand redock validation gate.

Before docking a library, re-dock the co-crystal ligand into its own receptor
and check that the docking setup reproduces the crystallographic pose. If it
cannot reproduce a pose it has already seen, ranking a million unseen molecules
against that setup is not meaningful.

The gate is deliberately **core-aware and two-tier**:

* *all-atom RMSD* - the standard number everyone reports, but it is dominated by
  flexible solvent-exposed tails that no docking program is expected to place.
* *buried rigid-core RMSD* - restricted to ring-system / rigid atoms that are
  buried in the pocket. This is the number that actually says "the binding mode
  is right".

A run passes when **either** all-atom RMSD < threshold **or** buried-core RMSD <
threshold. Failure is loud (logged as an error, written to
``redock_validation.json``, added to the manifest warnings) but by default it
does **not** abort: many legitimate targets - shallow pockets, induced fit,
crystallographic waters - fail a naive redock while still supporting useful
relative ranking. Set ``receptor.redock_fail_hard: true`` to make it fatal.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vcdd.docking.base import DockParams, DockingEngine
from vcdd.receptor.box import Box, rmsd
from vcdd.runtime.logging import get_logger
from vcdd.runtime.paths import atomic_write_text

LOG = get_logger()


@dataclass
class RedockResult:
    passed: bool
    all_atom_rmsd: float | None = None
    core_rmsd: float | None = None
    n_core_atoms: int = 0
    n_matched_atoms: int = 0
    score: float | None = None
    threshold: float = 2.0
    engine: str = ""
    seconds: float = 0.0
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return atomic_write_text(path, json.dumps(self.as_dict(), indent=2))


def _load_reference_mol(path: str | Path, resname: str | None = None) -> Any | None:
    """Load the native ligand with 3D coordinates."""
    from rdkit import Chem

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".sdf", ".mol"):
        for m in Chem.SDMolSupplier(str(p), sanitize=False, removeHs=True):
            if m is not None:
                return m
        return None
    if suffix == ".mol2":
        return Chem.MolFromMol2File(str(p), sanitize=False, removeHs=True)
    if suffix in (".pdb", ".ent"):
        text = p.read_text()
        if resname:
            lines = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == resname]
            text = "\n".join(lines) + "\nEND\n"
        return Chem.MolFromPDBBlock(text, sanitize=False, removeHs=True)
    return None


def _heavy_coords(mol: Any) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer()
    return [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]


def rigid_core_indices(mol: Any) -> list[int]:
    """Heavy-atom indices belonging to the rigid core (ring systems + attached).

    Terminal rotatable substituents are excluded: their placement is not what a
    redock gate should be testing.
    """
    ring_info = mol.GetRingInfo()
    core: set[int] = set()
    for ring in ring_info.AtomRings():
        core.update(ring)
    if not core:
        # Acyclic ligand: fall back to atoms not in a terminal rotatable branch.
        core = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetDegree() > 1}
    else:
        # Include heavy atoms directly bonded to a ring (they define the vector).
        for idx in list(core):
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetAtomicNum() > 1:
                    core.add(nbr.GetIdx())
    return sorted(i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def symmetry_corrected_rmsd(ref_mol: Any, probe_mol: Any, indices: list[int] | None = None) -> float | None:
    """Best RMSD over the ligand's automorphism group, without superposition.

    *indices* always refer to atoms of ``ref_mol`` (after ``RemoveHs``), so the
    caller can restrict the calculation to e.g. the rigid core. Returns ``None``
    when no substructure correspondence can be established, rather than a
    meaningless number computed over a partial match.
    """
    from rdkit import Chem

    try:
        ref_noh = Chem.RemoveHs(ref_mol)
        probe_noh = Chem.RemoveHs(probe_mol)
        if ref_noh.GetNumAtoms() != probe_noh.GetNumAtoms():
            return None
        # match[i] = probe atom index corresponding to ref atom i
        matches = probe_noh.GetSubstructMatches(ref_noh, uniquify=False, useChirality=False, maxMatches=5000)
        if not matches:
            # Bond perception may differ; retry on the connectivity skeleton.
            ref_skel = Chem.MolFromSmarts(Chem.MolToSmarts(ref_noh))
            if ref_skel is not None:
                matches = probe_noh.GetSubstructMatches(ref_skel, uniquify=False, maxMatches=5000)
        if not matches:
            return None

        ref_conf = ref_noh.GetConformer()
        probe_conf = probe_noh.GetConformer()
        keep = set(indices) if indices is not None else set(range(ref_noh.GetNumAtoms()))
        if not keep:
            return None

        best: float | None = None
        for match in matches:
            if len(match) != ref_noh.GetNumAtoms():
                continue
            total, n = 0.0, 0
            for ref_i in keep:
                p = ref_conf.GetAtomPosition(ref_i)
                q = probe_conf.GetAtomPosition(match[ref_i])
                total += (p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2
                n += 1
            if n == 0:
                continue
            value = (total / n) ** 0.5
            best = value if best is None else min(best, value)
        return best
    except Exception:  # noqa: BLE001
        return None


def run_redock_gate(
    engine: DockingEngine,
    receptor_pdbqt: str | Path,
    reference_ligand_path: str | Path,
    box: Box,
    params: DockParams,
    threshold: float = 2.0,
    reference_resname: str | None = None,
) -> RedockResult:
    """Re-dock the native ligand and evaluate the two-tier RMSD gate."""
    t0 = time.time()
    from rdkit import Chem

    ref = _load_reference_mol(reference_ligand_path, reference_resname)
    if ref is None or ref.GetNumConformers() == 0:
        return RedockResult(
            passed=False, threshold=threshold, engine=engine.name,
            error=f"could not read reference ligand with 3D coordinates from {reference_ligand_path}",
        )
    try:
        Chem.SanitizeMol(ref, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:  # noqa: BLE001
        pass

    ref_smiles = None
    try:
        ref_smiles = Chem.MolToSmiles(Chem.RemoveHs(ref))
    except Exception:  # noqa: BLE001
        pass

    # Prepare the native ligand exactly as a library molecule would be prepared,
    # so the gate tests the real pipeline, not a privileged shortcut.
    from vcdd.docking.ligprep import mol_to_pdbqt, pdbqt_to_mol

    ref_h = Chem.AddHs(ref, addCoords=True)
    lig_pdbqt, err = mol_to_pdbqt(ref_h, conf_id=0)
    if lig_pdbqt is None:
        return RedockResult(passed=False, threshold=threshold, engine=engine.name,
                            error=f"native ligand PDBQT preparation failed: {err}")

    result = engine.dock(lig_pdbqt, str(receptor_pdbqt), box, params, mol_id="native_ligand")
    if not result.ok or result.pose_pdbqt is None:
        return RedockResult(passed=False, threshold=threshold, engine=engine.name,
                            seconds=time.time() - t0, error=result.error or "redock produced no pose")

    pose = pdbqt_to_mol(result.pose_pdbqt, template_smiles=ref_smiles)
    if pose is None or pose.GetNumConformers() == 0:
        return RedockResult(passed=False, threshold=threshold, engine=engine.name, score=result.score,
                            seconds=time.time() - t0, error="could not parse the redocked pose")

    warnings: list[str] = []
    all_atom = symmetry_corrected_rmsd(ref, pose)
    if all_atom is None:
        ref_c, pose_c = _heavy_coords(ref), _heavy_coords(pose)
        if len(ref_c) == len(pose_c):
            all_atom = rmsd(ref_c, pose_c)
            warnings.append("no substructure match between reference and pose; used positional RMSD in file order")
        else:
            return RedockResult(passed=False, threshold=threshold, engine=engine.name, score=result.score,
                                seconds=time.time() - t0,
                                error=f"atom count mismatch reference={len(ref_c)} pose={len(pose_c)}")

    core_idx = rigid_core_indices(Chem.RemoveHs(ref))
    core = symmetry_corrected_rmsd(ref, pose, indices=core_idx) if core_idx else None

    passed = (all_atom is not None and all_atom < threshold) or (core is not None and core < threshold)
    if passed and all_atom is not None and all_atom >= threshold:
        verdict = (
            f"PASS on buried rigid core ({core:.2f} A) but all-atom RMSD is {all_atom:.2f} A: "
            "the binding mode is reproduced, flexible substituents are not."
        )
    elif passed:
        verdict = f"PASS: all-atom RMSD {all_atom:.2f} A < {threshold} A."
    else:
        verdict = (
            f"FAIL: all-atom RMSD {all_atom:.2f} A"
            + (f", rigid-core RMSD {core:.2f} A" if core is not None else "")
            + f" (threshold {threshold} A). The docking setup does not reproduce the crystallographic pose. "
            "Check the box, protonation, missing cofactors/metals, or induced fit before trusting the ranking."
        )

    return RedockResult(
        passed=passed,
        all_atom_rmsd=round(all_atom, 3) if all_atom is not None else None,
        core_rmsd=round(core, 3) if core is not None else None,
        n_core_atoms=len(core_idx),
        n_matched_atoms=Chem.RemoveHs(ref).GetNumAtoms(),
        score=result.score,
        threshold=threshold,
        engine=engine.name,
        seconds=round(time.time() - t0, 2),
        verdict=verdict,
        warnings=warnings,
    )
