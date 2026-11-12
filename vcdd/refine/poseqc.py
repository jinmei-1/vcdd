"""Pose quality control.

A docking score says nothing about whether the pose is physically sensible. The
common failure modes are cheap to detect and are checked here:

``in_box``            pose escaped or sits on the box wall (box too small / wrong site)
``clash``             a heavy atom closer than ~2.0 A to a receptor heavy atom
``buried_fraction``   a pose floating on the surface with few contacts
``strain``            internal energy of the docked conformer above its own relaxed minimum
``ligand_efficiency`` score per heavy atom, to stop large greasy molecules winning on size alone

None of these are pass/fail for the pipeline. They are flags attached to each
compound so a bad pose is visible before someone orders the compound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vcdd.receptor.box import Box

#: Heavy-atom distance below which a contact is a clash, in Angstrom.
CLASH_DISTANCE = 2.0
#: Distance defining a receptor contact, in Angstrom.
CONTACT_DISTANCE = 4.0
#: Strain above this (kcal/mol, MMFF) is flagged.
STRAIN_WARN = 10.0
#: Ligand efficiency below this magnitude is flagged as size-driven.
LE_WARN = 0.20


@dataclass
class PoseQC:
    mol_id: str
    n_heavy: int = 0
    in_box: bool = True
    fraction_in_box: float = 1.0
    min_receptor_distance: float | None = None
    n_clashes: int = 0
    n_contacts: int = 0
    buried_fraction: float | None = None
    strain_kcal: float | None = None
    ligand_efficiency: float | None = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def pose_ok(self) -> bool:
        return self.error is None and not self.flags

    def as_dict(self) -> dict[str, Any]:
        d = {
            "mol_id": self.mol_id,
            "n_heavy": self.n_heavy,
            "in_box": self.in_box,
            "fraction_in_box": self.fraction_in_box,
            "min_receptor_distance": self.min_receptor_distance,
            "n_clashes": self.n_clashes,
            "n_contacts": self.n_contacts,
            "buried_fraction": self.buried_fraction,
            "strain_kcal": self.strain_kcal,
            "ligand_efficiency": self.ligand_efficiency,
            "pose_ok": self.pose_ok,
            "flags": ";".join(self.flags),
            "error": self.error,
        }
        return d


def receptor_heavy_coords(receptor_pdbqt_path: str | Path) -> np.ndarray:
    """Heavy-atom coordinates of the prepared receptor."""
    from vcdd.docking.ligprep import AD_TYPE_TO_ELEMENT

    coords: list[tuple[float, float, float]] = []
    for line in Path(receptor_pdbqt_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = line[77:79].strip() if len(line) > 77 else line[76:78].strip()
        if AD_TYPE_TO_ELEMENT.get(ad_type, ad_type.capitalize()) == "H":
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return np.asarray(coords, dtype=float) if coords else np.zeros((0, 3))


def conformer_strain(mol: Any, max_iters: int = 500) -> float | None:
    """MMFF energy of the pose minus the energy of its own relaxed copy.

    A docked pose comes back from PDBQT with only polar hydrogens, so the
    non-polar hydrogens have to be re-added at idealised positions. Scoring that
    geometry directly charges the ligand for bad hydrogen placement and inflates
    "strain" by tens of kcal/mol. The fix is a two-step reference:

    1. relax hydrogens only, with every heavy atom fixed -> ``E_pose``
       (the energy of the *docked heavy-atom geometry*)
    2. relax everything -> ``E_relaxed``

    The difference is *local* strain: how far the docked heavy-atom geometry
    sits above its own nearest minimum. It is not the global conformational
    penalty, which needs a full conformer search over the free ligand.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        work = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(work)
        if props is None:
            return None

        # Step 1: hydrogens only.
        ff_h = AllChem.MMFFGetMoleculeForceField(work, props)
        if ff_h is None:
            return None
        for atom in work.GetAtoms():
            if atom.GetAtomicNum() > 1:
                ff_h.AddFixedPoint(atom.GetIdx())
        ff_h.Minimize(maxIts=max_iters)
        e_pose = ff_h.CalcEnergy()

        # Step 2: everything.
        ff_all = AllChem.MMFFGetMoleculeForceField(work, props)
        if ff_all is None:
            return None
        ff_all.Minimize(maxIts=max_iters * 4)
        e_relaxed = ff_all.CalcEnergy()
        return round(float(e_pose - e_relaxed), 3)
    except Exception:  # noqa: BLE001
        return None


def assess_pose(
    mol_id: str,
    pose_mol: Any,
    receptor_coords: np.ndarray,
    box: Box,
    score: float | None = None,
    box_tolerance: float = 0.5,
    compute_strain: bool = True,
) -> PoseQC:
    """Run every check on one pose and return the flag set."""
    from rdkit import Chem

    qc = PoseQC(mol_id=mol_id)
    if pose_mol is None:
        qc.error = "pose_missing"
        qc.flags.append("pose_missing")
        return qc

    try:
        heavy = Chem.RemoveHs(pose_mol)
        conf = heavy.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(heavy.GetNumAtoms())])
    except Exception as exc:  # noqa: BLE001
        qc.error = f"pose_unreadable:{type(exc).__name__}"
        qc.flags.append("pose_unreadable")
        return qc

    qc.n_heavy = int(len(coords))
    if qc.n_heavy == 0:
        qc.error = "no_heavy_atoms"
        qc.flags.append("no_heavy_atoms")
        return qc

    # ---- box containment ---------------------------------------------------
    lo = np.array(box.center) - np.array(box.size) / 2.0 - box_tolerance
    hi = np.array(box.center) + np.array(box.size) / 2.0 + box_tolerance
    inside = np.all((coords >= lo) & (coords <= hi), axis=1)
    qc.fraction_in_box = round(float(inside.mean()), 3)
    qc.in_box = bool(inside.all())
    if not qc.in_box:
        qc.flags.append(f"outside_box({qc.fraction_in_box:.2f}_inside)")

    # ---- receptor contacts and clashes -------------------------------------
    if len(receptor_coords):
        d = np.linalg.norm(coords[:, None, :] - receptor_coords[None, :, :], axis=2)
        per_atom_min = d.min(axis=1)
        qc.min_receptor_distance = round(float(per_atom_min.min()), 3)
        qc.n_clashes = int((per_atom_min < CLASH_DISTANCE).sum())
        qc.n_contacts = int((d < CONTACT_DISTANCE).sum())
        qc.buried_fraction = round(float((per_atom_min < CONTACT_DISTANCE).mean()), 3)
        if qc.n_clashes:
            qc.flags.append(f"clash({qc.n_clashes}_atoms<{CLASH_DISTANCE}A)")
        if qc.buried_fraction is not None and qc.buried_fraction < 0.5:
            qc.flags.append(f"surface_pose(buried={qc.buried_fraction:.2f})")

    # ---- internal strain ---------------------------------------------------
    if compute_strain:
        qc.strain_kcal = conformer_strain(pose_mol)
        if qc.strain_kcal is not None and qc.strain_kcal > STRAIN_WARN:
            qc.flags.append(f"strained({qc.strain_kcal:.1f}kcal)")

    # ---- ligand efficiency -------------------------------------------------
    if score is not None and qc.n_heavy:
        qc.ligand_efficiency = round(float(score) / qc.n_heavy, 4)
        if abs(qc.ligand_efficiency) < LE_WARN:
            qc.flags.append(f"low_ligand_efficiency({qc.ligand_efficiency:.3f})")

    return qc


def qc_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate flag counts for the manifest and the report."""
    if not rows:
        return {"n_poses": 0}
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("pose_ok"))
    counts: dict[str, int] = {}
    for r in rows:
        for flag in str(r.get("flags") or "").split(";"):
            if not flag:
                continue
            key = flag.split("(")[0]
            counts[key] = counts.get(key, 0) + 1
    return {
        "n_poses": n,
        "n_pose_ok": n_ok,
        "fraction_pose_ok": round(n_ok / n, 3),
        "flag_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
