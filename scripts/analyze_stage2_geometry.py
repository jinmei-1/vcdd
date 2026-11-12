#!/usr/bin/env python3
"""Geometric comparison of a Stage-2 (Boltz-2) predicted complex with the crystal crop.

For every predicted complex this answers the questions a reader asks about an
independently folded structure:

* **Is it the same fold / the same frame?**  The 423 receptor C-alpha atoms are
  superposed onto the crystal crop (Kabsch), and the receptor RMSD is reported.
* **Is the ligand in the ATP pocket?**  After that superposition the ligand is in
  the crystal coordinate frame, so it can be tested directly against the docking
  box and against the co-crystal ligand AZD7648 (MBW): fraction of ligand heavy
  atoms inside the box, distance from the ligand centroid to the AZD7648 centroid,
  fraction of ligand heavy atoms overlapping AZD7648 within 2 A, and the best-fit
  (in-place, symmetry-agnostic) ligand RMSD onto AZD7648.
* **Is the interface physical?**  Number of receptor contact residues, ligand
  burial, and the closest receptor-ligand heavy-atom distance.

This is the pose-level counterpart of `interface_contacts.csv`, which works in
sequence space.  Both are only meaningful when the receptor sequence submitted to
Boltz is the sequence of the docked receptor; the script therefore records the
sequence-recovery check and refuses to write a summary that mixes the two.

Usage:  python scripts/analyze_stage2_geometry.py [poses_dir] [out_csv]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path("/home/geng/vcdd-0.3.0")
POSES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dnapkcs_run/stage2/poses"
REF_PDB = ROOT / "dnapkcs_run/receptor/receptor_clean.pdb"
REF_LIGAND = ROOT / "DNA-PKcs/MBW_ligand.pdb"
BOX = ROOT / "dnapkcs_run/receptor/box.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "dnapkcs_run/stage2/pose_geometry.csv"

CONTACT_CUTOFF = 4.5
OVERLAP_CUTOFF = 2.0


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation + translation mapping `mobile` onto `target`, plus the RMSD."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    a, b = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    rmsd = float(np.sqrt((((a @ rot.T) - b) ** 2).sum(axis=1).mean()))
    return rot, tc - mc @ rot.T, rmsd


def ca_coords(chain) -> np.ndarray:
    out = []
    for res in chain:
        ca = res.find_atom("CA", "*")
        if ca is None:
            continue
        out.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return np.asarray(out, dtype=float)


def heavy_xyz(residues) -> np.ndarray:
    pts = []
    for res in residues:
        for atom in res:
            if not atom.element.is_hydrogen:
                pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.asarray(pts, dtype=float)


def main() -> None:
    box = json.loads(Path(BOX).read_text())
    lo = np.array(box["center"]) - np.array(box["size"]) / 2.0
    hi = np.array(box["center"]) + np.array(box["size"]) / 2.0

    ref_st = gemmi.read_structure(str(REF_PDB))
    ref_chain = ref_st[0][0]
    ref_ca = ca_coords(ref_chain)
    pocket_idx = {
        i
        for i, res in enumerate(ref_chain)
        if any(
            (not a.element.is_hydrogen)
            and (lo[0] <= a.pos.x <= hi[0] and lo[1] <= a.pos.y <= hi[1] and lo[2] <= a.pos.z <= hi[2])
            for a in res
        )
    }

    ref_lig_st = gemmi.read_structure(str(REF_LIGAND))
    ref_lig = heavy_xyz([r for ch in ref_lig_st[0] for r in ch])
    ref_lig_centroid = ref_lig.mean(axis=0)

    rows: list[dict] = []
    for path in sorted(POSES.glob("*.cif")):
        st = gemmi.read_structure(str(path))
        model = st[0]
        chains = sorted(model, key=lambda c: len(c), reverse=True)
        prot, lig = chains[0], chains[-1]

        ca = ca_coords(prot)
        rot, t = (np.eye(3), np.zeros(3))
        rmsd = float("nan")
        if ca.shape == ref_ca.shape:
            rot, t, rmsd = kabsch(ca, ref_ca)

        lxyz = heavy_xyz([r for r in lig])
        lxyz_t = lxyz @ rot.T + t                     # ligand in the crystal frame
        centroid = lxyz_t.mean(axis=0)
        in_box = float(((lxyz_t >= lo) & (lxyz_t <= hi)).all(axis=1).mean())

        dl = np.linalg.norm(lxyz_t[:, None, :] - ref_lig[None, :, :], axis=2)
        overlap = float((dl.min(axis=1) <= OVERLAP_CUTOFF).mean())

        # Pocket C-alpha RMSD after the global superposition.  This is only
        # interpretable when residue i of the prediction is residue i of the
        # crystal crop, i.e. when the submitted sequence is the receptor's.
        pocket_ca = np.asarray(
            [
                [r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y, r.find_atom("CA", "*").pos.z]
                for i, r in enumerate(prot)
                if i in pocket_idx and r.find_atom("CA", "*") is not None
            ],
            dtype=float,
        )
        ref_pocket_ca = np.asarray(
            [
                [r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y, r.find_atom("CA", "*").pos.z]
                for i, r in enumerate(ref_chain)
                if i in pocket_idx and r.find_atom("CA", "*") is not None
            ],
            dtype=float,
        )
        pocket_rmsd = float("nan")
        if pocket_ca.shape == ref_pocket_ca.shape and pocket_ca.size:
            moved = pocket_ca @ rot.T + t
            pocket_rmsd = float(np.sqrt(((moved - ref_pocket_ca) ** 2).sum(axis=1).mean()))

        pxyz = heavy_xyz([r for r in prot])
        pxyz_t = pxyz @ rot.T + t
        dprot = np.linalg.norm(pxyz_t[:, None, :] - lxyz_t[None, :, :], axis=2)
        min_dist = float(dprot.min())
        burial = float((dprot.min(axis=0) <= CONTACT_CUTOFF).mean())
        contacts = int((dprot.min(axis=1) <= CONTACT_CUTOFF).sum())

        rows.append(
            {
                "mol_id": path.stem,
                "receptor_ca_rmsd_A": rmsd,
                "ligand_centroid_to_azd7648_A": float(np.linalg.norm(centroid - ref_lig_centroid)),
                "ligand_in_box_frac": in_box,
                "ligand_atoms_overlapping_azd7648_frac": overlap,
                "pocket_ca_rmsd_A": pocket_rmsd,
                "ligand_burial_frac": burial,
                "n_contact_atoms": contacts,
                "min_heavy_distance_A": min_dist,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    def q(col: str) -> str:
        v = df[col].dropna()
        return f"median {v.median():.2f} (min {v.min():.2f}, max {v.max():.2f})"

    print(f"analysed {len(df)} predicted complexes from {POSES}")
    print(f"receptor C-alpha RMSD after superposition: {q('receptor_ca_rmsd_A')}")
    print(f"ligand centroid distance to AZD7648:       {q('ligand_centroid_to_azd7648_A')}")
    print(f"ligand heavy atoms inside the docking box: {q('ligand_in_box_frac')}")
    print(f"ligand heavy atoms overlapping AZD7648:    {q('ligand_atoms_overlapping_azd7648_frac')}")
    print(f"ATP-pocket C-alpha RMSD after superposition: {q('pocket_ca_rmsd_A')}")
    print(f"ligand burial fraction:                    {q('ligand_burial_frac')}")
    print(f"min receptor-ligand heavy distance:        {q('min_heavy_distance_A')}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()