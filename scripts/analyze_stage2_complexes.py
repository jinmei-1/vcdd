#!/usr/bin/env python3
"""Structural analysis of the Stage-2 Boltz-2 predicted complexes.

Answers three questions per complex, then aggregates:

1. **Is the predicted receptor the right protein, in the right frame?**
   The 423 receptor C-alpha atoms are superposed (Kabsch) onto the crystal crop
   the docking receptor came from (PDB 7OTW residues 3701-4123), giving a global
   C-alpha RMSD and, separately, the RMSD of the 39 box-defined ATP-site
   residues.  A correct fold gives a low global RMSD; a low *pocket* RMSD
   additionally says the ATP site itself is reproduced.
2. **Is the ligand in the ATP pocket?**  After superposing, the predicted ligand
   sits in the crystal frame, so it can be tested directly against the docking
   box and against the co-crystal ligand AZD7648 (MBW): fraction of ligand heavy
   atoms inside the box, fraction of its receptor contact residues that are
   box-defined ATP-site residues, centroid displacement from AZD7648, and heavy
   atom overlap with AZD7648.
3. **Is the binding geometry physically reasonable?**  Receptor-ligand heavy-atom
   clashes below 2.2 A, polar contacts below 3.5 A, ligand burial, and an RDKit
   bond-length sanity check of the predicted ligand against an ETKDG conformer of
   the same SMILES.

Every complex is predicted *without* the docking pose, box, template or
reference ligand, so these numbers measure independent localisation and
plausibility, not pose recovery.

Usage:
    python scripts/analyze_stage2_complexes.py [--pattern GLOB] [--out CSV]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path("/home/geng/vcdd-0.3.0")
REF_PDB = ROOT / "dnapkcs_run/receptor/receptor_clean.pdb"
REF_LIGAND = ROOT / "DNA-PKcs/MBW_ligand.pdb"
BOX = ROOT / "dnapkcs_run/receptor/box.json"
DEFAULT_PATTERN = "dnapkcs_run/stage2/boltz/local/*/out/boltz_results_input/predictions/input/input_model_0.cif"

CA_CACHE: dict[str, np.ndarray] = {}
POCKET_CACHE: dict[str, np.ndarray] = {}

CONTACT_CUTOFF = 4.5      # receptor-ligand contact
CLASH_CUTOFF = 2.2        # heavy-atom clash
POLAR_CUTOFF = 3.5        # N/O ... N/O contact
OVERLAP_CUTOFF = 2.0      # ligand atom overlapping AZD7648


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation + translation mapping `mobile` onto `target`, plus the RMSD."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    a, b = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    rmsd = float(np.sqrt((((a @ rot.T) - b) ** 2).sum(axis=1).mean()))
    return rot, tc - mc @ rot.T, rmsd


def heavy_atoms(residues):
    return [(a, np.array([a.pos.x, a.pos.y, a.pos.z])) for res in residues for a in res
            if not a.element.is_hydrogen]


def ca_matrix(chain) -> np.ndarray:
    pts = []
    for res in chain:
        ca = res.find_atom("CA", "*")
        pts.append([ca.pos.x, ca.pos.y, ca.pos.z] if ca is not None else [np.nan] * 3)
    return np.asarray(pts, dtype=float)


def split_chains(model):
    """(protein chain, ligand chain) - order-independent, unlike a positional guess."""
    chains = list(model)
    protein = max(chains, key=len)
    others = [c for c in chains if c is not protein]
    ligand = min(others, key=len) if others else None
    return protein, ligand


def ligand_bond_deviation(smiles: str, ligand_res) -> float:
    """Median |bond length - ETKDG reference| over atom-mapped bonds, in A.

    A predicted pose can be confidently placed and still carry distorted internal
    geometry; this catches that.  Atoms are mapped through a substructure match to
    an ETKDG conformer of the same SMILES, so the comparison is bond-by-bond rather
    than a length distribution.  Failures return NaN instead of breaking the run.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        st = gemmi.Structure()
        st.add_model(gemmi.Model("1"))
        ch = gemmi.Chain("X")
        res = gemmi.Residue()
        res.name = ligand_res.name
        res.seqid = gemmi.SeqId(1, " ")
        for a in ligand_res:
            res.add_atom(a)
        ch.add_residue(res)
        st[0].add_chain(ch)
        block = st.make_pdb_string()

        mol = Chem.MolFromPDBBlock(block, sanitize=False, proximityBonding=True)
        ref = Chem.MolFromSmiles(smiles)
        if mol is None or ref is None:
            return float("nan")
        mol = Chem.RemoveHs(mol)
        ref = Chem.RemoveHs(Chem.AddHs(ref))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(ref, params) != 0:
            return float("nan")

        def lengths(m):
            conf = m.GetConformer()
            out = {}
            for b in m.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                p, q = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
                out[frozenset((i, j))] = float(np.linalg.norm([p.x - q.x, p.y - q.y, p.z - q.z]))
            return out

        # map each reference atom onto the predicted ligand
        match = mol.GetSubstructMatch(ref)
        if match and len(match) == ref.GetNumAtoms():
            inv = {m: r for r, m in enumerate(match)}
            pred, gold = lengths(mol), lengths(ref)
            devs = [
                abs(pred[frozenset((inv[i], inv[j]))] - gold[frozenset((i, j))])
                for i, j in (tuple(b.GetBeginAtomIdx() for b in [bd]) + (b.GetEndAtomIdx(),) for bd in ref.GetBonds())
                if frozenset((inv[i], inv[j])) in pred
            ]
            return float(np.median(devs)) if devs else float("nan")
        # graph mismatch: fall back to a distribution comparison
        a = sorted(lengths(mol).values())
        b = sorted(lengths(ref).values())
        if len(a) != len(b) or not a:
            return float("nan")
        return float(np.median([abs(x - y) for x, y in zip(a, b)]))
    except Exception:  # noqa: BLE001 - a sanity check must never break the analysis
        return float("nan")


def residue_index_per_atom(chain) -> list[int]:
    """Residue index (0-based, chain order) for each heavy atom of `chain`."""
    out = []
    for i, res in enumerate(chain):
        out.extend([i] * sum(1 for a in res if not a.element.is_hydrogen))
    return out


def reference_contact_shell(ref_chain, ref_lig_xyz, cutoff: float = CONTACT_CUTOFF) -> set[int]:
    """Crystal residues within `cutoff` of the co-crystal ligand (the ATP-site shell)."""
    shell = set()
    for i, res in enumerate(ref_chain):
        pts = [np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res if not a.element.is_hydrogen]
        if not pts:
            continue
        d = np.linalg.norm(np.asarray(pts)[:, None, :] - ref_lig_xyz[None, :, :], axis=2)
        if d.min() <= cutoff:
            shell.add(i)
    return shell


def analyse(cif: Path, box, ref_chain, ref_lig_xyz, ref_lig_centroid, pocket_idx, ref_shell, smiles) -> dict:
    lo = np.array(box["center"]) - np.array(box["size"]) / 2.0
    hi = np.array(box["center"]) + np.array(box["size"]) / 2.0
    st = gemmi.read_structure(str(cif))
    protein, ligand = split_chains(st[0])
    if ligand is None:
        raise ValueError(f"{cif}: no ligand chain")

    ca = ca_matrix(protein)
    rot, t, rmsd = kabsch(ca, ca_matrix(ref_chain))

    lig = heavy_atoms([r for r in ligand])
    lxyz = np.asarray([p for _, p in lig])
    lxyz_t = lxyz @ rot.T + t
    centroid = lxyz_t.mean(axis=0)

    dl = np.linalg.norm(lxyz_t[:, None, :] - ref_lig_xyz[None, :, :], axis=2)
    prot_xyz = np.asarray([p for _, p in heavy_atoms([r for r in protein])])
    prot_xyz_t = prot_xyz @ rot.T + t
    dprot = np.linalg.norm(prot_xyz_t[:, None, :] - lxyz_t[None, :, :], axis=2)

    contact_mask = dprot <= CONTACT_CUTOFF
    res_of_atom = residue_index_per_atom(protein)
    contact_res_idx = {res_of_atom[i] for i, hit in enumerate(contact_mask.any(axis=1)) if hit}
    lig_contact_idx = {j for j, hit in enumerate(contact_mask.any(axis=0)) if hit}
    shell_hits = contact_res_idx & ref_shell

    # polar contacts: ligand N/O within 3.5 A of a protein N/O
    ligand_hetero = [k for k, (a, _) in enumerate(lig) if a.element.name in ("N", "O")]
    prot_atoms = heavy_atoms([r for r in protein])
    prot_hetero = [k for k, (a, _) in enumerate(prot_atoms) if a.element.name in ("N", "O")]
    n_polar = 0
    if ligand_hetero and prot_hetero:
        sub = dprot[np.ix_(prot_hetero, ligand_hetero)]
        n_polar = int((sub <= POLAR_CUTOFF).sum())

    pocket_ca = np.asarray([
        [r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y, r.find_atom("CA", "*").pos.z]
        for i, r in enumerate(protein) if i in pocket_idx and r.find_atom("CA", "*") is not None
    ])
    ref_pocket_ca = np.asarray([
        [r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y, r.find_atom("CA", "*").pos.z]
        for i, r in enumerate(ref_chain) if i in pocket_idx and r.find_atom("CA", "*") is not None
    ])
    pocket_rmsd = float(np.sqrt((((pocket_ca @ rot.T + t) - ref_pocket_ca) ** 2).sum(axis=1).mean()))

    CA_CACHE[cif.parents[4].name] = ca @ rot.T + t   # superposed onto the crystal frame
    POCKET_CACHE[cif.parents[4].name] = (ca @ rot.T + t)[sorted(pocket_idx)]

    conf_path = next(cif.parent.glob("confidence_*_model_*.json"), None)
    conf = json.loads(conf_path.read_text()) if conf_path else {}

    return {
        "mol_id": cif.parents[4].name,
        "confidence_score": conf.get("confidence_score"),
        "iptm": conf.get("iptm"),
        "ptm": conf.get("ptm"),
        "complex_plddt": conf.get("complex_plddt"),
        "complex_iplddt": conf.get("complex_iplddt"),
        "complex_pde": conf.get("complex_pde"),
        "receptor_ca_rmsd_A": rmsd,
        "pocket_ca_rmsd_A": pocket_rmsd,
        "ligand_in_box_frac": float(((lxyz_t >= lo) & (lxyz_t <= hi)).all(axis=1).mean()),
        "ligand_centroid_to_azd7648_A": float(np.linalg.norm(centroid - ref_lig_centroid)),
        "ligand_overlap_with_azd7648_frac": float((dl.min(axis=1) <= OVERLAP_CUTOFF).mean()),
        "n_contact_residues": len(contact_res_idx),
        "n_atp_shell_residues": len(ref_shell),
        "atp_shell_coverage": len(shell_hits) / len(ref_shell) if ref_shell else float("nan"),
        "atp_shell_jaccard": (
            len(shell_hits) / len(contact_res_idx | ref_shell) if (contact_res_idx | ref_shell) else float("nan")
        ),
        "contact_residues_in_box_pocket_frac": (
            len(contact_res_idx & pocket_idx) / len(contact_res_idx) if contact_res_idx else float("nan")
        ),
        "ligand_burial_frac": len(lig_contact_idx) / len(lig),
        "n_clashes": int((dprot < CLASH_CUTOFF).sum()),
        "min_heavy_distance_A": float(dprot.min()),
        "n_polar_contacts": n_polar,
        "ligand_bond_median_dev_A": ligand_bond_deviation(smiles, [r for r in ligand][0]),
        "ligand_mean_plddt": float(np.mean([a.b_iso for res in ligand for a in res])) / 100.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--out", default="dnapkcs_run/stage2/complex_analysis.csv")
    ap.add_argument("--progress", action="store_true",
                    help="per-molecule timing, plus a stack dump every 120 s while stuck")
    args = ap.parse_args()

    box = json.loads(BOX.read_text())
    lo = np.array(box["center"]) - np.array(box["size"]) / 2.0
    hi = np.array(box["center"]) + np.array(box["size"]) / 2.0

    ref_chain = gemmi.read_structure(str(REF_PDB))[0][0]
    ref_lig_xyz = np.asarray([p for _, p in heavy_atoms(list(gemmi.read_structure(str(REF_LIGAND))[0][0]))])
    ref_lig_centroid = ref_lig_xyz.mean(axis=0)
    ref_shell = reference_contact_shell(ref_chain, ref_lig_xyz)
    pocket_idx = {
        i for i, res in enumerate(ref_chain)
        if any((not a.element.is_hydrogen) and lo[0] <= a.pos.x <= hi[0]
               and lo[1] <= a.pos.y <= hi[1] and lo[2] <= a.pos.z <= hi[2] for a in res)
    }

    import glob as _glob
    pattern = args.pattern if Path(args.pattern).is_absolute() else str(ROOT / args.pattern)
    cifs = [Path(x) for x in sorted(_glob.glob(pattern))]
    if args.progress:
        # A molecule that takes minutes (ETKDG embedding of a hard ligand) used to
        # look exactly like a dead process. Report per-molecule timing, and dump
        # the Python stack every 120 s while stuck, so a slow molecule names itself.
        import faulthandler

        faulthandler.dump_traceback_later(120, repeat=True)
    rows = []
    for i, cif in enumerate(cifs):
        import yaml
        t0 = time.time()

        wd = cif.parents[4]
        y = yaml.safe_load((wd / "input.yaml").read_text())
        smiles = next(s["ligand"]["smiles"] for s in y["sequences"] if "ligand" in s)
        try:
            rows.append(analyse(cif, box, ref_chain, ref_lig_xyz, ref_lig_centroid, pocket_idx, ref_shell, smiles))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {cif.parents[4].name}: {type(exc).__name__}: {exc}")
        if args.progress:
            dt = time.time() - t0
            if dt > 5 or (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(cifs)}] {cif.parents[4].name} {dt:.1f} s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_csv(out, index=False)

    def stat(col: str) -> str:
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            return "n/a"
        return f"{v.median():.2f} (IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}; min {v.min():.2f}, max {v.max():.2f})"

    print(f"analysed {len(df)} complex(es) from {args.pattern}\n")
    for col, label in [
        ("confidence_score", "Boltz composite confidence"),
        ("iptm", "ipTM"),
        ("complex_plddt", "complex pLDDT"),
        ("complex_iplddt", "interface pLDDT"),
        ("receptor_ca_rmsd_A", "receptor C-alpha RMSD vs 7OTW 3701-4123"),
        ("pocket_ca_rmsd_A", "ATP-site (39 res) C-alpha RMSD"),
        ("ligand_in_box_frac", "ligand heavy atoms inside the docking box"),
        ("ligand_centroid_to_azd7648_A", "ligand centroid to AZD7648 centroid"),
        ("ligand_overlap_with_azd7648_frac", "ligand atoms overlapping AZD7648"),
        ("n_contact_residues", "receptor contact residues"),
        ("n_atp_shell_residues", "ATP-site shell from co-crystal AZD7648 (<4.5 A)"),
        ("atp_shell_coverage", "coverage of that AZD7648 contact shell"),
        ("atp_shell_jaccard", "Jaccard vs that shell"),
        ("contact_residues_in_box_pocket_frac", "contact residues inside the 39-residue docking box set"),
        ("ligand_burial_frac", "ligand atoms with a receptor contact"),
        ("n_clashes", "heavy-atom clashes < 2.2 A"),
        ("min_heavy_distance_A", "closest receptor-ligand heavy distance"),
        ("n_polar_contacts", "polar contacts < 3.5 A"),
        ("ligand_bond_median_dev_A", "ligand bond-length deviation vs ETKDG"),
        ("ligand_mean_plddt", "ligand mean pLDDT (CIF B-factor)"),
    ]:
        if col in df:
            print(f"  {label:52s} {stat(col)}")

    if len(df) > 2:
        for a, b in [("iptm", "pocket_ca_rmsd_A"), ("iptm", "ligand_in_box_frac"),
                     ("confidence_score", "atp_shell_coverage")]:
            s = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(s) > 2:
                print(f"  Spearman rho({a}, {b}) = {s[a].corr(s[b], method='spearman'):+.3f}  (n={len(s)})")

    names = sorted(CA_CACHE)
    globals_, pockets = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            globals_.append(float(np.sqrt(((CA_CACHE[names[i]] - CA_CACHE[names[j]]) ** 2).sum(axis=1).mean())))
            pockets.append(float(np.sqrt(((POCKET_CACHE[names[i]] - POCKET_CACHE[names[j]]) ** 2).sum(axis=1).mean())))
    if globals_:
        print(f"  run-to-run consistency ({len(names)} models, {len(globals_)} pairs, no further superposition):")
        print(f"    whole chain C-alpha RMSD  {np.median(globals_):.2f} A (max {np.max(globals_):.2f})")
        print(f"    ATP-site C-alpha RMSD     {np.median(pockets):.2f} A (max {np.max(pockets):.2f})")

    in_pocket = df[(df["ligand_in_box_frac"] >= 0.5) | (df["atp_shell_coverage"] >= 0.5)]
    print(f"\n  ATP-site localisation: {len(in_pocket)}/{len(df)} complexes have >=50% of the ligand inside the "
          f"docking box, or >=50% coverage of the AZD7648 contact shell")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()