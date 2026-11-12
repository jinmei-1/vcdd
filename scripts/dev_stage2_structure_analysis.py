#!/usr/bin/env python3
"""Sequence-space interface analysis of the Stage-2 (Boltz-2) predicted complexes.

Why this analysis exists
------------------------
The Stage-2 complexes are folded independently of the docking run, so they share no
coordinate frame with the docking box: the in-box ligand test therefore reports 0/50 and
the centroid offset is ~312 A (see `vcdd/refine/boltz/poseqc.py`).  That diagnostic cannot
answer the question a reader actually asks -- *did the model put the ligand where the ATP
site is?* -- so we answer it in **sequence space instead of coordinate space**:

* the docking box defines a set of receptor residues with a heavy atom inside it (the
  ATP-competitive site; 0-indexed positions in the 423-residue submitted sequence);
* each predicted complex defines the set of residues in contact with the ligand;
* the two sets are comparable without any superposition, because both are indexed by
  position in the same submitted sequence.

Reported per complex: number of contact residues, fraction of contacts that fall in the
box-defined pocket, interface pLDDT against the chain mean, ligand burial, and the closest
receptor-ligand heavy-atom distance (a stereochemical sanity check).

Outputs
-------
`dnapkcs_run/stage2/interface_contacts.csv`  -- one row per analysed complex
stdout summary                               -- cohort-level statistics to quote in the text
"""
from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path("/home/geng/vcdd-0.3.0")
POSES = ROOT / "dnapkcs_run/stage2/poses"
RECEPTOR = ROOT / "dnapkcs_run/receptor/receptor_clean.pdb"
BOX = ROOT / "dnapkcs_run/receptor/box.json"
CONSENSUS = ROOT / "dnapkcs_run/stage2/consensus_ranking.parquet"
OUT = ROOT / "dnapkcs_run/stage2/interface_contacts.csv"

CONTACT_CUTOFF = 4.5      # A, heavy-atom..heavy-atom
CLASH_CUTOFF = 2.0        # A, heavy-atom..heavy-atom


def pocket_index_set() -> tuple[set[int], list[str]]:
    """0-indexed sequence positions with a heavy atom inside the docking box."""
    box = json.loads(Path(BOX).read_text())
    cx, cy, cz = box["center"]
    hx, hy, hz = (s / 2.0 for s in box["size"])
    lo = (cx - hx, cy - hy, cz - hz)
    hi = (cx + hx, cy + hy, cz + hz)

    st = gemmi.read_structure(str(RECEPTOR))
    chain = st[0][0]
    hits: set[int] = set()
    names: list[str] = []
    for i, res in enumerate(chain):
        names.append(res.name)
        for atom in res:
            if atom.element.is_hydrogen:
                continue
            p = atom.pos
            if lo[0] <= p.x <= hi[0] and lo[1] <= p.y <= hi[1] and lo[2] <= p.z <= hi[2]:
                hits.add(i)
                break
    return hits, names


def chain_residue_names(chain) -> list[str]:
    return [r.name for r in chain]


def analyse(path: Path, pocket: set[int], pocket_names: list[str]) -> dict:
    st = gemmi.read_structure(str(path))
    model = st[0]
    chains = sorted(model, key=lambda c: len(c), reverse=True)
    prot_chain, lig_chain = chains[0], chains[-1]
    assert len(lig_chain) == 1, f"{path.name}: expected one ligand residue, got {len(lig_chain)}"

    lig_atoms = [a for r in lig_chain for a in r if not a.element.is_hydrogen]
    lig_xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig_atoms])

    contacts: list[int] = []
    min_dist = np.inf
    n_buried = 0
    res_plddt: dict[int, float] = {}
    prot_plddt: list[float] = []
    for i, res in enumerate(prot_chain):
        heavy = [a for a in res if not a.element.is_hydrogen]
        vals = [a.b_iso for a in res if not a.element.is_hydrogen]
        if vals:
            prot_plddt.extend(vals)
            res_plddt[i] = float(np.mean(vals))
        if not heavy:
            continue
        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in heavy])
        d = np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
        dmin = float(d.min())
        min_dist = min(min_dist, dmin)
        if dmin <= CONTACT_CUTOFF:
            contacts.append(i)

    # ligand burial: fraction of ligand heavy atoms within CONTACT_CUTOFF of the protein
    prot_all = np.array(
        [[a.pos.x, a.pos.y, a.pos.z] for r in prot_chain for a in r if not a.element.is_hydrogen]
    )
    dl = np.linalg.norm(lig_xyz[:, None, :] - prot_all[None, :, :], axis=2)
    n_buried = int((dl.min(axis=1) <= CONTACT_CUTOFF).sum())

    cset = set(contacts)
    inside = cset & pocket
    return {
        "mol_id": path.stem,
        "n_lig_atoms": len(lig_atoms),
        "n_contact_residues": len(cset),
        "n_contacts_in_pocket": len(inside),
        "frac_contacts_in_pocket": len(inside) / len(cset) if cset else np.nan,
        "jaccard_vs_pocket": len(inside) / len(cset | pocket) if cset else np.nan,
        "contact_pocket_recall": len(inside) / len(pocket),
        "touches_atp_site": bool(inside),
        "ligand_burial_frac": n_buried / len(lig_atoms) if len(lig_atoms) else np.nan,
        "min_heavy_distance_A": min_dist,
        "n_clashes_lt2A": int(min_dist < CLASH_CUTOFF),
        "interface_plddt_mean": float(np.mean([res_plddt[i] for i in cset])) if cset else np.nan,
        "chain_plddt_mean": float(np.mean(prot_plddt)) if prot_plddt else np.nan,
        "pocket_plddt_mean": float(np.mean([res_plddt[i] for i in pocket if i in res_plddt])),
        "n_residues_chain": len(prot_chain),
        "sequence_identity_vs_submitted": (
            chain_residue_names(prot_chain) == pocket_names
        ),
    }


def main() -> None:
    pocket, pocket_names = pocket_index_set()
    rows = [analyse(p, pocket, pocket_names) for p in sorted(POSES.glob("*.cif"))]
    df = pd.DataFrame(rows)

    cons = pd.read_parquet(CONSENSUS)[["mol_id", "consensus_rank", "iptm", "binding_confidence"]]
    df = df.merge(cons, on="mol_id", how="left").sort_values("consensus_rank")
    df.to_csv(OUT, index=False)

    def q(s: pd.Series) -> str:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return f"median {s.median():.3f} (IQR {np.percentile(s, 25):.3f}-{np.percentile(s, 75):.3f}; n={len(s)})"

    print(f"analysed complexes: {len(df)}  |  ATP-site pocket residues from box: {len(pocket)}")
    print(f"consensus ranks covered: {int(df.consensus_rank.min())}-{int(df.consensus_rank.max())}")
    print(f"sequences match submitted receptor: {bool(df.sequence_identity_vs_submitted.all())}")
    print(f"contact residues per complex: {q(df.n_contact_residues)}")
    print(f"fraction of contacts inside ATP-site pocket: {q(df.frac_contacts_in_pocket)}")
    print(f"ligand burial fraction: {q(df.ligand_burial_frac)}")
    print(f"min heavy-atom distance (A): {q(df.min_heavy_distance_A)}")
    print(f"clashes < 2.0 A: {int(df.n_clashes_lt2A.sum())} across {len(df)} complexes")
    print(f"interface pLDDT: {q(df.interface_plddt_mean)} | chain pLDDT {q(df.chain_plddt_mean)}")
    print(
        f"complexes touching the ATP site: {int(df.touches_atp_site.sum())}/{len(df)}; "
        f">=50% of contacts in pocket: {int((df.frac_contacts_in_pocket >= 0.5).sum())}/{len(df)}; "
        f">=80%: {int((df.frac_contacts_in_pocket >= 0.8).sum())}/{len(df)}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()