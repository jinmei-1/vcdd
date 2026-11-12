#!/usr/bin/env python
"""Generate the bundled test fixtures.

Creates, under ``tests/fixtures/``:

* ``library.sdf``, ``library.smi``, ``library.csv``, ``library.parquet`` -
  the *same* 200 molecules in all four supported input formats, so the reader
  tests can assert format-independence.
* ``library_messy.sdf`` - deliberately broken records (bad valence, empty mol)
  plus salts and mixtures, to test the rejection accounting.
* ``receptor.pdb`` + ``reference_ligand.sdf`` - a small protein-ligand complex
  fetched from the RCSB PDB (CC0) and split into receptor and native ligand.

Run:  python scripts/make_fixtures.py [--pdb 3ERT] [--n 200]

The molecule set is a deterministic, self-contained enumeration (no network
needed) built from a small set of drug-like scaffolds and substituents, so the
fixtures are reproducible and carry no licence obligations.
"""

from __future__ import annotations

import argparse
import gzip
import io
import random
import sys
import urllib.request
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# Small, deliberately drug-like building blocks. Combinatorial enumeration of
# these gives a set with a realistic spread of MW / logP / HBD / ring counts.
SCAFFOLDS = [
    "c1ccc(-c2ccncc2)cc1{R}",
    "O=C(N{R})c1ccc(Cl)cc1",
    "c1ccc2[nH]c(-c3ccc({R})cc3)nc2c1",
    "O=S(=O)(N{R})c1ccccc1",
    "c1cnc2c(c1)ccc(n2){R}",
    "O=C1CCN(C1){R}",
    "c1ccc(Oc2ccc({R})cc2)cc1",
    "N#Cc1ccc({R})cc1",
    "O=C(Nc1ccccc1){R}",
    "c1ccc(-c2nnc(o2){R})cc1",
    "O=C(N1CCCC1)c1ccc({R})nc1",
    "c1ccc(-n2cnc(c2)-c2ccc({R})cc2)cc1",
    "O=C(O)c1ccc(NC(=O){R})cc1",
    "Cc1nc(N{R})sc1-c1ccccc1",
    "c1ccc2c(c1)nc(-c1ccc({R})cc1)[nH]2",
    "O=C1Nc2ccccc2N1{R}",
]

SUBSTITUENTS = [
    "C", "CC", "CCC", "C(C)C", "CCO", "CN", "CC(=O)O", "CN(C)C", "c1ccccc1",
    "Cc1ccccc1", "CF", "C(F)(F)F", "Cl", "Br", "OC", "N1CCOCC1", "N1CCNCC1",
    "c1ccncc1", "CC(N)=O", "CS(C)=O", "CCN", "CCS", "OCC", "C1CC1", "C1CCCC1",
    "c1ccc(F)cc1", "c1ccc(OC)cc1", "CC(C)(C)O", "CCOC(C)=O", "CNC(C)=O",
    "N1CCCC1", "N1CCCCC1", "c1cc(Cl)ccn1", "CC#N", "COC(=O)C",
]

MESSY_EXTRA = [
    ("SALT-1", "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"),
    ("SALT-2", "CN1CCC[C@H]1c1cccnc1.Cl.Cl"),
    ("MIXTURE-1", "c1ccccc1.CCO"),
    ("BAD-VALENCE", "C(C)(C)(C)(C)C"),
    ("EMPTY", ""),
    ("GIBBERISH", "not_a_smiles"),
    ("HUGE", "C" * 90),
    ("METAL", "[Pt](Cl)(Cl)(N)N"),
]


def enumerate_molecules(n: int, seed: int = 7) -> list[tuple[str, str]]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        scaffold = rng.choice(SCAFFOLDS)
        sub = rng.choice(SUBSTITUENTS)
        smi = scaffold.replace("{R}", sub)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        if not (12 <= mol.GetNumHeavyAtoms() <= 45):
            continue
        seen.add(canonical)
        out.append((f"FIX{len(out) + 1:05d}", canonical))
    return out


def write_all_formats(mols: list[tuple[str, str]], out_dir: Path) -> None:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SDF (with property tags and 3D coordinates on a subset) -------------
    sdf_path = out_dir / "library.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for i, (mol_id, smi) in enumerate(mols):
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("_Name", mol_id)
        mol.SetProp("ID", mol_id)
        mol.SetProp("source", "vcdd-fixture")
        mol.SetProp("batch", str(i % 4))
        mol.SetProp("mw_tag", f"{Descriptors.MolWt(mol):.2f}")
        if i < 5:  # a few records with real 3D coordinates
            mh = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mh, randomSeed=0xC0FFEE) == 0:
                AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
                mol = Chem.RemoveHs(mh)
                mol.SetProp("_Name", mol_id)
                mol.SetProp("ID", mol_id)
                mol.SetProp("source", "vcdd-fixture")
                mol.SetProp("batch", str(i % 4))
        writer.write(mol)
    writer.close()

    # gzipped copy, to exercise the .sdf.gz path
    with open(sdf_path, "rb") as fin, gzip.open(out_dir / "library.sdf.gz", "wb") as fout:
        fout.write(fin.read())

    # --- SMI ---------------------------------------------------------------
    with open(out_dir / "library.smi", "w") as fh:
        for mol_id, smi in mols:
            fh.write(f"{smi}\t{mol_id}\n")

    # --- CSV (BOM + extra columns, mimicking a real vendor export) ---------
    rows = []
    for mol_id, smi in mols:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            {
                "compound_id": mol_id,
                "canonical_smiles": smi,
                "vendor": "fixture-chem",
                "price_usd": round(50 + Descriptors.MolWt(mol) / 10, 2),
                "clogp": round(Crippen.MolLogP(mol), 3),
            }
        )
    df = pd.DataFrame(rows)
    with open(out_dir / "library.csv", "w", encoding="utf-8-sig", newline="") as fh:
        df.to_csv(fh, index=False)

    # --- Parquet -----------------------------------------------------------
    df.to_parquet(out_dir / "library.parquet", index=False)

    # --- messy SDF ---------------------------------------------------------
    messy = out_dir / "library_messy.sdf"
    with open(messy, "w") as fh:
        w = Chem.SDWriter(fh)
        for mol_id, smi in mols[:20]:
            mol = Chem.MolFromSmiles(smi)
            mol.SetProp("_Name", mol_id)
            w.write(mol)
        for mol_id, smi in MESSY_EXTRA:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            mol.SetProp("_Name", mol_id)
            w.write(mol)
        w.close()
        # Append a truly malformed record that RDKit cannot parse.
        fh.write("BROKEN-RECORD\n  fake\n\n  not a counts line\nM  END\n$$$$\n")

    # --- messy CSV / SMI with the un-parseable entries ----------------------
    messy_rows = [{"compound_id": mid, "smiles": smi} for mid, smi in MESSY_EXTRA]
    messy_rows += [{"compound_id": mid, "smiles": smi} for mid, smi in mols[:20]]
    pd.DataFrame(messy_rows).to_csv(out_dir / "library_messy.csv", index=False)


def fetch_receptor(pdb_id: str, out_dir: Path, ligand_resname: str | None = None) -> tuple[Path, Path] | None:
    """Download a PDB entry and split it into receptor + native ligand.

    RCSB PDB data is CC0. Returns ``None`` (with a message) if offline, so that
    fixture generation still succeeds for the chemistry-only tests.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"[make_fixtures] could not download {pdb_id}: {exc}", file=sys.stderr)
        return None

    from collections import Counter

    skip_het = {
        "HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT", "CL",
        "NA", "K", "MG", "CA", "ZN", "MN", "TRS", "DMS", "IOD", "BR", "NO3",
        "FMT", "ACY", "IMD", "EPE", "PGE", "1PE", "P6G",
    }
    het_counts: Counter[str] = Counter()
    for line in text.splitlines():
        if line.startswith("HETATM"):
            res = line[17:20].strip()
            if res not in skip_het:
                het_counts[res] += 1
    if ligand_resname is None:
        if not het_counts:
            print(f"[make_fixtures] no ligand-like HETATM in {pdb_id}", file=sys.stderr)
            return None
        ligand_resname = het_counts.most_common(1)[0][0]

    # Keep the first chain that contains the ligand, to make a compact fixture.
    lig_chain = None
    for line in text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == ligand_resname:
            lig_chain = line[21]
            break

    receptor_lines, ligand_lines = [], []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            res = line[17:20].strip()
            chain = line[21]
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            if res == ligand_resname and (lig_chain is None or chain == lig_chain):
                ligand_lines.append(line)
            elif line.startswith("ATOM") and (lig_chain is None or chain == lig_chain):
                receptor_lines.append(line)
        elif line.startswith(("HEADER", "TITLE", "CRYST1")):
            receptor_lines.append(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = out_dir / "receptor.pdb"
    ligand_pdb = out_dir / "reference_ligand.pdb"
    receptor_path.write_text("\n".join(receptor_lines) + "\nEND\n")
    ligand_pdb.write_text("\n".join(ligand_lines) + "\nEND\n")

    # Convert the ligand to SDF with bond orders where possible.
    ligand_sdf = out_dir / "reference_ligand.sdf"
    try:
        from rdkit import Chem

        mol = Chem.MolFromPDBBlock(ligand_pdb.read_text(), removeHs=False, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            w = Chem.SDWriter(str(ligand_sdf))
            mol.SetProp("_Name", f"{pdb_id}_{ligand_resname}")
            w.write(mol)
            w.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[make_fixtures] ligand SDF conversion failed: {exc}", file=sys.stderr)

    meta = out_dir / "receptor_meta.txt"
    meta.write_text(
        f"pdb_id={pdb_id}\nligand_resname={ligand_resname}\nchain={lig_chain}\n"
        f"receptor_atoms={sum(1 for l in receptor_lines if l.startswith('ATOM'))}\n"
        f"ligand_atoms={len(ligand_lines)}\nsource=RCSB PDB (CC0)\n"
    )
    print(f"[make_fixtures] receptor {pdb_id} ligand {ligand_resname}: "
          f"{sum(1 for l in receptor_lines if l.startswith('ATOM'))} protein atoms, {len(ligand_lines)} ligand atoms")
    return receptor_path, ligand_sdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pdb", default="1STP", help="PDB id for the receptor fixture")
    ap.add_argument("--ligand", default=None, help="ligand residue name override")
    ap.add_argument("--out", default=str(FIXTURES))
    ap.add_argument("--skip-receptor", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    mols = enumerate_molecules(args.n)
    print(f"[make_fixtures] enumerated {len(mols)} molecules")
    write_all_formats(mols, out_dir)
    print(f"[make_fixtures] wrote library.{{sdf,sdf.gz,smi,csv,parquet}} to {out_dir}")
    if not args.skip_receptor:
        fetch_receptor(args.pdb, out_dir, args.ligand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
