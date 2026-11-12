"""Receptor preparation: PDB -> cleaned PDB -> PDBQT.

Three interchangeable backends, tried in order of fidelity:

1. ``prepare_receptor`` (ADFR suite) - the reference implementation.
2. ``meeko``'s ``MoleculePreparation`` via RDKit - pure Python.
3. ``obabel -xr`` - last resort; adds hydrogens and Gasteiger charges.

Whichever is used is recorded in the manifest, because receptor protonation and
charge assignment materially change docking scores and the choice must not be
invisible.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcdd.runtime.logging import get_logger
from vcdd.runtime.paths import ensure_dir

LOG = get_logger()

#: Het residues discarded during cleaning (crystallisation additives, buffers).
DEFAULT_STRIP_HET = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT",
    "TRS", "DMS", "IOD", "NO3", "FMT", "ACY", "IMD", "EPE", "PGE", "1PE",
    "P6G", "BME", "MES", "CIT", "TLA", "MLI", "SIN",
}
#: Metals retained when ``keep_metals`` is true (often structural / catalytic).
METAL_RESIDUES = {"ZN", "MG", "MN", "FE", "FE2", "CA", "NA", "K", "CU", "NI", "CO", "CD"}


@dataclass
class ReceptorPrepResult:
    pdbqt_path: Path
    clean_pdb_path: Path
    method: str
    n_atoms: int
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pdbqt_path": str(self.pdbqt_path),
            "clean_pdb_path": str(self.clean_pdb_path),
            "method": self.method,
            "n_atoms": self.n_atoms,
            "warnings": self.warnings,
        }


def clean_pdb(
    src: str | Path,
    dst: str | Path,
    chain: str | None = None,
    keep_waters: bool = False,
    keep_metals: bool = True,
    strip_resnames: set[str] | None = None,
) -> tuple[Path, int, list[str]]:
    """Strip ligands/solvent/alt-locs and optionally select one chain."""
    strip = set(DEFAULT_STRIP_HET if strip_resnames is None else strip_resnames)
    if keep_waters:
        strip -= {"HOH", "WAT", "DOD"}

    kept: list[str] = []
    warnings: list[str] = []
    n_atoms = 0
    seen_chains: set[str] = set()
    dropped_het: dict[str, int] = {}

    for line in Path(src).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            ch = line[21]
            altloc = line[16]
            seen_chains.add(ch)
            if altloc not in (" ", "A"):
                continue
            if chain is not None and ch != chain:
                continue
            if line.startswith("HETATM"):
                if resname in strip:
                    continue
                if resname in METAL_RESIDUES:
                    if not keep_metals:
                        continue
                else:
                    # Any other HETATM is a ligand/cofactor: drop from receptor.
                    dropped_het[resname] = dropped_het.get(resname, 0) + 1
                    continue
            kept.append(line)
            n_atoms += 1
        elif line.startswith(("TER", "SSBOND", "CRYST1", "HEADER", "TITLE")):
            kept.append(line)

    if chain is not None and chain not in seen_chains:
        warnings.append(f"requested chain {chain!r} not present; chains found: {sorted(seen_chains)}")
    if dropped_het:
        warnings.append(f"removed non-protein HETATM residues from receptor: {dropped_het}")
    if n_atoms == 0:
        raise ValueError(f"receptor cleaning produced 0 atoms from {src}")

    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\nEND\n")
    return out, n_atoms, warnings


def _residue_order(pdb_text: str) -> dict[str, list[tuple[int, str, str]]]:
    """Residues per chain as (number, icode, name), sorted by number.

    Sorted rather than in file order: a PDBQT may interleave or reorder its
    records (the campaign receptor does), and only the numbering - not the line
    order - is meaningful. Two files can still be compared positionally because
    every writer emits residues in ascending order.
    """
    order: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            try:
                num = int(line[22:26])
            except ValueError:
                num = 0
            order.setdefault(line[21], []).append((num, line[26], line[17:20]))
    for chain in order:
        order[chain].sort(key=lambda t: (t[0], t[1]))
    return order


def restore_residue_names(pdbqt: Path, clean_pdb: Path) -> int:
    """Rewrite PDBQT residue names back to the crystal's own names.

    A writer that re-perceives a residue from the atoms present - Open Babel's
    ``-xr`` here, the backend actually used on this machine, since ADFR is not
    installed - concludes that a residue modelled as backbone plus CB is an
    alanine and writes that name. 7OTW leaves ASN4037 and ASP4109 in exactly that
    state, so both came back as ALA. The atoms are unchanged, so docking is barely
    affected, but stage 2 derives the sequence it sends to Boltz from these names:
    a two-residue difference against the MSA that was built from the real sequence
    is enough to misfold the predicted complex (measured: 21.8 A receptor C-alpha
    RMSD instead of 4.7 A, for all 70 complexes of the benchmark run). Restoring
    the names closes that path.

    Matching is positional within each chain because the writer also *renumbers*
    residues (1..N) while the cleaned PDB keeps the crystal numbering
    (3701..4123), so residue numbers cannot be compared directly. A chain whose
    residue count changed during preparation is left alone rather than guessed at.

    Returns the number of residues renamed.
    """
    want = _residue_order(clean_pdb.read_text())
    have = _residue_order(pdbqt.read_text())
    for chain, residues in have.items():
        if len(want.get(chain, [])) != len(residues):
            LOG.warning(
                "not restoring residue names on chain %s: %d residue(s) in %s vs %d in %s",
                chain, len(residues), pdbqt.name, len(want.get(chain, [])), clean_pdb.name,
            )
            want[chain] = residues       # identity mapping: change nothing

    # positional map from each PDBQT residue key to the crystal's name
    rename: dict[tuple[str, str], str] = {}
    for chain, residues in have.items():
        names = [t[2] for t in want.get(chain, [])]
        for i, (num, icode, name) in enumerate(residues):
            rename[(chain, f"{num:>4}{icode}")] = names[i] if i < len(names) else name
    fixed = sum(1 for k, v in rename.items() if v != dict(
        ((c, f"{n:>4}{i}"), nm) for c, rs in have.items() for n, i, nm in rs)[k])

    out_lines = []
    for line in pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:27])
            want_name = rename.get(key)
            if want_name and line[17:20] != want_name:
                line = f"{line[:17]}{want_name}{line[20:]}"
        out_lines.append(line)
    if fixed:
        pdbqt.write_text("\n".join(out_lines) + "\n")
        LOG.info("restored %d residue name(s) in %s from %s", fixed, pdbqt.name, clean_pdb.name)
    return fixed


def _prep_with_adfr(pdb: Path, pdbqt: Path) -> bool:
    exe = shutil.which("prepare_receptor")
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [exe, "-r", str(pdb), "-o", str(pdbqt), "-A", "hydrogens", "-U", "nphs_lps_waters"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode == 0 and pdbqt.exists() and pdbqt.stat().st_size > 0:
            return True
        LOG.debug("prepare_receptor failed: %s", (proc.stderr or proc.stdout)[:500])
    except Exception as exc:  # noqa: BLE001
        LOG.debug("prepare_receptor raised: %s", exc)
    return False


def _prep_with_meeko(pdb: Path, pdbqt: Path) -> bool:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
    except Exception:  # noqa: BLE001
        return False
    try:
        mol = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        # Meeko's MoleculePreparation requires *explicit* hydrogens and raises
        # "RDKit molecule has implicit Hs" otherwise. A PDB-derived receptor has
        # none by default, so this was dead code in practice whenever ADFR's
        # prepare_receptor and obabel were both absent.
        mol = Chem.AddHs(mol)
        prep = MoleculePreparation(rigid_macrocycles=True)
        setups = prep.prepare(mol)
        text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            LOG.debug("meeko receptor write failed: %s", err)
            return False
        # Meeko writes a flexible-ligand PDBQT; strip torsion records so Vina
        # reads it as a rigid receptor.
        lines = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
        pdbqt.write_text("\n".join(lines) + "\n")
        return pdbqt.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        LOG.debug("meeko receptor prep raised: %s", exc)
        return False


def _prep_with_obabel(pdb: Path, pdbqt: Path) -> bool:
    exe = shutil.which("obabel")
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [exe, str(pdb), "-O", str(pdbqt), "-xr", "-p", "7.4", "--partialcharge", "gasteiger"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        return proc.returncode == 0 and pdbqt.exists() and pdbqt.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        LOG.debug("obabel receptor prep raised: %s", exc)
        return False


def prepare_receptor(
    receptor_path: str | Path,
    out_dir: str | Path,
    chain: str | None = None,
    keep_waters: bool = False,
    keep_metals: bool = True,
    force: bool = False,
) -> ReceptorPrepResult:
    """Clean and convert a receptor to PDBQT. Returns the method actually used."""
    src = Path(receptor_path)
    if not src.exists():
        raise FileNotFoundError(f"receptor not found: {src}")
    out = ensure_dir(out_dir)
    clean_path = out / "receptor_clean.pdb"
    pdbqt_path = out / "receptor.pdbqt"

    if pdbqt_path.exists() and not force:
        n_atoms = sum(1 for l in pdbqt_path.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
        return ReceptorPrepResult(pdbqt_path, clean_path, "cached", n_atoms, [])

    if src.suffix.lower() == ".pdbqt":
        shutil.copyfile(src, pdbqt_path)
        n_atoms = sum(1 for l in pdbqt_path.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
        return ReceptorPrepResult(pdbqt_path, src, "provided_pdbqt", n_atoms, [])

    clean_path, n_atoms, warnings = clean_pdb(
        src, clean_path, chain=chain, keep_waters=keep_waters, keep_metals=keep_metals
    )

    for method, fn in (("prepare_receptor", _prep_with_adfr), ("meeko", _prep_with_meeko), ("obabel", _prep_with_obabel)):
        if fn(clean_path, pdbqt_path):
            renamed = restore_residue_names(pdbqt_path, clean_path)
            n_pdbqt = sum(1 for l in pdbqt_path.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
            if method != "prepare_receptor":
                warnings.append(
                    f"receptor prepared with {method} because prepare_receptor (ADFR) was unavailable; "
                    "protonation and charges may differ from the reference implementation"
                )
            if renamed:
                warnings.append(
                    f"{method} renamed {renamed} residue(s) whose side chain the crystal never "
                    f"modelled; the names were restored from {clean_path.name}. Without this, "
                    "stage 2 sends the wrong sequence to Boltz and the predicted complex misfolds."
                )
            return ReceptorPrepResult(pdbqt_path, clean_path, method, n_pdbqt, warnings)

    raise RuntimeError(
        "receptor PDBQT preparation failed with every backend "
        "(prepare_receptor / meeko / obabel). Install the ADFR suite or Open Babel."
    )
