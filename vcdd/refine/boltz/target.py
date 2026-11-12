"""Build the Boltz ``Target`` payload from the prepared receptor and the box.

Boltz takes a *sequence*, not a structure, so stage 2 has to recover one from
the receptor file stage 1 docked into. Two things then have to be got right or
stage 2 silently folds the wrong thing:

**Pocket residues are 0-indexed positions in the submitted sequence**, keyed by
chain ID - not PDB residue numbers. Crystal structures have gaps, insertion
codes, and numbering that starts wherever the depositor chose, so
``resSeq - 1`` is wrong in the general case and wrong in a way that still
produces plausible output. This module builds the sequence and the index map in
one pass so the two can never disagree.

**Sequences come from the receptor that was actually docked.** Reading the
canonical UniProt sequence instead would point Boltz at residues that are not in
the structure and drop the ones that are.

Parsing is done directly from the PDB/PDBQT text rather than through a structure
library: the only fields needed are chain, residue number, residue name and
coordinates, all at fixed columns, and the prepared receptor is by construction
a clean single-model file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from vcdd.receptor.box import Box

#: Three-letter to one-letter, including the protonation-state and modified
#: residue names that receptor preparation (ADFR / pdb2pqr / Amber) emits.
AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # protonation variants
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R", "TYM": "Y",
    # common modified residues that belong in the chain
    "MSE": "M", "SEC": "U", "PYL": "O", "SEP": "S", "TPO": "T", "PTR": "Y",
}

#: Residue names that are never part of the protein chain.
_NON_POLYMER = {"HOH", "WAT", "DOD", "SOL"}


@dataclass
class ChainSequence:
    """One protein chain recovered from the receptor file."""

    chain_id: str
    sequence: str = ""
    #: ``(resSeq, iCode)`` for each position, in sequence order.
    residue_keys: list[tuple[int, str]] = field(default_factory=list)
    #: Heavy-atom coordinates per position.
    coords: list[list[tuple[float, float, float]]] = field(default_factory=list)
    #: Residue names the parser could not map to a one-letter code.
    unknown: list[str] = field(default_factory=list)
    #: ``False`` when the input file listed residues out of ascending order
    #: (see :func:`parse_receptor_chains`).  The sequence is sorted either way;
    #: this flag exists so a reordered receptor is visible in the provenance.
    file_order_matches_sequence: bool = True

    def __len__(self) -> int:
        return len(self.sequence)


def parse_receptor_chains(path: str | Path) -> dict[str, ChainSequence]:
    """Recover per-chain sequences and heavy-atom coordinates from a PDB/PDBQT.

    Only ``ATOM`` records are read. ``HETATM`` is skipped, which is what we
    want: cofactors, ions and the co-crystal ligand are not part of the chain
    Boltz should fold, and including them would shift every downstream pocket
    index.

    Residues are ordered by ``(resSeq, iCode)``, **not** by the order in which
    they appear in the file.  This is not a detail: docking preparation tools do
    not all preserve residue order.  ADFR's ``prepare_receptor``, for example,
    writes a PDBQT whose atoms are grouped by atom name, so the residues appear
    in reverse order in the file.  Reading such a file sequentially returns a
    *permutation* of the real chain - the right composition and a plausible
    looking sequence, but not the protein.  Sorting by residue number makes the
    recovered sequence independent of writer conventions, and
    ``ChainSequence.file_order_matches_sequence`` records whether the input
    happened to be ordered, so a reordered receptor surfaces in the stage-2
    provenance instead of silently changing what Boltz folds.
    """
    #: chain -> (resSeq, iCode) -> [resname, [(x, y, z), ...]]
    residues: dict[str, dict[tuple[int, str], list]] = {}
    last_key: dict[str, tuple[int, str]] = {}
    out_of_order: set[str] = set()

    for raw in Path(path).read_text(errors="replace").splitlines():
        if not raw.startswith("ATOM"):
            continue
        if len(raw) < 54:
            continue
        resname = raw[17:20].strip().upper()
        if resname in _NON_POLYMER:
            continue
        chain_id = (raw[21] or " ").strip() or "A"
        try:
            resseq = int(raw[22:26])
            x, y, z = float(raw[30:38]), float(raw[38:46]), float(raw[46:54])
        except ValueError:
            continue
        icode = raw[26].strip()
        element = raw[76:78].strip().upper() if len(raw) >= 78 else ""
        atom_name = raw[12:16].strip()
        # PDBQT replaces the element column with an AutoDock atom type
        # ('HD', 'NA', 'OA', 'A'), so fall back to the atom name's first letter.
        is_hydrogen = element in {"H", "HD", "HS"} or (not element and atom_name[:1] == "H")
        if is_hydrogen:
            continue

        key = (resseq, icode)
        store = residues.setdefault(chain_id, {})
        if key not in store:
            store[key] = [resname, []]
        store[key][1].append((x, y, z))

        prev = last_key.get(chain_id)
        if prev is not None and key < prev:
            out_of_order.add(chain_id)
        last_key[chain_id] = key

    chains: dict[str, ChainSequence] = {}
    for chain_id, store in residues.items():
        ch = ChainSequence(
            chain_id=chain_id, file_order_matches_sequence=chain_id not in out_of_order
        )
        for key in sorted(store):
            resname, coords = store[key]
            one = AA3TO1.get(resname)
            if one is None:
                if resname not in ch.unknown:
                    ch.unknown.append(resname)
                continue
            ch.sequence += one
            ch.residue_keys.append(key)
            ch.coords.append(list(coords))
        chains[chain_id] = ch

    return {k: v for k, v in chains.items() if v.sequence}


def pocket_residues_from_box(
    chains: dict[str, ChainSequence],
    box: Box,
    margin: float = 0.0,
) -> dict[str, list[int]]:
    """Residues with at least one heavy atom inside the docking box.

    Returns **0-indexed positions into each chain's submitted sequence**, which
    is what the Boltz API expects. Chains with no residue in the box are omitted
    rather than mapped to an empty list, so the payload says "this chain is not
    part of the pocket" instead of "this chain's pocket is empty".
    """
    cx, cy, cz = box.center
    sx, sy, sz = box.size
    hx, hy, hz = sx / 2.0 + margin, sy / 2.0 + margin, sz / 2.0 + margin
    lo = (cx - hx, cy - hy, cz - hz)
    hi = (cx + hx, cy + hy, cz + hz)

    out: dict[str, list[int]] = {}
    for chain_id, ch in chains.items():
        hits = [
            i
            for i, atoms in enumerate(ch.coords)
            if any(
                lo[0] <= a[0] <= hi[0] and lo[1] <= a[1] <= hi[1] and lo[2] <= a[2] <= hi[2]
                for a in atoms
            )
        ]
        if hits:
            out[chain_id] = hits
    return out


def build_target(
    receptor_path: str | Path,
    box: Box | None = None,
    pocket_from_box: bool = True,
    pocket_margin: float = 0.0,
    reference_ligands: Iterable[str] | None = None,
    chains: dict[str, ChainSequence] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the API ``Target`` dict plus a human-readable provenance record.

    The second return value goes into ``stage2_summary.json``: chain lengths,
    pocket sizes and any residue names the parser could not map. That record is
    what makes a silently truncated chain visible instead of showing up as
    inexplicably poor binding confidence.
    """
    chains = chains if chains is not None else parse_receptor_chains(receptor_path)
    if not chains:
        raise ValueError(
            f"no protein chains parsed from {receptor_path}. Boltz needs a sequence; "
            f"check that the prepared receptor still has ATOM records."
        )

    entities = [
        {"chain_ids": [ch.chain_id], "type": "protein", "value": ch.sequence}
        for ch in (chains[k] for k in sorted(chains))
    ]
    target: dict[str, Any] = {"entities": entities, "type": "no_template"}

    pockets: dict[str, list[int]] = {}
    if pocket_from_box and box is not None:
        pockets = pocket_residues_from_box(chains, box, margin=pocket_margin)
        if pockets:
            target["pocket_residues"] = pockets

    refs = [r for r in (reference_ligands or []) if r]
    if refs:
        target["reference_ligands"] = list(refs)

    provenance = {
        "receptor": str(receptor_path),
        # A list, not a dict keyed by chain id: this record is written straight
        # into stage2_summary.json and read back by the report, and a list keeps
        # the chain order stable and the JSON diffable.
        "chains": [
            {
                "chain_id": k,
                "length": len(chains[k]),
                "first_residue": chains[k].residue_keys[0] if chains[k].residue_keys else None,
                "last_residue": chains[k].residue_keys[-1] if chains[k].residue_keys else None,
                "unmapped_residue_names": chains[k].unknown,
                # False means the receptor file listed residues out of order (a
                # reordered PDBQT, say).  The sequence above is sorted by residue
                # number regardless; this flag makes the input quirk auditable.
                "file_order_matches_sequence": chains[k].file_order_matches_sequence,
            }
            for k in sorted(chains)
        ],
        "pocket_sizes": {k: len(v) for k, v in pockets.items()},
        "pocket_from_box": bool(pocket_from_box and box is not None),
        "pocket_margin_a": pocket_margin,
        "reference_ligands": len(refs),
        "note": "pocket_residues are 0-indexed positions in the submitted sequence, not PDB "
                "residue numbers; sequences are taken from the docked receptor, so any "
                "unresolved loop is absent here exactly as it was absent from docking.",
    }
    if pocket_from_box and box is not None and not pockets:
        provenance["warning"] = (
            "no residue had a heavy atom inside the docking box, so no pocket hint was sent "
            "and Boltz will place the ligand without one. Check that the receptor and the box "
            "are in the same coordinate frame."
        )
    return target, provenance
