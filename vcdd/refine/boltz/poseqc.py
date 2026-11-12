"""Does the predicted complex actually put the ligand in the pocket we docked into?

This is the one structural check that stage 2 can still make in 0.2.0, and it
matters more than it did when stage 2 was a docking run. A docking program is
*constrained* to the box; Boltz is not. It folds the complex from sequence plus
SMILES and is free to place the ligand anywhere on the protein - a lipid groove,
a crystal-contact crevice, an allosteric site. A high ``binding_confidence`` for
a ligand sitting 25 A from the site you care about is not a hit, but nothing in
the returned metrics will tell you that.

So: download the predicted ligand coordinates, measure them against the stage-1
docking box, and record the offset. The result is advisory - it is written to
``pose_qc.parquet`` and surfaced in the report, and it does not silently drop
rows - but a shortlist where most poses land outside the box is telling you the
pocket constraint did not hold.

Downloading is optional and capped at ``keep_structures_top_k``: the artifact
URLs are presigned and expire, and pulling 500 structures to check 500 centroids
is a lot of transfer for a number that only matters at the top of the list.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from vcdd.runtime.logging import get_logger
from vcdd.runtime.paths import ensure_dir

LOG = get_logger()

QC_COLUMNS = [
    "mol_id", "ligand_in_box", "centroid_offset_A", "frac_atoms_in_box",
    "max_axis_excess_A", "pose_n_atoms", "pose_path", "pose_error",
    "pose_frame_aligned", "receptor_ca_rmsd_A",
]


# ------------------------------------------------------------------ download
def http_get_text(url: str, timeout_s: float = 60.0) -> str:
    """Fetch a presigned artifact URL as text."""
    try:
        import httpx

        resp = httpx.get(url, timeout=timeout_s, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except ImportError:  # pragma: no cover - httpx ships with the SDK
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout_s) as fh:  # noqa: S310
            return fh.read().decode("utf-8", errors="replace")


# ------------------------------------------------------------------- parsing
def _parse_cif_atoms(text: str) -> list[dict[str, Any]]:
    """Read the ``_atom_site`` loop of an mmCIF.

    Hand-rolled rather than pulled from gemmi/Biopython: this needs one loop out
    of one file, the dependency is not otherwise required, and the failure mode
    of a missing parser at stage 2 is worse than the failure mode of a simple one.
    """
    lines = text.splitlines()
    atoms: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        j = i + 1
        headers: list[str] = []
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if not any(h.startswith("_atom_site.") for h in headers):
            i = j
            continue
        cols = {h.split(".", 1)[1]: k for k, h in enumerate(headers) if h.startswith("_atom_site.")}
        need = ("Cartn_x", "Cartn_y", "Cartn_z")
        if not all(c in cols for c in need):
            i = j
            continue
        while j < len(lines):
            raw = lines[j]
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                break
            parts = s.split()
            if len(parts) < len(headers):
                break
            try:
                atom = {
                    "x": float(parts[cols["Cartn_x"]]),
                    "y": float(parts[cols["Cartn_y"]]),
                    "z": float(parts[cols["Cartn_z"]]),
                    "element": (parts[cols["type_symbol"]] if "type_symbol" in cols else "").strip(),
                    "group": (parts[cols["group_PDB"]] if "group_PDB" in cols else "").strip(),
                    "comp": (parts[cols["label_comp_id"]] if "label_comp_id" in cols else "").strip(),
                    "chain": (
                        parts[cols["auth_asym_id"]] if "auth_asym_id" in cols
                        else parts[cols["label_asym_id"]] if "label_asym_id" in cols
                        else ""
                    ).strip(),
                    "name": (
                        parts[cols["label_atom_id"]] if "label_atom_id" in cols
                        else parts[cols["auth_atom_id"]] if "auth_atom_id" in cols
                        else ""
                    ).strip(),
                }
            except (ValueError, IndexError):
                j += 1
                continue
            atoms.append(atom)
            j += 1
        i = j
    return atoms


def _parse_pdb_atoms(text: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atoms.append({
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "element": line[76:78].strip() or line[12:16].strip()[:1],
                "group": line[:6].strip(),
                "comp": line[17:20].strip(),
                "chain": (line[21:22].strip() or "A"),
                "name": line[12:16].strip(),
            })
        except ValueError:
            continue
    return atoms


def _parse_sdf_atoms(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if len(lines) < 4:
        return []
    try:
        n_atoms = int(lines[3][0:3])
    except ValueError:
        return []
    atoms: list[dict[str, Any]] = []
    for line in lines[4:4 + n_atoms]:
        try:
            atoms.append({
                "x": float(line[0:10]), "y": float(line[10:20]), "z": float(line[20:30]),
                "element": line[31:34].strip(), "group": "HETATM", "comp": "LIG", "chain": "",
                "name": "",
            })
        except ValueError:
            continue
    return atoms


def parse_structure_atoms(text: str) -> list[dict[str, Any]]:
    """Parse mmCIF, PDB or SDF/MOL text into a flat atom list."""
    head = text[:4000]
    if "_atom_site." in head or head.lstrip().startswith("data_"):
        return _parse_cif_atoms(text)
    if "V2000" in head or "V3000" in head:
        return _parse_sdf_atoms(text)
    return _parse_pdb_atoms(text)


def ligand_heavy_coords(
    text: str,
    chain_id: str | None = None,
    ligand_comp: str | None = None,
    hetatm_only: bool = False,
) -> np.ndarray:
    """Heavy-atom coordinates of the ligand.

    Boltz's API ``ligand_structure`` artifact contains only the ligand, so
    usually no filter is needed; the chain and residue filters are for the
    full-complex file, where picking the wrong chain would silently measure the
    protein's centroid instead. The local backend's structure file is the whole
    complex, in which Boltz tags the ligand ``HETATM`` (and the protein
    ``ATOM``), so ``hetatm_only`` isolates it without knowing its chain label.
    """
    atoms = parse_structure_atoms(text)
    if hetatm_only:
        atoms = [a for a in atoms if a["group"] == "HETATM"]
    if chain_id:
        subset = [a for a in atoms if a["chain"] == chain_id]
        if subset:
            atoms = subset
    if ligand_comp:
        subset = [a for a in atoms if a["comp"] == ligand_comp]
        if subset:
            atoms = subset
    heavy = [a for a in atoms if (a["element"] or "X").upper() not in {"H", "D"}]
    if not heavy:
        return np.zeros((0, 3), dtype=float)
    return np.array([[a["x"], a["y"], a["z"]] for a in heavy], dtype=float)


def protein_ca_coords(text: str, chain_id: str | None = None) -> np.ndarray:
    """C-alpha coordinates of the protein chain of a predicted complex.

    Co-folding backends return a *model*, not a pose: the complex sits in
    whatever frame the network chose, unrelated to the receptor file the docking
    box was defined on.  These coordinates are what makes the frames
    comparable again (see :func:`superpose_to_reference`).
    """
    atoms = [
        a for a in parse_structure_atoms(text)
        if a["group"] == "ATOM" and (a.get("name") or "").upper() == "CA"
    ]
    if chain_id:
        subset = [a for a in atoms if a["chain"] == chain_id]
        if subset:
            atoms = subset
    if not atoms:
        return np.zeros((0, 3), dtype=float)
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for a in atoms:
        by_chain.setdefault(a["chain"], []).append(a)
    # a co-folded complex has one protein chain; if the file somehow has more,
    # the largest is the receptor and the small ones are peptides/ions.
    largest = max(by_chain.values(), key=len)
    return np.array([[a["x"], a["y"], a["z"]] for a in largest], dtype=float)


def kabsch(mobile: Any, target: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation and translation mapping ``mobile`` onto ``target``, plus the RMSD."""
    mobile = np.asarray(mobile, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    a, b = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    rmsd = float(np.sqrt((((a @ rot.T) - b) ** 2).sum(axis=1).mean()))
    return rot, tc - mc @ rot.T, rmsd


def superpose_to_reference(
    coords: Any, mobile_ca: Any, reference_ca: Any
) -> tuple[np.ndarray, float] | None:
    """Move ``coords`` from the model's frame into the receptor's frame.

    Returns ``(moved_coords, receptor_ca_rmsd)`` or ``None`` when the model's
    protein chain cannot be matched to the reference (different length, or no
    protein in the file at all) - in that case the box test would be measuring
    an arbitrary coordinate frame, not a binding site.
    """
    mobile_ca = np.asarray(mobile_ca, dtype=float).reshape(-1, 3)
    reference_ca = np.asarray(reference_ca, dtype=float).reshape(-1, 3)
    if mobile_ca.shape != reference_ca.shape or mobile_ca.shape[0] < 3:
        return None
    rot, trans, rmsd = kabsch(mobile_ca, reference_ca)
    return np.asarray(coords, dtype=float).reshape(-1, 3) @ rot.T + trans, rmsd


# -------------------------------------------------------------------- checks
def box_check(coords: Any, box: Any, tolerance_a: float = 0.0) -> dict[str, Any]:
    """Measure a ligand against the stage-1 docking box.

    ``ligand_in_box`` is deliberately strict - every heavy atom inside - because
    the interesting failure is "the model put it somewhere else entirely", and a
    centroid-only test passes for a ligand draped across the box edge.

    ``coords`` is any ``(n, 3)`` array-like; a plain list of tuples is fine.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    center = np.asarray(box.center, dtype=float)
    half = np.asarray(box.size, dtype=float) / 2.0 + float(tolerance_a)
    if coords.shape[0] == 0:
        return {
            "ligand_in_box": None, "centroid_offset_A": None, "frac_atoms_in_box": None,
            "max_axis_excess_A": None, "pose_n_atoms": 0,
        }
    delta = np.abs(coords - center)
    inside = np.all(delta <= half, axis=1)
    excess = np.max(delta - half)
    centroid = coords.mean(axis=0)
    return {
        "ligand_in_box": bool(inside.all()),
        "centroid_offset_A": float(np.linalg.norm(centroid - center)),
        "frac_atoms_in_box": float(inside.mean()),
        "max_axis_excess_A": float(max(0.0, excess)),
        "pose_n_atoms": int(coords.shape[0]),
    }


def check_pose_url(
    url: str | None,
    box: Any,
    mol_id: str,
    save_to: Path | None = None,
    chain_id: str | None = None,
    hetatm_only: bool = False,
    timeout_s: float = 60.0,
    fetch: Any = http_get_text,
    reference_ca: Any = None,
) -> dict[str, Any]:
    """Download one predicted ligand structure and measure it against the box.

    ``reference_ca`` is the receptor's C-alpha trace (the file the box was
    defined on). When it is given *and* the structure contains a protein chain,
    that chain is superposed onto the reference first and the ligand is measured
    in the receptor's frame; without that step a co-folded complex is compared
    against the box in an unrelated coordinate frame, which reports every ligand
    as hundreds of Angstroms away.

    A structure with no protein chain is a different case: the hosted backends
    return a ligand-only file, produced from a request that carried the pocket,
    so its coordinates are already in the receptor's frame and nothing needs to
    be superposed. ``pose_frame_aligned`` therefore distinguishes the three
    outcomes - ``True`` (superposed), ``False`` (a chain was there and could not
    be matched, so no box number is reported), ``None`` (nothing to superpose).
    """
    row: dict[str, Any] = {
        "mol_id": str(mol_id), "ligand_in_box": None, "centroid_offset_A": None,
        "frac_atoms_in_box": None, "max_axis_excess_A": None, "pose_n_atoms": 0,
        "pose_path": None, "pose_error": None,
        "pose_frame_aligned": None, "receptor_ca_rmsd_A": None,
    }
    if not url:
        row["pose_error"] = "no_structure_url"
        return row
    try:
        text = fetch(url, timeout_s)
    except Exception as exc:  # noqa: BLE001
        row["pose_error"] = f"download_failed:{type(exc).__name__}"
        return row
    if save_to is not None:
        try:
            ensure_dir(save_to.parent)
            save_to.write_text(text)
            row["pose_path"] = str(save_to)
        except OSError as exc:
            row["pose_error"] = f"save_failed:{exc.strerror}"
    try:
        coords = ligand_heavy_coords(text, chain_id=chain_id, hetatm_only=hetatm_only)
    except Exception as exc:  # noqa: BLE001
        row["pose_error"] = f"parse_failed:{type(exc).__name__}"
        return row
    if reference_ca is not None and np.asarray(reference_ca).size:
        try:
            mobile_ca = protein_ca_coords(text, chain_id=chain_id)
            moved = (superpose_to_reference(coords, mobile_ca, reference_ca)
                     if len(mobile_ca) else None)
        except Exception as exc:  # noqa: BLE001
            row["pose_error"] = f"alignment_failed:{type(exc).__name__}"
            return row
        if moved is None and len(mobile_ca):
            # A protein chain is present but does not match the receptor (a
            # different chain, or a different length): the box test would be
            # frame-arbitrary, so report the pose as unmeasured instead of
            # inventing a distance.
            row["pose_frame_aligned"] = False
            row["pose_error"] = "frame_not_aligned"
            return row
        if moved is not None:
            coords, row["receptor_ca_rmsd_A"] = moved
            row["pose_frame_aligned"] = True
    row.update(box_check(coords, box))
    if row["pose_n_atoms"] == 0:
        row["pose_error"] = "no_heavy_atoms_parsed"
    return row


def qc_poses(
    rows: Sequence[dict[str, Any]],
    box: Any,
    out_dir: str | Path,
    top_k: int = 50,
    url_field: str = "url_ligand_structure",
    chain_id: str | None = None,
    hetatm_only: bool = False,
    timeout_s: float = 60.0,
    fetch: Any = http_get_text,
    reference_ca: Any = None,
) -> list[dict[str, Any]]:
    """QC the top ``top_k`` rows, in the order given, saving each structure.

    ``rows`` must already be ranked; this does not re-rank. Poses are written as
    ``<mol_id>.cif`` under ``out_dir``. ``hetatm_only`` restricts the measured
    atoms to ``HETATM`` lines (the local backend's structure file is the full
    complex, and Boltz tags the ligand HETATM). ``reference_ca`` is passed
    through to :func:`check_pose_url` so the box test happens in the receptor's
    frame.
    """
    if top_k <= 0 or not rows:
        return []
    poses = ensure_dir(out_dir)
    out: list[dict[str, Any]] = []
    for row in list(rows)[:top_k]:
        mol_id = str(row.get("mol_id"))
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in mol_id)[:120]
        out.append(check_pose_url(
            row.get(url_field), box, mol_id,
            save_to=poses / f"{safe}.cif", chain_id=chain_id,
            hetatm_only=hetatm_only, timeout_s=timeout_s, fetch=fetch,
            reference_ca=reference_ca,
        ))
    return out


def qc_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pose QC into the numbers worth putting in the summary."""
    rows = list(rows)
    if not rows:
        return {"n_checked": 0, "n_in_box": 0, "fraction_in_box": None,
                "median_centroid_offset_A": None, "n_errors": 0,
                "n_frame_aligned": 0, "n_frame_unaligned": 0,
                "median_receptor_ca_rmsd_A": None}
    checked = [r for r in rows if r.get("ligand_in_box") is not None]
    offsets = [r["centroid_offset_A"] for r in rows
               if isinstance(r.get("centroid_offset_A"), (int, float))
               and not math.isnan(r["centroid_offset_A"])]
    rmsds = [r["receptor_ca_rmsd_A"] for r in rows
             if isinstance(r.get("receptor_ca_rmsd_A"), (int, float))
             and not math.isnan(r["receptor_ca_rmsd_A"])]
    n_in = sum(1 for r in checked if r["ligand_in_box"])
    return {
        "n_checked": len(checked),
        "n_in_box": n_in,
        "fraction_in_box": round(n_in / len(checked), 3) if checked else None,
        "median_centroid_offset_A": round(float(np.median(offsets)), 2) if offsets else None,
        "n_errors": sum(1 for r in rows if r.get("pose_error")),
        # How many poses were measured after a superposition. A ligand-only
        # structure needs none (it is already in the receptor's frame) and is
        # counted as measured but not as aligned; a complex whose chain could
        # not be matched is reported separately, because for those the box
        # distance is not merely missing - it would be arbitrary.
        "n_frame_aligned": sum(1 for r in rows if r.get("pose_frame_aligned")),
        "n_frame_unaligned": sum(1 for r in rows if r.get("pose_frame_aligned") is False),
        "median_receptor_ca_rmsd_A": round(float(np.median(rmsds)), 2) if rmsds else None,
    }
