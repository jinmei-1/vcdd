"""Docking box definition and geometry helpers.

A box that is too small truncates poses; a box that is too large wastes
search effort on empty space and degrades ranking quality. The default
behaviour derives a near-cubic box from the co-crystal ligand's bounding box
plus padding, then clamps each edge into ``[min_edge, max_edge]`` and warns when
the clamp actually bites - a silently clamped box is a silently wrong result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from vcdd.runtime.paths import atomic_write_text


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source: str = "explicit"
    warnings: tuple[str, ...] = ()

    @property
    def volume(self) -> float:
        return float(self.size[0] * self.size[1] * self.size[2])

    def contains(self, point: Sequence[float], tol: float = 0.0) -> bool:
        return all(
            abs(point[i] - self.center[i]) <= self.size[i] / 2.0 + tol for i in range(3)
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center"] = list(self.center)
        d["size"] = list(self.size)
        d["warnings"] = list(self.warnings)
        d["volume"] = self.volume
        return d

    def save(self, path: str | Path) -> Path:
        return atomic_write_text(path, json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Box":
        data = json.loads(Path(path).read_text())
        return cls(
            center=tuple(data["center"]),  # type: ignore[arg-type]
            size=tuple(data["size"]),  # type: ignore[arg-type]
            source=data.get("source", "loaded"),
            warnings=tuple(data.get("warnings", [])),
        )


def coords_from_pdb_block(block: str, resname: str | None = None) -> list[tuple[float, float, float]]:
    """Extract HETATM/ATOM coordinates from a PDB block, optionally filtered."""
    out: list[tuple[float, float, float]] = []
    for line in block.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if resname is not None and line[17:20].strip() != resname:
            continue
        if line[76:78].strip().upper() == "H":
            continue
        try:
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return out


def coords_from_file(path: str | Path, resname: str | None = None) -> list[tuple[float, float, float]]:
    """Heavy-atom coordinates from a PDB / SDF / MOL2 file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".pdb", ".ent"):
        return coords_from_pdb_block(p.read_text(), resname)

    from rdkit import Chem

    mol = None
    if suffix in (".sdf", ".mol"):
        supplier = Chem.SDMolSupplier(str(p), sanitize=False, removeHs=False)
        for m in supplier:
            if m is not None:
                mol = m
                break
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(p), sanitize=False, removeHs=False)
    elif suffix == ".pdbqt":
        return coords_from_pdb_block(p.read_text(), resname)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"no 3D coordinates found in {p}")
    conf = mol.GetConformer()
    return [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]


def box_from_coords(
    coords: Sequence[Sequence[float]],
    padding: float = 4.0,
    min_edge: float = 16.0,
    max_edge: float = 22.0,
    cubic: bool = True,
) -> Box:
    """Derive a docking box from a set of reference coordinates."""
    if not coords:
        raise ValueError("cannot derive a box from zero coordinates")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    center = (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    )
    extents = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    raw = [e + 2.0 * padding for e in extents]
    if cubic:
        edge = max(raw)
        raw = [edge, edge, edge]

    warnings: list[str] = []
    size = []
    for axis, value in zip("xyz", raw):
        clamped = min(max(value, min_edge), max_edge)
        if abs(clamped - value) > 1e-6:
            direction = "up to" if clamped > value else "down to"
            warnings.append(
                f"box edge {axis} clamped {direction} {clamped:.1f} A from {value:.1f} A "
                f"(ligand extent {extents['xyz'.index(axis)]:.1f} A + 2x{padding:.1f} A padding)"
            )
        size.append(round(clamped, 3))

    # A clamp that cuts into the ligand itself is a hard geometry problem.
    for axis_i, axis in enumerate("xyz"):
        if size[axis_i] < extents[axis_i]:
            warnings.append(
                f"BOX TOO SMALL on {axis}: edge {size[axis_i]:.1f} A < ligand extent "
                f"{extents[axis_i]:.1f} A. Raise receptor.box.max_edge."
            )

    return Box(
        center=(round(center[0], 3), round(center[1], 3), round(center[2], 3)),
        size=(size[0], size[1], size[2]),
        source="reference_ligand",
        warnings=tuple(warnings),
    )


def resolve_box(receptor_cfg: Any, reference_path: str | Path | None = None) -> Box:
    """Build the box from config: explicit centre/size wins, else the ligand."""
    box_cfg = receptor_cfg.box
    if box_cfg.center is not None and box_cfg.size is not None:
        return Box(
            center=tuple(float(v) for v in box_cfg.center),  # type: ignore[arg-type]
            size=tuple(float(v) for v in box_cfg.size),  # type: ignore[arg-type]
            source="explicit",
        )

    ref = reference_path or receptor_cfg.reference_ligand_path
    if ref is not None:
        coords = coords_from_file(ref)
    elif receptor_cfg.reference_ligand and receptor_cfg.path:
        coords = coords_from_file(receptor_cfg.path, resname=receptor_cfg.reference_ligand)
        if not coords:
            raise ValueError(
                f"ligand residue {receptor_cfg.reference_ligand!r} not found in {receptor_cfg.path}"
            )
    else:
        raise ValueError(
            "cannot determine docking box: set receptor.box.center + receptor.box.size, "
            "or receptor.reference_ligand / receptor.reference_ligand_path"
        )
    return box_from_coords(
        coords,
        padding=box_cfg.padding,
        min_edge=box_cfg.min_edge,
        max_edge=box_cfg.max_edge,
    )


def rmsd(coords_a: Sequence[Sequence[float]], coords_b: Sequence[Sequence[float]]) -> float:
    """Plain (non-superposed, no symmetry) coordinate RMSD."""
    if len(coords_a) != len(coords_b):
        raise ValueError(f"coordinate count mismatch: {len(coords_a)} vs {len(coords_b)}")
    if not coords_a:
        raise ValueError("empty coordinate sets")
    total = 0.0
    for a, b in zip(coords_a, coords_b):
        total += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return float((total / len(coords_a)) ** 0.5)
