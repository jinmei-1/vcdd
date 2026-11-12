#!/usr/bin/env python
"""A scripted ``boltz`` CLI for the local-backend contract tests.

The point is the same as :mod:`fake_boltz`: exercise vcdd's *real* calling
code - the argv it builds, the subprocess contract, the output files it parses
- without a model. This script is executed by :func:`vcdd.refine.boltz.local.
run_predict` through the exact ``subprocess.run`` path, so a wrong flag name,
a wrong argument shape or a wrong output-file convention fails here rather
than after the user has queued a real prediction on a GPU.

Behaviour is controlled by environment variables so the tests can script
scenarios without editing this file:

``FAKE_BOLTZ_CENTER``  "x,y,z" - where the predicted ligand is placed (default
                       0,0,0). The test fixture uses the small_box centre so
                       ``ligand_in_box`` is exercised for both outcomes.
``FAKE_BOLTZ_RECEPTOR`` path to the receptor file the caller docked against.
                       When set, the fake complex contains that receptor's
                       C-alpha trace - the way a co-folding model returns the
                       whole complex rather than a pose - and both the trace and
                       the ligand are written in a *different* frame (rotated and
                       translated). Pose QC can therefore only put the ligand in
                       the box after superposing the trace back onto the
                       receptor; without that step every offset is hundreds of
                       Angstroms.
``FAKE_BOLTZ_FAIL``    non-empty - exit 2 with an error on stderr (a predict
                       that crashed).
``FAKE_BOLTZ_TIMEOUT`` non-empty - sleep past the caller's timeout.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path


def _model_frame(shift: tuple[float, float, float]):
    """A rotation + translation standing in for the model's own coordinate frame."""
    angle = math.radians(30.0)
    cos, sin = math.cos(angle), math.sin(angle)

    def apply(x: float, y: float, z: float) -> tuple[float, float, float]:
        xr, yr = cos * x - sin * y, sin * x + cos * y
        return (xr + shift[0], yr + shift[1], z + shift[2])

    return apply


def _receptor_ca_trace(path: str) -> list[tuple[float, float, float]]:
    """C-alpha coordinates of a PDB/PDBQT receptor, in *sequence* order.

    Residues are keyed by ``(resSeq, iCode)`` and sorted, because a prepared
    PDBQT is not necessarily written in residue order (the real one for this
    campaign is not) while a predicted model always returns the chain in
    sequence order. Reading the file order here would pair the wrong residues
    during superposition and turn a rigid transform into a ~14 A mismatch.
    """
    per_residue: dict[tuple[int, str], tuple[float, float, float]] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[12:16].strip().upper() != "CA":
            continue
        try:
            key = (int(line[22:26]), line[26].strip())
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        per_residue[key] = xyz
    return [per_residue[k] for k in sorted(per_residue)]

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--version":
        print("boltz 2.1.1")
        sys.exit(0)

    if not args or args[0] != "predict":
        print(f"fake_boltz_cli: unexpected argv {args!r}", file=sys.stderr)
        sys.exit(2)

    # ---- parse the predict invocation -----------------------------------
    argv = args[1:]
    out_dir = None
    while argv:
        tok = argv.pop(0)
        if tok == "--out_dir":
            out_dir = argv.pop(0)
        elif tok == "--use_msa_server":
            pass  # flag without a value
        elif tok in ("--devices", "--diffusion_samples", "--cache",
                     "--msa_server_url", "--msa_server_username",
                     "--msa_server_password") and argv:
            argv.pop(0)  # consumed by real boltz; ignored here
        # --half / --no-half / extra_args are accepted and ignored
    if out_dir is None:
        print("fake_boltz_cli: missing --out_dir", file=sys.stderr)
        sys.exit(2)

    if os.environ.get("FAKE_BOLTZ_FAIL"):
        print("fake_boltz_cli: injected failure", file=sys.stderr)
        sys.exit(2)
    if os.environ.get("FAKE_BOLTZ_TIMEOUT"):
        import time

        time.sleep(3600)

    # ---- read the input YAML ---------------------------------------------
    import yaml

    input_yaml = Path(args[1])  # argv[0] is the "predict" subcommand
    doc = yaml.safe_load(input_yaml.read_text())
    seqs = doc["sequences"]
    ligand = next(s for s in seqs if "ligand" in s)
    smiles = ligand["ligand"]["smiles"]

    # ---- deterministic metrics from the SMILES ---------------------------
    digest = hashlib.sha256(str(smiles).encode()).digest()
    bc = 0.5 + (int.from_bytes(digest[:4], "big") / 2**32) * 0.49
    # Mirrors the boltz-2.2 writer (data/write/writer.py): the headline is
    # `confidence_score`, with the interface metrics as siblings.
    conf = {
        "confidence_score": round(bc, 4),
        "ptm": 0.72,
        "iptm": round(bc - 0.1, 4),
        "ligand_iptm": round(bc, 4),
        "protein_iptm": round(bc, 4),
        "complex_plddt": 0.71,
        "complex_iplddt": round(bc - 0.05, 4),
        "complex_pde": 0.4,
        "complex_ipde": 0.5,
    }

    # ---- predicted full-complex CIF at the requested centre --------------
    # Ligand chains are HETATM (and placed at FAKE_BOLTZ_CENTER); the protein is
    # an ATOM chain. Both are written in a model frame that is rotated and
    # translated away from the receptor frame, so pose QC only passes if it (a)
    # filters HETATM - the local run must never measure protein atoms - and (b)
    # superposes the predicted trace back onto the receptor before testing the
    # box. With FAKE_BOLTZ_RECEPTOR the protein chain is the real trace, so the
    # superposition is a genuine (recoverable) rigid transform; without it the
    # chain is two dummy CAs, which is the unalignable case.
    centre = [0.0, 0.0, 0.0]
    if os.environ.get("FAKE_BOLTZ_CENTER"):
        centre = [float(v) for v in os.environ["FAKE_BOLTZ_CENTER"].split(",")[:3]]
    cx, cy, cz = centre
    shift = (999.0, -123.0, 77.0)
    if os.environ.get("FAKE_BOLTZ_SHIFT"):
        shift = tuple(float(v) for v in os.environ["FAKE_BOLTZ_SHIFT"].split(",")[:3])
    to_model_frame = _model_frame(shift)
    receptor_path = os.environ.get("FAKE_BOLTZ_RECEPTOR")
    if receptor_path:
        protein_xyz = _receptor_ca_trace(receptor_path)
    else:
        protein_xyz = [(cx, cy, cz), (cx, cy + 1.0, cz)]
    heads = [
        "data_complex", "loop_",
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.auth_asym_id",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
    ]
    prot = "\n".join(
        f"ATOM  {i + 1} C CA ALA A A {x:.3f} {y:.3f} {z:.3f}"
        for i, (x, y, z) in enumerate(to_model_frame(*xyz) for xyz in protein_xyz)
    )
    lig = "\n".join(
        f"HETATM {i + 1} C C{i + 1} LIG L L {x:.3f} {y:.3f} {z:.3f}"
        for i, (x, y, z) in enumerate(
            to_model_frame(*xyz)
            for xyz in [(cx, cy, cz), (cx + 1.0, cy, cz), (cx, cy + 1.0, cz)]
        )
    )
    cif = "\n".join(heads + [prot, lig]) + "\n"

    stem = input_yaml.stem  # boltz uses the YAML stem as record.id
    results = Path(out_dir) / f"boltz_results_{stem}" / "predictions" / stem
    results.mkdir(parents=True, exist_ok=True)
    (results / f"confidence_{stem}_model_0.json").write_text(
        __import__("json").dumps(conf)
    )
    # boltz writes an affinity JSON only when the input document asked for the
    # affinity head (a ``properties`` block naming the ligand as binder).
    if any("affinity" in (prop or {}) for prop in (doc.get("properties") or [])):
        binder = next(
            prop["affinity"]["binder"] for prop in doc["properties"] if "affinity" in prop
        )
        affinity = {
            "affinity_pred_value": round(1.0 - bc, 4),      # lower = stronger
            "affinity_probability_binary": round(bc, 4),
            "affinity_pred_value1": round(1.0 - bc, 4),
            "affinity_probability_binary1": round(bc, 4),
            "affinity_pred_value2": round(1.0 - bc, 4),
            "affinity_probability_binary2": round(bc, 4),
            "binder": binder,
        }
        # real boltz writes ``affinity_<record.id>.json`` (no _model_ suffix)
        (results / f"affinity_{stem}.json").write_text(
            __import__("json").dumps(affinity)
        )
    (results / f"{stem}_model_0.cif").write_text(cif)
    # record which GPU the caller pinned this process to, so the round-robin
    # over local_boltz.devices is observable in tests
    (Path(out_dir) / "cuda_visible.txt").write_text(
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    # Append-only invocation log *outside* out_dir, so it survives the stale
    # output wipe run_predict performs when the input changed: this is how a
    # test tells "boltz really ran again" from "boltz skipped and vcdd read the
    # old files".
    with (Path(out_dir).parent / "invocations.log").open("a") as fh:
        fh.write(f"{stem}\t{smiles}\n")
    sys.exit(0)
