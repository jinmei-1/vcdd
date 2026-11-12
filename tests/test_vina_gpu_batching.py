"""Vina-GPU batch robustness: one unparsable ligand must not sink its batch.

Regression tests for two defects found on the production campaign:

* Open Babel's PDBQT fallback annotates atom types with the Gasteiger charge
  (``CG0``), which the binary rejects with "not a valid AutoDock type".
* The engine docked a whole directory in one invocation and, on a non-zero
  exit, marked *every* ligand in the batch as failed - so a single bad ligand
  silently cost ``gpu_batch_size`` molecules (200 in the campaign, ~3% of the
  library) and the driver never retried.

The fake binary below reproduces the upstream behaviour: it parses every ligand
in the directory first and aborts if any of them is unparsable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vcdd.docking.base import DockParams
from vcdd.docking.engines.vina_gpu import VinaGPUEngine
from vcdd.docking.ligprep import sanitize_pdbqt_types
from vcdd.receptor.box import Box

VARIANT = "quickvina2_gpu"

# A ligand the fake binary refuses to parse, mimicking Vina's type table.
POISON = "CG0"
LIGAND = (
    "ATOM      1  C1  LIG A   1      10.000  10.000  10.000  1.00  0.00     0.000 {t}\n"
    "ATOM      2  O1  LIG A   1      11.000  10.000  10.000  1.00  0.00    -0.300 OA\n"
)
POSE = ("MODEL 1\nREMARK VINA RESULT:     -9.0      0.000      0.000\n"
        "ATOM      1  C1  LIG A   1      10.000  10.000  10.000  1.00  0.00     0.000 C\n"
        "ENDMDL\n")


def _fake_binary(tmp_path: Path) -> Path:
    """A stand-in that aborts on the first ligand containing ``POISON``."""
    script = tmp_path / "fake-vina-gpu"
    script.write_text(f'''#!{__import__("sys").executable}
import pathlib, sys

argv = sys.argv[1:]
lig = pathlib.Path(argv[argv.index("--ligand_directory") + 1])
out = pathlib.Path(argv[argv.index("--output_directory") + 1])
texts = {{p.name: p.read_text() for p in sorted(lig.glob("*.pdbqt"))}}
if any("{POISON}" in t for t in texts.values()):
    print('Parse error: "{POISON}" is not a valid AutoDock type', file=sys.stderr)
    sys.exit(2)
for name in texts:
    (out / name.replace(".pdbqt", "_out.pdbqt")).write_text({POSE!r})
''')
    script.chmod(0o755)
    return script


@pytest.fixture()
def small_box() -> Box:
    return Box(center=(10.0, 10.0, 10.0), size=(16.0, 16.0, 16.0))


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VinaGPUEngine:
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        "vcdd.docking.engines.vina_gpu.shutil.which", lambda name: str(binary))
    return VinaGPUEngine()


def _receptor(tmp_path: Path) -> str:
    path = tmp_path / "receptor.pdbqt"
    path.write_text("ATOM      1  C   REC A   1      10.000  10.000  10.000  1.00  0.00     0.000 C\n")
    return str(path)


def test_a_poisoned_batch_loses_only_the_poisoned_ligand(tmp_path, monkeypatch, small_box):
    engine = _engine(tmp_path, monkeypatch)
    items = [(f"L{i}", LIGAND.format(t=POISON if i == 3 else "C")) for i in range(8)]

    results = engine.dock_batch(items, _receptor(tmp_path), small_box, DockParams(variant=VARIANT))

    assert [r.mol_id for r in results] == [m for m, _ in items], "input order must be preserved"
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error]
    assert len(ok) == 7, "the other seven ligands must still be scored"
    assert [r.mol_id for r in failed] == ["L3"]
    assert all(r.score == pytest.approx(-9.0) for r in ok)
    assert "not a valid AutoDock type" in failed[0].error


def test_isolation_retries_in_halves(tmp_path, monkeypatch, small_box):
    """Bisection, not one call per ligand: recovering 1 poison from 32 ligands
    costs O(log n) invocations (a per-ligand retry would cost 32)."""
    engine = _engine(tmp_path, monkeypatch)
    calls: list[int] = []
    real = engine._invoke_once

    def counting(items, *args, **kwargs):
        calls.append(len(items))
        return real(items, *args, **kwargs)

    monkeypatch.setattr(engine, "_invoke_once", counting)
    items = [(f"L{i}", LIGAND.format(t=POISON if i == 17 else "C")) for i in range(32)]
    results = engine.dock_batch(items, _receptor(tmp_path), small_box, DockParams(variant=VARIANT))

    assert calls[0] == 32, "the full batch is attempted first"
    assert calls[1] == 16, "the clean half is re-docked in one call, not ligand by ligand"
    assert len(calls) <= 14, f"expected O(log n) invocations, got {calls}"
    assert sum(1 for r in results if r.error is None) == 31, "31 healthy ligands must be scored"
    assert [r.mol_id for r in results if r.error] == ["L17"]


def test_a_clean_batch_is_docked_in_one_invocation(tmp_path, monkeypatch, small_box):
    engine = _engine(tmp_path, monkeypatch)
    calls: list[int] = []
    real = engine._invoke_once
    monkeypatch.setattr(engine, "_invoke_once",
                        lambda items, *a, **k: (calls.append(len(items)), real(items, *a, **k))[1])
    items = [(f"L{i}", LIGAND.format(t="C")) for i in range(5)]
    results = engine.dock_batch(items, _receptor(tmp_path), small_box, DockParams(variant=VARIANT))

    assert calls == [5] and all(r.error is None for r in results)


def test_empty_ligand_is_rejected_without_invoking_the_binary(tmp_path, monkeypatch, small_box):
    """A failed preparation used to be written as an empty file, which also
    aborted the batch; it must be reported per ligand instead."""
    engine = _engine(tmp_path, monkeypatch)
    invoked: list[int] = []
    real = engine._invoke_once
    monkeypatch.setattr(engine, "_invoke_once",
                        lambda items, *a, **k: (invoked.append(len(items)), real(items, *a, **k))[1])
    items = [("OK", LIGAND.format(t="C")), ("BAD", ""), ("OK2", LIGAND.format(t="C"))]

    results = engine.dock_batch(items, _receptor(tmp_path), small_box, DockParams(variant=VARIANT))

    assert [r.mol_id for r in results] == ["OK", "BAD", "OK2"]
    assert results[1].error == "empty_ligand_pdbqt" and results[1].score is None
    assert invoked == [2], "the empty ligand must not reach the binary"


def test_binary_missing_reports_every_ligand(tmp_path, monkeypatch, small_box):
    monkeypatch.setattr("vcdd.docking.engines.vina_gpu.shutil.which", lambda name: None)
    results = VinaGPUEngine().dock_batch(
        [("A", LIGAND.format(t="C")), ("B", LIGAND.format(t="C"))],
        _receptor(tmp_path), small_box, DockParams(variant=VARIANT))
    assert len(results) == 2 and all("binary_not_found" in r.error for r in results)


# ------------------------------------------------------- ligand type sanitising
def test_charge_annotated_types_are_normalised():
    """A Gasteiger charge suffix is stripped, not kept: Vina wants ``OA``."""
    for raw, want in (("OA-", "OA"), ("CG+", "C"), ("N1", "N"), ("CG0", None)):
        text, err = sanitize_pdbqt_types(LIGAND.format(t=raw))
        if want is None:
            continue
        assert err is None, (raw, err)
        assert text.splitlines()[0].endswith(f" {want}"), (raw, text.splitlines()[0][-4:])


def test_standard_types_survive_untouched():
    text, err = sanitize_pdbqt_types(LIGAND.format(t="A"))
    assert err is None and text.splitlines()[0].endswith(" A")
    assert text.splitlines()[1].endswith(" OA")


def test_halogen_types_keep_vina_case():
    """Vina's type table is case-sensitive: ``Cl``, not ``CL``."""
    text, err = sanitize_pdbqt_types(LIGAND.format(t="CL"))
    assert err is None and text.splitlines()[0].endswith(" Cl")


def test_unknown_type_rejects_the_ligand_instead_of_guessing():
    text, err = sanitize_pdbqt_types(LIGAND.format(t="Zz"))
    assert text is None and err == "unmappable_atom_type:'Zz'"


def test_metals_are_rejected_not_read_as_carbon():
    """``CA`` would otherwise be taken for carbon by a prefix match."""
    assert sanitize_pdbqt_types(LIGAND.format(t="CA"))[1] == "unmappable_atom_type:'CA'"


# ------------------------------------------------- macrocycles and glue atoms
MACROCYCLE = "O=C(C1=CC=C2CC1=CC=CC=CC=N2)N1CCCC1"   # meeko opens this ring


def test_macrocycle_ligands_are_docked_with_rigid_rings_not_dropped():
    """Meeko's macrocycle handling emits glue pseudo-atoms (G0/CG0) that Vina
    cannot parse; the ligand must be recovered with rigid rings, not lost."""
    from vcdd.docking.ligprep import mol_to_pdbqt
    from vcdd.docking.ligprep import embed_conformers

    mol = embed_conformers(MACROCYCLE, n_conformers=1, minimize=False, seed=42)
    assert mol is not None
    pdbqt, err = mol_to_pdbqt(mol, conf_id=0)

    assert pdbqt is not None, f"macrocycle ligand was dropped: {err}"
    types = {line[77:].strip() for line in pdbqt.splitlines()
             if line.startswith(("ATOM", "HETATM"))}
    assert not any("G" in t for t in types), f"glue atom types survived: {types}"


def test_glue_types_are_reported_as_glue_not_as_carbon():
    """``CG0`` is meeko's glue anchor, not a carbon: reading it as ``C`` would
    silently keep a pseudo-atom in the ligand."""
    for raw in ("CG0", "G0", "G1", "CG3"):
        text, err = sanitize_pdbqt_types(LIGAND.format(t=raw))
        assert text is None and err == f"glue_atom_type:'{raw}'", (raw, err)


def test_a_glue_typed_ligand_is_still_rejected_by_the_sanitiser():
    """The safety net stays: if a glue type ever reaches the engine it must be
    reported per ligand, never written into a batch."""
    line = ("ATOM      1  C   LIG A   1      10.000  10.000  10.000  1.00  0.00     0.000 Zz9")
    text, err = sanitize_pdbqt_types(line + "\n")
    assert text is None and err == "unmappable_atom_type:'Zz9'"
