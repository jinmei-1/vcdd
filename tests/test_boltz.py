"""Stage-2 Boltz layer: credentials, target assembly, both backends, pose QC.

Every test that talks to the API talks to it through the *real* ``boltz-api``
SDK, wired to :mod:`fake_boltz` with an ``httpx.MockTransport``. That is
deliberate. The bug 0.2.0 exists to fix is a 0.1.0 engine wrapper that passed
flags the tool did not accept - it had unit tests, but they mocked the call
instead of making it, so nothing ever checked the contract. Driving the SDK
means a wrong argument name, a wrong payload shape or a wrong response field
fails here rather than after the user has paid for a screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fake_boltz import SCREEN, FakeBoltz, make_fake_client, pose_in_box, pose_outside_box
from vcdd.receptor.box import Box
from vcdd.refine.boltz import client as bclient
from vcdd.refine.boltz import poseqc as bposeqc
from vcdd.refine.boltz import sab as bsab
from vcdd.refine.boltz import screen as bscreen
from vcdd.refine.boltz import target as btarget

MOL_IDS = ["M000", "M001", "M002", "M003", "M004"]
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCO"]

TARGET = {
    "entities": [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}],
    "type": "no_template",
}


def _molecules(n: int = 5):
    return bscreen.make_molecules(MOL_IDS[:n], SMILES[:n])


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #


def test_credentials_report_absence_without_raising(monkeypatch):
    monkeypatch.delenv("BOLTZ_API_KEY", raising=False)
    creds = bclient.resolve_credentials()
    assert creds.present is False
    assert creds.suffix is None
    with pytest.raises(bclient.BoltzUnavailable) as excinfo:
        creds.require()
    assert "BOLTZ_API_KEY" in str(excinfo.value)


@pytest.mark.parametrize("key,livemode", [
    ("sk_bc_abcdefghijklmnop_test_0123456789", False),
    ("sk_test_abcdefghijklmnopqrstuvwx", False),
    ("sk_bc_abcdefghijklmnopqrstuvwxyz0123", True),
])
def test_test_mode_keys_are_detected(monkeypatch, key, livemode):
    monkeypatch.setenv("BOLTZ_API_KEY", key)
    creds = bclient.resolve_credentials()
    assert creds.present is True
    assert creds.livemode is livemode


def test_credentials_never_expose_the_key(monkeypatch):
    secret = "sk_bc_supersecretvalue_test_zzzz9999"
    monkeypatch.setenv("BOLTZ_API_KEY", secret)
    creds = bclient.resolve_credentials()
    blob = json.dumps(creds.as_dict()) + repr(creds)
    assert "supersecretvalue" not in blob
    assert creds.suffix == "9999"          # enough to tell two keys apart
    assert len(creds.suffix) == 4


# --------------------------------------------------------------------------- #
# target assembly
# --------------------------------------------------------------------------- #


def test_receptor_chains_are_parsed_into_one_letter_sequences(prepared_receptor: Path):
    chains = btarget.parse_receptor_chains(prepared_receptor)
    assert chains, "no chains parsed from the prepared receptor"
    chain = chains[sorted(chains)[0]]
    assert chain.sequence and chain.sequence.isupper()
    assert len(chain.sequence) == len(chain.residue_keys) == len(chain.coords)
    assert set(chain.sequence) <= set("ACDEFGHIKLMNPQRSTVWYX")


def test_residue_order_in_the_file_does_not_change_the_sequence(tmp_path: Path):
    """A reordered receptor must yield the real chain, not a permutation.

    ADFR's ``prepare_receptor`` writes PDBQTs whose residues appear in reverse
    order; reading such a file sequentially returns a scrambled sequence with
    the right composition.  This is the regression test for that failure: the
    shuffled and in-order files must give byte-identical sequences, and the
    provenance flag must record that the input was reordered.
    """
    residues = [
        ("ALA", 10, [("N", 0.0, 0.0, 0.0), ("CA", 1.0, 0.0, 0.0)]),
        ("GLY", 11, [("N", 2.0, 0.0, 0.0), ("CA", 3.0, 0.0, 0.0)]),
        ("SER", 12, [("N", 4.0, 0.0, 0.0), ("CA", 5.0, 0.0, 0.0)]),
    ]

    def write(path: Path, order) -> Path:
        lines = []
        serial = 0
        for resname, resseq, atoms in order:
            for atom_name, x, y, z in atoms:
                serial += 1
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} A{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     0.000 C "
                )
        path.write_text("\n".join(lines) + "\n")
        return path

    in_order = btarget.parse_receptor_chains(write(tmp_path / "in_order.pdbqt", residues))
    shuffled = btarget.parse_receptor_chains(
        write(tmp_path / "shuffled.pdbqt", list(reversed(residues)))
    )

    assert in_order["A"].sequence == shuffled["A"].sequence == "AGS"
    assert in_order["A"].residue_keys == shuffled["A"].residue_keys == [(10, ""), (11, ""), (12, "")]
    assert in_order["A"].coords[0][0] == shuffled["A"].coords[0][0]
    assert in_order["A"].file_order_matches_sequence is True
    assert shuffled["A"].file_order_matches_sequence is False


def test_pocket_residues_are_zero_indexed_and_inside_the_box(prepared_receptor, small_box):
    chains = btarget.parse_receptor_chains(prepared_receptor)
    pockets = btarget.pocket_residues_from_box(chains, small_box)
    assert pockets, "the fixture box should select some residues"
    for chain_id, positions in pockets.items():
        assert positions == sorted(set(positions))
        assert min(positions) >= 0
        assert max(positions) < len(chains[chain_id].sequence)
        # every selected residue really has an atom in the box
        for pos in positions:
            assert any(small_box.contains(a) for a in chains[chain_id].coords[pos])


def test_pocket_is_a_strict_subset_of_the_receptor(prepared_receptor, small_box):
    """A pocket that is the whole protein is not a pocket; it is a bug."""
    chains = btarget.parse_receptor_chains(prepared_receptor)
    pockets = btarget.pocket_residues_from_box(chains, small_box)
    total = sum(len(c.sequence) for c in chains.values())
    selected = sum(len(v) for v in pockets.values())
    assert 0 < selected < total


def test_build_target_carries_the_box_into_the_payload(prepared_receptor, small_box):
    target, prov = btarget.build_target(prepared_receptor, box=small_box, pocket_from_box=True)
    assert target["type"] == "no_template"
    assert target["entities"] and target["entities"][0]["type"] == "protein"
    assert set(target["pocket_residues"]) == set(prov["pocket_sizes"])
    assert all(len(v) == prov["pocket_sizes"][k] for k, v in target["pocket_residues"].items())
    assert all(n > 0 for n in prov["pocket_sizes"].values())
    assert prov["chains"][0]["length"] > 0


def test_build_target_without_a_box_sends_no_pocket(prepared_receptor):
    target, prov = btarget.build_target(prepared_receptor, box=None, pocket_from_box=False)
    assert "pocket_residues" not in target
    assert not prov.get("pocket_sizes")


def test_build_target_rejects_an_unparseable_receptor(tmp_path: Path):
    stub = tmp_path / "empty.pdbqt"
    stub.write_text("REMARK nothing here\n")
    with pytest.raises(ValueError):
        btarget.build_target(stub, box=None, pocket_from_box=False)


# --------------------------------------------------------------------------- #
# library_screen backend
# --------------------------------------------------------------------------- #


def test_screen_round_trip_through_the_real_sdk(fake_boltz, boltz_client):
    mols = _molecules()
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=mols, chunk_index=0)
    assert run.status == "running"
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=0, sleep=lambda _: None)
    assert run.status == "succeeded"
    assert run.engine == "boltzmol" and run.engine_version == "1.0"

    rows = bscreen.fetch_results(boltz_client, run.run_id)
    assert [r["mol_id"] for r in rows] == MOL_IDS          # pagination reassembled in order
    assert all(0.0 <= r["binding_confidence"] <= 1.0 for r in rows)
    assert all(r["url_ligand_structure"] for r in rows)
    # the SDK really paginated: page_size=2 over 5 results
    result_calls = [c for c in fake_boltz.calls if c[1].endswith("/results")]
    assert len(result_calls) >= 3


def test_polling_actually_polls(fake_boltz, boltz_client):
    """A poll loop that exits on the first response would pass a naive mock."""
    fake_boltz.running_polls = 3
    naps: list[float] = []
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=_molecules(2), chunk_index=0)
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=7.0,
                                      sleep=naps.append)
    assert run.status == "succeeded"
    assert naps == [7.0, 7.0, 7.0]


def test_poll_timeout_is_recorded_not_raised(fake_boltz, boltz_client):
    fake_boltz.running_polls = 10_000
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=_molecules(2), chunk_index=0)
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=0,
                                      timeout_s=-1.0, sleep=lambda _: None)
    assert run.status == "running"
    assert run.error["code"] == "vcdd_poll_timeout"


def test_idempotency_key_is_stable_for_the_same_chunk_and_differs_otherwise():
    a = bscreen.idempotency_key(TARGET, _molecules(3))
    b = bscreen.idempotency_key(TARGET, _molecules(3))
    c = bscreen.idempotency_key(TARGET, _molecules(4))
    other = dict(TARGET, entities=[{"chain_ids": ["A"], "type": "protein", "value": "MKV"}])
    d = bscreen.idempotency_key(other, _molecules(3))
    assert a == b
    assert a != c and a != d
    assert a.startswith("vcdd-")


def test_resubmitting_the_same_chunk_reattaches_instead_of_paying_twice(fake_boltz, boltz_client):
    mols = _molecules(3)
    first = bscreen.start_chunk(boltz_client, target=TARGET, molecules=mols, chunk_index=0)
    second = bscreen.start_chunk(boltz_client, target=TARGET, molecules=mols, chunk_index=0)
    assert first.run_id == second.run_id
    assert len(fake_boltz.runs) == 1


def test_filtered_molecules_are_reported_not_silently_dropped(fake_boltz, boltz_client):
    """The API omits SMARTS-filtered molecules from results entirely."""
    fake_boltz.filtered_ids = {"M002"}
    mols = _molecules()
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=mols, chunk_index=0)
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=0, sleep=lambda _: None)
    rows = bscreen.fetch_results(boltz_client, run.run_id)
    returned = {r["mol_id"] for r in rows}
    assert "M002" not in returned

    missing = bscreen.reconcile_missing(mols, returned, run)
    assert [m["mol_id"] for m in missing] == ["M002"]
    assert "filter" in missing[0]["reason"]


def test_failed_molecules_come_back_without_metrics(fake_boltz, boltz_client):
    fake_boltz.failed_ids = {"M001"}
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=_molecules(), chunk_index=0)
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=0, sleep=lambda _: None)
    rows = {r["mol_id"]: r for r in bscreen.fetch_results(boltz_client, run.run_id)}
    assert rows["M001"]["binding_confidence"] is None
    assert rows["M000"]["binding_confidence"] is not None


def test_a_stopped_run_keeps_the_partial_results(fake_boltz, boltz_client):
    fake_boltz.terminal_status = "stopped"
    fake_boltz.partial_ids = {"M000", "M001"}
    run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=_molecules(), chunk_index=0)
    run = bscreen.poll_until_terminal(boltz_client, run, poll_interval_s=0, sleep=lambda _: None)
    assert run.status == "stopped"
    rows = bscreen.fetch_results(boltz_client, run.run_id)
    assert {r["mol_id"] for r in rows} == {"M000", "M001"}


# ------------------------------------------------------------------- spending
def test_cost_estimate_comes_from_the_api(fake_boltz, boltz_client):
    est = bscreen.estimate_cost(boltz_client, TARGET, _molecules())
    assert float(est["estimated_cost_usd"]) == pytest.approx(5 * 0.025)
    assert ("POST", f"{SCREEN}/estimate-cost") in fake_boltz.calls


def test_spend_gate_blocks_above_the_ceiling():
    estimates = [{"estimated_cost_usd": "12.50"}, {"estimated_cost_usd": "12.50"}]
    assert bscreen.check_spend(estimates, max_spend_usd=30.0) == pytest.approx(25.0)
    with pytest.raises(bscreen.SpendGateExceeded) as excinfo:
        bscreen.check_spend(estimates, max_spend_usd=20.0)
    assert "max_spend_usd" in str(excinfo.value)
    # the override is explicit, and it returns the same total
    assert bscreen.check_spend(estimates, max_spend_usd=20.0, allow=True) == pytest.approx(25.0)


def test_an_unknown_estimate_blocks_rather_than_counting_as_zero():
    """Treating a missing price as $0 is how an unbounded bill happens."""
    with pytest.raises(bscreen.SpendGateExceeded) as excinfo:
        bscreen.check_spend([{"estimated_cost_usd": None}], max_spend_usd=1000.0)
    assert "no cost estimate" in str(excinfo.value)
    assert bscreen.check_spend([{"estimated_cost_usd": None}], 1000.0, allow=True) == 0.0


def test_estimate_failure_surfaces_as_an_exception(boltz_client, fake_boltz):
    fake_boltz.estimate_error = True
    with pytest.raises(Exception):
        bscreen.estimate_cost(boltz_client, TARGET, _molecules(2))


# --------------------------------------------------------------- persistence
def test_results_jsonl_is_rewritten_and_deduped(tmp_path: Path):
    """S3-backed run dirs have no POSIX append, so this rewrites the whole file."""
    path = tmp_path / "results.jsonl"
    rows = [{"boltz_result_id": "r1", "mol_id": "M000"},
            {"boltz_result_id": "r2", "mol_id": "M001"}]
    assert bscreen.append_results_jsonl(path, rows) == 2
    # a resumed chunk re-submits rows it already wrote
    assert bscreen.append_results_jsonl(path, rows + [{"boltz_result_id": "r3",
                                                       "mol_id": "M002"}]) == 3
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert [x["boltz_result_id"] for x in lines] == ["r1", "r2", "r3"]


def test_runs_round_trip_through_json(tmp_path: Path):
    run = bscreen.ScreenRun(run_id="ls_1", chunk_index=0, n_molecules=3,
                            idempotency_key="vcdd-abc", estimated_cost_usd="0.08",
                            status="succeeded", mol_ids=["M000", "M001", "M002"])
    path = tmp_path / "runs.json"
    bscreen.save_runs(path, [run])
    back = bscreen.load_runs(path)
    assert len(back) == 1
    assert back[0].run_id == "ls_1"
    assert back[0].mol_ids == ["M000", "M001", "M002"]


# --------------------------------------------------------------------------- #
# structure_and_binding backend
# --------------------------------------------------------------------------- #


def test_binder_size_limit_is_enforced_before_submission():
    assert bsab.heavy_atom_count("CCO") == 3
    assert bsab.heavy_atom_count("not a smiles") is None
    big = "C" * (bsab.MAX_BINDER_HEAVY_ATOMS + 20)
    assert bsab.heavy_atom_count(big) > bsab.MAX_BINDER_HEAVY_ATOMS


def test_ligand_chain_id_never_collides_with_the_receptor():
    assert bsab.pick_ligand_chain_id(["A", "B"]) == "L"
    assert bsab.pick_ligand_chain_id(["A", "L"]) != "L"
    # a 26-chain receptor is a hard stop, not a silent collision
    with pytest.raises(ValueError):
        bsab.pick_ligand_chain_id(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def test_sab_input_puts_the_docking_box_in_a_pocket_constraint():
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}]
    payload = bsab.build_input(entities, "CCO", pocket_residues={"A": [2, 3, 4]},
                               num_samples=2)
    assert payload["num_samples"] == 2
    assert payload["binding"] == {"type": "ligand_protein_binding", "binder_chain_id": "L"}
    ligand = [e for e in payload["entities"] if e["type"] == "ligand_smiles"]
    assert ligand and ligand[0]["value"] == "CCO"
    constraint = payload["constraints"][0]
    assert constraint["type"] == "pocket"
    assert constraint["contact_residues"] == {"A": [2, 3, 4]}


@pytest.mark.parametrize("bad", [0, 11])
def test_sab_rejects_out_of_range_sample_counts(bad):
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKV"}]
    with pytest.raises(ValueError):
        bsab.build_input(entities, "CCO", num_samples=bad)


def test_sab_refuses_a_ligand_chain_that_collides():
    entities = [{"chain_ids": ["L"], "type": "protein", "value": "MKV"}]
    with pytest.raises(ValueError):
        bsab.build_input(entities, "CCO", ligand_chain_id="L")


def test_sab_screens_molecules_and_pre_rejects_the_impossible(fake_boltz, boltz_client):
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}]
    mol_ids = ["M000", "M001", "BAD", "HUGE"]
    smiles = ["CCO", "c1ccccc1", "this is not a smiles", "C" * 150]
    rows, runs, rejected = bsab.screen_molecules(
        boltz_client, entities, mol_ids, smiles,
        pocket_residues={"A": [2, 3]}, poll_interval_s=0, sleep=lambda _: None,
        max_concurrency=2,
    )
    assert {r["mol_id"] for r in rows} == {"M000", "M001"}
    assert len(runs) == 2
    reasons = {r["mol_id"]: r["reason"] for r in rejected}
    assert reasons["BAD"] == "invalid_smiles"
    assert "binder_too_large" in reasons["HUGE"]
    assert all(r["run_status"] == "not_submitted" for r in rejected)
    # nothing impossible was ever sent
    assert len(fake_boltz.started_payloads) == 2


def test_sab_rows_have_the_same_shape_as_screen_rows(fake_boltz, boltz_client):
    """Both backends must feed the same downstream table."""
    entities = [{"chain_ids": ["A"], "type": "protein", "value": "MKVLAAGIVGAWQ"}]
    rows, _, _ = bsab.screen_molecules(boltz_client, entities, ["M000"], ["CCO"],
                                       poll_interval_s=0, sleep=lambda _: None)
    screen_run = bscreen.start_chunk(boltz_client, target=TARGET, molecules=_molecules(1),
                                     chunk_index=0)
    screen_run = bscreen.poll_until_terminal(boltz_client, screen_run, poll_interval_s=0,
                                             sleep=lambda _: None)
    screen_rows = bscreen.fetch_results(boltz_client, screen_run.run_id)
    shared = {"mol_id", "binding_confidence", "optimization_score", "boltz_run_id",
              "boltz_result_id", "url_ligand_structure"}
    assert shared <= set(rows[0])
    assert shared <= set(screen_rows[0])


# --------------------------------------------------------------------------- #
# pose QC
# --------------------------------------------------------------------------- #


def test_box_check_is_strict_about_every_heavy_atom(small_box):
    cx, cy, cz = small_box.center
    inside = [(cx, cy, cz), (cx + 1, cy, cz)]
    straddling = [(cx, cy, cz), (cx + 50, cy, cz)]
    assert bposeqc.box_check(inside, small_box)["ligand_in_box"] is True
    out = bposeqc.box_check(straddling, small_box)
    assert out["ligand_in_box"] is False
    assert out["frac_atoms_in_box"] == pytest.approx(0.5)
    assert out["max_axis_excess_A"] > 0


def test_cif_and_pdb_and_sdf_all_parse(small_box):
    cif = pose_in_box(small_box.center)
    assert len(bposeqc.parse_structure_atoms(cif)) == 3
    pdb = "\n".join(
        f"HETATM {i:>4}  C   LIG L   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    ) + "\nEND\n"
    assert len(bposeqc.parse_structure_atoms(pdb)) == 2


def test_pose_qc_flags_a_prediction_placed_outside_the_docking_box(tmp_path, small_box):
    """Boltz is free to bind anywhere on the protein; this is the check that catches it."""
    server = FakeBoltz(box_center=small_box.center, outside_box_ids={"M001"})
    rows = [{"mol_id": m, "url_ligand_structure": f"https://files.boltz.test/run/{m}.cif"}
            for m in ("M000", "M001")]
    qc = bposeqc.qc_poses(rows, small_box, tmp_path / "poses", top_k=10, fetch=server.fetch)
    by_id = {r["mol_id"]: r for r in qc}
    assert by_id["M000"]["ligand_in_box"] is True
    assert by_id["M001"]["ligand_in_box"] is False
    assert by_id["M001"]["centroid_offset_A"] > 20
    assert (tmp_path / "poses" / "M000.cif").exists()

    summary = bposeqc.qc_summary(qc)
    assert summary["n_checked"] == 2 and summary["n_in_box"] == 1
    assert summary["fraction_in_box"] == pytest.approx(0.5)


def _complex_cif(protein_ca, ligand_xyz) -> str:
    """Minimal co-folded complex: a protein CA trace plus one HETATM ligand."""
    lines = [
        "data_test", "loop_",
        "_atom_site.group_PDB", "_atom_site.label_atom_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.type_symbol",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
    ]
    for x, y, z in protein_ca:
        lines.append(f"ATOM CA ALA A C {x:.3f} {y:.3f} {z:.3f}")
    for i, (x, y, z) in enumerate(ligand_xyz, start=1):
        lines.append(f"HETATM C{i} LIG L C {x:.3f} {y:.3f} {z:.3f}")
    return "\n".join(lines) + "\n"


def test_cofolded_pose_qc_superposes_onto_the_receptor_frame(small_box):
    """A predicted complex arrives in its own frame; the box test needs a superposition.

    Without it every co-folded ligand looks hundreds of Angstroms from the box,
    which is a false alarm about the model rather than a statement about binding.
    """
    ref_ca = np.array([
        [0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 3.8, 0.0], [0.0, 0.0, 3.8], [3.8, 3.8, 0.0],
    ])
    cx, cy, cz = small_box.center
    shift = np.array([310.0, -20.0, 15.0])
    text = _complex_cif(ref_ca + shift, np.array([[cx, cy, cz], [cx + 1.0, cy, cz]]) + shift)
    fetch = lambda url, timeout_s=60.0: text  # noqa: E731

    blind = bposeqc.check_pose_url("mem://m", small_box, "M1", hetatm_only=True, fetch=fetch)
    assert blind["ligand_in_box"] is False
    assert blind["centroid_offset_A"] > 200

    aligned = bposeqc.check_pose_url(
        "mem://m", small_box, "M1", hetatm_only=True, fetch=fetch, reference_ca=ref_ca
    )
    assert aligned["pose_frame_aligned"] is True
    assert aligned["receptor_ca_rmsd_A"] == pytest.approx(0.0, abs=1e-6)
    assert aligned["ligand_in_box"] is True
    assert aligned["centroid_offset_A"] < 5.0
    assert bposeqc.qc_summary([aligned])["n_frame_aligned"] == 1


def test_pose_qc_reports_an_unalignable_model_instead_of_a_bogus_box_distance(small_box):
    """A protein chain that does not match the receptor: not measurable."""
    ref_ca = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 3.8, 0.0]])
    shorter = _complex_cif(ref_ca[:2], np.array([[0.0, 0.0, 0.0]]))
    fetch = lambda url, timeout_s=60.0: shorter  # noqa: E731

    row = bposeqc.check_pose_url(
        "mem://m", small_box, "M1", hetatm_only=True, fetch=fetch, reference_ca=ref_ca
    )
    assert row["pose_frame_aligned"] is False
    assert row["pose_error"] == "frame_not_aligned"
    assert row["ligand_in_box"] is None          # not a measurement, so not reported as one
    summary = bposeqc.qc_summary([row])
    assert summary["n_checked"] == 0 and summary["n_frame_unaligned"] == 1


def test_ligand_only_pose_is_measured_without_a_superposition(small_box):
    """The hosted backends return a ligand-only file, already in the receptor frame.

    There is no protein chain to superpose, and requiring one would silently drop
    a check that those backends can legitimately perform - which is what happened
    before ``pose_frame_aligned`` distinguished the two cases.
    """
    cx, cy, cz = small_box.center
    ligand_only = _complex_cif([], np.array([[cx, cy, cz], [cx + 1.0, cy, cz]]))
    fetch = lambda url, timeout_s=60.0: ligand_only  # noqa: E731
    ref_ca = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 3.8, 0.0]])

    row = bposeqc.check_pose_url(
        "mem://m", small_box, "M1", hetatm_only=True, fetch=fetch, reference_ca=ref_ca
    )
    assert row["pose_frame_aligned"] is None, "nothing to superpose, so nothing to claim"
    assert row["pose_error"] is None
    assert row["ligand_in_box"] is True
    assert row["receptor_ca_rmsd_A"] is None
    summary = bposeqc.qc_summary([row])
    assert summary["n_checked"] == 1 and summary["fraction_in_box"] == 1.0
    assert summary["n_frame_aligned"] == 0 and summary["n_frame_unaligned"] == 0


def test_pose_qc_records_the_failure_instead_of_raising(tmp_path, small_box):
    def boom(url, timeout_s=60.0):
        raise RuntimeError("connection reset")

    rows = [{"mol_id": "M000", "url_ligand_structure": "https://x/y.cif"},
            {"mol_id": "M001", "url_ligand_structure": None}]
    qc = bposeqc.qc_poses(rows, small_box, tmp_path / "poses", top_k=10, fetch=boom)
    assert qc[0]["pose_error"].startswith("download_failed")
    assert qc[1]["pose_error"] == "no_structure_url"
    assert all(r["ligand_in_box"] is None for r in qc)
    assert bposeqc.qc_summary(qc)["n_errors"] == 2


def test_pose_qc_sanitises_hostile_molecule_ids(tmp_path, small_box):
    server = FakeBoltz(box_center=small_box.center)
    rows = [{"mol_id": "../../etc/passwd",
             "url_ligand_structure": "https://files.boltz.test/run/x.cif"}]
    bposeqc.qc_poses(rows, small_box, tmp_path / "poses", top_k=1, fetch=server.fetch)
    written = list((tmp_path / "poses").glob("*"))
    assert len(written) == 1
    assert written[0].parent == tmp_path / "poses"
    assert "/" not in written[0].name


def test_pose_qc_respects_top_k(tmp_path, small_box):
    server = FakeBoltz(box_center=small_box.center)
    rows = [{"mol_id": f"M{i:03d}",
             "url_ligand_structure": f"https://files.boltz.test/run/M{i:03d}.cif"}
            for i in range(10)]
    qc = bposeqc.qc_poses(rows, small_box, tmp_path / "poses", top_k=3, fetch=server.fetch)
    assert len(qc) == 3
    assert len(list((tmp_path / "poses").glob("*.cif"))) == 3
