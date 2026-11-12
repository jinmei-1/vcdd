"""The ``structure_and_binding`` backend: one Boltz-2.1 co-folding job per ligand.

How this differs from :mod:`vcdd.refine.boltz.screen`, and when it is worth the
extra money:

* **Model.** ``library_screen`` runs BoltzMol-1, a screening-tuned pipeline.
  ``structure_and_binding`` runs boltz-2.1 itself - the model the published
  benchmarks describe. If you want your numbers to be comparable to the Boltz-2
  literature, this is the endpoint that produces them.
* **Granularity.** One prediction per molecule, not one job per chunk. That
  means per-molecule failures, per-molecule idempotency, and real concurrency
  control, but also more requests and a higher unit price.
* **Control.** Pocket constraints, sample count, receptor templates and
  diffusion options are all exposed. ``library_screen`` fixes them.
* **Cost.** Priced per *sample* by receptor crop size rather than per molecule,
  so ``num_samples`` multiplies the bill directly.

Two API limits worth knowing before you submit 500 jobs:

* The binder chain must have **exactly one copy and at most 100 heavy atoms**.
  Larger ligands are rejected by the API; they are screened out here first, with
  the reason recorded, so a macrocycle in the shortlist does not silently become
  a missing row.
* ``num_samples`` is 1-10. More samples give a better best-of-N structure and
  cost proportionally more.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

from vcdd.refine.boltz.screen import ScreenRun, _as_float, _get, idempotency_key
from vcdd.runtime.logging import get_logger

LOG = get_logger()

MODEL = "boltz-2.1"
#: ``structure_and_binding`` has no ``stopped`` state, unlike ``library_screen``.
TERMINAL = {"succeeded", "failed"}
#: API limit on the binder chain.
MAX_BINDER_HEAVY_ATOMS = 100
#: Chain id given to the ligand. Receptor chains come from the PDBQT, which in
#: practice never uses this one; :func:`build_input` checks anyway.
LIGAND_CHAIN_ID = "L"


def heavy_atom_count(smiles: str) -> int | None:
    """Heavy atoms in a SMILES, or ``None`` if RDKit cannot parse it."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001
        return None
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def pick_ligand_chain_id(receptor_chain_ids: Iterable[str]) -> str:
    """A chain id for the ligand that does not collide with the receptor."""
    used = {str(c) for c in receptor_chain_ids}
    for candidate in [LIGAND_CHAIN_ID, "X", "Z", "Y", "W"]:
        if candidate not in used:
            return candidate
    for code in range(ord("A"), ord("Z") + 1):
        if chr(code) not in used:
            return chr(code)
    raise ValueError("no free chain id for the ligand; receptor uses A-Z")


def build_input(
    protein_entities: Sequence[dict[str, Any]],
    smiles: str,
    ligand_chain_id: str = LIGAND_CHAIN_ID,
    pocket_residues: dict[str, Sequence[int]] | None = None,
    pocket_max_distance_a: float = 6.0,
    pocket_force: bool = False,
    num_samples: int = 1,
    templates: Sequence[dict[str, Any]] | None = None,
    model_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``Input`` payload for one ligand.

    ``protein_entities`` are the receptor entities produced by
    :func:`vcdd.refine.boltz.target.build_target` - the same sequences and chain
    ids both backends use, so the two are comparable on the same receptor.

    The pocket constraint is what carries the stage-1 docking box across into
    Boltz. Without it the model is free to place the ligand at any site on the
    protein, and a confident prediction at the wrong pocket is worse than no
    prediction: it looks like a hit.
    """
    if not 1 <= int(num_samples) <= 10:
        raise ValueError(f"num_samples must be 1-10, got {num_samples}")
    entities: list[dict[str, Any]] = [dict(e) for e in protein_entities]
    used = {c for e in entities for c in e.get("chain_ids", [])}
    if ligand_chain_id in used:
        raise ValueError(
            f"ligand chain id {ligand_chain_id!r} collides with a receptor chain; "
            f"pass a free one (receptor uses {sorted(used)})"
        )
    entities.append({
        "type": "ligand_smiles",
        "chain_ids": [ligand_chain_id],
        "value": smiles,
    })

    payload: dict[str, Any] = {
        "entities": entities,
        "binding": {"type": "ligand_protein_binding", "binder_chain_id": ligand_chain_id},
        "num_samples": int(num_samples),
    }
    if pocket_residues:
        payload["constraints"] = [{
            "type": "pocket",
            "binder_chain_id": ligand_chain_id,
            "contact_residues": {k: list(v) for k, v in pocket_residues.items()},
            "max_distance_angstrom": float(pocket_max_distance_a),
            "force": bool(pocket_force),
        }]
    if templates:
        payload["templates"] = [dict(t) for t in templates]
    if model_options:
        payload["model_options"] = dict(model_options)
    return payload


def receptor_template(
    cif_or_pdb_text: str,
    chain_map: dict[str, str],
    media_type: str = "chemical/x-pdb",
    force_threshold_angstroms: float | None = None,
) -> dict[str, Any]:
    """A base64 template entry pinning protein geometry to the docked receptor.

    Only useful when you want Boltz to reproduce *this* receptor conformation
    rather than predict one. It biases the structure, so it also removes any
    independent structural evidence the prediction would otherwise provide -
    which is exactly why it is off by default.
    """
    import base64

    data = base64.b64encode(cif_or_pdb_text.encode("utf-8")).decode("ascii")
    entry: dict[str, Any] = {
        "template_chains": [
            {"input_chain_id": k, "template_chain_id": v} for k, v in chain_map.items()
        ],
        "template_structure": {"type": "base64", "media_type": media_type, "data": data},
    }
    if force_threshold_angstroms is not None:
        entry["force_threshold_angstroms"] = float(force_threshold_angstroms)
    return entry


def estimate_cost(
    client: Any,
    payload: dict[str, Any],
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Cost for one prediction. Free, allocates no GPU, is not billed."""
    kwargs: dict[str, Any] = {"input": payload, "model": MODEL}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    resp = client.predictions.structure_and_binding.estimate_cost(**kwargs)
    breakdown = _get(resp, "breakdown")
    return {
        "estimated_cost_usd": _get(resp, "estimated_cost_usd"),
        "disclaimer": _get(resp, "disclaimer"),
        "breakdown": {
            "application": _get(breakdown, "application"),
            "cost_per_unit_usd": _get(breakdown, "cost_per_unit_usd"),
            "num_units": _get(breakdown, "num_units"),
        },
    }


def start_one(
    client: Any,
    payload: dict[str, Any],
    mol_id: str,
    index: int = 0,
    workspace_id: str | None = None,
    estimated_cost_usd: str | None = None,
) -> ScreenRun:
    """Submit one prediction. The idempotency key makes a retry free."""
    key = idempotency_key({"model": MODEL}, [{"id": str(mol_id), "smiles": ""}],
                          salt=str(sorted(payload.get("entities", []), key=str)))
    kwargs: dict[str, Any] = {"input": payload, "model": MODEL, "idempotency_key": key}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    resp = client.predictions.structure_and_binding.start(**kwargs)
    return ScreenRun(
        run_id=str(_get(resp, "id")),
        chunk_index=index,
        n_molecules=1,
        idempotency_key=key,
        estimated_cost_usd=estimated_cost_usd,
        status=str(_get(resp, "status", "pending")),
        engine=MODEL,
        engine_version=_get(resp, "version"),
        livemode=_get(resp, "livemode"),
        mol_ids=[str(mol_id)],
    )


def retrieve(client: Any, prediction_id: str, workspace_id: str | None = None) -> Any:
    kwargs = {"workspace_id": workspace_id} if workspace_id else {}
    return client.predictions.structure_and_binding.retrieve(prediction_id, **kwargs)


def poll_until_terminal(
    client: Any,
    run: ScreenRun,
    poll_interval_s: float = 20.0,
    timeout_s: float | None = None,
    workspace_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ScreenRun, Any]:
    """Poll one prediction to a terminal state; return the run and the response."""
    t0 = time.time()
    resp = None
    while True:
        resp = retrieve(client, run.run_id, workspace_id=workspace_id)
        run.status = str(_get(resp, "status", "pending"))
        run.livemode = _get(resp, "livemode", run.livemode)
        run.engine_version = _get(resp, "version", run.engine_version)
        err = _get(resp, "error")
        if err is not None:
            run.error = {"code": _get(err, "code"), "message": _get(err, "message")}
        if run.status in TERMINAL:
            return run, resp
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            run.error = {"code": "vcdd_poll_timeout",
                         "message": f"still {run.status} after {timeout_s:.0f}s"}
            return run, resp
        sleep(poll_interval_s)


def flatten_result(resp: Any, mol_id: str, smiles: str) -> dict[str, Any]:
    """Flatten one prediction into the same row shape ``library_screen`` produces.

    Both backends land in the same ``scores.parquet``, so downstream code, the
    report and stage 3 never have to know which one ran.
    """
    output = _get(resp, "output")
    best = _get(output, "best_sample")
    metrics = _get(best, "metrics")
    binding = _get(output, "binding_metrics")
    structure = _get(best, "structure")
    ligand_structure = _get(best, "ligand_structure")
    archive = _get(output, "archive")
    err = _get(resp, "error")

    row: dict[str, Any] = {
        "boltz_run_id": str(_get(resp, "id")),
        "boltz_result_id": str(_get(resp, "id")),
        "mol_id": str(mol_id),
        "boltz_smiles": smiles,
        "created_at": str(_get(resp, "created_at") or ""),
        "binding_confidence": _as_float(_get(binding, "binding_confidence")),
        "optimization_score": _as_float(_get(binding, "optimization_score")),
        "structure_confidence": _as_float(_get(metrics, "structure_confidence")),
        "iptm": _as_float(_get(metrics, "iptm")),
        "ptm": _as_float(_get(metrics, "ptm")),
        "complex_plddt": _as_float(_get(metrics, "complex_plddt")),
        "complex_iplddt": _as_float(_get(metrics, "complex_iplddt")),
        # boltz-2.1 reports these; BoltzMol-1 does not. Extra columns are
        # harmless, and dropping them would throw away the interface-quality
        # signal that distinguishes this endpoint.
        "ligand_iptm": _as_float(_get(metrics, "ligand_iptm")),
        "protein_iptm": _as_float(_get(metrics, "protein_iptm")),
        "complex_pde": _as_float(_get(metrics, "complex_pde")),
        "complex_ipde": _as_float(_get(metrics, "complex_ipde")),
        "adme_lipophilicity": None,
        "adme_permeability": None,
        "adme_solubility": None,
        "url_structure": _get(structure, "url"),
        "url_ligand_structure": _get(ligand_structure, "url"),
        "url_archive": _get(archive, "url"),
        "n_samples": len(_get(output, "all_sample_results") or []) or None,
        "warnings": None,
    }
    if err is not None:
        row["warnings"] = f"{_get(err, 'code')}:{_get(err, 'message')}"
    return row


def screen_molecules(
    client: Any,
    protein_entities: Sequence[dict[str, Any]],
    mol_ids: Sequence[str],
    smiles: Sequence[str],
    pocket_residues: dict[str, Sequence[int]] | None = None,
    ligand_chain_id: str = LIGAND_CHAIN_ID,
    num_samples: int = 1,
    pocket_max_distance_a: float = 6.0,
    templates: Sequence[dict[str, Any]] | None = None,
    workspace_id: str | None = None,
    max_concurrency: int = 8,
    poll_interval_s: float = 20.0,
    timeout_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    estimates: Sequence[str | None] | None = None,
) -> tuple[list[dict[str, Any]], list[ScreenRun], list[dict[str, Any]]]:
    """Submit, poll and collect one prediction per molecule.

    Returns ``(rows, runs, rejected)``. Concurrency is bounded because the org
    rate limit is 100 requests / 10 s and the queue caps out at 5,000 pending
    predictions; firing 500 submissions at once wastes the budget on 429s that
    the SDK then has to back off from.
    """
    rejected: list[dict[str, Any]] = []
    work: list[tuple[int, str, str]] = []
    for i, (mid, smi) in enumerate(zip(mol_ids, smiles)):
        n_heavy = heavy_atom_count(smi)
        if n_heavy is None:
            rejected.append({"mol_id": str(mid), "smiles": smi, "boltz_run_id": None,
                             "reason": "invalid_smiles", "run_status": "not_submitted"})
            continue
        if n_heavy > MAX_BINDER_HEAVY_ATOMS:
            rejected.append({
                "mol_id": str(mid), "smiles": smi, "boltz_run_id": None,
                "reason": f"binder_too_large ({n_heavy} heavy atoms > {MAX_BINDER_HEAVY_ATOMS})",
                "run_status": "not_submitted",
            })
            continue
        work.append((i, str(mid), smi))

    if rejected:
        LOG.warning("stage2/boltz: %d molecule(s) not submitted to %s (see rejected.parquet)",
                    len(rejected), MODEL)

    rows: list[dict[str, Any]] = []
    runs: list[ScreenRun] = []

    def _one(item: tuple[int, str, str]) -> tuple[ScreenRun, dict[str, Any]]:
        idx, mid, smi = item
        payload = build_input(
            protein_entities, smi,
            ligand_chain_id=ligand_chain_id,
            pocket_residues=pocket_residues,
            pocket_max_distance_a=pocket_max_distance_a,
            num_samples=num_samples,
            templates=templates,
        )
        est = estimates[idx] if estimates is not None and idx < len(estimates) else None
        run = start_one(client, payload, mid, index=idx, workspace_id=workspace_id,
                        estimated_cost_usd=est)
        run, resp = poll_until_terminal(
            client, run, poll_interval_s=poll_interval_s, timeout_s=timeout_s,
            workspace_id=workspace_id, sleep=sleep,
        )
        return run, flatten_result(resp, mid, smi)

    n_workers = max(1, min(int(max_concurrency), len(work) or 1))
    if work:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for run, row in pool.map(_one, work):
                runs.append(run)
                if run.status == "succeeded":
                    rows.append(row)
                else:
                    rejected.append({
                        "mol_id": row["mol_id"], "smiles": row["boltz_smiles"],
                        "boltz_run_id": run.run_id,
                        "reason": (run.error or {}).get("code") or run.status,
                        "run_status": run.status,
                    })
    return rows, runs, rejected
