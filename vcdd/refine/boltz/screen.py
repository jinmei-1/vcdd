"""The ``library_screen`` backend: BoltzMol-1 over the stage-1 shortlist.

Lifecycle, in the order the driver walks it:

1. ``estimate_cost`` - free, allocates no GPU, is not billed. Used as the spend
   gate *before* anything is submitted.
2. ``start`` with a deterministic ``idempotency_key`` derived from the molecule
   set and the target. Re-running the same shortlist re-attaches to the existing
   run instead of paying for it twice.
3. ``retrieve`` on a poll loop until the run reaches a terminal state.
4. ``list_results`` paginated, joined back to the shortlist on ``external_id``.

Three failure modes that are easy to get wrong and expensive to discover late:

* **Filtered molecules disappear.** With ``molecule_filters`` enabled the API
  rejects molecules up-front and they are simply absent from the results, so a
  naive join turns "rejected by a SMARTS catalogue" into "no data". Every
  submitted molecule is reconciled against the returned set and the difference
  is written to ``rejected.parquet``.
* **A stopped run still has results.** ``stopped`` is terminal but partial. The
  results fetched so far are kept and labelled, not discarded.
* **Cost is per molecule, and it is real money.** The gate is checked against
  the API's own estimate, not against an arithmetic guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from vcdd.runtime.hashing import hash_obj
from vcdd.runtime.logging import get_logger
from vcdd.runtime.paths import atomic_write_text, ensure_dir

LOG = get_logger()

TERMINAL = {"succeeded", "failed", "stopped"}

#: Flattened metric columns pulled out of each result item.
METRIC_FIELDS = (
    "binding_confidence",
    "optimization_score",
    "structure_confidence",
    "iptm",
    "ptm",
    "complex_plddt",
    "complex_iplddt",
)
ADME_FIELDS = ("lipophilicity", "permeability", "solubility")


class SpendGateExceeded(RuntimeError):
    """The API's own cost estimate is above the configured ceiling."""


@dataclass
class ScreenRun:
    """Bookkeeping for one submitted chunk, persisted so a run can re-attach."""

    run_id: str
    chunk_index: int
    n_molecules: int
    idempotency_key: str
    estimated_cost_usd: str | None = None
    status: str = "pending"
    engine: str | None = None
    engine_version: str | None = None
    livemode: bool | None = None
    error: dict[str, Any] | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    #: Molecule ids submitted in this run. Persisted so a resumed stage 2 knows
    #: exactly which molecules a run covers even if ``chunk_size`` changed.
    mol_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "chunk_index": self.chunk_index,
            "n_molecules": self.n_molecules,
            "idempotency_key": self.idempotency_key,
            "estimated_cost_usd": self.estimated_cost_usd,
            "status": self.status,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "livemode": self.livemode,
            "error": self.error,
            "progress": self.progress,
            "mol_ids": list(self.mol_ids),
        }


# ------------------------------------------------------------------- helpers
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute off an SDK model or a key off a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def make_molecules(mol_ids: Sequence[str], smiles: Sequence[str]) -> list[dict[str, str]]:
    """``[{smiles, id}]`` - the ``id`` comes back as ``external_id`` on results."""
    return [{"smiles": str(s), "id": str(m)} for m, s in zip(mol_ids, smiles)]


def idempotency_key(target: dict[str, Any], molecules: Sequence[dict[str, str]], salt: str = "") -> str:
    """Deterministic key over the exact request payload.

    Re-submitting an identical chunk re-attaches to the existing run rather than
    starting - and paying for - a second one. Including the target means a new
    receptor or a new pocket correctly starts a new run.
    """
    return "vcdd-" + hash_obj({"target": target, "molecules": list(molecules), "salt": salt})[:32]


def estimate_cost(
    client: Any,
    target: dict[str, Any],
    molecules: Sequence[dict[str, str]],
    molecule_filters: dict[str, Any] | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Ask the API what this chunk would cost. Free, no GPU, not billed."""
    kwargs: dict[str, Any] = {"molecules": list(molecules), "target": target}
    if molecule_filters:
        kwargs["molecule_filters"] = molecule_filters
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    resp = client.small_molecule.library_screen.estimate_cost(**kwargs)
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


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_spend(
    estimates: Iterable[dict[str, Any]],
    max_spend_usd: float,
    allow: bool = False,
) -> float:
    """Total the API estimates and enforce the ceiling.

    Returns the total. Raises :class:`SpendGateExceeded` above the ceiling
    unless ``allow`` (``--yes-spend``). An estimate the API declined to give is
    treated as unknown-and-therefore-blocking rather than as zero.
    """
    total = 0.0
    unknown = 0
    for est in estimates:
        value = _as_float(est.get("estimated_cost_usd"))
        if value is None:
            unknown += 1
        else:
            total += value
    if allow:
        return total
    if unknown:
        raise SpendGateExceeded(
            f"the API returned no cost estimate for {unknown} chunk(s), so the spend cannot be "
            f"bounded before submitting. Re-run with --yes-spend to proceed anyway."
        )
    if total > max_spend_usd:
        raise SpendGateExceeded(
            f"estimated cost ${total:.2f} exceeds stage2.max_spend_usd=${max_spend_usd:.2f}. "
            f"Either raise the ceiling, lower stage2.top_n, or re-run with --yes-spend."
        )
    return total


# -------------------------------------------------------------------- submit
def start_chunk(
    client: Any,
    target: dict[str, Any],
    molecules: Sequence[dict[str, str]],
    chunk_index: int,
    molecule_filters: dict[str, Any] | None = None,
    workspace_id: str | None = None,
    estimated_cost_usd: str | None = None,
) -> ScreenRun:
    """Submit one chunk and return its bookkeeping record."""
    key = idempotency_key(target, molecules)
    kwargs: dict[str, Any] = {
        "molecules": list(molecules),
        "target": target,
        "idempotency_key": key,
    }
    if molecule_filters:
        kwargs["molecule_filters"] = molecule_filters
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    resp = client.small_molecule.library_screen.start(**kwargs)
    return ScreenRun(
        run_id=str(_get(resp, "id")),
        chunk_index=chunk_index,
        n_molecules=len(molecules),
        idempotency_key=key,
        estimated_cost_usd=estimated_cost_usd,
        status=str(_get(resp, "status", "pending")),
        engine=_get(resp, "engine"),
        engine_version=_get(resp, "engine_version"),
        livemode=_get(resp, "livemode"),
        mol_ids=[str(m["id"]) for m in molecules],
    )


def poll_until_terminal(
    client: Any,
    run: ScreenRun,
    poll_interval_s: float = 20.0,
    timeout_s: float | None = None,
    workspace_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[ScreenRun], None] | None = None,
) -> ScreenRun:
    """Poll ``retrieve`` until the run reaches a terminal state.

    ``sleep`` is injected so the tests can drive the whole loop without waiting.
    """
    t0 = time.time()
    while True:
        kwargs = {"workspace_id": workspace_id} if workspace_id else {}
        resp = client.small_molecule.library_screen.retrieve(run.run_id, **kwargs)
        run.status = str(_get(resp, "status", "pending"))
        run.engine = _get(resp, "engine", run.engine)
        run.engine_version = _get(resp, "engine_version", run.engine_version)
        run.livemode = _get(resp, "livemode", run.livemode)
        progress = _get(resp, "progress")
        if progress is not None:
            rejection = _get(progress, "rejection_summary")
            run.progress = {
                "num_molecules_screened": _get(progress, "num_molecules_screened"),
                "num_molecules_failed": _get(progress, "num_molecules_failed"),
                "total_molecules_to_screen": _get(progress, "total_molecules_to_screen"),
                "rejection_filtered_count": _get(rejection, "filtered_count"),
                "rejection_invalid_count": _get(rejection, "invalid_count"),
            }
        err = _get(resp, "error")
        if err is not None:
            run.error = {"code": _get(err, "code"), "message": _get(err, "message")}
        if on_progress:
            on_progress(run)
        if run.status in TERMINAL:
            return run
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            run.error = {"code": "vcdd_poll_timeout",
                         "message": f"still {run.status} after {timeout_s:.0f}s"}
            return run
        sleep(poll_interval_s)


def fetch_results(
    client: Any,
    run_id: str,
    workspace_id: str | None = None,
    page_limit: int = 100,
) -> list[dict[str, Any]]:
    """Page through ``list_results`` and flatten each item into a plain dict.

    The SDK's ``SyncCursorPage`` is iterable and handles the cursor itself, so
    iterating it is the paginated fetch.
    """
    kwargs: dict[str, Any] = {"limit": page_limit}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    page = client.small_molecule.library_screen.list_results(run_id, **kwargs)

    rows: list[dict[str, Any]] = []
    for item in page:
        metrics = _get(item, "metrics")
        adme = _get(item, "adme")
        artifacts = _get(item, "artifacts")
        warnings = _get(item, "warnings") or []
        row: dict[str, Any] = {
            "boltz_run_id": run_id,
            "boltz_result_id": _get(item, "id"),
            "mol_id": _get(item, "external_id"),
            "boltz_smiles": _get(item, "smiles"),
            "created_at": str(_get(item, "created_at") or ""),
        }
        for name in METRIC_FIELDS:
            row[name] = _as_float(_get(metrics, name))
        for name in ADME_FIELDS:
            value = _get(adme, name)
            row[f"adme_{name}"] = value if name == "solubility" else _as_float(value)
        structure = _get(artifacts, "structure")
        ligand_structure = _get(artifacts, "ligand_structure")
        archive = _get(artifacts, "archive")
        row["url_structure"] = _get(structure, "url")
        row["url_ligand_structure"] = _get(ligand_structure, "url")
        row["url_archive"] = _get(archive, "url")
        row["warnings"] = "; ".join(
            f"{_get(w, 'code')}:{_get(w, 'message')}" for w in warnings
        ) or None
        rows.append(row)
    return rows


# ----------------------------------------------------------------- artefacts
def append_results_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write result rows to ``results.jsonl``.

    Rewrites the whole file rather than appending: ``/mnt/results`` style
    S3-backed mounts do not support POSIX append, and a silently truncated
    results file is the worst possible artefact here. Existing rows are read
    back first and de-duplicated on ``boltz_result_id``.
    """
    p = Path(path)
    ensure_dir(p.parent)
    existing: dict[str, dict[str, Any]] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[str(rec.get("boltz_result_id"))] = rec
    for row in rows:
        existing[str(row.get("boltz_result_id"))] = row
    atomic_write_text(p, "\n".join(json.dumps(r, default=str) for r in existing.values()) + "\n")
    return len(existing)


def save_runs(path: str | Path, runs: Sequence[ScreenRun]) -> None:
    """Persist run ids so an interrupted stage 2 re-attaches instead of re-paying."""
    atomic_write_text(Path(path), json.dumps([r.as_dict() for r in runs], indent=2, default=str))


def load_runs(path: str | Path) -> list[ScreenRun]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[ScreenRun] = []
    for rec in json.loads(p.read_text() or "[]"):
        out.append(ScreenRun(
            run_id=rec["run_id"],
            chunk_index=int(rec.get("chunk_index", 0)),
            n_molecules=int(rec.get("n_molecules", 0)),
            idempotency_key=rec.get("idempotency_key", ""),
            estimated_cost_usd=rec.get("estimated_cost_usd"),
            status=rec.get("status", "pending"),
            engine=rec.get("engine"),
            engine_version=rec.get("engine_version"),
            livemode=rec.get("livemode"),
            error=rec.get("error"),
            progress=rec.get("progress") or {},
            mol_ids=[str(m) for m in (rec.get("mol_ids") or [])],
        ))
    return out


def reconcile_missing(
    submitted: Sequence[dict[str, str]],
    returned_mol_ids: Iterable[str],
    run: ScreenRun,
) -> list[dict[str, Any]]:
    """Molecules that were submitted but came back with no result.

    The API drops filtered and invalid molecules from the results rather than
    returning them with an error, so without this reconciliation a SMARTS-filter
    rejection is indistinguishable from a missing row. ``progress.rejection_summary``
    gives the counts; this attributes them per molecule as best the API allows.
    """
    have = {str(m) for m in returned_mol_ids if m is not None}
    missing = [m for m in submitted if str(m["id"]) not in have]
    if not missing:
        return []
    filtered = run.progress.get("rejection_filtered_count") or 0
    invalid = run.progress.get("rejection_invalid_count") or 0
    if filtered and not invalid:
        reason = "filtered_by_smarts_catalog"
    elif invalid and not filtered:
        reason = "invalid_smiles"
    elif filtered or invalid:
        reason = f"rejected (run reported {filtered} filtered, {invalid} invalid)"
    else:
        reason = "no_result_returned"
    return [
        {
            "mol_id": m["id"],
            "smiles": m["smiles"],
            "boltz_run_id": run.run_id,
            "reason": reason,
            "run_status": run.status,
        }
        for m in missing
    ]
