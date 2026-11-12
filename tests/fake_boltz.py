"""A scripted Boltz API, served through ``httpx.MockTransport``.

The point of this module is that stage 2 is exercised through the *real* SDK:
the client is constructed by :func:`vcdd.refine.boltz.client.make_client`, the
request bodies are serialised by the SDK, the responses are parsed and validated
by the SDK's own models, and pagination is driven by the SDK's cursor logic.
A hand-rolled ``Mock()`` would pass while the real call signature was wrong,
which is exactly the class of bug 0.1.0 shipped in its Vina-GPU wrapper.

Everything the tests need to vary - failures, SMARTS-filtered molecules, a
``stopped`` run with partial results, pagination, cost - is a constructor
argument, so a test states its scenario instead of monkeypatching internals.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import httpx

BASE = "https://api.boltz.bio"
SCREEN = "/compute/v1/small-molecule/library-screen"
SAB = "/compute/v1/predictions/structure-and-binding"


def _cif(coords: Sequence[tuple[float, float, float]], comp: str = "LIG",
         chain: str = "L") -> str:
    """Minimal mmCIF with an ``_atom_site`` loop, which is all pose QC parses."""
    head = [
        "data_pose",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    body = [
        f"HETATM {i + 1} C C{i + 1} {comp} {chain} {chain} "
        f"{x:.3f} {y:.3f} {z:.3f}"
        for i, (x, y, z) in enumerate(coords)
    ]
    return "\n".join(head + body) + "\n"


def pose_in_box(centre: Sequence[float]) -> str:
    cx, cy, cz = centre
    return _cif([(cx, cy, cz), (cx + 1.0, cy, cz), (cx, cy + 1.0, cz)])


def pose_outside_box(centre: Sequence[float]) -> str:
    cx, cy, cz = centre
    return _cif([(cx + 40.0, cy, cz), (cx + 41.0, cy, cz), (cx + 40.0, cy + 1.0, cz)])


class FakeBoltz:
    """Scripted server for the two stage-2 backends.

    Parameters
    ----------
    running_polls:
        How many times ``retrieve`` reports ``running`` before the terminal
        status, so the polling loop is genuinely exercised.
    terminal_status:
        ``succeeded``, ``failed`` or ``stopped`` (``library_screen`` only).
    failed_ids / filtered_ids:
        Molecule ids that come back with no metrics, and ids that never come
        back at all because the SMARTS catalogue filter dropped them.
    page_size:
        Results are paginated at this size, so the SDK's cursor is exercised.
    outside_box_ids:
        Molecules whose predicted pose is served far outside the docking box.
    """

    def __init__(
        self,
        *,
        box_center: Sequence[float] = (0.0, 0.0, 0.0),
        running_polls: int = 1,
        terminal_status: str = "succeeded",
        failed_ids: Iterable[str] = (),
        filtered_ids: Iterable[str] = (),
        partial_ids: Iterable[str] = (),
        outside_box_ids: Iterable[str] = (),
        page_size: int = 2,
        cost_per_molecule: float = 0.025,
        cost_per_sample: float = 0.05,
        livemode: bool = True,
        estimate_error: bool = False,
    ) -> None:
        self.box_center = tuple(box_center)
        self.running_polls = running_polls
        self.terminal_status = terminal_status
        self.failed_ids = set(failed_ids)
        self.filtered_ids = set(filtered_ids)
        self.partial_ids = set(partial_ids)
        self.outside_box_ids = set(outside_box_ids)
        self.page_size = page_size
        self.cost_per_molecule = cost_per_molecule
        self.cost_per_sample = cost_per_sample
        self.livemode = livemode
        self.estimate_error = estimate_error

        #: run_id -> {"molecules": [...], "polls": int, "idempotency_key": str}
        self.runs: dict[str, dict[str, Any]] = {}
        #: every request the SDK actually made, for assertions
        self.calls: list[tuple[str, str]] = []
        self.started_payloads: list[dict[str, Any]] = []
        self._by_key: dict[str, str] = {}
        self._n = 0

    # ------------------------------------------------------------ transport
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def structure_text(self, mol_id: str) -> str:
        if mol_id in self.outside_box_ids:
            return pose_outside_box(self.box_center)
        return pose_in_box(self.box_center)

    def fetch(self, url: str, timeout_s: float = 60.0) -> str:
        """Drop-in for ``poseqc.http_get_text``; the URL carries the mol id."""
        self.calls.append(("GET", url))
        mol_id = url.rsplit("/", 1)[-1].removesuffix(".cif")
        return self.structure_text(mol_id)

    # -------------------------------------------------------------- routing
    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        self.calls.append((method, path))
        body = json.loads(request.content) if request.content else {}

        if path == f"{SCREEN}/estimate-cost":
            return self._estimate(len(body.get("molecules") or []), self.cost_per_molecule,
                                  "small_molecule_library_screen")
        if path == f"{SAB}/estimate-cost":
            n = int((body.get("input") or {}).get("num_samples") or 1)
            return self._estimate(n, self.cost_per_sample, "structure_and_binding")
        if path == SCREEN and method == "POST":
            return self._start_screen(body)
        if path == SAB and method == "POST":
            return self._start_sab(body)
        if path.startswith(f"{SCREEN}/") and path.endswith("/results"):
            return self._results(path.split("/")[-2], request)
        if path.startswith(f"{SCREEN}/") and method == "GET":
            return self._retrieve_screen(path.rsplit("/", 1)[-1])
        if path.startswith(f"{SAB}/") and method == "GET":
            return self._retrieve_sab(path.rsplit("/", 1)[-1])
        return httpx.Response(404, json={"error": {"code": "not_found", "message": path}})

    # --------------------------------------------------------------- pieces
    def _estimate(self, units: int, per_unit: float, application: str) -> httpx.Response:
        if self.estimate_error:
            return httpx.Response(
                500, json={"error": {"code": "server_error", "message": "no price"}})
        return httpx.Response(200, json={
            "estimated_cost_usd": f"{units * per_unit:.4f}",
            "disclaimer": "estimate only",
            "breakdown": {"application": application,
                          "cost_per_unit_usd": f"{per_unit:.4f}",
                          "num_units": units},
        })

    def _new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:04d}"

    def _start_screen(self, body: dict[str, Any]) -> httpx.Response:
        self.started_payloads.append(body)
        key = body.get("idempotency_key") or ""
        if key and key in self._by_key:                    # re-attach, do not re-run
            run_id = self._by_key[key]
        else:
            run_id = self._new_id("ls")
            self._by_key[key] = run_id
            self.runs[run_id] = {
                "molecules": list(body.get("molecules") or []),
                "polls": 0, "kind": "screen",
            }
        return httpx.Response(200, json=self._screen_state(run_id))

    def _screen_state(self, run_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        mols = run["molecules"]
        kept = [m for m in mols if m.get("id") not in self.filtered_ids]
        status = "running" if run["polls"] < self.running_polls else self.terminal_status
        return {
            "id": run_id,
            "status": status,
            "engine": "boltzmol",
            "engine_version": "1.0",
            "pipeline": "boltzmol",
            "pipeline_version": "1.0",
            "livemode": self.livemode,
            "created_at": "2026-01-01T00:00:00Z",
            "progress": {
                "num_molecules_screened": len(kept) if status != "running" else 0,
                "num_molecules_failed": len(self.failed_ids & {m.get("id") for m in mols}),
                "total_molecules_to_screen": len(kept),
                "latest_result_id": None,
                "rejection_summary": {
                    "filtered_count": len(mols) - len(kept), "invalid_count": 0},
            },
        }

    def _retrieve_screen(self, run_id: str) -> httpx.Response:
        if run_id not in self.runs:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": run_id}})
        state = self._screen_state(run_id)
        self.runs[run_id]["polls"] += 1
        return httpx.Response(200, json=state)

    def _result_item(self, run_id: str, mol: dict[str, Any], n: int) -> dict[str, Any]:
        mol_id = mol.get("id")
        rid = f"{run_id}_r{n:04d}"
        failed = mol_id in self.failed_ids
        url = f"https://files.boltz.test/{run_id}/{mol_id}.cif"
        metrics = None if failed else {
            # deterministic, ordered by position so ranking assertions are stable
            "binding_confidence": round(0.95 - 0.07 * n, 4),
            "complex_iplddt": 0.8, "complex_plddt": 0.85,
            "iptm": 0.7, "optimization_score": round(0.9 - 0.05 * n, 4),
            "ptm": 0.75, "structure_confidence": 0.82,
        }
        return {
            "id": rid,
            "smiles": mol.get("smiles"),
            "external_id": mol_id,
            "created_at": "2026-01-01T00:00:00Z",
            "metrics": metrics,
            "adme": {"lipophilicity": 2.1, "permeability": -5.0, "solubility": -3.2},
            "artifacts": {
                "archive": {"url": f"{url}.tar.gz", "url_expires_at": None},
                "structure": {"url": url, "url_expires_at": None},
                "ligand_structure": {"url": url, "url_expires_at": None},
            },
            "warnings": [{"code": "no_metrics", "message": "prediction failed"}] if failed else [],
        }

    def _results(self, run_id: str, request: httpx.Request) -> httpx.Response:
        if run_id not in self.runs:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": run_id}})
        mols = [m for m in self.runs[run_id]["molecules"]
                if m.get("id") not in self.filtered_ids]
        if self.terminal_status == "stopped":
            keep = self.partial_ids or {m.get("id") for m in mols[:1]}
            mols = [m for m in mols if m.get("id") in keep]
        items = [self._result_item(run_id, m, i) for i, m in enumerate(mols)]

        after = request.url.params.get("after_id")
        if after:
            ids = [it["id"] for it in items]
            items = items[ids.index(after) + 1:] if after in ids else []
        page = items[: self.page_size]
        has_more = len(items) > len(page)
        return httpx.Response(200, json={
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": has_more,
        })

    # ------------------------------------------------- structure_and_binding
    def _start_sab(self, body: dict[str, Any]) -> httpx.Response:
        self.started_payloads.append(body)
        key = body.get("idempotency_key") or ""
        if key and key in self._by_key:
            run_id = self._by_key[key]
        else:
            run_id = self._new_id("sab")
            self._by_key[key] = run_id
            self.runs[run_id] = {"input": body.get("input"), "polls": 0, "kind": "sab"}
        return httpx.Response(200, json=self._sab_state(run_id))

    def _sab_state(self, run_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        status = "running" if run["polls"] < self.running_polls else self.terminal_status
        if status == "stopped":       # not a state this endpoint has
            status = "failed"
        out: dict[str, Any] = {
            "id": run_id, "status": status, "model": "boltz-2.1", "version": "2.1.0",
            "livemode": self.livemode, "created_at": "2026-01-01T00:00:00Z",
        }
        if status != "succeeded":
            return out
        n = int(run.get("_n") or len(self.runs))
        metrics = {"complex_iplddt": 0.8, "complex_plddt": 0.85, "iptm": 0.7,
                   "ligand_iptm": 0.65, "protein_iptm": 0.72, "ptm": 0.75,
                   "complex_pde": 1.2, "complex_ipde": 1.4, "structure_confidence": 0.82}
        url = f"https://files.boltz.test/{run_id}/{run_id}.cif"
        sample = {"metrics": metrics,
                  "structure": {"url": url, "url_expires_at": None},
                  "ligand_structure": {"url": url, "url_expires_at": None}}
        out["output"] = {
            "all_sample_results": [sample],
            "best_sample": sample,
            "archive": {"url": f"{url}.tar.gz", "url_expires_at": None},
            "binding_metrics": {"type": "ligand_protein_binding_metrics",
                                "binding_confidence": round(0.9 - 0.05 * n, 4),
                                "optimization_score": 0.6},
        }
        return out

    def _retrieve_sab(self, run_id: str) -> httpx.Response:
        if run_id not in self.runs:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": run_id}})
        state = self._sab_state(run_id)
        self.runs[run_id]["polls"] += 1
        return httpx.Response(200, json=state)


def make_fake_client(server: FakeBoltz, env_var: str = "BOLTZ_API_KEY"):
    """Build a real SDK client whose transport is ``server``."""
    from vcdd.refine.boltz.client import make_client

    return make_client(env_var, http_client=httpx.Client(
        transport=server.transport(), base_url=BASE))
