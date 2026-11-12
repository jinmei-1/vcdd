"""The ``local`` backend: run the boltz-2 model on this machine.

``stage2.backend: local`` replaces the hosted API with a ``boltz predict``
subprocess. The science is the same model the ``structure_and_binding`` API
backend runs, so the confidence columns are comparable to that backend's - and
are *not* comparable to ``library_screen`` (BoltzMol-1 is a hosted-only
screening pipeline). What local inference buys and costs:

* **No key, no billing, no spend gate.** ``stage2.max_spend_usd`` is skipped
  entirely: the only ceiling is wall clock.
* **Your GPU, your queue.** The knob that matters is ``max_concurrency`` x
  ``devices``: one ``boltz predict`` process per molecule, round-robin across
  the configured GPU ids. Each subprocess is pinned to its GPU via
  ``CUDA_VISIBLE_DEVICES`` (boltz's ``--devices`` is a *count*, not a list of
  ids), so ``max_concurrency <= len(devices)`` means no GPU is oversubscribed.
  Samples are requested with ``--diffusion_samples``.
* **MSA is mandatory and the sequence leaves the machine by default.** boltz
  2.2 refuses MSA-less inputs, so each predict launches with
  ``--use_msa_server`` (default server: the public ColabFold API). Point
  ``local_boltz.msa_server_url`` at a self-hosted MMSeqs2 server, or provide
  precomputed ``.a3m`` per chain in ``local_boltz.protein_msa_files`` (also the
  way to query once per receptor instead of once per molecule).
* **The API-only hints disappear.** ``pocket_residues`` and
  ``reference_ligands`` cannot be expressed in the ``boltz predict`` YAML, so
  the stage-1 docking box is *not* passed to the model. The
  :mod:`vcdd.refine.boltz.poseqc` ligand-in-box check is therefore not
  optional decoration here - it is the only thing that tells you the model put
  the ligand in the pocket you docked into, and stage 2 runs it on every
  predicted structure.
* **Not bit-reproducible.** Boltz's diffusion sampling is stochastic; nothing
  here seeds it. Treat the numbers like stage-1 Vina scores: stable in
  protocol, noisy in value.

Requirements: the ``boltz`` package (``pip install boltz``) with a CUDA GPU of
~14-20 GB VRAM for a typical target, and weights (downloaded automatically on
first run, or pre-fetched with ``boltz download_weights``). ``vcdd doctor``
reports whether the binary is findable and which version it is.

The interface deliberately mirrors :mod:`vcdd.refine.boltz.sab` so
:func:`vcdd.refine.stage2.run_stage2` dispatches on ``backend`` in one place:
``screen_molecules`` returns ``(rows, runs, rejected)`` where ``rows`` are the
same flattened columns ``sab.flatten_result`` produces, and ``runs`` are
``ScreenRun`` records so ``runs.json`` stays uniform.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from vcdd.refine.boltz.screen import ScreenRun, _as_float, _get
from vcdd.runtime.logging import get_logger
from vcdd.runtime.paths import ensure_dir

LOG = get_logger()

#: The model family this backend drives. Reported as the engine everywhere a
#: run surfaces, so a local run can never be mistaken for BoltzMol-1. The
#: version is deliberately omitted here - ``engine_version`` carries the real
#: ``boltz --version`` / importlib string (e.g. 2.2.1), which is better than a
#: label that could lie.
MODEL = "boltz-2 (local)"

#: Sequence id given to the ligand in the input YAML. It is put first so Boltz
#: names the outputs after it, which makes the per-run files predictable.
#: Kept short on purpose: boltz-2's ``Chain`` dtype stores ``name`` as `<U5`
#: (a fixed 5-char field, verified against boltz/data/types.py in 2.2.1), so a
#: longer id is silently truncated to 5 chars and the resulting numpy-key vs
#: dict-key mismatch crashes the parser with `KeyError`. Record ids in the
#: output are the input YAML *stem*, not this id, so there is no cost to
#: keeping it short.
LIGAND_ID = "LIG"

_SEMVER_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


# ------------------------------------------------------------------- binary
def _invoke_prefix(binary: str) -> list[str]:
    """How to run *binary*: directly, or through the current interpreter.

    ``boltz`` itself is a console script, but ``stage2.local_boltz.binary`` may
    point at a wrapper script without the exec bit (or a ``python -m`` style
    entry); launching a non-executable file raises PermissionError instead of
    a readable error, so detect that case and invoke it via ``sys.executable``.
    """
    if "/" in binary:
        p = Path(binary)
        if p.exists() and p.suffix == ".py" and not os.access(p, os.X_OK):
            return [sys.executable, binary]
    return [binary]


def find_binary(binary: str = "boltz") -> str | None:
    """The configured executable, or ``None`` if it cannot be found."""
    if "/" in binary:
        return binary if Path(binary).exists() else None
    return shutil.which(binary)


def version(binary: str = "boltz") -> str | None:
    """The semver of *binary*, or ``None`` if it cannot be established.

    The binary itself is probed first (``--version``), because that is
    authoritative for what vcdd will actually run. Some boltz builds - 2.2.x
    among them - do not have a ``--version`` flag; when boltz is installed in
    the interpreter running vcdd, ``importlib.metadata`` is then the ground
    truth. (If ``binary`` points at a boltz in *another* environment, the
    importlib fallback may report a different version than that binary runs;
    acceptable for a display value.)
    """
    exe = find_binary(binary)
    if exe is None:
        return None
    proc = None
    try:
        proc = subprocess.run(
            [*_invoke_prefix(exe), "--version"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except Exception:  # noqa: BLE001
        pass
    if proc is not None and proc.returncode == 0:
        text = (proc.stdout or "") + " " + (proc.stderr or "")
        match = _SEMVER_RE.search(text)
        if match:
            return match.group(1)
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        return pkg_version("boltz")
    except (PackageNotFoundError, Exception):  # noqa: BLE001
        return None


# ------------------------------------------------------------------ inputs
def build_yaml(
    protein_entities: Sequence[dict[str, Any]],
    smiles: str,
    ligand_id: str = LIGAND_ID,
    msa_files: dict[str, str] | None = None,
    affinity: bool = False,
) -> dict[str, Any]:
    """The ``boltz predict`` input document for one ligand.

    ``protein_entities`` are the receptor entities produced by
    :func:`vcdd.refine.boltz.target.build_target` - the same sequences and chain
    ids the API backends use, so all three backends predict against the same
    receptor. The document follows boltz's schema exactly (verified against
    ``boltz/data/parse/schema.py`` in 2.2.1): the entity type is the key of
    each ``sequences`` item and ``id`` lives *inside* it - a sibling ``id``
    raises ``Invalid entity type: id`` - and chain ids must be <= 5 chars
    (the ``Chain.name`` dtype is ``<U5``). The ligand is listed first so Boltz
    names the output files after it.

    ``msa_files`` maps chain id -> ``.a3m`` path, injected as the ``msa:``
    field of that protein entity. boltz 2.2 refuses to run MSA-less, so a chain
    without ``msa:`` needs ``--use_msa_server`` (see :func:`predict_command`).

    ``affinity`` adds the ``properties`` block that switches on Boltz-2's
    affinity head; it must name the ligand chain as the binder, and boltz
    accepts only one affinity ligand per document.
    """
    sequences: list[dict[str, Any]] = [
        {"ligand": {"id": ligand_id, "smiles": str(smiles)}}
    ]
    msa_files = dict(msa_files or {})
    used = {ligand_id}
    for ent in protein_entities:
        chain_ids = [str(c) for c in (ent.get("chain_ids") or [])]
        chain_id = chain_ids[0] if chain_ids else ent.get("id", "PROTEIN")
        if chain_id in used:
            chain_id = f"{chain_id}_2"
        used.add(chain_id)
        protein: dict[str, Any] = {"id": chain_id, "sequence": str(ent.get("value", ""))}
        if chain_id in msa_files:
            protein["msa"] = msa_files[chain_id]
        sequences.append({"protein": protein})
    doc: dict[str, Any] = {"version": 1, "sequences": sequences}
    if affinity:
        doc["properties"] = [{"affinity": {"binder": ligand_id}}]
    return doc


def build_yaml_text(
    protein_entities: Sequence[dict[str, Any]],
    smiles: str,
    ligand_id: str = LIGAND_ID,
    msa_files: dict[str, str] | None = None,
    affinity: bool = False,
) -> str:
    """Render :func:`build_yaml` as YAML text with safe SMILES quoting."""
    return yaml.safe_dump(build_yaml(protein_entities, smiles, ligand_id=ligand_id,
                                     msa_files=msa_files, affinity=affinity),
                          sort_keys=False, default_flow_style=False)


# ------------------------------------------------------------------ running
def predict_command(
    binary: str,
    input_yaml: str | Path,
    out_dir: str | Path,
    device: int,
    cfg: Any,
    num_samples: int = 1,
) -> list[str]:
    """The exact ``boltz predict`` argv for one molecule.

    Boltz takes a device *count* (``--devices``, default 1), not a list of GPU
    ids: :func:`run_predict` pins each subprocess to one physical GPU through
    ``CUDA_VISIBLE_DEVICES`` first, so the count is always 1 here. Samples are
    requested with ``--diffusion_samples`` (``--num_samples`` does not exist in
    boltz >= 2). ``--half``/``--no-half`` are not options in these builds
    either; ``half_precision`` is therefore ignored and should stay ``null``.
    """
    cmd = [
        *_invoke_prefix(binary), "predict", str(input_yaml),
        "--out_dir", str(out_dir),
        "--devices", "1",
        "--diffusion_samples", str(max(1, int(num_samples))),
    ]
    if cfg.cache_dir:
        cmd += ["--cache", str(cfg.cache_dir)]
    # boltz 2.2 refuses to run without per-chain MSA. A chain with a precomputed
    # ``msa:`` file (see build_yaml / protein_msa_files) needs no server; the
    # rest do, so emit the server flags unless every chain is covered. The
    # sequence IS sent to the configured server - a self-hosted ``msa_server_url``
    # keeps proprietary targets in-house.
    if cfg.use_msa_server:
        cmd.append("--use_msa_server")
        if cfg.msa_server_url:
            cmd += ["--msa_server_url", str(cfg.msa_server_url)]
        if cfg.msa_server_username:
            cmd += ["--msa_server_username", str(cfg.msa_server_username)]
        if cfg.msa_server_password:
            cmd += ["--msa_server_password", str(cfg.msa_server_password)]
    # The affinity head is switched on by the input document's ``properties``
    # block (see build_yaml); these flags only tune it.
    if getattr(cfg, "affinity", False):
        cmd += ["--diffusion_samples_affinity", str(max(1, int(cfg.affinity_samples)))]
        cmd += ["--sampling_steps_affinity", str(max(1, int(cfg.affinity_sampling_steps)))]
        if getattr(cfg, "affinity_mw_correction", False):
            cmd.append("--affinity_mw_correction")
    cmd += [str(a) for a in (cfg.extra_args or ())]
    return cmd


def run_predict(
    binary: str,
    protein_entities: Sequence[dict[str, Any]],
    smiles: str,
    work_dir: str | Path,
    device: int,
    cfg: Any,
    mol_id: str = "",
    num_samples: int = 1,
    keep_raw: bool = True,
    force: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One ``boltz predict`` subprocess; returns ``(result_row, meta)``.

    *work_dir* is a per-molecule directory: ``input.yaml`` plus ``out/`` (the
    Boltz ``--out_dir``). On success the confidence JSON and the predicted
    ligand structure are read back and flattened into the same row shape
    :func:`vcdd.refine.boltz.sab.flatten_result` produces. ``meta`` carries the
    ``ScreenRun`` bookkeeping and any error message.

    Stale outputs are a hazard here, not just a waste: ``boltz predict`` skips
    whatever already exists in its output directory ("Found some existing
    predictions ... skipping and running only the missing ones"), and this
    directory is keyed by molecule id alone. Re-running a molecule with a
    different SMILES, MSA or affinity setting would therefore report the *old*
    numbers as if they were fresh. So the rendered input is compared with the
    one on disk and ``out/`` is wiped whenever it changed - or whenever
    *force* says the caller wants a genuine re-prediction.
    """
    wd = ensure_dir(work_dir)
    yaml_path = wd / "input.yaml"
    out_dir = wd / "out"
    yaml_text = build_yaml_text(
        protein_entities, smiles, msa_files=dict(getattr(cfg, "protein_msa_files", None) or {}),
        affinity=bool(getattr(cfg, "affinity", False)),
    )
    previous = yaml_path.read_text() if yaml_path.exists() else None
    if out_dir.exists() and (force or previous != yaml_text):
        shutil.rmtree(out_dir, ignore_errors=True)
    yaml_path.write_text(yaml_text)
    out_dir = ensure_dir(out_dir)
    cmd = predict_command(binary, yaml_path, out_dir, device, cfg, num_samples=num_samples)
    # Boltz has no way to pick a GPU by id; ``CUDA_VISIBLE_DEVICES`` does.
    # Pinning the subprocess to *device* is what makes the round-robin over
    # ``cfg.devices`` honest: with ``max_concurrency <= len(devices)`` each GPU
    # hosts at most one predict at a time.
    subprocess_env = dict(os.environ)
    subprocess_env["CUDA_VISIBLE_DEVICES"] = str(int(device))
    try:
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=float(cfg.timeout_s), check=False,
                env=subprocess_env,
            )
        except subprocess.TimeoutExpired:
            meta = {"status": "failed", "error": {"code": "vcdd_predict_timeout",
                                                  "message": f"boltz predict exceeded {cfg.timeout_s}s"}}
            return None, meta
        except Exception as exc:  # noqa: BLE001 - subprocess plumbing
            meta = {"status": "failed", "error": {"code": "vcdd_predict_launch",
                                                  "message": f"{type(exc).__name__}: {exc}"}}
            return None, meta
        seconds = round(time.time() - t0, 1)

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            meta = {"status": "failed",
                    "error": {"code": "boltz_predict_nonzero",
                              "message": f"exit {proc.returncode}: {' | '.join(tail)}"}}
            return None, meta

        conf_path = _find_output(out_dir, "confidence_*_model_*.json")
        if conf_path is None:
            meta = {"status": "failed",
                    "error": {"code": "boltz_output_missing",
                              "message": f"no confidence_*_model_*.json under {out_dir}"}}
            return None, meta

        try:
            row = parse_confidence(conf_path, mol_id=mol_id, smiles=smiles, seconds=seconds)
        except Exception as exc:  # noqa: BLE001 - a broken JSON must not abort the chunk
            meta = {"status": "failed",
                    "error": {"code": "boltz_parse_failed",
                              "message": f"{type(exc).__name__}: {exc}"}}
            return None, meta

        # The affinity head writes its own JSON next to the confidence one. A
        # missing file is not an error when the head was not requested, and not
        # fatal when it was: the structural result is still usable.
        if getattr(cfg, "affinity", False):
            # boltz names this one ``affinity_<record.id>.json`` - unlike the
            # confidence file it carries no ``_model_<n>`` suffix, because the
            # affinity head is not a diffusion sample of the structure.
            aff_path = _find_output(out_dir, "affinity_*.json")
            if aff_path is not None:
                try:
                    row.update(parse_affinity(aff_path))
                except Exception as exc:  # noqa: BLE001 - keep the structural row
                    row["affinity_error"] = f"{type(exc).__name__}: {exc}"

        # The only structure boltz writes is the full complex (ligand = HETATM
        # chains); pose QC must know it is not a ligand-only file (hetatm_only).
        complex_cif = _find_output(out_dir, "*_model_*.cif")
        row["url_ligand_structure"] = str(complex_cif) if complex_cif else None
        row["local_work_dir"] = str(wd)

        meta = {
            "status": "succeeded",
            "engine": MODEL,
            "engine_version": version(binary),
            "livemode": True,
            "seconds": seconds,
        }
        return row, meta
    finally:
        if not keep_raw and wd.exists():
            # keep_raw_outputs=false deletes the per-molecule directory on every
            # path - success or failure - which is what the flag promises. The
            # parsed row already holds the numbers; stage 2 skips pose QC when
            # the structures are gone (see stage2.run_stage2).
            shutil.rmtree(wd, ignore_errors=True)


def _find_output(out_dir: Path, pattern: str) -> Path | None:
    """The first file under *out_dir* matching *pattern* (recursive glob).

    boltz-2.2's layout, verified against its source, is::

        <out>/boltz_results_<stem>/predictions/<record.id>/
            <record.id>_model_<n>.cif            # the whole complex
            confidence_<record.id>_model_<n>.json

    where ``record.id`` is the input YAML's stem and ``<n>`` ranks the
    diffusion samples (the best sample is always ``model_0``). The caller never
    hardcodes the stem or the id - one file per run is guaranteed because each
    run has its own out directory.
    """
    try:
        return sorted(out_dir.rglob(pattern))[0]
    except IndexError:
        return None


def read_local_file(path: str, timeout_s: float | None = None) -> str:
    """Fetch seam for pose QC: a local CIF path instead of a signed URL.

    :func:`vcdd.refine.boltz.poseqc.qc_poses` calls ``fetch(url, timeout_s)``
    on every row's ``url_ligand_structure``; the local backend stores the
    on-disk path there, so injecting this function reuses the whole pose-QC
    machinery unchanged.
    """
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ parsing
def parse_confidence(
    path: str | Path,
    mol_id: str,
    smiles: str,
    seconds: float | None = None,
) -> dict[str, Any]:
    """Flatten one Boltz-2 ``confidence_<id>_model_<n>.json`` into the sab row shape.

    The schema is boltz-2.2's (verified against its source): the headline number
    is ``confidence_score``, with ``iptm``/``ptm``/``ligand_iptm``/
    ``protein_iptm``/``complex_plddt``/``complex_iplddt``/``complex_pde``/
    ``complex_ipde`` as siblings. ``optimization_score`` is a BoltzMol-1
    (hosted-only) metric and is always ``None`` here; ADME and artifact URLs are
    API-only and also ``None``. Extra local-only columns are harmless: stage 2
    merges on ``mol_id`` and downstream code reads only ``BASE_COLUMNS``.
    """
    data = json.loads(Path(path).read_text())
    run_id = f"local_{mol_id}"

    def _num(key: str) -> float | None:
        value = _get(data, key)
        return _as_float(value)

    binding_confidence = _num("confidence_score")
    if binding_confidence is None:
        # A release that renames the head (or an older writer) must not lose it.
        binding_confidence = _num("binding_confidence")
    if binding_confidence is None:
        lig_iptm = _num("ligand_iptm")
        prot_iptm = _num("protein_iptm")
        if lig_iptm is not None and prot_iptm is not None:
            binding_confidence = round(0.5 * (lig_iptm + prot_iptm), 4)
    return {
        "boltz_run_id": run_id,
        "boltz_result_id": run_id,
        "mol_id": str(mol_id),
        "boltz_smiles": str(smiles),
        "created_at": "",
        "binding_confidence": binding_confidence,
        "optimization_score": None,
        "structure_confidence": None,
        "iptm": _num("iptm"),
        "ptm": _num("ptm"),
        "complex_plddt": _num("complex_plddt"),
        "complex_iplddt": _num("complex_iplddt"),
        "ligand_iptm": _num("ligand_iptm"),
        "protein_iptm": _num("protein_iptm"),
        "complex_pde": _num("complex_pde"),
        "complex_ipde": _num("complex_ipde"),
        "adme_lipophilicity": None,
        "adme_permeability": None,
        "adme_solubility": None,
        "url_structure": None,
        "url_ligand_structure": None,
        "url_archive": None,
        "n_samples": None,
        "local_seconds": seconds,
        "warnings": None,
        # Filled in by parse_affinity when the affinity head ran.
        "affinity_pred_value": None,
        "affinity_probability_binary": None,
        "affinity_error": None,
    }


def parse_affinity(path: str | Path) -> dict[str, Any]:
    """Read one Boltz-2 ``affinity_<id>_model_<n>.json``.

    Two numbers, with different meanings and different training data (boltz's
    own guidance): ``affinity_probability_binary`` is the probability that the
    ligand is a binder and is meant for separating binders from decoys, while
    ``affinity_pred_value`` is a potency value, ``log10(IC50)`` with IC50 in
    uM, meant for comparing binders with one another - which is what analog
    ranking needs. When boltz ran both affinity models it also writes
    ``..._value1/_value2``; those are kept for inspection but are not what
    downstream code reads.
    """
    data = json.loads(Path(path).read_text())
    out: dict[str, Any] = {
        "affinity_pred_value": _as_float(_get(data, "affinity_pred_value")),
        "affinity_probability_binary": _as_float(_get(data, "affinity_probability_binary")),
        "affinity_error": None,
    }
    for key in ("affinity_pred_value1", "affinity_probability_binary1",
                "affinity_pred_value2", "affinity_probability_binary2"):
        value = _as_float(_get(data, key))
        if value is not None:
            out[key] = value
    return out


# -------------------------------------------------------------------- driver
def check_msa_agreement(
    protein_entities: Sequence[dict[str, Any]],
    msa_files: Mapping[str, Any] | None,
    max_reported: int = 8,
) -> dict[str, Any]:
    """Refuse to hand Boltz a sequence that disagrees with its own MSA.

    A precomputed ``.a3m`` carries the query sequence as its first record. If the
    entity sequence differs from that query - even by one residue - Boltz aligns
    the wrong sequence against the alignment and the predicted fold collapses
    (measured on 7OTW 3701-4123: two substituted residues turned a 4.7 A receptor
    C-alpha RMSD into 21.8 A, for every one of 70 complexes). That failure is
    silent and expensive, so it is checked here, once per batch, before any GPU
    time is spent.

    Returns a report dict; raises ``ValueError`` on a mismatch so a mis-prepared
    receptor cannot quietly produce a screenful of misfolded complexes.
    """
    if not msa_files:
        return {"checked": 0, "mismatches": 0}
    report: dict[str, Any] = {"checked": 0, "mismatches": 0, "details": []}
    for ent in protein_entities:
        for chain_id in ent.get("chain_ids") or []:
            path = msa_files.get(chain_id)
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                LOG.warning("stage2/boltz-local: MSA for chain %s not found at %s", chain_id, p)
                continue
            query = ""
            for line in p.read_text().splitlines():
                if line.startswith(">"):
                    continue
                query = line.strip().replace("-", "").upper()
                break
            if not query:
                continue
            report["checked"] += 1
            entity_seq = str(ent.get("value", "")).upper()
            if query == entity_seq:
                continue
            diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(query, entity_seq)) if a != b]
            report["mismatches"] += 1
            detail = {
                "chain": chain_id, "msa": str(p), "msa_len": len(query),
                "entity_len": len(entity_seq),
                "differences": [f"pos{i}:{a}->{b}" for i, a, b in diff[:max_reported]],
                "n_differences": len(diff),
            }
            report["details"].append(detail)
            LOG.error(
                "stage2/boltz-local: chain %s sequence disagrees with its MSA %s "
                "(%d difference(s): %s). Boltz would fold the wrong sequence; refusing to run. "
                "Usually the receptor PDBQT was written by a tool that renames residues whose "
                "side chain the crystal never modelled (see vcdd.receptor.prepare."
                "restore_residue_names).",
                chain_id, p.name, len(diff), detail["differences"],
            )
    if report["mismatches"]:
        raise ValueError(
            f"{report['mismatches']} chain(s) disagree with their MSA: "
            f"{report['details'][0]['differences']} - fix the receptor preparation "
            f"rather than spending GPU time on a misfolded screen"
        )
    return report


def screen_molecules(
    cfg: Any,
    protein_entities: Sequence[dict[str, Any]],
    mol_ids: Sequence[str],
    smiles: Sequence[str],
    out_dir: str | Path,
    chunk_index: int = 0,
    num_samples: int = 1,
    max_concurrency: int | None = None,
    keep_raw: bool | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[ScreenRun], list[dict[str, Any]]]:
    """Run one ``boltz predict`` per molecule and collect the results.

    Returns ``(rows, runs, rejected)`` like :func:`vcdd.refine.boltz.sab.
    screen_molecules`. Each predict gets its own directory under *out_dir*
    named after the molecule, so a failed molecule never touches its
    neighbours' outputs and a re-run can be resumed per molecule by the
    stage-2 chunk ledger.

    Concurrency is one process per worker, round-robin over ``cfg.devices``:
    keep ``max_concurrency`` <= ``len(devices)`` so no GPU is oversubscribed.
    """
    out_root = ensure_dir(out_dir)
    concurrency = max(1, int(max_concurrency if max_concurrency is not None else cfg.max_concurrency))
    devices = [int(d) for d in (cfg.devices or [0])]
    keep_raw = bool(cfg.keep_raw_outputs) if keep_raw is None else bool(keep_raw)
    if cfg.half_precision is not None:
        LOG.warning(
            "stage2/boltz-local: this boltz build has no --half/--no-half flag; "
            "local_boltz.half_precision is ignored (leave it null)."
        )

    rejected: list[dict[str, Any]] = []
    check_msa_agreement(protein_entities, getattr(cfg, "protein_msa_files", None))
    work: list[tuple[int, str, str]] = []
    for i, (mid, smi) in enumerate(zip(mol_ids, smiles)):
        if _smiles_parses(smi):
            work.append((i, str(mid), str(smi)))
        else:
            rejected.append({"mol_id": str(mid), "smiles": str(smi), "boltz_run_id": None,
                             "reason": "invalid_smiles", "run_status": "not_submitted"})
    if rejected:
        LOG.warning("stage2/boltz-local: %d molecule(s) skipped (see rejected.parquet)",
                    len(rejected))

    rows: list[dict[str, Any]] = []
    runs: list[ScreenRun] = []

    def _one(item: tuple[int, str, str]) -> tuple[ScreenRun, dict[str, Any] | None]:
        idx, mid, smi = item
        device = devices[idx % len(devices)]
        safe = _safe_name(mid)
        row, meta = run_predict(
            cfg.binary, protein_entities, smi, out_root / safe, device, cfg,
            mol_id=mid, num_samples=num_samples, keep_raw=keep_raw, force=force,
        )
        run = ScreenRun(
            run_id=f"local_{safe}",
            chunk_index=chunk_index,
            n_molecules=1,
            idempotency_key=f"local_{safe}",
            estimated_cost_usd=None,
            status=str(meta.get("status", "failed")),
            engine=meta.get("engine"),
            engine_version=meta.get("engine_version"),
            livemode=meta.get("livemode", True),
            error=meta.get("error"),
            mol_ids=[mid],
        )
        return run, row

    if work:
        n_workers = max(1, min(int(concurrency), len(work)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for run, row in pool.map(_one, work):
                runs.append(run)
                if run.status == "succeeded" and row is not None:
                    rows.append(row)
                else:
                    rejected.append({
                        "mol_id": run.mol_ids[0],
                        "smiles": next((s for _, m, s in work if m == run.mol_ids[0]), ""),
                        "boltz_run_id": run.run_id,
                        "reason": (run.error or {}).get("code") or run.status,
                        "run_status": run.status,
                    })
    return rows, runs, rejected


# ------------------------------------------------------------------- helpers
def _smiles_parses(smiles: str) -> bool:
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(str(smiles)) is not None
    except Exception:  # noqa: BLE001
        return False


def _safe_name(mol_id: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(mol_id))[:120] or "mol"
