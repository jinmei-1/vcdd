"""Load, merge, validate and persist run configurations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from vcdd.config.schema import RunConfig
from vcdd.runtime.paths import atomic_write_text

TEMPLATE_PATH = Path(__file__).with_name("template.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (override wins on scalars)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["stage2.top_n=50", "cell.cell_line=A375"]`` into a nested dict.

    Values are parsed as YAML scalars so ``true``, ``3``, ``[A,B]`` all work.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cursor = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


# --------------------------------------------------------------- 0.1.0 -> 0.2.0
#: Keys removed in 0.2.0, mapped to what to do instead. The schema is
#: ``extra="forbid"``, so a 0.1.0 config fails validation anyway - but with
#: pydantic's generic "extra inputs are not permitted", which tells a user
#: nothing about *why* their exhaustiveness setting vanished. These messages do.
REMOVED_KEYS: dict[str, str] = {
    # ---- stage 1: CPU-Vina flags Vina-GPU 2.1 does not accept --------------
    "stage1.exhaustiveness":
        "Vina-GPU 2.1 has no --exhaustiveness. Search effort is set with "
        "stage1.thread (parallel Monte-Carlo threads, default 8000) and "
        "stage1.search_depth (depth per thread; null = adaptive).",
    "stage1.cpu_per_ligand":
        "removed - the GPU engine does not take --cpu. CPU workers now only prepare "
        "ligands and are sized by the top-level n_workers.",
    "stage1.num_modes":
        "removed - Vina-GPU 2.1 does not accept --num_modes. It writes all poses it kept.",
    "stage1.energy_range":
        "removed - Vina-GPU 2.1 does not accept --energy_range.",
    # ---- stage 2: no longer a docking stage --------------------------------
    "stage2.engine":
        "stage 2 no longer docks. Choose the Boltz endpoint with "
        "stage2.backend: library_screen | structure_and_binding.",
    "stage2.exhaustiveness":
        "stage 2 no longer docks; there is nothing to set exhaustiveness on.",
    "stage2.num_modes":
        "stage 2 no longer docks.",
    "stage2.energy_range":
        "stage 2 no longer docks.",
    "stage2.cpu_per_ligand":
        "stage 2 is an API call; concurrency is bounded by the client, not by CPU count.",
    "stage2.seed":
        "the Boltz API takes no seed. Re-attaching to a run is done through the "
        "idempotency key, which vcdd derives from the molecule set.",
    "stage2.n_conformers":
        "removed - Boltz predicts the complex from SMILES; there is no conformer "
        "enumeration step to configure.",
    "stage2.enumerate_tautomers":
        "removed - Boltz takes one SMILES per molecule.",
    "stage2.max_tautomers":
        "removed - Boltz takes one SMILES per molecule.",
    "stage2.timeout_s":
        "replaced by stage2.request_timeout_s (per HTTP request) and "
        "stage2.poll_interval_s (how often the run is polled).",
    "stage2.rescorers":
        "the rescoring plugin subsystem was removed. Stage 2 is Boltz only.",
    "stage2.consensus":
        "replaced by stage2.fusion: boltz_only | rank_fusion | gate.",
    "stage2.min_scorer_agreement":
        "there is only one scorer now, so there is no inter-scorer agreement to "
        "threshold. Use stage2.gate_threshold on binding_confidence instead.",
    "stage2.pose_qc":
        "always on, and cheap: the predicted ligand centroid is checked against the "
        "docking box for every molecule with a downloaded structure.",
    "stage2.use_ensemble":
        "removed - there is no docking step to run over a receptor ensemble.",
    "stage2.ensemble_reduce":
        "removed - see stage2.use_ensemble.",
}


def check_migration(raw: dict[str, Any]) -> None:
    """Raise a targeted error when a config still uses 0.1.0 keys."""
    problems: list[str] = []
    for dotted, advice in REMOVED_KEYS.items():
        section, _, key = dotted.partition(".")
        block = raw.get(section)
        if isinstance(block, dict) and key in block:
            problems.append(f"  {dotted}: {advice}")
    if (raw.get("stage4") or {}).get("redock_mode") == "stage2":
        problems.append(
            "  stage4.redock_mode=stage2: stage 2 is a Boltz prediction, not a docking run, "
            "so it has no docking parameters to reuse. Use 'stage1' or 'none'."
        )
    if problems:
        raise ValueError(
            "this config uses keys that were removed in vcdd 0.2.0 "
            "(stage 1 is Vina-GPU only; stage 2 is Boltz only):\n"
            + "\n".join(problems)
            + "\n\nSee MIGRATION-0.2.0.md, or start from `vcdd config`."
        )


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Load a YAML config, apply overrides, and validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping, got {type(loaded).__name__}")
        raw = loaded
    if overrides:
        raw = deep_merge(raw, overrides)
    check_migration(raw)
    return RunConfig(**raw)


def save_resolved(cfg: RunConfig, run_dir: str | Path | None = None) -> Path:
    """Write the fully resolved config into the run directory."""
    target = Path(run_dir) if run_dir is not None else cfg.run_dir
    target.mkdir(parents=True, exist_ok=True)
    path = target / "config.resolved.yaml"
    atomic_write_text(path, yaml.safe_dump(cfg.resolved_dict(), sort_keys=False, default_flow_style=False))
    return path


def template_text() -> str:
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text()
    # Fallback: dump defaults if the annotated template is missing.
    return yaml.safe_dump(RunConfig().resolved_dict(), sort_keys=False)
