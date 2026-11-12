"""Run provenance manifest.

Everything needed to explain *how* a result was produced and *what may be
redistributed*: package version, git hash, tool versions, seeds, input file
hashes, and the per-source licence notices for any data the run touched.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vcdd.runtime.hashing import hash_file
from vcdd.runtime.paths import atomic_write_text
from vcdd.version import __version__

#: Licence notices for data sources the pipeline can touch. Keyed by the source
#: identifier used in configs. ``needs_commercial_review`` marks sources whose
#: terms are not obviously compatible with commercial redistribution.
LICENSE_NOTICES: dict[str, dict[str, Any]] = {
    "chembl": {
        "name": "ChEMBL",
        "license": "CC BY-SA 3.0",
        "obligations": [
            "Attribute ChEMBL and record the release version.",
            "Share-alike: derived datasets distributed publicly inherit CC BY-SA 3.0.",
            "Preserve CHEMBL IDs so records remain traceable.",
        ],
        "needs_commercial_review": False,
    },
    "lincs": {
        "name": "LINCS L1000 / SigCom LINCS",
        "license": "NIH LINCS Consortium data release policy",
        "obligations": [
            "Cite the LINCS Consortium and the SigCom LINCS resource.",
            "Signature values are read-across evidence, not measurements of the query compound.",
        ],
        "needs_commercial_review": True,
    },
    "broad_repurposing_hub": {
        "name": "Broad Drug Repurposing Hub",
        "license": "CC BY 4.0 with a research-use clause",
        "obligations": ["Attribute the Broad Institute.", "Check the research-use clause before commercial use."],
        "needs_commercial_review": True,
    },
    "pdb": {
        "name": "RCSB Protein Data Bank",
        "license": "CC0 1.0",
        "obligations": ["No restrictions; citation of the depositing study is good practice."],
        "needs_commercial_review": False,
    },
    "msigdb_hallmark": {
        "name": "MSigDB Hallmark collection",
        "license": "CC BY 4.0 (Hallmark only)",
        "obligations": [
            "Only the Hallmark collection is used. KEGG-derived gene sets are deliberately excluded "
            "because their licence is not compatible with unrestricted redistribution.",
        ],
        "needs_commercial_review": False,
    },
    "progeny": {
        "name": "PROGENy pathway footprints",
        "license": "Apache-2.0 (saezlab)",
        "obligations": ["Cite Schubert et al. 2018."],
        "needs_commercial_review": False,
    },
    "collectri": {
        "name": "CollecTRI TF-target regulons",
        "license": "GPL-3.0 / source-specific (saezlab)",
        "obligations": ["Cite Muller-Dott et al. 2023.", "Some constituent resources are academic-use only."],
        "needs_commercial_review": True,
    },
}


def _git_hash(package_root: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _binary_version(name: str, args: list[str]) -> str | None:
    exe = shutil.which(name)
    if exe is None:
        return None
    try:
        out = subprocess.run([exe] + args, capture_output=True, text=True, timeout=20)
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0] if text else exe
    except Exception:  # pragma: no cover
        return exe


def tool_versions() -> dict[str, str | None]:
    """Versions of every external tool the pipeline may shell out to.

    Vina-GPU 2.1 has no ``--version`` flag, so the three variants are recorded
    as resolved paths. That is the only provenance available for them, which is
    also why stage-1 scores are not bit-reproducible: the upstream binaries
    expose no seed either, so the manifest records ``stage1_deterministic:
    false`` rather than pretending otherwise.
    """
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vina": _binary_version("vina", ["--version"]),
        "obabel": _binary_version("obabel", ["-V"]),
        "prepare_receptor": "present" if shutil.which("prepare_receptor") else None,
    }

    try:  # stage 1: the Vina-GPU 2.1 family, located but not version-queryable
        from vcdd.docking.engines.vina_gpu import VinaGPUEngine

        for key, path in VinaGPUEngine.variant_binaries().items():
            versions[f"vina_gpu:{key}"] = path
    except Exception:  # noqa: BLE001 - provenance must never break a run
        pass
    versions["stage1_deterministic"] = "false"

    try:  # stage 2: the hosted Boltz API, so only the SDK is local
        from vcdd.refine.boltz.client import sdk_version

        versions["boltz-api"] = sdk_version()
    except Exception:  # noqa: BLE001
        versions["boltz-api"] = None

    for mod in ("rdkit", "numpy", "pandas", "scipy", "sklearn", "meeko", "gemmi",
                "openmm", "torch", "crem", "httpx"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            versions[mod] = None
    return versions


class Manifest:
    """Accumulates provenance and writes ``manifest.json``."""

    def __init__(self, run_dir: str | Path, config: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "manifest.json"
        self.data: dict[str, Any] = {
            "vcdd_version": __version__,
            "git_hash": _git_hash(Path(__file__).resolve().parents[2]),
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool_versions": tool_versions(),
            "config": config or {},
            "inputs": {},
            "stages": {},
            "licenses": {},
            "warnings": [],
        }
        if self.path.exists():
            try:
                prior = json.loads(self.path.read_text())
                prior.update({k: v for k, v in self.data.items() if k in ("vcdd_version", "git_hash", "tool_versions")})
                prior["config"] = self.data["config"] or prior.get("config", {})
                self.data = prior
            except Exception:  # pragma: no cover
                pass

    def add_input(self, role: str, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        p = Path(path)
        rec: dict[str, Any] = {"path": str(p), "exists": p.exists()}
        if p.exists() and p.is_file():
            rec["bytes"] = p.stat().st_size
            rec["sha256_16"] = hash_file(p)
        if extra:
            rec.update(extra)
        self.data.setdefault("inputs", {})[role] = rec

    def add_stage(self, stage: str, info: dict[str, Any]) -> None:
        entry = self.data.setdefault("stages", {}).setdefault(stage, {})
        entry.update(info)
        entry["updated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def add_license(self, source_key: str) -> None:
        notice = LICENSE_NOTICES.get(source_key)
        if notice:
            self.data.setdefault("licenses", {})[source_key] = notice

    def warn(self, message: str, **fields: Any) -> None:
        self.data.setdefault("warnings", []).append({"message": message, **fields})

    def save(self) -> Path:
        atomic_write_text(self.path, json.dumps(self.data, indent=2, default=str))
        return self.path
