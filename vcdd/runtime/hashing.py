"""Stable hashing used for run-ledger keys and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json(obj: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no whitespace jitter)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any, length: int = 16) -> str:
    """Hash an arbitrary JSON-serialisable object."""
    return hashlib.sha256(stable_json(obj).encode()).hexdigest()[:length]


def hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def hash_file(path: str | Path, length: int = 16, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (never loads the whole file)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:length]


def param_hash(params: dict[str, Any], ignore: tuple[str, ...] = ()) -> str:
    """Hash of the parameters that actually change a computation's result.

    Keys in *ignore* (e.g. ``n_workers``, ``verbose``) are excluded so that
    re-running with different parallelism still hits the resume cache.
    """
    filtered = {k: v for k, v in sorted(params.items()) if k not in ignore}
    return hash_obj(filtered)
