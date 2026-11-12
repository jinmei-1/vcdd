"""Filesystem policy helpers.

Rationale
---------
The run directory may live on an S3-backed FUSE mount. Those mounts do **not**
support POSIX append or random-access rewrites: SQLite databases, HDF5 files and
incremental ``open(..., "a")`` writers either fail or silently corrupt. Every
random-access or streaming writer in ``vcdd`` therefore writes to a local
scratch directory first and is copied to its final destination in one shot.

Public contract
---------------
``scratch_for(path)``      local scratch twin of a final path
``atomic_write_bytes``     write bytes via tmp-file + ``os.replace``
``atomic_write_text``      same for text
``stage_out(local, final)``copy a finished local file to its final location
``LocalStagedFile``        context manager: yields a local path, copies on exit
``is_network_fs(path)``    heuristic for "this path cannot take random writes"
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

#: Prefixes that are known S3/FUSE-backed on the Biomni platform. Anything under
#: these is treated as "no random-access writes allowed".
NETWORK_FS_PREFIXES = ("/mnt/results", "/mnt/shared-workspace", "/mnt/user-uploads", "/mnt/user-results")

#: Local scratch root. Overridable for tests / unusual deployments.
SCRATCH_ROOT_ENV = "VCDD_SCRATCH_ROOT"


def default_scratch_root() -> Path:
    root = os.environ.get(SCRATCH_ROOT_ENV)
    if root:
        return Path(root)
    for candidate in ("/workspace", "/tmp"):
        if Path(candidate).is_dir() and os.access(candidate, os.W_OK):
            return Path(candidate) / ".vcdd_scratch"
    return Path(tempfile.gettempdir()) / "vcdd_scratch"


def is_network_fs(path: str | os.PathLike[str]) -> bool:
    """True when *path* lives on a mount that cannot take random-access writes."""
    resolved = str(Path(path).absolute())
    return any(resolved == p or resolved.startswith(p + "/") for p in NETWORK_FS_PREFIXES)


def scratch_for(final_path: str | os.PathLike[str]) -> Path:
    """Return a deterministic local scratch twin for *final_path*.

    The twin keeps the original file name (so extension-sniffing libraries still
    work) under a hash of the absolute destination, which keeps concurrent runs
    from colliding.
    """
    final = Path(final_path).absolute()
    digest = hashlib.sha1(str(final.parent).encode()).hexdigest()[:16]
    scratch_dir = default_scratch_root() / digest
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir / final.name


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write *data* to *path* atomically.

    On a network mount ``os.replace`` across the FUSE layer is still the safest
    primitive available: we write a sibling temp file and rename it.
    """
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def stage_out(local_path: str | os.PathLike[str], final_path: str | os.PathLike[str]) -> Path:
    """Copy a finished local file to its final destination."""
    local = Path(local_path)
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if local.resolve() == final.resolve():
        return final
    shutil.copyfile(local, final)
    return final


@contextmanager
def LocalStagedFile(final_path: str | os.PathLike[str], keep_local: bool = False) -> Iterator[Path]:
    """Yield a local path to write to; copy it to *final_path* on clean exit.

    Use for every format that needs random access (sqlite, hdf5, zip-based
    formats) or incremental appends. If the destination is already local the
    file is written in place and no copy happens.
    """
    final = Path(final_path)
    if not is_network_fs(final):
        final.parent.mkdir(parents=True, exist_ok=True)
        yield final
        return
    local = scratch_for(final)
    try:
        yield local
        stage_out(local, final)
    finally:
        if not keep_local and local.exists():
            try:
                local.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass


def writable_db_path(final_path: str | os.PathLike[str]) -> Path:
    """Return a path safe for a SQLite database backing *final_path*.

    SQLite needs random access and file locking, which FUSE mounts do not give
    us. When the requested location is on a network mount we keep the live
    database on local scratch and only mirror it out on close (see
    :class:`vcdd.runtime.ledger.RunLedger`).
    """
    final = Path(final_path)
    if is_network_fs(final):
        return scratch_for(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    return final
