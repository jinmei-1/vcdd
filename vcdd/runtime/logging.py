"""Console + JSON-lines logging.

Two sinks:

* a human-readable console stream (rich when available, plain otherwise);
* a machine-readable ``logs/vcdd.jsonl`` in the run directory so a long
  screening run can be audited or parsed after the fact.

The JSONL sink buffers to local scratch and is flushed to the run directory,
because S3/FUSE mounts do not support append.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from vcdd.runtime.paths import is_network_fs, scratch_for, stage_out

_LOGGER_NAME = "vcdd"


class JsonlSink:
    """Append-only JSONL writer that is safe on network filesystems."""

    def __init__(self, final_path: str | Path, flush_every: int = 50):
        self.final_path = Path(final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path = scratch_for(self.final_path) if is_network_fs(self.final_path) else self.final_path
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.local_path, "a", encoding="utf-8")
        self._n = 0
        self._flush_every = flush_every

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        if self.local_path != self.final_path:
            stage_out(self.local_path, self.final_path)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()


class _JsonlHandler(logging.Handler):
    def __init__(self, sink: JsonlSink):
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin adapter
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "vcdd_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        try:
            self.sink.write(payload)
        except Exception:
            pass


_sink: JsonlSink | None = None


def setup_logging(run_dir: str | Path | None = None, level: str = "INFO", quiet: bool = False) -> logging.Logger:
    """Configure the ``vcdd`` logger. Idempotent."""
    global _sink
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    if not quiet:
        console_handler: logging.Handler
        try:
            from rich.logging import RichHandler  # type: ignore

            console_handler = RichHandler(rich_tracebacks=False, show_path=False, markup=False)
            console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        except Exception:  # pragma: no cover
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        logger.addHandler(console_handler)

    if run_dir is not None:
        if _sink is not None:
            _sink.close()
        _sink = JsonlSink(Path(run_dir) / "logs" / "vcdd.jsonl")
        logger.addHandler(_JsonlHandler(_sink))
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def log_event(msg: str, level: str = "INFO", **fields: Any) -> None:
    """Log a message with structured fields attached to the JSONL sink."""
    logger = get_logger()
    logger.log(getattr(logging, level.upper(), logging.INFO), msg, extra={"vcdd_extra": fields})


def flush_logs() -> None:
    if _sink is not None:
        _sink.flush()


def close_logs() -> None:
    global _sink
    if _sink is not None:
        _sink.close()
        _sink = None


def worker_quiet() -> None:  # pragma: no cover - used inside worker processes
    """Silence the logger inside forked workers to avoid interleaved output."""
    logging.getLogger(_LOGGER_NAME).handlers.clear()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
