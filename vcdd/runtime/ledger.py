"""SQLite work ledger that makes every stage resumable and idempotent.

A *unit* of work is identified by ``(stage, unit_id, input_hash, param_hash)``.
Before computing, a driver asks the ledger whether that exact tuple already
completed; if so it reuses the recorded artifact path and skips the work. The
key includes the parameter hash so changing e.g. ``thread`` correctly
invalidates the cache rather than silently reusing stale results.

The database lives on local disk even when the run directory is on an S3/FUSE
mount (SQLite needs random access + locking); it is mirrored to the run
directory on ``close()`` and on demand via ``sync()``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vcdd.runtime.paths import is_network_fs, stage_out, writable_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    stage       TEXT NOT NULL,
    unit_id     TEXT NOT NULL,
    input_hash  TEXT NOT NULL,
    param_hash  TEXT NOT NULL,
    status      TEXT NOT NULL,
    artifact    TEXT,
    n_items     INTEGER,
    error       TEXT,
    started_at  REAL,
    finished_at REAL,
    PRIMARY KEY (stage, unit_id, input_hash, param_hash)
);
CREATE INDEX IF NOT EXISTS idx_units_stage_status ON units(stage, status);

CREATE TABLE IF NOT EXISTS events (
    ts     REAL NOT NULL,
    stage  TEXT,
    kind   TEXT,
    payload TEXT
);
"""

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"


@dataclass(frozen=True)
class UnitKey:
    stage: str
    unit_id: str
    input_hash: str
    param_hash: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.stage, self.unit_id, self.input_hash, self.param_hash)


class RunLedger:
    """Small, dependency-free resume ledger."""

    def __init__(self, db_path: str | Path):
        self.final_path = Path(db_path)
        self.local_path = writable_db_path(self.final_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        # If a previous run left a mirrored db in the run dir, restore it first.
        if is_network_fs(self.final_path) and self.final_path.exists() and not self.local_path.exists():
            stage_out(self.final_path, self.local_path)
        self.conn = sqlite3.connect(str(self.local_path), timeout=60.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ query
    def status(self, key: UnitKey) -> tuple[str | None, str | None]:
        row = self.conn.execute(
            "SELECT status, artifact FROM units WHERE stage=? AND unit_id=? AND input_hash=? AND param_hash=?",
            key.as_tuple(),
        ).fetchone()
        return (None, None) if row is None else (row[0], row[1])

    def is_done(self, key: UnitKey) -> bool:
        status, artifact = self.status(key)
        if status != STATUS_DONE:
            return False
        if artifact and not Path(artifact).exists():
            # Artifact was deleted out from under us: treat as not done.
            return False
        return True

    def completed_units(self, stage: str) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT unit_id, artifact, n_items FROM units WHERE stage=? AND status=? ORDER BY unit_id",
            (stage, STATUS_DONE),
        )
        return [{"unit_id": r[0], "artifact": r[1], "n_items": r[2]} for r in cur.fetchall()]

    # ----------------------------------------------------------------- update
    def mark_running(self, key: UnitKey) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO units(stage,unit_id,input_hash,param_hash,status,artifact,n_items,error,started_at,finished_at)"
            " VALUES (?,?,?,?,?,NULL,NULL,NULL,?,NULL)",
            (*key.as_tuple(), STATUS_RUNNING, time.time()),
        )

    def mark_done(self, key: UnitKey, artifact: str | Path | None = None, n_items: int | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO units(stage,unit_id,input_hash,param_hash,status,artifact,n_items,error,started_at,finished_at)"
            " VALUES (?,?,?,?,?,?,?,NULL,COALESCE((SELECT started_at FROM units WHERE stage=? AND unit_id=? AND input_hash=? AND param_hash=?), ?),?)",
            (
                *key.as_tuple(),
                STATUS_DONE,
                str(artifact) if artifact is not None else None,
                n_items,
                *key.as_tuple(),
                time.time(),
                time.time(),
            ),
        )

    def mark_failed(self, key: UnitKey, error: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO units(stage,unit_id,input_hash,param_hash,status,artifact,n_items,error,started_at,finished_at)"
            " VALUES (?,?,?,?,?,NULL,NULL,?,NULL,?)",
            (*key.as_tuple(), STATUS_FAILED, error[:4000], time.time()),
        )

    def log_event(self, stage: str, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events(ts,stage,kind,payload) VALUES (?,?,?,?)",
            (time.time(), stage, kind, json.dumps(payload or {}, default=str)),
        )

    def events(self, stage: str | None = None) -> list[dict[str, Any]]:
        if stage:
            cur = self.conn.execute("SELECT ts,stage,kind,payload FROM events WHERE stage=? ORDER BY ts", (stage,))
        else:
            cur = self.conn.execute("SELECT ts,stage,kind,payload FROM events ORDER BY ts")
        return [{"ts": r[0], "stage": r[1], "kind": r[2], "payload": json.loads(r[3])} for r in cur.fetchall()]

    def reset_stage(self, stage: str) -> int:
        cur = self.conn.execute("DELETE FROM units WHERE stage=?", (stage,))
        return cur.rowcount

    def summary(self) -> dict[str, dict[str, int]]:
        cur = self.conn.execute("SELECT stage, status, COUNT(*) FROM units GROUP BY stage, status")
        out: dict[str, dict[str, int]] = {}
        for stage, status, n in cur.fetchall():
            out.setdefault(stage, {})[status] = n
        return out

    # ---------------------------------------------------------------- plumbing
    def sync(self) -> None:
        """Mirror the local database out to the run directory."""
        if is_network_fs(self.final_path):
            try:
                self.conn.execute("PRAGMA wal_checkpoint(FULL)")
            except sqlite3.Error:  # pragma: no cover
                pass
            stage_out(self.local_path, self.final_path)

    def close(self) -> None:
        try:
            self.conn.commit()
        except sqlite3.Error:  # pragma: no cover
            pass
        self.sync()
        self.conn.close()

    def __enter__(self) -> "RunLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def filter_pending(ledger: RunLedger, keys: Iterable[UnitKey], force: bool = False) -> list[UnitKey]:
    """Return the subset of *keys* that still need computing."""
    if force:
        return list(keys)
    return [k for k in keys if not ledger.is_done(k)]
