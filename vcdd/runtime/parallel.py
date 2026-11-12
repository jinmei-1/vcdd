"""Process-pool helpers with deterministic ordering and shard support.

Docking is embarrassingly parallel: one ligand per core. The helpers here keep
three properties that matter for a screening pipeline:

* **Deterministic output order** - results are re-sorted by input index so a
  ranking never depends on scheduler timing.
* **Failure isolation** - one bad ligand raises inside its worker and is
  recorded as a failed item, it does not abort the shard.
* **Shardable** - ``shard_indices`` lets N independent machines each take a
  disjoint slice of the same library with no coordination.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ItemResult:
    index: int
    ok: bool
    value: Any = None
    error: str | None = None


def resolve_workers(requested: int | None) -> int:
    """Resolve a worker count. ``None``/0/negative means "all available cores"."""
    if requested and requested > 0:
        return int(requested)
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def shard_indices(n_items: int, shard: int, nshards: int) -> list[int]:
    """Round-robin assignment of item indices to shard *shard* of *nshards*.

    Round-robin (rather than contiguous blocks) keeps per-shard runtime even
    when a library is sorted by size or complexity.
    """
    if nshards <= 1:
        return list(range(n_items))
    if not (0 <= shard < nshards):
        raise ValueError(f"shard must be in [0,{nshards}), got {shard}")
    return list(range(shard, n_items, nshards))


def _safe_call(fn: Callable[[T], R], index: int, item: T) -> ItemResult:
    try:
        return ItemResult(index=index, ok=True, value=fn(item))
    except Exception as exc:  # noqa: BLE001 - deliberate: isolate per-item failures
        return ItemResult(index=index, ok=False, error=f"{type(exc).__name__}: {exc}")


def map_items(
    fn: Callable[[T], R],
    items: Sequence[T],
    n_workers: int | None = None,
    chunk_report: Callable[[int, int], None] | None = None,
    ordered: bool = True,
) -> list[ItemResult]:
    """Map *fn* over *items*, isolating failures, preserving input order.

    ``n_workers <= 1`` runs in-process, which keeps tracebacks readable and is
    what the unit tests use.
    """
    workers = resolve_workers(n_workers)
    total = len(items)
    results: list[ItemResult] = []

    if workers <= 1 or total <= 1:
        for i, item in enumerate(items):
            results.append(_safe_call(fn, i, item))
            if chunk_report and (i + 1) % 25 == 0:
                chunk_report(i + 1, total)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_safe_call, fn, i, item): i for i, item in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if chunk_report and done % 25 == 0:
                chunk_report(done, total)

    if ordered:
        results.sort(key=lambda r: r.index)
    return results


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of at most *size* items from *iterable* (streaming)."""
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
