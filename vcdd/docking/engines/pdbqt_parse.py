"""Parsing helpers for Vina-family PDBQT output.

Split out of the removed ``vina_like`` module so the Vina-GPU engine and the
tests can share exactly one implementation of "what is the affinity in this
file". The regexes are carried over verbatim from 0.1.0: Vina-GPU 2.1 writes the
same ``REMARK VINA RESULT`` header as CPU Vina, which is the one piece of the
CPU code path that remains correct.
"""

from __future__ import annotations

import re

#: ``REMARK VINA RESULT:  -8.123  0.000  0.000`` - one per MODEL, best first.
_AFFINITY_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+\.?\d*)")

#: Emitted on stdout by ``--score_only`` / ``--local_only``, which write no
#: output file at all. Without this fallback those modes look like a failure.
_STDOUT_AFFINITY_RE = re.compile(
    r"(?:Estimated Free Energy of Binding|Affinity)\s*:\s*(-?\d+\.?\d*)"
)


def parse_pdbqt_scores(pdbqt_text: str) -> list[float]:
    """Every ``REMARK VINA RESULT`` affinity in a docked PDBQT, in file order."""
    return [float(m) for m in _AFFINITY_RE.findall(pdbqt_text or "")]


def parse_stdout_affinity(stdout: str) -> list[float]:
    """Affinities printed on stdout, for the modes that write no output file."""
    return [float(m) for m in _STDOUT_AFFINITY_RE.findall(stdout or "")]


def best_score(scores: list[float]) -> float | None:
    """Most negative affinity, or ``None`` for an empty list.

    Vina writes modes best-first, but nothing in the file format guarantees it,
    and a truncated file can leave them out of order. Taking the minimum is one
    comparison and removes the assumption.
    """
    return min(scores) if scores else None
