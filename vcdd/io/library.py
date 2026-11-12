"""``prep-library``: read -> standardise -> filter -> dedup -> shard.

Output contract (all Parquet, all under ``<run>/library/``):

* ``shards/shard_00000.parquet`` - accepted molecules, columns
  ``mol_id, smiles, inchikey, mw, logp, hbd, hba, tpsa, rotatable, heavy_atoms,
  rings, fraction_csp3, formal_charge, source_index, shard``
* ``index.parquet``   - one row per shard: path, n_molecules
* ``rejected.parquet``- every rejected input with a machine-readable reason

Memory profile is flat: the reader streams, molecules are written out shard by
shard, and only the dedup key set grows (a ~27-byte string per unique molecule,
i.e. roughly 60 MB for 10^6 compounds - acceptable, and swappable for a Bloom
filter if a much larger library is ever needed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from vcdd.io.filters import apply_filters
from vcdd.io.readers import read_library
from vcdd.io.standardize import dedup_key, standardize_smiles
from vcdd.runtime.logging import get_logger, log_event
from vcdd.runtime.paths import ensure_dir

LOG = get_logger()

MOLECULE_COLUMNS = [
    "mol_id", "smiles", "inchikey", "mw", "logp", "hbd", "hba", "tpsa",
    "rotatable", "heavy_atoms", "rings", "fraction_csp3", "formal_charge",
    "source_index", "shard",
]


@dataclass
class PrepStats:
    n_input: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    n_shards: int = 0
    seconds: float = 0.0

    def reject(self, reason: str) -> None:
        self.n_rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_shards": self.n_shards,
            "seconds": round(self.seconds, 2),
            "top_rejection_reasons": dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])[:15]),
        }


def _make_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:08d}"


def iter_prepared(
    path: str | Path,
    lib_cfg: Any,
    id_prefix: str = "VCDD",
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Yield ``(accepted_row, rejected_row)`` pairs; exactly one is non-None."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    seen: set[str] = set()
    counter = 0

    for rec in read_library(
        path,
        smiles_column=lib_cfg.smiles_column,
        id_column=lib_cfg.id_column,
    ):
        if rec.error:
            yield None, {"mol_id": rec.mol_id, "smiles": rec.smiles, "reason": rec.error, "source_index": rec.index}
            continue

        if lib_cfg.standardize:
            std = standardize_smiles(
                rec.smiles or "",
                desalt=lib_cfg.desalt,
                neutralize=lib_cfg.neutralize,
                canonical_tautomer=lib_cfg.canonical_tautomer,
                compute_inchikey=lib_cfg.dedup_by.startswith("inchikey"),
            )
            if not std.ok:
                yield None, {
                    "mol_id": rec.mol_id,
                    "smiles": rec.smiles,
                    "reason": std.error or "standardization_failed",
                    "source_index": rec.index,
                }
                continue
            smiles = std.smiles
            inchikey = std.inchikey
        else:
            smiles = rec.smiles
            inchikey = None
            std = None

        key = dedup_key(std, lib_cfg.dedup_by) if std is not None else smiles
        if lib_cfg.dedup_by != "none" and key is not None:
            if key in seen:
                yield None, {
                    "mol_id": rec.mol_id,
                    "smiles": smiles,
                    "reason": "duplicate",
                    "source_index": rec.index,
                }
                continue
            seen.add(key)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            yield None, {"mol_id": rec.mol_id, "smiles": smiles, "reason": "post_std_unparseable", "source_index": rec.index}
            continue

        verdict = apply_filters(mol, lib_cfg.filters)
        if not verdict.passed:
            yield None, {
                "mol_id": rec.mol_id,
                "smiles": smiles,
                "reason": verdict.reason or "filtered",
                "source_index": rec.index,
            }
            continue

        counter += 1
        row: dict[str, Any] = {
            "mol_id": rec.mol_id or _make_id(id_prefix, counter),
            "smiles": smiles,
            "inchikey": inchikey,
            "source_index": rec.index,
        }
        row.update(verdict.props or {})
        yield row, None


def prepare_library(
    path: str | Path,
    out_dir: str | Path,
    lib_cfg: Any,
    id_prefix: str = "VCDD",
) -> PrepStats:
    """Run the full library preparation and write the shard contract."""
    t0 = time.time()
    out = ensure_dir(out_dir)
    shard_dir = ensure_dir(out / "shards")

    stats = PrepStats()
    buffer: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    shard_idx = 0
    max_mols = lib_cfg.max_molecules

    def flush_shard() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        for row in buffer:
            row["shard"] = shard_idx
        df = pd.DataFrame(buffer)
        for col in MOLECULE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[MOLECULE_COLUMNS]
        shard_path = shard_dir / f"shard_{shard_idx:05d}.parquet"
        df.to_parquet(shard_path, index=False)
        shard_rows.append({"shard": shard_idx, "path": str(shard_path), "n_molecules": len(df)})
        stats.n_shards += 1
        shard_idx += 1
        buffer = []

    for accepted, rejected in iter_prepared(path, lib_cfg, id_prefix=id_prefix):
        stats.n_input += 1
        if rejected is not None:
            stats.reject(str(rejected["reason"]).split(":")[0])
            if len(rejects) < 2_000_000:
                rejects.append(rejected)
            continue
        assert accepted is not None
        buffer.append(accepted)
        stats.n_accepted += 1
        if len(buffer) >= lib_cfg.shard_size:
            flush_shard()
        if max_mols is not None and stats.n_accepted >= max_mols:
            break

    flush_shard()

    index_df = pd.DataFrame(shard_rows, columns=["shard", "path", "n_molecules"])
    index_df.to_parquet(out / "index.parquet", index=False)
    rejected_df = pd.DataFrame(rejects, columns=["mol_id", "smiles", "reason", "source_index"])
    rejected_df.to_parquet(out / "rejected.parquet", index=False)

    stats.seconds = time.time() - t0
    log_event(
        f"prep-library: {stats.n_accepted}/{stats.n_input} accepted into {stats.n_shards} shard(s)",
        stage="library",
        **stats.as_dict(),
    )
    return stats


def load_shard_index(library_dir: str | Path) -> pd.DataFrame:
    idx = Path(library_dir) / "index.parquet"
    if not idx.exists():
        raise FileNotFoundError(f"library index not found: {idx}. Run `vcdd prep-library` first.")
    return pd.read_parquet(idx)


def iter_shards(library_dir: str | Path) -> Iterator[tuple[int, pd.DataFrame]]:
    for _, row in load_shard_index(library_dir).iterrows():
        yield int(row["shard"]), pd.read_parquet(row["path"])


def load_all_molecules(library_dir: str | Path) -> pd.DataFrame:
    """Concatenate every shard. Only for small libraries / reporting."""
    frames = [df for _, df in iter_shards(library_dir)]
    if not frames:
        return pd.DataFrame(columns=MOLECULE_COLUMNS)
    return pd.concat(frames, ignore_index=True)
