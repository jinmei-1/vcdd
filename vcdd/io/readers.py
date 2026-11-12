"""Streaming readers for the four supported library formats.

Supported: ``.sdf`` / ``.sdf.gz``, ``.smi`` / ``.ism`` / ``.txt``,
``.csv`` / ``.tsv`` (optionally ``.gz``), ``.parquet``.

Every reader yields :class:`RawRecord` objects lazily so a 10^6-molecule library
never has to fit in memory. Malformed records are yielded with ``error`` set
rather than dropped, so the pipeline can account for every input row.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SMILES_COLUMN_CANDIDATES = (
    "smiles", "smile", "canonical_smiles", "canonicalsmiles", "isomeric_smiles",
    "structure", "mol", "molecule", "smi",
)
ID_COLUMN_CANDIDATES = (
    "id", "mol_id", "molecule_id", "compound_id", "cid", "name", "title",
    "zinc_id", "chembl_id", "catalog_id", "idnumber", "identifier",
)

_BOM = "\ufeff"


@dataclass
class RawRecord:
    """One input row before standardisation."""

    index: int
    smiles: str | None = None
    mol_id: str | None = None
    props: dict[str, Any] = field(default_factory=dict)
    molblock: str | None = None
    error: str | None = None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lstrip(_BOM).lower())


def detect_columns(
    columns: list[str],
    smiles_column: str | None = None,
    id_column: str | None = None,
) -> tuple[str | None, str | None]:
    """Pick the SMILES and ID columns from a header.

    Explicit names win. Otherwise the first column whose normalised name matches
    a known alias is used; ties are broken by the order of the candidate lists,
    not by column order, so ``canonical_smiles`` is preferred over a stray
    ``structure`` column.
    """
    normalized = {_norm(c): c for c in columns}

    smi: str | None = None
    if smiles_column:
        if smiles_column in columns:
            smi = smiles_column
        elif _norm(smiles_column) in normalized:
            smi = normalized[_norm(smiles_column)]
        else:
            raise KeyError(f"smiles_column {smiles_column!r} not in {columns}")
    else:
        for cand in SMILES_COLUMN_CANDIDATES:
            key = _norm(cand)
            if key in normalized:
                smi = normalized[key]
                break

    mid: str | None = None
    if id_column:
        if id_column in columns:
            mid = id_column
        elif _norm(id_column) in normalized:
            mid = normalized[_norm(id_column)]
        else:
            raise KeyError(f"id_column {id_column!r} not in {columns}")
    else:
        for cand in ID_COLUMN_CANDIDATES:
            key = _norm(cand)
            if key in normalized and normalized[key] != smi:
                mid = normalized[key]
                break
    return smi, mid


def _open_text(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")  # type: ignore[return-value]
    return open(path, "r", encoding="utf-8", errors="replace")


def sniff_format(path: str | os.PathLike[str]) -> str:
    """Return one of ``sdf|smi|csv|parquet`` for *path*."""
    name = str(path).lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith((".sdf", ".sd", ".mol")):
        return "sdf"
    if name.endswith((".smi", ".ism", ".smiles")):
        return "smi"
    if name.endswith((".csv", ".tsv", ".txt")):
        return "csv"
    if name.endswith((".parquet", ".pq")):
        return "parquet"
    raise ValueError(f"unrecognised library format: {path}")


# ------------------------------------------------------------------------- SDF
def read_sdf(path: str | os.PathLike[str], keep_molblock: bool = False) -> Iterator[RawRecord]:
    """Stream an SDF, preserving property tags.

    Uses RDKit's ``ForwardSDMolSupplier`` (constant memory). A record RDKit
    cannot parse still yields a ``RawRecord`` with ``error`` set so the count of
    inputs always reconciles.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    p = Path(path)
    fh = gzip.open(p, "rb") if str(p).endswith(".gz") else open(p, "rb")
    try:
        supplier = Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=False)
        for i, mol in enumerate(supplier):
            if mol is None:
                yield RawRecord(index=i, error="rdkit_parse_failed")
                continue
            props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            mol_id = name or props.get("ID") or props.get("id")
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception as exc:  # noqa: BLE001
                yield RawRecord(index=i, mol_id=mol_id, props=props, error=f"smiles_failed:{exc}")
                continue
            yield RawRecord(
                index=i,
                smiles=smi,
                mol_id=mol_id,
                props=props,
                molblock=Chem.MolToMolBlock(mol) if keep_molblock else None,
            )
    finally:
        fh.close()


# ------------------------------------------------------------------------- SMI
def read_smi(path: str | os.PathLike[str]) -> Iterator[RawRecord]:
    """Stream a whitespace- or tab-delimited SMILES file.

    Accepts ``SMILES`` or ``SMILES <id>`` per line; skips blanks and ``#``
    comments; tolerates a header line whose first token is not a SMILES.
    """
    p = Path(path)
    with _open_text(p) as fh:
        idx = 0
        for lineno, line in enumerate(fh):
            line = line.strip().lstrip(_BOM)
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            if lineno == 0 and _norm(smi) in SMILES_COLUMN_CANDIDATES:
                continue  # header row
            mol_id = parts[1] if len(parts) > 1 else None
            yield RawRecord(index=idx, smiles=smi, mol_id=mol_id)
            idx += 1


# ------------------------------------------------------------------------- CSV
def read_csv(
    path: str | os.PathLike[str],
    smiles_column: str | None = None,
    id_column: str | None = None,
) -> Iterator[RawRecord]:
    """Stream a CSV/TSV with column auto-detection and BOM tolerance."""
    p = Path(path)
    with _open_text(p) as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = "\t" if str(p).lower().endswith((".tsv", ".tab")) else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        raw_fields = list(reader.fieldnames or [])
        if not raw_fields:
            return
        # Strip a UTF-8 BOM from the first header cell.
        clean_fields = [f.lstrip(_BOM) if f else f for f in raw_fields]
        smi_col, mid_col = detect_columns(clean_fields, smiles_column, id_column)
        if smi_col is None:
            raise KeyError(f"no SMILES column found in {clean_fields}; set library.smiles_column")
        raw_to_clean = dict(zip(raw_fields, clean_fields))
        for i, row in enumerate(reader):
            clean = {raw_to_clean.get(k, k): v for k, v in row.items() if k is not None}
            smi = (clean.get(smi_col) or "").strip()
            mol_id = (clean.get(mid_col) or "").strip() if mid_col else None
            props = {k: v for k, v in clean.items() if k not in (smi_col, mid_col) and v not in (None, "")}
            if not smi:
                yield RawRecord(index=i, mol_id=mol_id or None, props=props, error="empty_smiles")
                continue
            yield RawRecord(index=i, smiles=smi, mol_id=mol_id or None, props=props)


# --------------------------------------------------------------------- PARQUET
def read_parquet(
    path: str | os.PathLike[str],
    smiles_column: str | None = None,
    id_column: str | None = None,
    batch_size: int = 20_000,
) -> Iterator[RawRecord]:
    """Stream a Parquet library in row-group batches."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    columns = list(pf.schema_arrow.names)
    smi_col, mid_col = detect_columns(columns, smiles_column, id_column)
    if smi_col is None:
        raise KeyError(f"no SMILES column found in {columns}; set library.smiles_column")

    idx = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        table = batch.to_pydict()
        n = len(table[smi_col])
        other = [c for c in table if c not in (smi_col, mid_col)]
        for j in range(n):
            smi = table[smi_col][j]
            mol_id = str(table[mid_col][j]) if mid_col and table[mid_col][j] is not None else None
            props = {c: table[c][j] for c in other if table[c][j] is not None}
            if not smi:
                yield RawRecord(index=idx, mol_id=mol_id, props=props, error="empty_smiles")
            else:
                yield RawRecord(index=idx, smiles=str(smi), mol_id=mol_id, props=props)
            idx += 1


# -------------------------------------------------------------------- dispatch
def read_library(
    path: str | os.PathLike[str],
    smiles_column: str | None = None,
    id_column: str | None = None,
    keep_molblock: bool = False,
) -> Iterator[RawRecord]:
    """Format-dispatching streaming reader for any supported library file."""
    fmt = sniff_format(path)
    if fmt == "sdf":
        yield from read_sdf(path, keep_molblock=keep_molblock)
    elif fmt == "smi":
        yield from read_smi(path)
    elif fmt == "csv":
        yield from read_csv(path, smiles_column, id_column)
    elif fmt == "parquet":
        yield from read_parquet(path, smiles_column, id_column)
    else:  # pragma: no cover - sniff_format already raised
        raise ValueError(fmt)
