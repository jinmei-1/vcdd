"""Build the offline resource bundle used by the ``vcdd`` Stage 3 cell layer.

The script downloads a small number of public, redistributable reference files
and converts them into compact tables under ``vcdd/cell/resources``.  Running it
is optional: every consumer degrades gracefully when a resource is missing, but
the zero-weight Tier-1 backends (``lincs_readacross`` and ``target_knockdown``)
and the protein-activity transform become available only once the bundle
exists.

Sources
-------
L1000 landmark genes
    GEO GSE92742 supplementary ``gene_info`` table (Broad LINCS).
PROGENy pathway footprints
    OmniPath ``annotations`` endpoint, ``resources=PROGENy``.
CollecTRI TF-target network
    OmniPath ``interactions`` endpoint, ``datasets=collectri``.
MSigDB Hallmark 2020
    Enrichr gene-set library ``MSigDB_Hallmark_2020``.
LINCS L1000 consensus signatures
    Enrichr libraries ``LINCS_L1000_Chem_Pert_Consensus_Sigs`` and
    ``LINCS_L1000_CRISPR_KO_Consensus_Sigs``.
Compound structures
    Broad Drug Repurposing Hub molecule table (local datalake mirror, or a
    user-supplied parquet/CSV via ``--repurposing-hub``).

Usage
-----
    python scripts/build_cell_resources.py --out vcdd/cell/resources
    python scripts/build_cell_resources.py --only progeny collectri
    python scripts/build_cell_resources.py --check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# --------------------------------------------------------------------------- #
# Source definitions
# --------------------------------------------------------------------------- #

L1000_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_gene_info.txt.gz"
)
OMNIPATH_ANNOTATIONS = "https://omnipathdb.org/annotations"
OMNIPATH_INTERACTIONS = "https://omnipathdb.org/interactions"
ENRICHR_GMT = "https://maayanlab.cloud/Enrichr/geneSetLibrary"

HALLMARK_LIBRARY = "MSigDB_Hallmark_2020"
CHEM_LIBRARY = "LINCS_L1000_Chem_Pert_Consensus_Sigs"
CRISPR_LIBRARY = "LINCS_L1000_CRISPR_KO_Consensus_Sigs"

#: Public source of the compound structures used to attach SMILES to the LINCS
#: consensus terms. The Broad Drug Repurposing Hub (Corsello et al. 2020) is
#: distributed at <https://repo-hub.broadinstitute.org/repurposing>.
#:
#: This used to be a hard-coded absolute path on the machine where the bundle was
#: first built. That path is meaningless to anyone else and leaked internal
#: infrastructure into the shipped MANIFEST.json, so it is now an environment
#: variable with no default: set VCDD_REPURPOSING_HUB to a local parquet/CSV
#: (``--repurposing-hub`` also works) or the structures are simply left unjoined,
#: which is a visible, non-fatal degradation.
HUB_PARQUET_ENV = "VCDD_REPURPOSING_HUB"
HUB_PUBLIC_SOURCE = "https://repo-hub.broadinstitute.org/repurposing"

#: Number of highest-confidence footprint genes retained per PROGENy pathway.
PROGENY_TOP_N = 100

HTTP_TIMEOUT = 300


@dataclass
class BuildResult:
    """Outcome of a single resource build step."""

    name: str
    path: Path | None = None
    rows: int = 0
    ok: bool = False
    message: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _requests():
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - requests is a hard dep
        raise SystemExit("The 'requests' package is required to build resources.") from exc
    return requests


def _get(url: str, params: dict | None = None) -> bytes:
    requests = _requests()
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def _parse_gmt(text: str) -> list[tuple[str, str, list[str]]]:
    """Return ``(term, description, genes)`` triples from GMT text."""
    out: list[tuple[str, str, list[str]]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term, desc = parts[0], parts[1]
        genes = [g.split(",")[0].strip() for g in parts[2:] if g.strip()]
        out.append((term, desc, [g for g in genes if g]))
    return out


def _split_direction(term: str) -> tuple[str, str] | None:
    """Split an Enrichr consensus term such as ``"TP53 Up"``."""
    stem, _, tail = term.rpartition(" ")
    tail = tail.strip().lower()
    if not stem or tail not in {"up", "down"}:
        return None
    return stem.strip(), tail


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_landmark(out_dir: Path) -> BuildResult:
    """L1000 landmark / BING gene annotation."""
    raw = _get(L1000_GENE_INFO_URL)
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    keep = df.rename(
        columns={
            "pr_gene_id": "entrez_id",
            "pr_gene_symbol": "gene_symbol",
            "pr_gene_title": "gene_title",
            "pr_is_lm": "is_landmark",
            "pr_is_bing": "is_bing",
        }
    )
    keep["is_landmark"] = keep["is_landmark"].astype(int)
    keep["is_bing"] = keep["is_bing"].astype(int)
    keep = keep[keep["is_bing"] == 1][
        ["entrez_id", "gene_symbol", "gene_title", "is_landmark", "is_bing"]
    ]
    keep = keep.sort_values(["is_landmark", "gene_symbol"], ascending=[False, True])
    path = out_dir / "l1000_genes.tsv"
    _write_tsv(keep, path)
    n_lm = int(keep["is_landmark"].sum())
    return BuildResult(
        "l1000_genes",
        path,
        len(keep),
        True,
        f"{n_lm} landmark / {len(keep)} BING genes",
        {"source": L1000_GENE_INFO_URL, "sha256": _sha256(raw), "n_landmark": n_lm},
    )


def build_progeny(out_dir: Path) -> BuildResult:
    """PROGENy pathway footprints, pivoted from the OmniPath long format.

    OmniPath returns one row per (record, attribute) pair; ``record_id`` groups
    the ``weight`` / ``p_value`` / ``pathway`` attributes of a single
    gene-pathway assignment.  ``genesymbols=yes`` is *not* accepted by the
    annotations endpoint, so the symbol comes from the ``genesymbol`` column.
    """
    raw = _get(OMNIPATH_ANNOTATIONS, {"resources": "PROGENy", "format": "tsv"})
    long = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
    wide = long.pivot_table(
        index=["record_id", "genesymbol", "uniprot"],
        columns="label",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    missing = {"pathway", "weight", "p_value"} - set(wide.columns)
    if missing:
        raise RuntimeError(f"PROGENy response missing attributes: {sorted(missing)}")
    wide = wide.rename(columns={"genesymbol": "gene_symbol"})
    wide["weight"] = pd.to_numeric(wide["weight"], errors="coerce")
    wide["p_value"] = pd.to_numeric(wide["p_value"], errors="coerce")
    wide = wide.dropna(subset=["pathway", "gene_symbol", "weight", "p_value"])
    wide = wide.sort_values(["pathway", "p_value"])
    top = wide.groupby("pathway", group_keys=False).head(PROGENY_TOP_N)
    top = top[["pathway", "gene_symbol", "uniprot", "weight", "p_value"]].reset_index(drop=True)
    path = out_dir / "progeny.tsv"
    _write_tsv(top, path)
    return BuildResult(
        "progeny",
        path,
        len(top),
        True,
        f"{top['pathway'].nunique()} pathways x top {PROGENY_TOP_N} genes",
        {
            "source": OMNIPATH_ANNOTATIONS + "?resources=PROGENy",
            "sha256": _sha256(raw),
            "pathways": sorted(top["pathway"].unique().tolist()),
            "top_n": PROGENY_TOP_N,
        },
    )


def build_collectri(out_dir: Path) -> BuildResult:
    """CollecTRI signed TF-target regulon."""
    raw = _get(
        OMNIPATH_INTERACTIONS,
        {
            "datasets": "collectri",
            "genesymbols": "yes",
            "format": "tsv",
            "fields": "sources",
        },
    )
    df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)

    def _flag(col: str) -> pd.Series:
        # OmniPath serialises these as the strings "True"/"False" (or 1/0).
        if col not in df:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    stim, inhib = _flag("is_stimulation"), _flag("is_inhibition")
    # CollecTRI encodes sign as a pair of booleans; both-set or neither-set
    # records are treated as activating, matching decoupleR's convention.
    mor = pd.Series(1, index=df.index, dtype=int)
    mor[inhib & ~stim] = -1
    # Select before renaming: OmniPath already has uniprot-keyed `source` /
    # `target` columns that would collide with the gene-symbol names.
    out = pd.DataFrame(
        {
            "tf": df["source_genesymbol"].astype(str).str.strip(),
            "target": df["target_genesymbol"].astype(str).str.strip(),
            "mor": mor,
        }
    ).dropna()
    out = out[(out["tf"] != "") & (out["target"] != "")]
    # Drop protein-complex regulators; the transform operates on single symbols.
    out = out[~out["tf"].str.contains("COMPLEX", case=False, na=False)]
    out = out.drop_duplicates(subset=["tf", "target"]).sort_values(["tf", "target"])
    path = out_dir / "collectri.tsv"
    _write_tsv(out.reset_index(drop=True), path)
    return BuildResult(
        "collectri",
        path,
        len(out),
        True,
        f"{out['tf'].nunique()} TFs / {len(out)} edges",
        {
            "source": OMNIPATH_INTERACTIONS + "?datasets=collectri",
            "sha256": _sha256(raw),
            "n_tf": int(out["tf"].nunique()),
        },
    )


def build_hallmark(out_dir: Path) -> BuildResult:
    """MSigDB Hallmark 2020 gene sets (Enrichr mirror)."""
    raw = _get(ENRICHR_GMT, {"mode": "text", "libraryName": HALLMARK_LIBRARY})
    sets = _parse_gmt(raw.decode("utf-8", "replace"))
    rows = [
        {"gene_set": term, "gene_symbol": gene}
        for term, _desc, genes in sets
        for gene in genes
    ]
    df = pd.DataFrame(rows).drop_duplicates()
    path = out_dir / "hallmark.tsv"
    _write_tsv(df, path)
    return BuildResult(
        "hallmark",
        path,
        len(df),
        True,
        f"{df['gene_set'].nunique()} hallmark sets",
        {
            "source": f"{ENRICHR_GMT}?libraryName={HALLMARK_LIBRARY}",
            "sha256": _sha256(raw),
        },
    )


def _consensus_from_gmt(raw: bytes) -> pd.DataFrame:
    """Collapse an Enrichr ``<name> Up`` / ``<name> Down`` library to one row per name."""
    sets = _parse_gmt(raw.decode("utf-8", "replace"))
    acc: dict[str, dict[str, list[str]]] = {}
    for term, _desc, genes in sets:
        split = _split_direction(term)
        if split is None:
            continue
        name, direction = split
        acc.setdefault(name, {"up": [], "down": []})[direction] = genes
    rows = [
        {
            "name": name,
            "up_genes": sorted(set(d["up"])),
            "down_genes": sorted(set(d["down"])),
            "n_up": len(set(d["up"])),
            "n_down": len(set(d["down"])),
        }
        for name, d in sorted(acc.items())
    ]
    return pd.DataFrame(rows)


def build_lincs_chem(out_dir: Path, hub_path: str | None) -> BuildResult:
    """Compound consensus signatures joined to Repurposing-Hub structures."""
    raw = _get(ENRICHR_GMT, {"mode": "text", "libraryName": CHEM_LIBRARY})
    df = _consensus_from_gmt(raw)
    if df.empty:
        raise RuntimeError("no Up/Down consensus terms parsed from the chemical library")
    df["name_key"] = df["name"].str.lower().str.strip()

    smiles_map: dict[str, str] = {}
    hub_source = None
    candidate = hub_path or os.environ.get(HUB_PARQUET_ENV)
    if candidate and Path(candidate).exists():
        hub = (
            pd.read_parquet(candidate)
            if candidate.endswith(".parquet")
            else pd.read_csv(candidate)
        )
        name_col = next(
            (c for c in ("pert_iname", "name", "compound_name") if c in hub.columns), None
        )
        smi_col = next(
            (c for c in ("smiles", "SMILES", "canonical_smiles") if c in hub.columns), None
        )
        if name_col and smi_col:
            sub = hub[[name_col, smi_col]].dropna()
            sub = sub[sub[smi_col].astype(str).str.len() > 0]
            sub["key"] = sub[name_col].astype(str).str.lower().str.strip()
            # Keep the first structure per name; the hub lists one row per vendor lot.
            smiles_map = sub.drop_duplicates("key").set_index("key")[smi_col].to_dict()
            hub_source = candidate

    df["smiles"] = df["name_key"].map(smiles_map)
    n_mapped = int(df["smiles"].notna().sum())
    path = out_dir / "lincs_chem_consensus.parquet"
    _write_parquet(
        df[["name", "name_key", "smiles", "up_genes", "down_genes", "n_up", "n_down"]], path
    )
    return BuildResult(
        "lincs_chem_consensus",
        path,
        len(df),
        True,
        f"{len(df)} compounds, {n_mapped} with SMILES",
        {
            "source": f"{ENRICHR_GMT}?libraryName={CHEM_LIBRARY}",
            "sha256": _sha256(raw),
            # Record the public provenance and the file's own name, never the
            # absolute path it happened to be read from.
            "structure_source": HUB_PUBLIC_SOURCE if hub_source else None,
            "structure_file": Path(hub_source).name if hub_source else None,
            "n_with_smiles": n_mapped,
        },
    )


def build_lincs_crispr(out_dir: Path) -> BuildResult:
    """CRISPR knockout consensus signatures, keyed by knocked-out gene."""
    raw = _get(ENRICHR_GMT, {"mode": "text", "libraryName": CRISPR_LIBRARY})
    df = _consensus_from_gmt(raw)
    if df.empty:
        raise RuntimeError("no Up/Down consensus terms parsed from the CRISPR library")
    df = df.rename(columns={"name": "gene"})
    df["gene"] = df["gene"].str.upper().str.strip()
    path = out_dir / "lincs_crispr_ko_consensus.parquet"
    _write_parquet(df[["gene", "up_genes", "down_genes", "n_up", "n_down"]], path)
    return BuildResult(
        "lincs_crispr_ko_consensus",
        path,
        len(df),
        True,
        f"{len(df)} knocked-out genes",
        {"source": f"{ENRICHR_GMT}?libraryName={CRISPR_LIBRARY}", "sha256": _sha256(raw)},
    )


BUILDERS: dict[str, Callable[..., BuildResult]] = {
    "l1000_genes": build_landmark,
    "progeny": build_progeny,
    "collectri": build_collectri,
    "hallmark": build_hallmark,
    "lincs_chem": build_lincs_chem,
    "lincs_crispr": build_lincs_crispr,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "vcdd" / "cell" / "resources"


def run(names: Iterable[str], out_dir: Path, hub_path: str | None) -> list[BuildResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BuildResult] = []
    for name in names:
        builder = BUILDERS[name]
        try:
            res = (
                builder(out_dir, hub_path) if name == "lincs_chem" else builder(out_dir)
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            res = BuildResult(name, None, 0, False, f"{type(exc).__name__}: {exc}")
        results.append(res)
        status = "ok " if res.ok else "FAIL"
        print(f"[{status}] {name:24s} {res.message}", flush=True)
    return results


def write_manifest(out_dir: Path, results: list[BuildResult]) -> Path:
    manifest_path = out_dir / "MANIFEST.json"
    previous: dict = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            previous = {}
    entries = dict(previous.get("resources", {}))
    for res in results:
        if not res.ok or res.path is None:
            continue
        extra = dict(res.extra)
        # ``sha256`` here is the digest of the *downloaded source*, which is NOT
        # the digest of the file we write (the payload is re-serialised). Shipping
        # it under the bare name ``sha256`` next to the written file's size invited
        # the reader to treat it as a file-integrity check, which it never was.
        # Name both explicitly so either can be verified.
        if "sha256" in extra:
            extra["source_sha256"] = extra.pop("sha256")
        entries[res.name] = {
            "file": res.path.name,
            "rows": res.rows,
            "file_bytes": res.path.stat().st_size,
            "file_sha256": _sha256(res.path.read_bytes()),
            "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }
    payload = {
        "schema": 1,
        "generator": "scripts/build_cell_resources.py",
        "resources": entries,
        "notice": (
            "Derived from public sources; original licences apply. "
            "See LICENSE-NOTICES.md."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="output resource directory")
    ap.add_argument(
        "--only", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS)
    )
    ap.add_argument("--repurposing-hub", default=None, help="parquet/CSV with name+SMILES")
    ap.add_argument("--check", action="store_true", help="report bundle status and exit")
    args = ap.parse_args(argv)

    out_dir = args.out or default_out_dir()
    if args.check:
        manifest = out_dir / "MANIFEST.json"
        if not manifest.exists():
            print(f"no manifest at {manifest}")
            return 1
        data = json.loads(manifest.read_text())
        for name, meta in sorted(data.get("resources", {}).items()):
            exists = (out_dir / meta["file"]).exists()
            print(
                f"{'ok ' if exists else 'MISSING'} {name:24s} "
                f"{meta['rows']:>8} rows  {meta['bytes'] / 1e6:6.2f} MB  {meta['file']}"
            )
        return 0

    results = run(args.only, out_dir, args.repurposing_hub)
    path = write_manifest(out_dir, results)
    total = sum(r.path.stat().st_size for r in results if r.ok and r.path)
    print(f"\nmanifest: {path}")
    print(f"bundle written this run: {total / 1e6:.2f} MB")
    failed = [r.name for r in results if not r.ok]
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
