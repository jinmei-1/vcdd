#!/usr/bin/env python
"""Build a relative-ranking benchmark for the DNA-PKcs scoring functions.

The campaign's central claim is relative: "our molecules are better than the
positive control (AZD7648)". A relative claim is only as good as the ranking
function's ability to place a known potent inhibitor above known weak ones, so
this script assembles a set where the answer is known from experiment:

* actives   - measured DNA-PKcs IC50 from ChEMBL (target CHEMBL3142), spanning
              sub-nanomolar to micromolar;
* inactives - compounds measured in the *same* assays with IC50 > 10 uM, i.e.
              real experimental negatives rather than computational decoys;
* AZD7648   - the campaign's positive control, added explicitly.

Output: benchmark/known_inhibitors.csv (measured values, for scoring the
scores) and benchmark/known_inhibitors.smi (the same set as a vcdd library, so
stage 1 and stage 2 run exactly as they did for the campaign).
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmark"
TARGET = "CHEMBL3142"  # DNA-dependent protein kinase catalytic subunit
API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
AZD7648 = "CC1=CC2=NC=NN2C=C1NC3=NC=C4C(=N3)N(C(=O)N4C)C5CCOCC5"

UNIT_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "M": 1e9, "mM": 1e6}


def fetch(offset: int, limit: int = 1000) -> dict:
    query = urllib.parse.urlencode({
        "target_chembl_id": TARGET,
        "standard_type": "IC50",
        "limit": limit,
        "offset": offset,
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=120) as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001 - transient API failures
            print(f"  retry {attempt + 1} at offset {offset}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ChEMBL fetch failed at offset {offset}")


def collect() -> list[dict]:
    rows: list[dict] = []
    offset, total = 0, None
    while total is None or offset < total:
        page = fetch(offset)
        total = page["page_meta"]["total_count"]
        for act in page["activities"]:
            units = act.get("standard_units")
            value = act.get("standard_value")
            if not units or value in (None, "") or units not in UNIT_TO_NM:
                continue
            if float(value) <= 0:  # a handful of records carry a zero value
                continue
            if act.get("standard_flag") != 1:  # keep one curated value per record
                continue
            if not act.get("canonical_smiles"):
                continue
            rows.append({
                "chembl_id": act["molecule_chembl_id"],
                "name": act.get("molecule_pref_name") or "",
                "smiles": act["canonical_smiles"],
                "relation": act.get("standard_relation") or "=",
                "value_nM": float(value) * UNIT_TO_NM[units],
                "assay": (act.get("assay_description") or "")[:120],
                "doc_year": act.get("document_year"),
            })
        offset += 1000
        print(f"  fetched {offset}/{total}")
    return rows


def classify(records: list[dict]) -> dict | None:
    """Assign a ground-truth tier from all measurements of one molecule.

    Only unambiguous records count. ChEMBL holds single-concentration
    depositions that appear as both ``IC50 > 30 nM`` and ``IC50 < 30 nM`` for
    the same compound; those are neither a potency nor a clean negative and are
    dropped rather than guessed at.

    * active   - an exact (``=``) IC50 below 10 uM;
    * inactive - the most potent statement is ``> 10 uM``, i.e. no inhibition
                 measured at a concentration where a real inhibitor would show;
    * otherwise the molecule is dropped.
    """
    exact = [r for r in records if r["relation"] == "=" and r["value_nM"] < 10_000]
    if exact:
        best = min(exact, key=lambda r: r["value_nM"])
        best = dict(best)
        best["tier"] = ("potent" if best["value_nM"] <= 10
                        else "mid" if best["value_nM"] <= 1000 else "weak")
        return best
    negatives = [r for r in records
                 if r["relation"] in (">", ">=") and r["value_nM"] >= 10_000]
    if negatives and not [r for r in records if r["relation"] == "<"]:
        best = dict(min(negatives, key=lambda r: r["value_nM"]))
        best["tier"] = "inactive"
        return best
    return None


def main() -> None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    OUT.mkdir(exist_ok=True)
    print(f"fetching DNA-PKcs IC50 records for {TARGET} ...")
    raw = collect()
    print(f"usable activity records: {len(raw)}")

    by_mol: dict[str, list[dict]] = {}
    for row in raw:
        by_mol.setdefault(row["chembl_id"], []).append(row)
    print(f"distinct molecules with DNA-PKcs IC50 records: {len(by_mol)}")

    classified = [c for c in (classify(rs) for rs in by_mol.values()) if c]
    print("classified: " + ", ".join(
        f"{t}={sum(1 for c in classified if c['tier'] == t)}"
        for t in ("potent", "mid", "weak", "inactive")))

    seen: set[str] = set()
    picked: dict[str, list[dict]] = {"potent": [], "mid": [], "weak": [], "inactive": []}
    for row in sorted(classified, key=lambda r: r["value_nM"]):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        if not 250 <= mw <= 650:
            continue
        if rdMolDescriptors.CalcNumHBD(mol) > 5 or rdMolDescriptors.CalcNumHBA(mol) > 12:
            continue
        key = Chem.MolToInchiKey(mol)
        if key in seen:
            continue
        seen.add(key)
        row["mw"] = round(mw, 1)
        cap = 30 if row["tier"] == "inactive" else 15
        if len(picked[row["tier"]]) < cap:
            picked[row["tier"]].append(row)

    chosen = [r for t in ("potent", "mid", "weak", "inactive") for r in picked[t]]
    azd = {
        "chembl_id": "CHEMBL4439259", "name": "AZD-7648", "smiles": AZD7648,
        "relation": "=", "value_nM": 0.631,
        "assay": "Inhibition of human HeLa cell-derived full length DNA-PK catalytic subunit (TR-FRET)",
        "doc_year": 2020, "mw": 380.4, "tier": "control",
    }

    labels = {"potent": "ACT_P", "mid": "ACT_M", "weak": "ACT_W", "inactive": "INACT"}
    with (OUT / "known_inhibitors.smi").open("w") as fh:
        for row in chosen:
            fh.write(f"{row['smiles']} {labels[row['tier']]}_{row['chembl_id']}\n")
        fh.write(f"{AZD7648} AZD7648\n")

    with (OUT / "known_inhibitors.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "mol_id", "tier", "chembl_id", "name", "smiles", "relation",
            "value_nM", "pIC50", "mw", "doc_year", "assay",
        ])
        writer.writeheader()
        for row in chosen + [azd]:
            mol_id = f"{labels[row['tier']]}_{row['chembl_id']}" if row["tier"] in labels else "AZD7648"
            writer.writerow({
                **{k: row.get(k) for k in writer.fieldnames if k in row},
                "mol_id": mol_id,
                "pIC50": round(9 - __import__("math").log10(row["value_nM"]), 2),
            })

    counts = {t: len(picked[t]) for t in picked}
    print(f"benchmark built: {counts} + AZD7648 = {len(chosen) + 1} molecules")
    for row in chosen:
        print(f"  {labels[row['tier']]:6s} {row['chembl_id']:15s} {row['value_nM']:10.1f} nM  MW {row['mw']}")
    print(f"  {'CTRL':6s} CHEMBL4439259   {0.631:10.1f} nM  AZD7648")
    print(f"  -> {OUT / 'known_inhibitors.smi'}")
    print(f"  -> {OUT / 'known_inhibitors.csv'}")


if __name__ == "__main__":
    main()
