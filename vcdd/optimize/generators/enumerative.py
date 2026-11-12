"""Enumerative analog generators: R-group replacement, CReM, matched pairs.

These are the conservative generators.  They make one interpretable change at a
time, which is exactly what a medicinal chemist can act on, and they keep the
parent scaffold intact by construction.

``rgroup``
    Substituent swaps at aromatic and aliphatic C-H / C-X positions, drawn from
    a curated bioisostere table. Always available -- pure RDKit.
``crem``
    CReM mutations using fragments mined from ChEMBL with matched chemical
    context, so replacements are ones that actually occur in real molecules.
    Needs the ``crem`` package and its fragment database.
``mmpdb``
    Matched-molecular-pair transformations learned from a reference set.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

from .base import AnalogGenerator, Proposal, register_generator

__all__ = [
    "RGroupGenerator",
    "CRemGenerator",
    "MMPDBGenerator",
    "TRANSFORMS",
    "compiled_transforms",
    "apply_transform",
]

#: (reaction SMARTS, human-readable label).
#:
#: These are *mapped reactions*, not substructure replacements. The distinction
#: matters: ``ReplaceSubstructs`` with pattern ``[cH1]`` and replacement ``N``
#: swaps the ring carbon for nitrogen (an aza-scan) rather than hanging an
#: amine off it, so a table written that way silently mislabels its own output.
#: With ``[cH1:1]>>[c:1]N`` the mapped aromatic carbon is retained and the
#: substituent is appended, which is what the label says.
#:
#: The set covers the moves a medicinal chemist actually makes first: aromatic
#: C-H functionalisation (halogen and polarity scans), the F/Cl/CF3
#: lipophilicity ladder, aza-scanning, alkyl homologation, de-alkylation, and
#: three classic acid/amide/sulfur bioisosteres.
TRANSFORMS: tuple[tuple[str, str], ...] = (
    # --- aromatic C-H functionalisation (positional scan) ---
    ("[cH1:1]>>[c:1]F", "aromatic C-H -> C-F"),
    ("[cH1:1]>>[c:1]Cl", "aromatic C-H -> C-Cl"),
    ("[cH1:1]>>[c:1]C", "aromatic C-H -> C-CH3"),
    ("[cH1:1]>>[c:1]OC", "aromatic C-H -> C-OMe"),
    ("[cH1:1]>>[c:1]C(F)(F)F", "aromatic C-H -> C-CF3"),
    ("[cH1:1]>>[c:1]C#N", "aromatic C-H -> C-CN"),
    ("[cH1:1]>>[c:1]N", "aromatic C-H -> C-NH2"),
    ("[cH1:1]>>[c:1]O", "aromatic C-H -> C-OH"),
    # --- aza-scan: swap a ring CH for N ---
    ("[cH1:1]1[c:2][c:3][c:4][c:5][c:6]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1",
     "benzene C-H -> ring N (aza-scan)"),
    # --- halogen / lipophilicity ladder ---
    ("[c:1][F]>>[c:1]Cl", "aryl-F -> aryl-Cl"),
    ("[c:1][Cl]>>[c:1]F", "aryl-Cl -> aryl-F"),
    ("[c:1][Cl]>>[c:1]C(F)(F)F", "aryl-Cl -> aryl-CF3"),
    ("[c:1][Cl]>>[c:1]C#N", "aryl-Cl -> aryl-CN"),
    ("[c:1][Br]>>[c:1]Cl", "aryl-Br -> aryl-Cl"),
    # --- polar group edits ---
    ("[c:1][OX2H1]>>[c:1]OC", "aryl-OH -> aryl-OMe"),
    ("[CH3][O:1][c:2]>>[O:1][c:2]", "aryl-OMe -> aryl-OH (O-demethylation)"),
    ("[CH3][NX3;H0,H1:1]>>[NX3:1]", "N-methyl -> N-H (N-demethylation)"),
    # ``$(N[#6;!$([C]=[O,N,S])])`` keeps this on genuine amines: without it the
    # pattern also fires on sulfonamide and amide N-H2 and mislabels the result.
    ("[NX3H2;$(N[#6;!$([C]=[O,N,S])]):1]>>[N:1]C", "primary amine -> N-methyl amine"),
    ("[SX4:1](=O)(=O)[NX3H2:2]>>[S:1](=O)(=O)[N:2]C",
     "sulfonamide -> N-methyl sulfonamide"),
    ("[C:1](=O)[NX3H1:2][#6:3]>>[C:1](=O)[N:2](C)[#6:3]",
     "amide N-H -> N-methyl amide (donor masking)"),
    ("[c:1][N+](=O)[O-]>>[c:1]N", "aryl-NO2 -> aryl-NH2"),
    # --- alkyl homologation ---
    ("[CH3:1]>>[CH2:1]C", "methyl -> ethyl"),
    ("[CH3:1][c:2]>>[c:2]C(F)(F)F", "aryl-methyl -> aryl-CF3"),
    # --- classic bioisosteres ---
    ("[CX3:1](=O)[OX2H1]>>[C:1](=O)NC", "carboxylic acid -> N-methyl amide"),
    ("[CX3:1](=O)[OX2H1]>>[C:1]1=NN=N[NH]1", "carboxylic acid -> tetrazole"),
    ("[SX2:1][CH3]>>[S:1](=O)(=O)C", "thioether -> methyl sulfone"),
    ("[SX4:1](=O)(=O)[NX3H2]>>[S:1](=O)(=O)C", "sulfonamide -> methyl sulfone"),
)


@lru_cache(maxsize=1)
def compiled_transforms() -> tuple[tuple[object, str], ...]:
    """Compile the transform table once. Malformed entries are dropped."""
    from rdkit.Chem import AllChem

    out = []
    for smarts, label in TRANSFORMS:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
        except Exception:  # noqa: BLE001
            continue
        if rxn is None:
            continue
        try:
            rxn.Initialize()
        except Exception:  # noqa: BLE001
            continue
        out.append((rxn, label))
    return tuple(out)


def apply_transform(mol, rxn, max_products: int = 32) -> list[str]:
    """Run one reaction over every match, returning unique canonical SMILES.

    Products that fail sanitisation are dropped silently: a transform is not
    valid at every site it matches, and that is expected rather than an error.
    """
    from rdkit import Chem

    try:
        product_sets = rxn.RunReactants((mol,))
    except Exception:  # noqa: BLE001
        return []
    seen: list[str] = []
    for products in product_sets:
        for prod in products:
            try:
                prod.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(Chem.RemoveHs(prod))
            except Exception:  # noqa: BLE001
                continue
            if smi and "." not in smi and smi not in seen:
                seen.append(smi)
            if len(seen) >= max_products:
                return seen
    return seen


@register_generator
class RGroupGenerator(AnalogGenerator):
    """Single-point substituent replacement from a bioisostere table."""

    @classmethod
    def name(cls) -> str:
        return "rgroup"

    def generate(self, parent_id: str, parent_smiles: str, n: int) -> list[Proposal]:
        from rdkit import Chem

        parent = Chem.MolFromSmiles(parent_smiles)
        if parent is None:
            return []
        rng = random.Random(f"{self.seed}:{parent_id}")
        table = list(compiled_transforms())
        rng.shuffle(table)

        # Round-robin over transforms rather than draining one at a time, so a
        # transform with many matching sites (aromatic C-H) cannot consume the
        # whole budget and hide every other move from the chemist.
        buckets: list[tuple[str, list[str]]] = []
        for rxn, label in table:
            prods = apply_transform(parent, rxn)
            if prods:
                rng.shuffle(prods)
                buckets.append((label, prods))

        out: list[Proposal] = []
        seen: set[str] = {Chem.MolToSmiles(parent)}
        depth = 0
        while len(out) < n and buckets:
            progressed = False
            for label, prods in buckets:
                if depth >= len(prods):
                    continue
                progressed = True
                smi = prods[depth]
                if smi in seen:
                    continue
                seen.add(smi)
                out.append(Proposal(smi, parent_id, parent_smiles, self.name(), label))
                if len(out) >= n:
                    break
            if not progressed:
                break
            depth += 1
        return out


@register_generator
class CRemGenerator(AnalogGenerator):
    """CReM context-aware fragment mutation.

    Requires the fragment database, which is a separate multi-gigabyte
    download::

        pip install crem
        wget http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz
        gunzip replacements02_sc2.db.gz   # then set stage4.crem_db

    ``mutate_mol`` swaps a fragment for others seen in the same chemical
    context in ChEMBL, which keeps the result synthetically plausible in a way
    that a naive SMARTS edit does not.
    """

    @classmethod
    def name(cls) -> str:
        return "crem"

    def _db_path(self) -> Path | None:
        cand = getattr(self.cfg, "crem_db", None) or os.environ.get("VCDD_CREM_DB")
        if cand and Path(cand).expanduser().exists():
            return Path(cand).expanduser()
        return None

    def is_available(self) -> tuple[bool, str]:
        try:
            import crem  # noqa: F401,PLC0415
        except ImportError:
            return False, "pip install crem"
        db = self._db_path()
        if db is None:
            return False, (
                "no CReM fragment database; download replacements02_sc2.db from "
                "http://www.qsar4u.com/files/cremdb/ and set stage4.crem_db "
                "(or VCDD_CREM_DB)"
            )
        return True, f"fragment db {db.name}"

    def generate(self, parent_id: str, parent_smiles: str, n: int) -> list[Proposal]:
        from crem.crem import mutate_mol
        from rdkit import Chem

        parent = Chem.MolFromSmiles(parent_smiles)
        db = self._db_path()
        if parent is None or db is None:
            return []
        try:
            gen = mutate_mol(
                parent,
                db_name=str(db),
                radius=3,
                min_size=0,
                max_size=8,
                min_inc=-3,
                max_inc=3,
                return_mol=False,
                ncores=1,
            )
            smis = []
            for smi in gen:
                smis.append(smi)
                if len(smis) >= n * 3:
                    break
        except Exception:  # noqa: BLE001 - db issues degrade to no proposals
            return []
        rng = random.Random(f"{self.seed}:{parent_id}")
        rng.shuffle(smis)
        return [
            Proposal(s, parent_id, parent_smiles, self.name(), "CReM fragment mutation")
            for s in smis[:n]
        ]


@register_generator
class MMPDBGenerator(AnalogGenerator):
    """Matched-molecular-pair transformations from an mmpdb reference set.

    Requires ``mmpdb`` and a prebuilt database::

        pip install mmpdb
        mmpdb fragment reference.smi -o reference.fragments
        mmpdb index reference.fragments -o reference.mmpdb
        # then set stage4.mmpdb_reference
    """

    @classmethod
    def name(cls) -> str:
        return "mmpdb"

    def _db_path(self) -> Path | None:
        cand = getattr(self.cfg, "mmpdb_reference", None) or os.environ.get("VCDD_MMPDB")
        if cand and Path(cand).expanduser().exists():
            return Path(cand).expanduser()
        return None

    def is_available(self) -> tuple[bool, str]:
        try:
            import mmpdblib  # noqa: F401,PLC0415
        except ImportError:
            return False, "pip install mmpdb"
        if self._db_path() is None:
            return False, "set stage4.mmpdb_reference to a built .mmpdb database"
        return True, f"mmpdb {self._db_path().name}"

    def generate(self, parent_id: str, parent_smiles: str, n: int) -> list[Proposal]:
        import subprocess  # noqa: PLC0415

        db = self._db_path()
        if db is None:
            return []
        try:
            proc = subprocess.run(
                ["mmpdb", "transform", "--smiles", parent_smiles, str(db)],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if proc.returncode != 0:
            return []
        out: list[Proposal] = []
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            smi, rule = parts[0].strip(), parts[1].strip()
            if smi:
                out.append(
                    Proposal(smi, parent_id, parent_smiles, self.name(), f"MMP: {rule}")
                )
            if len(out) >= n:
                break
        return out
