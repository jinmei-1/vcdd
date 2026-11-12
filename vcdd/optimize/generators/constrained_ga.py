"""Constrained graph genetic algorithm for analog design.

An unconstrained molecular GA is a scoring-function exploit generator: given
enough generations it will find whatever pathology the surrogate rewards --
greasy polyaromatics that fill the docking box, or read-across neighbours of a
single high-scoring training compound.  Three constraints make it useful here:

1. **Similarity leash.** Every individual must stay within
   ``guardrails.similarity_floor_ga`` Tanimoto of its *parent* (not of the
   previous generation), so the population cannot drift away one small step at
   a time.  This is what makes the output an *analog series* rather than de
   novo design.
2. **Full guardrail set on every individual**, not just on the final answer, so
   the fitness landscape the GA searches never contains molecules a chemist
   would reject.
3. **Interpretable lineage.** Every operator records a human-readable label;
   an individual's ``transformations`` list is the ordered set of edits that
   produced it.  Those labels are the unit the SAR miner aggregates over, which
   is what turns the output into advice instead of a SMILES dump.

The fitness function is injected.  Standalone (in tests, or when no cell model
accepts novel SMILES) it falls back to a cheap drug-likeness fitness.  Stage 4
injects the virtual-cell objective, which is the score the user actually wants
improved; docking is deliberately *not* in the inner loop -- it is ~10 s per
molecule against ~1 ms for read-across, so it runs once on the survivors
instead of ~1000 times inside the search.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..guardrails import check_analog
from .base import AnalogGenerator, Proposal, register_generator
from .enumerative import apply_transform, compiled_transforms

__all__ = ["ConstrainedGA", "Individual", "FRAGMENTS", "default_fitness"]

#: Small fragments appended by the growth operator. Deliberately conservative:
#: common medicinal-chemistry caps and rings, nothing exotic.
FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("F", "add F"),
    ("Cl", "add Cl"),
    ("C", "add methyl"),
    ("OC", "add methoxy"),
    ("C(F)(F)F", "add CF3"),
    ("C#N", "add nitrile"),
    ("N", "add amine"),
    ("O", "add hydroxyl"),
    ("C(N)=O", "add primary amide"),
    ("S(C)(=O)=O", "add methylsulfonyl"),
    ("N1CCOCC1", "add morpholine"),
    ("N1CCNCC1", "add piperazine"),
    ("C1CC1", "add cyclopropyl"),
    ("c1ccncc1", "add pyridyl"),
)

#: Element swaps allowed by the atom-type operator, keyed by atomic number.
_ALIPHATIC_SWAPS: dict[int, tuple[int, ...]] = {
    6: (7, 8),
    7: (6, 8),
    8: (7, 6),
    16: (8,),
    9: (17,),
    17: (9, 35),
    35: (17,),
}
_AROMATIC_SWAPS: dict[int, tuple[int, ...]] = {6: (7,), 7: (6,)}
_SYMBOL = {6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}


@dataclass
class Individual:
    """One member of the population."""

    smiles: str
    fitness: float = float("-inf")
    similarity: float = float("nan")
    transformations: list[str] = field(default_factory=list)
    generation: int = 0

    @property
    def label(self) -> str:
        if not self.transformations:
            return "GA: parent"
        return "GA: " + " + ".join(self.transformations)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def _sanitized(mol) -> object | None:
    from rdkit import Chem

    try:
        m = mol.GetMol() if hasattr(mol, "GetMol") else mol
        Chem.SanitizeMol(m)
        smi = Chem.MolToSmiles(m)
        return Chem.MolFromSmiles(smi)
    except Exception:  # noqa: BLE001 - operators are allowed to produce garbage
        return None


def _mut_substituent(mol, rng: random.Random):
    """Apply one reaction from the shared medicinal-chemistry transform table."""
    from rdkit import Chem

    table = list(compiled_transforms())
    rng.shuffle(table)
    for rxn, label in table:
        prods = apply_transform(mol, rxn, max_products=8)
        if not prods:
            continue
        out = Chem.MolFromSmiles(rng.choice(prods))
        if out is not None:
            return out, label
    return None, ""


def _mut_atom_type(mol, rng: random.Random):
    """Single-atom element swap (aza scan, O/N interchange, halogen walk)."""
    from rdkit import Chem

    idxs = list(range(mol.GetNumAtoms()))
    rng.shuffle(idxs)
    for idx in idxs:
        atom = mol.GetAtomWithIdx(idx)
        z = atom.GetAtomicNum()
        table = _AROMATIC_SWAPS if atom.GetIsAromatic() else _ALIPHATIC_SWAPS
        options = table.get(z)
        if not options:
            continue
        new_z = rng.choice(options)
        rw = Chem.RWMol(mol)
        target = rw.GetAtomWithIdx(idx)
        target.SetAtomicNum(new_z)
        target.SetNumExplicitHs(0)
        target.SetNoImplicit(False)
        target.SetFormalCharge(0)
        out = _sanitized(rw)
        if out is not None:
            return out, f"{_SYMBOL.get(z, z)} -> {_SYMBOL.get(new_z, new_z)} at position {idx}"
    return None, ""


def _mut_grow(mol, rng: random.Random):
    """Append a small fragment at a hydrogen-bearing carbon or nitrogen."""
    from rdkit import Chem

    sites = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (6, 7)
    ]
    if not sites:
        return None, ""
    frag_smi, label = rng.choice(FRAGMENTS)
    frag = Chem.MolFromSmiles(frag_smi)
    if frag is None:
        return None, ""
    frag_site = next((a.GetIdx() for a in frag.GetAtoms() if a.GetTotalNumHs() > 0), None)
    if frag_site is None:
        return None, ""

    rng.shuffle(sites)
    for site in sites[:5]:
        rw = Chem.RWMol(Chem.CombineMols(mol, frag))
        a = rw.GetAtomWithIdx(site)
        b = rw.GetAtomWithIdx(mol.GetNumAtoms() + frag_site)
        for at in (a, b):
            if at.GetNumExplicitHs() > 0:
                at.SetNumExplicitHs(at.GetNumExplicitHs() - 1)
        try:
            rw.AddBond(site, mol.GetNumAtoms() + frag_site, Chem.BondType.SINGLE)
        except Exception:  # noqa: BLE001
            continue
        out = _sanitized(rw)
        if out is not None:
            return out, label
    return None, ""


def _mut_trim(mol, rng: random.Random):
    """Delete a terminal heavy atom, the cheapest way to shed dead weight."""
    from rdkit import Chem

    if mol.GetNumHeavyAtoms() <= 10:
        return None, ""
    sites = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetDegree() == 1 and not a.IsInRing()
    ]
    rng.shuffle(sites)
    for site in sites[:5]:
        sym = _SYMBOL.get(mol.GetAtomWithIdx(site).GetAtomicNum(), "atom")
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(site)
        out = _sanitized(rw)
        if out is not None:
            return out, f"delete terminal {sym}"
    return None, ""


_MUTATORS: tuple[tuple[Callable, float], ...] = (
    (_mut_substituent, 0.40),
    (_mut_atom_type, 0.25),
    (_mut_grow, 0.25),
    (_mut_trim, 0.10),
)


def mutate(mol, rng: random.Random):
    """Apply one randomly chosen mutation operator. Returns (mol, label)."""
    fns = [f for f, _ in _MUTATORS]
    weights = [w for _, w in _MUTATORS]
    for _ in range(4):  # a few attempts before giving up on this individual
        fn = rng.choices(fns, weights=weights, k=1)[0]
        out, label = fn(mol, rng)
        if out is not None:
            return out, label
    return None, ""


def _cuttable_bonds(mol) -> list[int]:
    from rdkit import Chem

    return [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and not b.IsInRing()
        and b.GetBeginAtom().GetDegree() > 1
        and b.GetEndAtom().GetDegree() > 1
    ]


def _fragment_on_bond(mol, bond_idx: int):
    from rdkit import Chem

    try:
        broken = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True, dummyLabels=[(0, 0)])
        return list(Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False))
    except Exception:  # noqa: BLE001
        return []


def _stitch(frag_a, frag_b):
    """Join two single-dummy fragments at their attachment points."""
    from rdkit import Chem

    combo = Chem.RWMol(Chem.CombineMols(frag_a, frag_b))
    dummies = [a.GetIdx() for a in combo.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    anchors = []
    for d in dummies:
        nbrs = combo.GetAtomWithIdx(d).GetNeighbors()
        if len(nbrs) != 1:
            return None
        anchors.append(nbrs[0].GetIdx())
    try:
        combo.AddBond(anchors[0], anchors[1], Chem.BondType.SINGLE)
        for d in sorted(dummies, reverse=True):
            combo.RemoveAtom(d)
    except Exception:  # noqa: BLE001
        return None
    return _sanitized(combo)


def crossover(mol_a, mol_b, rng: random.Random):
    """Graph crossover: cut each parent at one acyclic single bond, recombine."""
    bonds_a, bonds_b = _cuttable_bonds(mol_a), _cuttable_bonds(mol_b)
    if not bonds_a or not bonds_b:
        return None, ""
    for _ in range(5):
        frags_a = _fragment_on_bond(mol_a, rng.choice(bonds_a))
        frags_b = _fragment_on_bond(mol_b, rng.choice(bonds_b))
        if len(frags_a) != 2 or len(frags_b) != 2:
            continue
        child = _stitch(rng.choice(frags_a), rng.choice(frags_b))
        if child is not None and child.GetNumHeavyAtoms() >= 8:
            return child, "crossover"
    return None, ""


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #


def default_fitness(smiles: Sequence[str], parent_smiles: str) -> list[float]:
    """Cheap drug-likeness fitness used when no external scorer is injected.

    Rewards QED and synthetic accessibility. It does **not** reward similarity
    to the parent -- that is already a hard constraint, and rewarding it again
    would collapse the population onto the parent.
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    from ..guardrails import sa_score

    out: list[float] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(float("-inf"))
            continue
        qed = float(QED.qed(mol))
        sa = sa_score(mol)
        sa_term = 1.0 if math.isnan(sa) else max(0.0, min(1.0, (10.0 - sa) / 9.0))
        out.append(0.7 * qed + 0.3 * sa_term)
    return out


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


@register_generator
class ConstrainedGA(AnalogGenerator):
    """Similarity-constrained graph GA over mutation and crossover operators.

    Population and generation counts come from ``stage4.ga_population`` and
    ``stage4.ga_generations``; the similarity leash is
    ``stage4.guardrails.similarity_floor_ga``.
    """

    @classmethod
    def name(cls) -> str:
        return "constrained_ga"

    def __init__(self, cfg, seed: int = 42) -> None:
        super().__init__(cfg, seed=seed)
        self._fitness_fn: Callable[[Sequence[str], str], Sequence[float]] | None = None
        self.history: list[dict] = []

    def set_fitness(self, fn: Callable[[Sequence[str], str], Sequence[float]] | None) -> None:
        """Inject an external fitness function ``fn(smiles, parent_smiles) -> scores``.

        Higher is better. Called once per generation with the whole batch so a
        model-backed scorer can vectorise.
        """
        self._fitness_fn = fn

    def _score(self, smiles: Sequence[str], parent_smiles: str) -> list[float]:
        if not smiles:
            return []
        if self._fitness_fn is None:
            return list(default_fitness(smiles, parent_smiles))
        try:
            vals = list(self._fitness_fn(smiles, parent_smiles))
        except Exception:  # noqa: BLE001 - a scorer failure falls back, never aborts
            return list(default_fitness(smiles, parent_smiles))
        if len(vals) != len(smiles):
            return list(default_fitness(smiles, parent_smiles))
        return [float(v) if v is not None and math.isfinite(float(v)) else float("-inf") for v in vals]

    # ---------------------------------------------------------------- search
    def generate(self, parent_id: str, parent_smiles: str, n: int) -> list[Proposal]:
        from rdkit import Chem

        parent = Chem.MolFromSmiles(parent_smiles)
        if parent is None:
            return []

        guard = self.cfg.guardrails
        floor = guard.similarity_floor_ga
        rng = random.Random(f"{self.seed}:{parent_id}")
        pop_size = max(4, int(self.cfg.ga_population))
        n_gen = max(1, int(self.cfg.ga_generations))
        mut_rate = float(self.cfg.ga_mutation_rate)

        #: canonical SMILES -> Individual, every valid candidate ever seen
        archive: dict[str, Individual] = {}
        rejected = 0
        n_proposed = 0

        def admit(mol, lineage: list[str], generation: int) -> tuple[Individual | None, bool]:
            """Guardrail-check a candidate. Returns ``(individual, is_new)``."""
            nonlocal rejected, n_proposed
            n_proposed += 1
            smi = Chem.MolToSmiles(mol)
            if smi in archive:
                return archive[smi], False
            rep = check_analog(smi, parent, guard, similarity_floor=floor)
            if not rep.passed:
                rejected += 1
                return None, False
            if rep.smiles in archive:  # canonicalisation collapsed two candidates
                return archive[rep.smiles], False
            ind = Individual(
                smiles=rep.smiles,
                similarity=rep.similarity,
                transformations=list(lineage),
                generation=generation,
            )
            archive[rep.smiles] = ind
            return ind, True

        # ---- generation 0: mutate the parent until the population fills -----
        population: list[Individual] = []
        for _ in range(pop_size * 8):
            if len(population) >= pop_size:
                break
            child, label = mutate(parent, rng)
            if child is None:
                continue
            ind, is_new = admit(child, [label], 0)
            if ind is not None and is_new:
                population.append(ind)

        if not population:
            self.history.append(
                {"parent_id": parent_id, "generations": 0, "archive": 0,
                 "rejected": rejected, "note": "no valid generation-0 individual"}
            )
            return []

        scores = self._score([i.smiles for i in population], parent_smiles)
        for ind, sc in zip(population, scores):
            ind.fitness = sc
        population.sort(key=lambda i: (-i.fitness, i.smiles))

        def tournament() -> Individual:
            k = min(3, len(population))
            return max(rng.sample(population, k), key=lambda i: i.fitness)

        # ---- evolve ---------------------------------------------------------
        for gen in range(1, n_gen + 1):
            offspring: list[Individual] = []
            attempts = 0
            while len(offspring) < pop_size and attempts < pop_size * 10:
                attempts += 1
                a = tournament()
                mol_a = Chem.MolFromSmiles(a.smiles)
                if mol_a is None:
                    continue
                lineage = list(a.transformations)
                child = mol_a
                if rng.random() < 0.5 and len(population) > 1:
                    b = tournament()
                    mol_b = Chem.MolFromSmiles(b.smiles)
                    if mol_b is not None:
                        crossed, label = crossover(mol_a, mol_b, rng)
                        if crossed is not None:
                            child, lineage = crossed, lineage + [label]
                if rng.random() < mut_rate or child is mol_a:
                    mutated, label = mutate(child, rng)
                    if mutated is None:
                        continue
                    child, lineage = mutated, lineage + [label]
                # Keep lineages readable; long chains are unusable as SAR rules.
                ind, is_new = admit(child, lineage[-3:], gen)
                if ind is not None and is_new:
                    offspring.append(ind)

            if offspring:
                fresh = self._score([i.smiles for i in offspring], parent_smiles)
                for ind, sc in zip(offspring, fresh):
                    ind.fitness = sc

            merged = {i.smiles: i for i in population + offspring}
            population = sorted(merged.values(), key=lambda i: (-i.fitness, i.smiles))[:pop_size]
            self.history.append(
                {
                    "parent_id": parent_id,
                    "generation": gen,
                    "offspring": len(offspring),
                    "archive": len(archive),
                    "rejected": rejected,
                    "best_fitness": population[0].fitness if population else None,
                    "median_fitness": (
                        population[len(population) // 2].fitness if population else None
                    ),
                }
            )

        # ---- report the best of everything ever seen ------------------------
        best = sorted(
            (i for i in archive.values() if math.isfinite(i.fitness)),
            key=lambda i: (-i.fitness, i.smiles),
        )
        if not best:  # nothing was ever scored: fall back to the raw archive
            best = sorted(archive.values(), key=lambda i: i.smiles)
        return [
            Proposal(
                smiles=ind.smiles,
                parent_id=parent_id,
                parent_smiles=parent_smiles,
                generator=self.name(),
                transformation=ind.label,
                meta={
                    "transformations": list(ind.transformations),
                    "ga_fitness": ind.fitness,
                    "ga_generation": ind.generation,
                    "similarity_to_parent": ind.similarity,
                    "n_proposed": n_proposed,
                    "n_guardrail_rejected": rejected,
                },
            )
            for ind in best[:n]
        ]
