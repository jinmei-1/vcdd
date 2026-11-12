"""Dev probe: first execution of guardrails + every analog generator."""

from __future__ import annotations

import time

from vcdd.config.schema import Stage4Config
from vcdd.optimize.generators import list_generators, resolve_generators
from vcdd.optimize.guardrails import check_analog, filter_analogs, sa_score

PARENTS = {
    "P1_imatinib_like": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "P2_gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "P3_celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "P4_small": "CC(=O)Oc1ccccc1C(=O)O",
}


def main() -> None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    cfg = Stage4Config()

    # --- every transform must produce chemically sane, correctly-labelled output
    from vcdd.optimize.generators.enumerative import (
        TRANSFORMS,
        apply_transform,
        compiled_transforms,
    )

    print("=" * 72)
    print(f"transform table: {len(TRANSFORMS)} entries, "
          f"{len(compiled_transforms())} compiled")
    panel = {
        "toluene_probe": "Cc1ccccc1OC",
        "aryl_halide": "Clc1ccc(Br)cc1F",
        "acid": "OC(=O)c1ccc(N)cc1",
        "thio_sulfa": "CSc1ccc(S(N)(=O)=O)cc1[N+](=O)[O-]",
        "amine": "CN(C)CCNc1ccccc1O",
    }
    mols = {k: Chem.MolFromSmiles(v) for k, v in panel.items()}
    unfired = []
    for rxn, label in compiled_transforms():
        hits = []
        for name, m in mols.items():
            prods = apply_transform(m, rxn, max_products=2)
            if prods:
                hits.append(f"{name}:{prods[0]}")
        if not hits:
            unfired.append(label)
        print(f"  {label:<42} {' | '.join(hits[:2])[:110]}")
    print("  transforms that fired on nothing:", unfired or "none")

    print("\n" + "=" * 72)
    print("registered generators:", sorted(list_generators()))
    usable, report = resolve_generators(
        ["rgroup", "crem", "mmpdb", "constrained_ga", "nonexistent"], cfg, seed=42
    )
    for rec in report:
        print(f"  {rec['generator']:<16} available={rec['available']!s:<6} {rec['reason'][:80]}")
    print("usable:", [g.name() for g in usable])

    print("\n" + "=" * 72)
    print("guardrails on hand-picked cases (parent = celecoxib)")
    parent = PARENTS["P3_celecoxib"]
    cases = [
        (parent, "the parent itself"),
        ("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(C)(=O)=O)cc2)cc1", "sulfonamide -> sulfone"),
        ("CCCCCCCCCCCCCCCCCC", "greasy chain, far from parent"),
        ("not_a_smiles", "unparsable"),
        ("O=C(O)c1ccccc1O", "salicylic acid, unrelated"),
    ]
    for smi, note in cases:
        rep = check_analog(smi, Chem.MolFromSmiles(parent), cfg.guardrails)
        print(
            f"  pass={rep.passed!s:<6} sim={rep.similarity:.3f} sa={rep.sa:.2f} "
            f"mw={rep.mw:7.1f} :: {note} :: {';'.join(rep.reasons) or 'ok'}"
        )
    print("  sa_score sanity: aspirin=%.2f taxol-ish=%.2f"
          % (sa_score(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")),
             sa_score(Chem.MolFromSmiles("CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)O)OC(=O)c1ccccc1)(CO4)OC(C)=O)O)C)OC(=O)C"))))

    for gen in usable:
        print("\n" + "=" * 72)
        print(f"generator: {gen.name()}  --  {gen.description().strip().splitlines()[0]}")
        for pid, smi in PARENTS.items():
            t0 = time.time()
            props = gen.generate(pid, smi, 12)
            dt = time.time() - t0
            kept = filter_analogs(
                [p.smiles for p in props], smi, cfg.guardrails,
                similarity_floor=(cfg.guardrails.similarity_floor_ga
                                  if gen.name() == "constrained_ga" else None),
            )
            n_pass = sum(1 for r in kept if r.passed)
            print(f"  {pid:<20} n={len(props):>3}  pass={n_pass:>3}  {dt:5.2f}s")
            for p in props[:4]:
                print(f"      {p.transformation[:58]:<58} {p.smiles[:70]}")
        if gen.name() == "constrained_ga" and getattr(gen, "history", None):
            print("  GA history (last 3 entries):")
            for h in gen.history[-3:]:
                print("     ", {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in h.items()})


if __name__ == "__main__":
    main()
