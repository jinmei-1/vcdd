#!/usr/bin/env python3
"""Dock the DNA-PKcs benchmark (200 actives + 4524 decoys) with AutoDock Vina 1.2.5 (CPU).

- Receptor: dnapkcs_receptor.pdbqt (7OTW chain A, residues 3701-4123)
- Box: AZD7648 centroid, 17 x 17 x 17 A (manuscript protocol)
- Default search parameters (exhaustiveness 8), matching the manuscript's
  "upstream-default" stage-1 settings (quickvina2_gpu defaults).
- Ligands: SMILES -> RDKit 3D (ETKDG + MMFF) -> meeko PDBQT -> vina CLI

Writes: docking_out/scores.csv (set_id, score kcal/mol, lower = better)
Positive control: AZD7648 redock RMSD vs crystal pose printed to console.
"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

WS = '/workspace'
OUT = f'{WS}/docking_out'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/pdbqt', exist_ok=True)

box = json.load(open(f'{WS}/dnapkcs_box.json'))
CX, CY, CZ = box['center']
SIZE = box['size'][0]
REC = f'{WS}/dnapkcs_receptor.pdbqt'


def smi_to_pdbqt(smi, name):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 42
    if AllChem.EmbedMolecule(m, ps) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    prep = MoleculePreparation()
    try:
        mol_setups = prep.prepare(m)
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not ok:
            return None
    except Exception:
        return None
    path = f'{OUT}/pdbqt/{name}.pdbqt'
    with open(path, 'w') as f:
        f.write(pdbqt)
    return path


def dock(lig_pdbqt, name, exhaustiveness=8):
    out_pose = f'{OUT}/pdbqt/{name}_out.pdbqt'
    cmd = ['vina', '--receptor', REC, '--ligand', lig_pdbqt,
           '--center_x', str(CX), '--center_y', str(CY), '--center_z', str(CZ),
           '--size_x', str(SIZE), '--size_y', str(SIZE), '--size_z', str(SIZE),
           '--exhaustiveness', str(exhaustiveness), '--num_modes', '9',
           '--cpu', '1', '--out', out_pose, '--seed', '42']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    score = None
    for line in r.stdout.splitlines():
        # vina table: mode 1 line starts with '   1 '
        parts = line.split()
        if len(parts) >= 4 and parts[0] == '1':
            try:
                score = float(parts[1])
            except ValueError:
                pass
            break
    return score, out_pose if os.path.exists(out_pose) else None


def redock_control():
    act = pd.read_csv(f'{WS}/dnapkcs_actives.csv')
    smi = act.loc[act.mol_id == 'CHEMBL4439259', 'smiles'].iloc[0]
    p = smi_to_pdbqt(smi, 'AZD7648_redock')
    score, pose = dock(p, 'AZD7648_redock')
    # RMSD of best pose vs crystal (heavy atoms, no alignment needed - same frame)
    xtal = Chem.MolFromPDBFile(f'{WS}/azd7648_xtal.pdb', removeHs=True)
    dock_m = Chem.MolFromPDBFile(pose, removeHs=True) if pose else None
    rmsd = None
    if xtal and dock_m:
        from rdkit.Chem import rdMolAlign
        try:
            rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(dock_m), Chem.RemoveHs(xtal),
                                         maximizeIterations=False)
        except Exception as e:
            print('RMSD calc issue:', e)
    print(f'[CONTROL] AZD7648 redock: score={score} kcal/mol, RMSD vs crystal={rmsd}')
    return score, rmsd


def main():
    if '--control-only' in sys.argv:
        redock_control(); return
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0  # 0 = all

    bench = pd.read_csv(f'{WS}/dnapkcs_benchmark.csv')
    if limit:
        bench = bench.head(limit)
    done = set()
    scores_path = f'{OUT}/scores.csv'
    if os.path.exists(scores_path):
        done = set(pd.read_csv(scores_path).set_id)
        print('resuming;', len(done), 'already docked')

    t0 = time.time(); n_done = 0
    with open(scores_path, 'a') as fout:
        if not done:
            fout.write('set_id,score,label\n')
        for _, row in bench.iterrows():
            if row.set_id in done:
                continue
            p = smi_to_pdbqt(row.smiles, row.set_id)
            if p is None:
                fout.write(f'{row.set_id},NaN,{row.label}\n'); continue
            try:
                score, _ = dock(p, row.set_id)
            except subprocess.TimeoutExpired:
                score = float('nan')
            fout.write(f'{row.set_id},{score},{row.label}\n'); fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                dt = time.time() - t0
                print(f'{n_done} docked | {dt/n_done:.1f}s/mol | elapsed {dt/60:.1f} min', flush=True)
    print('ALL DONE')


if __name__ == '__main__':
    main()
