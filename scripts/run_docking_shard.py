#!/usr/bin/env python3
"""Shard runner for the DNA-PKcs docking benchmark.
Usage: python run_docking_shard.py <shard_idx> <n_shards>
Each shard writes docking_out/scores_shard<idx>.csv (resumable)."""
import os, sys, time
import pandas as pd
sys.path.insert(0, '/workspace')
from run_docking_benchmark import smi_to_pdbqt, dock, OUT

def main():
    idx, nsh = int(sys.argv[1]), int(sys.argv[2])
    bench = pd.read_csv('/workspace/dnapkcs_benchmark.csv')
    shard = bench.iloc[idx::nsh].reset_index(drop=True)
    path = f'{OUT}/scores_shard{idx:02d}.csv'
    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path).set_id)
    t0, n_done = time.time(), 0
    with open(path, 'a') as fout:
        if not done:
            fout.write('set_id,score,label\n')
        for _, row in shard.iterrows():
            if row.set_id in done:
                continue
            p = smi_to_pdbqt(row.smiles, f's{idx:02d}_{row.set_id}')
            if p is None:
                fout.write(f'{row.set_id},NaN,{row.label}\n'); fout.flush(); continue
            try:
                score, _ = dock(p, f's{idx:02d}_{row.set_id}')
            except Exception:
                score = float('nan')
            fout.write(f'{row.set_id},{score},{row.label}\n'); fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                dt = time.time() - t0
                print(f'[shard {idx}] {n_done} docked | {dt/max(n_done,1):.1f}s/mol', flush=True)
    print(f'[shard {idx}] DONE ({len(shard)} compounds)')

if __name__ == '__main__':
    main()
