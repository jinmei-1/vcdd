#!/usr/bin/env python3
"""Bundle the FULL project data (all runs, weights, research, deps) into one zip."""
import zipfile, time
from pathlib import Path

ROOT = Path('/home/geng/vcdd-0.3.0')
ZIP = ROOT / 'paper/Project_Data_Full_2026-08-25.zip'
TOP = 'vcdd-project-data'

include = ['dnapkcs_run', 'DNA-PKcs', 'paper', 'msa_out', 'scripts', 'research',
           'tmp_data', 'tmp_pydeps', 'tmp_xpert', 'tmp_unimol_weights',
           'run_dnapkcs.yaml', 'README.md', 'README-vcdd-upstream.md',
           'msa_receptor_vcdd.fasta', 'vina_fix.sh',
           # msa_fetch.py and tmp_pocket_crop.py live in scripts/, which is
           # already included above, so they are picked up from there.
           'run_all.py']
skip_dirs = {'__pycache__', '.git', '.pytest_cache', 'mplcache'}

t0 = time.time(); n = 0; mb = 0
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for item in include:
        p = ROOT / item
        if not p.exists():
            print('missing:', item); continue
        if p.is_file():
            z.write(p, f'{TOP}/{item}'); n += 1
            continue
        for f in p.rglob('*'):
            if f.is_file() and not any(part in skip_dirs for part in f.parts):
                z.write(f, f'{TOP}/{f.relative_to(ROOT)}')
                n += 1
                mb += f.stat().st_size
                if n % 20000 == 0:
                    print(f'  ... {n} files, {mb/2**30:.1f} GiB raw, elapsed {time.time()-t0:.0f}s', flush=True)
    print(f'done: {n} files, raw {mb/2**30:.1f} GiB, elapsed {time.time()-t0:.0f}s')
print('zip size:', ZIP.stat().st_size / 2**30, 'GiB')
