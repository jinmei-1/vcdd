"""Extract the AZD7648 (MBW) binding-pocket residues of DNA-PKcs (7OTW) as a
cropped receptor PDB + per-chain pocket FASTA, for local Boltz-2 scoring
(DNA-PKcs is 4128 aa; Boltz-2 caps at ~4096 tokens, and full-chain folding
would be hours per molecule anyway).
"""
import sys
from pathlib import Path

LIG = "MBW"
CUTOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("DNA-PKcs/7OTW.pdb")
OUT = SRC.parent / f"7OTW_pocket_{int(CUTOFF)}A.pdb"

# parse atoms: ligand heavy atoms and protein residues
lig = []
res_atoms = {}
for line in SRC.read_text().splitlines():
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    name = line[12:16].strip()
    resn = line[17:20].strip()
    resi = int(line[22:26])
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    is_lig = resn == LIG
    key = (resi, resn)
    if is_lig:
        lig.append((x, y, z))
    elif line.startswith("ATOM"):
        res_atoms.setdefault(key, []).append((name, x, y, z))

print(f"ligand atoms: {len(lig)} | protein residues: {len(res_atoms)}")

# select residues with any heavy atom within cutoff of any ligand atom
sel = {}
for (resi, resn), atoms in res_atoms.items():
    if any((a[1]-lx)**2 + (a[2]-ly)**2 + (a[3]-lz)**2 <= CUTOFF**2
           for a in atoms for (lx, ly, lz) in lig):
        sel[resi] = (resn, atoms)
print(f"pocket residues within {CUTOFF}A: {len(sel)} "
      f"({min(sel)}-{max(sel)})")

# contiguous segments
segs = []
start = prev = None
for r in sorted(sel):
    if start is None:
        start = prev = r
    elif r - prev > 1:
        segs.append((start, prev)); start = prev = r
    else:
        prev = r
if start is not None:
    segs.append((start, prev))
print("contiguous segments:", segs)

# write cropped PDB (keep ligand + selected residues + preserve order)
lines = []
for line in SRC.read_text().splitlines():
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    resi = int(line[22:26]) if (line.startswith("ATOM") or line.startswith("HETATM")) else -1
    resn = line[17:20].strip()
    if resn == LIG or (line.startswith("ATOM") and resi in sel):
        lines.append(line.rstrip())
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines)} atom lines)")

# write pocket FASTA (one entry per segment, fused gaps as iterated segments)
AA = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
      "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
      "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
      "SEC":"U","PYL":"O","ASX":"B","GLX":"Z","UNK":"X"}
fa = OUT.parent / f"7OTW_pocket_{int(CUTOFF)}A.fasta"
with fa.open("w") as f:
    for i, (a, b) in enumerate(segs):
        seq = "".join(AA.get(res_atoms[(r, sel[r][0])][1], "X") for r in range(a, b+1) if r in sel)
        f.write(f">pocket_seg{i} residues{a}-{b} len{len(seq)}\n{seq}\n")
print(f"wrote {fa}")
