#!/usr/bin/env bash
# Collect the Stage-2 Boltz-2 predicted complexes into one folder (+ zip) for download.
#
#   bash scripts/bundle_stage2_complexes.sh
#
# Safe to re-run while stage 2 is still running: it copies whatever is finished.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-stage2_complexes}"
SRC="dnapkcs_run/stage2/boltz/local"

rm -rf "$OUT"
mkdir -p "$OUT/reference"

n=0
for wd in "$SRC"/*/; do
  mol="$(basename "$wd")"
  cif="$wd/out/boltz_results_input/predictions/input/input_model_0.cif"
  conf="$wd/out/boltz_results_input/predictions/input/confidence_input_model_0.json"
  [ -f "$cif" ] || continue
  cp "$cif" "$OUT/${mol}.cif"
  [ -f "$conf" ] && cp "$conf" "$OUT/${mol}.confidence.json"
  n=$((n + 1))
done

# references: the docking receptor, the co-crystal inhibitor, the docking box
cp dnapkcs_run/receptor/receptor_clean.pdb "$OUT/reference/receptor_7OTW_3701-4123.pdb"
cp DNA-PKcs/MBW_ligand.pdb "$OUT/reference/AZD7648_MBW_ligand.pdb"
cp dnapkcs_run/receptor/box.json "$OUT/reference/box.json"
[ -f dnapkcs_run/stage2/complex_analysis.csv ] && \
  cp dnapkcs_run/stage2/complex_analysis.csv "$OUT/complex_analysis.csv"

cat > "$OUT/README.md" <<EOF
# Stage-2 predicted complexes (Boltz-2, corrected receptor sequence)

$n complex(es), one file per molecule: \`<mol_id>.cif\` (chain A = DNA-PKcs
residues 3701-4123 in input order, chain LIG = the molecule) plus
\`<mol_id>.confidence.json\` (Boltz confidence/ipTM/pLDDT/PDE).

\`reference/\` holds what the analysis compares against:
* \`receptor_7OTW_3701-4123.pdb\` - the docking receptor (the crystal crop)
* \`AZD7648_MBW_ligand.pdb\` - the co-crystal inhibitor, for pocket comparison
* \`box.json\` - centre and size of the docking box (ATP site)

\`complex_analysis.csv\` - per-complex geometry: receptor/pocket C-alpha RMSD,
ligand-in-box fraction, coverage of the AZD7648 contact shell, clashes, burial,
polar contacts, ligand bond-length deviation.

Viewing in PyMOL:

    pymol reference/receptor_7OTW_3701-4123.pdb VCDD00004316.cif

    # then, in the PyMOL command line:
    align VCDD00004316 and chain A, reference/receptor_7OTW_3701-4123
    show cartoon, reference/receptor_7OTW_3701-4123
    show sticks, VCDD00004316 and chain LIG
    show sticks, reference/AZD7648_MBW_ligand
    zoom (VCDD00004316 and chain LIG)

The CIF B-factor column is the per-atom pLDDT (0-100), so
\`spectrum b, blue_white_red, VCDD00004316 and chain LIG\` colours the ligand by
confidence.

Reproduce / refresh this bundle:

    bash scripts/bundle_stage2_complexes.sh
    ~/miniconda3/envs/vcdd/bin/python scripts/analyze_stage2_complexes.py
EOF

ZIP="${OUT}_$(date +%Y%m%d).zip"
rm -f "$ZIP"
zip -qr "$ZIP" "$OUT"
echo "bundled $n complex(es) -> $OUT/ ($(du -sh "$OUT" | cut -f1)) and $ZIP ($(du -h "$ZIP" | cut -f1))"