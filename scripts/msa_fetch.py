"""One-shot MSA fetch via colabfold's mmseqs2 client (no alphafold needed)."""
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from colabfold.input import parse_fasta
from colabfold.colabfold import run_mmseqs2

fasta = Path("msa_receptor_vcdd.fasta").read_text()
seqs, headers = parse_fasta(fasta)
print("query:", headers[0], "len", len(seqs[0]))

a3m_lines = run_mmseqs2(
    seqs, "msa_fetch", use_env=True, use_filter=True,
    host_url="https://api.colabfold.com", user_agent="vcdd-dnapkcs/1.0"
)
out = Path("msa_out"); out.mkdir(exist_ok=True)
for i, (h, a3m) in enumerate(zip(headers, a3m_lines)):
    name = h.replace("/", "_").replace(" ", "_")
    p = out / f"{name}.a3m"
    p.write_text(a3m + "\n")
    n_seqs = sum(1 for l in a3m.splitlines() if l and not l.startswith(">"))
    print(f"wrote {p} ({n_seqs} aligned sequences)")
