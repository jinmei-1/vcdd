#!/usr/bin/env python
"""Build the offline viewer for every Boltz-2 complex that has an affinity value.

Writes ``paper/affinity_gallery/``:

* ``affinity_structures.html`` - self-contained (structures inlined, 3Dmol.js
  local) so it opens straight from ``file://`` with no server and no network.
* ``affinity_structures.csv`` - the same numbers in machine-readable form.
* ``campaign_structures.csv`` - the 500 campaign complexes, which have no
  affinity (see paper/stage2_affinity_pilot.md section 5.1 for why).

The nine structures are the six-molecule pilot plus the three-molecule control,
each predicted twice; the card shows both runs so the repeatability is visible
next to the number it affects.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "paper" / "affinity_gallery"

RUNS = {
    "pilot": {
        "dir": ROOT / "dnapkcs_affinity_pilot",
        "title": "试点：stage-2 短名单 6 个分子",
        "roles": {
            "VCDD00130178": "短名单（对接第 1）",
            "VCDD00076009": "短名单",
            "VCDD00071494": "短名单（亲本）",
            "AN000092": "stage-4 类似物（亲本 VCDD00071494 的单原子编辑）",
            "VCDD00052532": "短名单（DDR 修复特征最强）",
            "VCDD00012880": "短名单（对接最弱，-10.4）",
        },
    },
    "control": {
        "dir": ROOT / "dnapkcs_affinity_control",
        "title": "对照：AZD7648 + 两个对接拒绝分子",
        "roles": {
            "AZD7648": "阳性对照：7OTW 共晶配体（已知 DNA-PK 抑制剂）",
            "VCDD00015044": "阴性对照：对接 -5.5 kcal/mol",
            "VCDD00017774": "阴性对照：对接 -5.6 kcal/mol",
        },
    },
}

# pLDDT colouring, the AlphaFold convention, applied to the B-factor column
# (Boltz writes per-atom pLDDT there).
PLDDT_FUNC = """
function(atom) {
  var b = atom.b;
  if (b === undefined || b === null) return 0xcccccc;
  if (b >= 90) return 0x0053d6;
  if (b >= 70) return 0x65cbf3;
  if (b >= 50) return 0xffdb13;
  return 0xff7d45;
}
"""


def _affinity(path: Path) -> dict[str, Any] | None:
    files = sorted(path.glob("out/**/affinity_*.json"))
    if not files:
        return None
    data = json.loads(files[0].read_text())
    return {
        "value": float(data["affinity_pred_value"]),
        "prob": float(data["affinity_probability_binary"]),
        "v1": data.get("affinity_pred_value1"),
        "v2": data.get("affinity_pred_value2"),
    }


def _cif(path: Path) -> Path | None:
    files = sorted(path.glob("out/**/*_model_0.cif"))
    return files[0] if files else None


def cif_to_pdb(text: str) -> str:
    """Convert a Boltz predicted complex CIF into PDB text for the viewer.

    3Dmol reads PDB reliably - chain, residue name and the B-factor column all
    land where a selector expects them - while its mmCIF handling of
    ``label_*`` vs ``auth_*`` fields is an extra thing to get wrong. The
    B-factor column is Boltz's per-atom pLDDT (66.8-98.7 for these complexes),
    which is what the viewer colours by.
    """
    lines = text.splitlines()
    try:
        start = next(k for k, l in enumerate(lines) if l.startswith("_atom_site.group_PDB"))
    except StopIteration:
        return ""
    names = [l.strip().split(".")[-1] for l in lines[start:] if l.startswith("_atom_site.")]
    col = {name: i for i, name in enumerate(names)}
    out: list[str] = []
    serial = 0
    for line in lines[start + len(names):]:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        f = line.split()
        if len(f) < len(names):
            continue
        serial += 1
        group = f[col["group_PDB"]][:6]
        name = f[col["label_atom_id"]][:4]
        resn = f[col["label_comp_id"]][:3]
        chain = (f[col["auth_asym_id"]] if "auth_asym_id" in col else f[col["label_asym_id"]])[:1]
        try:
            resseq = int(float(f[col.get("auth_seq_id", col.get("label_seq_id", 0))]))
        except (ValueError, IndexError):
            resseq = 0
        x, y, z = (float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]), float(f[col["Cartn_z"]]))
        b = float(f[col["B_iso_or_equiv"]]) if "B_iso_or_equiv" in col else 0.0
        element = f[col["type_symbol"]][:2] if "type_symbol" in col else name[:1]
        out.append(
            f"{group:<6s}{serial:5d} {name:<4s}{'':1s}{resn:>3s} {chain:1s}"
            f"{resseq:4d}{'':1s}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}"
            f"{'':10s}{element:>2s}"
        )
    return "\n".join(out) + "\n" if out else ""


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cards: list[dict[str, Any]] = []
    for group, spec in RUNS.items():
        base: Path = spec["dir"]
        scores = pd.read_parquet(base / "stage2" / "scores.parquet").set_index("mol_id")
        qc_path = base / "stage2" / "pose_qc.parquet"
        qc = (pd.read_parquet(qc_path).set_index("mol_id")
              if qc_path.exists() else pd.DataFrame())
        live = base / "stage2" / "boltz" / "local"
        first = base / "stage2" / "boltz" / "local_run1"
        for mol_dir in sorted(live.iterdir()):
            if not mol_dir.is_dir():
                continue
            mol_id = mol_dir.name
            now, before = _affinity(mol_dir), _affinity(first / mol_id)
            if now is None:
                continue
            cif = _cif(mol_dir)
            row = scores.loc[mol_id] if mol_id in scores.index else {}
            qrow = qc.loc[mol_id] if len(qc) and mol_id in qc.index else {}
            value = now["value"]
            cards.append({
                "group": group,
                "group_title": spec["title"],
                "mol_id": mol_id,
                "role": spec["roles"].get(mol_id, ""),
                "value": value,
                "ic50_um": 10 ** value,
                "prob": now["prob"],
                "value_run1": None if before is None else before["value"],
                "delta": None if before is None else value - before["value"],
                "model_gap": (abs(now["v1"] - now["v2"])
                              if now["v1"] is not None and now["v2"] is not None else None),
                "confidence": _f(row, "binding_confidence"),
                "iptm": _f(row, "iptm"),
                "stage1": _f(row, "stage1_score"),
                "in_box": _b(qrow, "ligand_in_box"),
                "offset": _f(qrow, "centroid_offset_A"),
                "ca_rmsd": _f(qrow, "receptor_ca_rmsd_A"),
                "cif": str(cif.relative_to(ROOT)) if cif else None,
                "cif_text": cif_to_pdb(cif.read_text()) if cif else "",
            })
    cards.sort(key=lambda c: c["value"])
    return cards, campaign_rows()


def _f(row: Any, key: str) -> float | None:
    try:
        value = row[key]
    except Exception:  # noqa: BLE001 - missing row/column
        return None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _b(row: Any, key: str) -> bool | None:
    value = _f(row, key)
    return None if value is None else bool(value)


def campaign_rows() -> list[dict[str, Any]]:
    """The 500 campaign complexes: structures yes, affinity no."""
    base = ROOT / "dnapkcs_run" / "stage2"
    scores = pd.read_parquet(base / "scores.parquet")
    qc = pd.read_parquet(base / "pose_qc.parquet").set_index("mol_id")
    out: list[dict[str, Any]] = []
    for _, row in scores.iterrows():
        mol_id = str(row["mol_id"])
        cif = _cif(base / "boltz" / "local" / mol_id)
        qrow = qc.loc[mol_id] if mol_id in qc.index else {}
        out.append({
            "mol_id": mol_id,
            "stage1_score": row.get("stage1_score"),
            "binding_confidence": row.get("binding_confidence"),
            "iptm": row.get("iptm"),
            "ligand_in_box": _b(qrow, "ligand_in_box"),
            "centroid_offset_A": _f(qrow, "centroid_offset_A"),
            "structure": str(cif.relative_to(ROOT)) if cif else None,
            "affinity_pred_value": None,
        })
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if digits and value < 0 else f"{value:.{digits}f}"
    return str(value)


def build_html(cards: list[dict[str, Any]], campaign: list[dict[str, Any]]) -> str:
    payload = [
        {k: v for k, v in card.items() if k != "cif_text"} | {"pdb": card["cif_text"]}
        for card in cards
    ]
    rows_html = []
    for i, card in enumerate(cards, start=1):
        delta = card["delta"]
        delta_html = ("–" if delta is None else
                      f'<span class="{"warn" if abs(delta) > 0.2 else ""}">{delta:+.3f}</span>')
        rows_html.append(f"""
    <tr>
      <td>{i}</td>
      <td><b>{card['mol_id']}</b><div class="role">{card['role']}</div></td>
      <td class="num strong">{card['value']:+.3f}</td>
      <td class="num">{card['ic50_um']:.3f}</td>
      <td class="num">{card['ic50_um'] * 1000:.0f}</td>
      <td class="num">{6 - card['value']:.2f}</td>
      <td class="num">{card['prob']:.3f}</td>
      <td class="num">{_fmt(card['value_run1'])}</td>
      <td class="num">{delta_html}</td>
      <td class="num">{_fmt(card['model_gap'])}</td>
      <td class="num">{_fmt(card['confidence'])}</td>
      <td class="num">{_fmt(card['iptm'])}</td>
      <td class="num">{_fmt(card['stage1'], 1)}</td>
      <td class="num">{_fmt(card['in_box'])}</td>
      <td class="num">{_fmt(card['offset'], 2)}</td>
    </tr>""")

    viewer_cards = []
    for i, card in enumerate(cards):
        viewer_cards.append(f"""
  <div class="card">
    <div class="cardhead">
      <span class="idx">{i + 1}</span>
      <b>{card['mol_id']}</b>
      <span class="badge {'ctl' if card['group'] == 'control' else ''}">{'对照' if card['group'] == 'control' else '短名单'}</span>
      <span class="aff">{card['value']:+.3f} · IC50 {card['ic50_um']:.3f} µM ({card['ic50_um'] * 1000:.0f} nM) · pIC50 {6 - card['value']:.2f} · P(binder) {card['prob']:.3f}</span>
    </div>
    <div class="role">{card['role']}</div>
    <div class="viewer" id="v{i}"></div>
    <div class="btns">
      <button onclick="zoomLig({i})">配体特写</button>
      <button onclick="zoomAll({i})">整体</button>
      <button onclick="toggleSpin({i}, this)">旋转</button>
      <button onclick="toggleSurface({i}, this)">口袋表面</button>
    </div>
    <div class="meta">
      confidence {_fmt(card['confidence'])} · ipTM {_fmt(card['iptm'])} ·
      stage1 {_fmt(card['stage1'], 1)} kcal/mol · 配体在 box 内 {_fmt(card['in_box'])} ·
      质心偏移 {_fmt(card['offset'], 2)} Å · 受体 Cα RMSD {_fmt(card['ca_rmsd'], 2)} Å ·
      两次运行 Δ {_fmt(card['delta'])} · 两模型分歧 {_fmt(card['model_gap'])}
    </div>
    <div class="path">{card['cif'] or ''}</div>
  </div>""")

    campaign_table = "\n".join(
        f"<tr><td>{r['mol_id']}</td><td class='num'>{_fmt(r['stage1_score'], 1)}</td>"
        f"<td class='num'>{_fmt(r['binding_confidence'])}</td>"
        f"<td class='num'>{_fmt(r['iptm'])}</td>"
        f"<td class='num'>{_fmt(r['ligand_in_box'])}</td>"
        f"<td class='path'>{r['structure'] or ''}</td>"
        f"<td class='num none'>未计算</td></tr>"
        for r in sorted(campaign, key=lambda r: (r["binding_confidence"] or 0), reverse=True)
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Boltz-2 预测复合体 + 亲和力头结果</title>
<script src="3Dmol-min.js"></script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
         margin: 0; padding: 24px 32px 80px; background: #f6f7f9; color: #1c1f23; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  h2 {{ font-size: 17px; margin: 32px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #dfe3e8; }}
  .sub {{ color: #5b6472; font-size: 13px; line-height: 1.7; max-width: 1100px; }}
  table {{ border-collapse: collapse; background: #fff; font-size: 13px; margin-top: 10px; }}
  th, td {{ border: 1px solid #e3e7ec; padding: 5px 8px; text-align: left; }}
  th {{ background: #eef1f5; position: sticky; top: 0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.strong {{ font-weight: 700; }}
  td.none {{ color: #a33; }}
  td.path, .path {{ font-family: ui-monospace, monospace; font-size: 11px; color: #6b7480; }}
  .role {{ color: #5b6472; font-size: 12px; }}
  .warn {{ color: #b3261e; font-weight: 700; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 18px; }}
  .card {{ background: #fff; border: 1px solid #e3e7ec; border-radius: 10px; padding: 12px 14px; }}
  .cardhead {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 14px; }}
  .idx {{ background: #1c1f23; color: #fff; border-radius: 50%; width: 20px; height: 20px;
          display: inline-flex; align-items: center; justify-content: center; font-size: 11px; }}
  .badge {{ background: #e7f0ff; color: #1a4f9c; border-radius: 10px; padding: 1px 8px; font-size: 11px; }}
  .badge.ctl {{ background: #ffe9d6; color: #9c4a1a; }}
  .aff {{ margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .viewer {{ height: 300px; margin: 8px 0; border: 1px solid #eef1f5; border-radius: 6px; position: relative; }}
  .btns {{ display: flex; gap: 6px; }}
  button {{ font: inherit; font-size: 12px; padding: 3px 10px; border: 1px solid #cfd6de;
            background: #fff; border-radius: 6px; cursor: pointer; }}
  button:hover {{ background: #f0f3f7; }}
  .meta {{ font-size: 11.5px; color: #5b6472; margin-top: 6px; line-height: 1.6; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f0b429; padding: 10px 14px;
           font-size: 13px; line-height: 1.75; margin: 12px 0; max-width: 1100px; }}
  .scroll {{ max-height: 460px; overflow: auto; }}
</style>
</head>
<body>
<h1>Boltz-2 预测复合体 + 亲和力头结果</h1>
<div class="sub">
  9 个分子的复合体结构（Boltz-2.2.1 本地，2×RTX 3090 Ti）与它们的亲和力头输出，按预测亲和力从强到弱排序。
  每个分子预测了两次，表中同时给出两次的值和差值。蛋白按 pLDDT 着色（深蓝 ≥90，浅蓝 ≥70，黄 ≥50，橙 &lt;50），
  配体为球棍模型，点击「配体特写」把视角收到配体上。
</div>

<div class="note">
  <b>单位。</b> <code>affinity_pred_value = log10(IC50)</code>，其中 IC50 以 <b>µM</b> 计（Boltz README 原文：
  "reports a binding affinity value as log10(IC50), derived from an IC50 measured in µM"），所以这个数本身是无量纲的对数。
  换算：<code>IC50(µM) = 10^value</code>，<code>IC50(nM) = 10^value × 1000</code>，<code>pIC50 = 6 − value</code>；
  <code>value = 0</code> 即 IC50 = 1 µM（pIC50 6.0）。它<b>不是</b> kcal/mol，也不是结合自由能。<br>
  <b>读数须知。</b> <b>越小越强</b>；
  <code>P(binder)</code> 是 Boltz 的 binder/decoy 判别概率（官方口径：前者用于比较 binder 之间的强弱，
  后者用于命中发现）。这两个数都是<b>预测值</b>，不是实测亲和力。本页最重要的一行是第 7 个分子
  <b>AZD7648</b>：它是 7OTW 的共晶配体、已知 DNA-PK 抑制剂，却被排在这批分子的中游；
  两次运行的差值（Δ 列）中位 0.10–0.16、最大 0.38 log 单位，亲本/类似物对（第 5、6 个）的差异符号在两次运行间翻转。
  因此这批数值只能当作粗筛信号，不能用来宣称效价或类似物优劣。详见 <code>paper/stage2_affinity_pilot.md</code>。<br><br>
  <b>AZD7648 的实测值（ChEMBL CHEMBL4439259，J. Med. Chem. 2020）：</b>
  生化 IC50 = 0.6 nM（pIC50 9.22），细胞内 IC50 = 90–100 nM（pIC50 ≈ 7.0）。
  模型给出 741 nM（pIC50 6.13）：对生化口径差约 3.1 log，对细胞口径差约 0.9 log。
  无论按哪个口径，AZD7648 都被排在这 6 个生成分子的第 5 位。
</div>

<h2>数值总表</h2>
<table>
  <tr><th>#</th><th>分子</th><th>affinity_pred_value<br>= log10(IC50/µM)，越小越强</th><th>IC50<br>(µM)</th>
      <th>IC50<br>(nM)</th><th>pIC50<br>= 6 − value</th>
      <th>P(binder)</th><th>第一次运行</th><th>Δ(两次)</th><th>两模型分歧</th>
      <th>confidence</th><th>ipTM</th><th>stage1<br>(kcal/mol)</th><th>配体<br>在 box 内</th><th>质心偏移<br>(Å)</th></tr>
{''.join(rows_html)}
</table>

<h2>结构（可交互）</h2>
<div class="grid">
{''.join(viewer_cards)}
</div>

<h2>主运行 500 个复合体（无亲和力）</h2>
<div class="sub">
  这 500 个复合体结构已经预测完成并已随 <code>paper/Data_Attachment_JCIM.zip</code> 打包，但<b>没有亲和力值</b>：
  Boltz 的亲和力头需要一个由「开启亲和力的结构预测那一趟」写出的
  <code>pre_affinity_&lt;id&gt;.npz</code>，这 500 个目录里没有该文件，事后补算会直接
  <code>FileNotFoundError</code>（详见 <code>paper/stage2_affinity_pilot.md</code> 第 5.1 节）。
  下表按 confidence 排序，<code>structure</code> 列是每个复合体 CIF 的路径。
</div>
<div class="scroll">
<table>
  <tr><th>mol_id</th><th>stage1</th><th>confidence</th><th>ipTM</th><th>配体在 box 内</th><th>结构文件</th><th>亲和力</th></tr>
{campaign_table}
</table>
</div>

<script>
const STRUCTURES = {json.dumps(payload, ensure_ascii=False)};
const viewers = {{}}, spinning = {{}}, surfaces = {{}};
const PLDDT = {PLDDT_FUNC};

function draw(i) {{
  const el = document.getElementById('v' + i);
  const v = $3Dmol.createViewer(el, {{backgroundColor: 'white'}});
  v.addModel(STRUCTURES[i].pdb, 'pdb');
  v.setStyle({{}}, {{cartoon: {{colorfunc: PLDDT}}}});
  v.setStyle({{resn: 'LIG'}}, {{stick: {{radius: 0.16, colorscheme: 'Jmol'}},
                              sphere: {{scale: 0.28, colorscheme: 'Jmol'}}}});
  v.zoomTo({{resn: 'LIG'}});
  v.render();
  viewers[i] = v;
}}
function zoomLig(i) {{ viewers[i].zoomTo({{resn: 'LIG'}}); viewers[i].render(); }}
function zoomAll(i) {{ viewers[i].zoomTo(); viewers[i].render(); }}
function toggleSpin(i, btn) {{
  spinning[i] = !spinning[i];
  viewers[i].spin(spinning[i] ? 'y' : false);
  btn.style.background = spinning[i] ? '#dbe9ff' : '#fff';
}}
function toggleSurface(i, btn) {{
  const v = viewers[i];
  if (surfaces[i]) {{ v.removeSurface(surfaces[i]); surfaces[i] = null; btn.style.background = '#fff'; }}
  else {{
    surfaces[i] = v.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.7, color: '#cfd8e3'}},
                               {{resn: 'LIG', within: {{distance: 5, sel: {{resn: 'LIG'}}}}}});
    btn.style.background = '#dbe9ff';
  }}
  v.render();
}}
STRUCTURES.forEach((s, i) => draw(i));
</script>
</body>
</html>
"""


def write_csv(cards: list[dict[str, Any]], campaign: list[dict[str, Any]]) -> None:
    with (GALLERY / "affinity_structures.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "group", "mol_id", "role", "value", "ic50_um", "prob", "value_run1",
            "delta", "model_gap", "confidence", "iptm", "stage1", "in_box",
            "offset", "ca_rmsd", "cif",
        ])
        writer.writeheader()
        for card in cards:
            writer.writerow({k: card[k] for k in writer.fieldnames})
    pd.DataFrame(campaign).to_csv(GALLERY / "campaign_structures.csv", index=False)


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    cards, campaign = collect()
    (GALLERY / "affinity_structures.html").write_text(build_html(cards, campaign))
    write_csv(cards, campaign)
    n_cif = sum(1 for c in cards if c["cif_text"])
    print(f"gallery: {len(cards)} structures with affinity ({n_cif} CIF inlined), "
          f"{len(campaign)} campaign structures listed without affinity")
    print(f"  -> {GALLERY / 'affinity_structures.html'}")
    print(f"  -> {GALLERY / 'affinity_structures.csv'}")
    print(f"  -> {GALLERY / 'campaign_structures.csv'}")


if __name__ == "__main__":
    main()
