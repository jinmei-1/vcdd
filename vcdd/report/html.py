"""Standalone HTML report for a vcdd run.

Design rules:

* **Self-contained.** SVGs are inlined, so a single file can be emailed.
* **Partial runs report.** Every section is optional; a run that stopped after
  stage 1 still gets a report saying so.
* **Caveats travel with the numbers.** Every stage summary carries a
  ``warnings`` list, and those are rendered next to the results rather than
  buried at the bottom.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from vcdd.report.figures import build_all
from vcdd.runtime.paths import atomic_write_text
from vcdd.version import __version__

__all__ = ["build_report", "render_html"]

CSS = """
:root { --ink:#101418; --muted:#5b6670; --rule:#dfe4e8; --accent:#0072B2;
        --warn:#8a5a00; --warnbg:#fff6e0; --ok:#0b6b3a; }
* { box-sizing:border-box; }
body { font-family:'Liberation Sans',Arimo,Arial,sans-serif; color:var(--ink);
       margin:0; padding:0 0 4rem; line-height:1.5; background:#fff; }
.wrap { max-width:1080px; margin:0 auto; padding:2rem 1.5rem; }
h1 { font-size:1.7rem; margin:0 0 .2rem; }
h2 { font-size:1.15rem; margin:2.2rem 0 .6rem; padding-bottom:.3rem;
     border-bottom:2px solid var(--rule); }
h3 { font-size:.98rem; margin:1.3rem 0 .4rem; color:var(--muted);
     text-transform:uppercase; letter-spacing:.04em; }
.sub { color:var(--muted); font-size:.86rem; margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; font-size:.82rem; margin:.5rem 0 1rem; }
th,td { text-align:left; padding:.34rem .5rem; border-bottom:1px solid var(--rule);
        vertical-align:top; }
th { background:#f5f7f9; font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.55rem;
      margin:.6rem 0 1rem; }
.kv div { border:1px solid var(--rule); border-radius:5px; padding:.5rem .65rem; }
.kv .k { font-size:.7rem; color:var(--muted); text-transform:uppercase;
         letter-spacing:.03em; }
.kv .v { font-size:1.05rem; font-variant-numeric:tabular-nums; }
.warn { background:var(--warnbg); border-left:3px solid var(--warn); color:#4a3600;
        padding:.55rem .8rem; margin:.4rem 0; font-size:.83rem; border-radius:0 4px 4px 0; }
.note { color:var(--muted); font-size:.83rem; margin:.4rem 0 1rem; }
.fig { margin:1rem 0 1.6rem; }
.fig svg { max-width:100%; height:auto; }
.smiles { font-family:'Liberation Mono',monospace; font-size:.72rem;
          word-break:break-all; color:var(--muted); }
.pill { display:inline-block; font-size:.7rem; padding:.1rem .45rem; border-radius:9px;
        background:#eef2f5; color:var(--muted); margin-right:.25rem; }
.pill.ok { background:#e4f4ea; color:var(--ok); }
.pill.no { background:#fdecea; color:#8a2018; }
footer { margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--rule);
         color:var(--muted); font-size:.78rem; }
details { margin:.5rem 0; } summary { cursor:pointer; font-size:.85rem; color:var(--accent); }
pre { background:#f6f8fa; padding:.7rem; border-radius:5px; overflow-x:auto;
      font-size:.74rem; }
"""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _load_table(path: Path, n: int | None = None) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if df.empty:
            return None
        return df.head(n) if n else df
    except Exception:  # noqa: BLE001
        return None


def _kv(items: list[tuple[str, Any]]) -> str:
    cells = []
    for k, v in items:
        if v is None:
            continue
        if isinstance(v, float):
            v = f"{v:,.4g}"
        elif isinstance(v, int):
            v = f"{v:,d}"
        cells.append(f'<div><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>')
    return f'<div class="kv">{"".join(cells)}</div>' if cells else ""


def _warnings(items: Any) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        items = [items]
    return "".join(f'<div class="warn">{_esc(w)}</div>' for w in items)


def _table(df: pd.DataFrame | None, cols: list[str] | None = None,
           round_to: int = 3, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return '<p class="note">no data</p>'
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    d = d.head(max_rows)
    num = {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])}
    head = "".join(
        f'<th class="num">{_esc(c)}</th>' if c in num else f"<th>{_esc(c)}</th>"
        for c in d.columns
    )
    body = []
    for row in d.itertuples(index=False):
        tds = []
        for c, v in zip(d.columns, row):
            if c in num and pd.notna(v):
                txt = f"{v:,.{round_to}f}" if isinstance(v, float) else f"{v:,}"
                tds.append(f'<td class="num">{_esc(txt)}</td>')
            elif "smiles" in c.lower():
                tds.append(f'<td class="smiles">{_esc(v)}</td>')
            else:
                tds.append(f"<td>{_esc('' if pd.isna(v) else v)}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    more = ""
    if len(df) > max_rows:
        more = f'<p class="note">showing {max_rows} of {len(df):,} rows</p>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def _fig(figs: dict[str, Path], name: str, caption: str = "") -> str:
    path = figs.get(name)
    if path is None or not Path(path).exists():
        return ""
    svg = Path(path).read_text()
    svg = svg[svg.find("<svg"):] if "<svg" in svg else svg
    cap = f'<p class="note">{_esc(caption)}</p>' if caption else ""
    return f'<div class="fig">{svg}{cap}</div>'


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _section_overview(run_dir: Path, cfg, manifest: dict) -> str:
    stages = manifest.get("stages", {})
    ran = [s for s in ("stage1", "stage2", "stage3", "stage4") if (run_dir / s).exists()]
    tools = manifest.get("tools", {})
    rows = [
        ("run name", getattr(cfg, "name", None) if cfg else manifest.get("config", {}).get("name")),
        ("stages present", ", ".join(ran) or "none"),
        ("vcdd version", __version__),
        ("total seconds", stages.get("run", {}).get("seconds")),
    ]
    tool_pills = "".join(
        f'<span class="pill">{_esc(k)} {_esc(v)}</span>'
        for k, v in sorted(tools.items()) if v
    )
    return (
        "<h2>Run overview</h2>"
        + _kv(rows)
        + (f"<p>{tool_pills}</p>" if tool_pills else "")
        + _warnings(manifest.get("warnings"))
    )


def _section_inputs(run_dir: Path, manifest: dict) -> str:
    lib = manifest.get("stages", {}).get("library", {})
    rec = manifest.get("stages", {}).get("receptor", {})
    redock = _load_json(run_dir / "receptor" / "redock_validation.json")
    out = ["<h2>Inputs and validation</h2>"]
    if lib:
        out.append("<h3>Compound library</h3>")
        out.append(_kv([
            ("input molecules", lib.get("n_input")),
            ("accepted", lib.get("n_accepted")),
            ("rejected", lib.get("n_rejected")),
            ("shards", lib.get("n_shards")),
        ]))
        reasons = lib.get("top_rejection_reasons") or {}
        if reasons:
            out.append(_table(
                pd.DataFrame({"reason": list(reasons), "n": list(reasons.values())}),
                max_rows=10,
            ))
    if rec or redock:
        out.append("<h3>Receptor and box</h3>")
        out.append(_kv([
            ("prep method", rec.get("method")),
            ("receptor atoms", rec.get("n_atoms")),
            ("box centre", rec.get("box_center")),
            ("box size", rec.get("box_size")),
        ]))
    if redock:
        passed = redock.get("passed")
        out.append("<h3>Native-ligand redock gate</h3>")
        out.append(_kv([
            ("verdict", redock.get("verdict")),
            ("all-atom RMSD", redock.get("all_atom_rmsd")),
            ("core RMSD", redock.get("core_rmsd")),
            ("threshold", redock.get("threshold")),
            ("redock score", redock.get("score")),
        ]))
        if passed is False:
            out.append(_warnings([
                "the docking protocol did not reproduce the crystallographic pose; "
                "scores from this receptor rank compounds but should not be read as "
                "affinities"
            ]))
        out.append(_warnings(redock.get("warnings")))
    return "".join(out)


def _section_stage1(run_dir: Path, manifest: dict, figs: dict) -> str:
    s = manifest.get("stages", {}).get("stage1", {})
    thr = _load_json(run_dir / "stage1" / "throughput.json")
    rank = _load_table(run_dir / "stage1" / "ranking.parquet")
    if rank is None and not s:
        return ""
    return "".join([
        "<h2>Stage 1 &mdash; fast screen</h2>",
        _kv([
            ("molecules docked",
             s.get("n_scored") or s.get("n_docked")
             or (len(rank) if rank is not None else None)),
            ("failed", s.get("n_failed")),
            ("engine", s.get("engine")),
            ("variant", s.get("variant")),
            ("thread", s.get("thread")),
            ("GPU batch size", s.get("gpu_batch_size")),
            ("reproducible", s.get("deterministic")),
            ("best score", s.get("best_score")),
            ("ligands/hour", thr.get("ligands_per_hour")),
            ("core-hours per 1M", thr.get("core_hours_per_million")),
            ("wall seconds", s.get("seconds")),
        ]),
        _fig(figs, "stage1_scores",
             "Left: score distribution over the whole library. Right: sorted score "
             "curve; the knee is where the shortlist stops being enriched."),
        "<h3>Top of the stage-1 ranking</h3>",
        _table(rank, ["mol_id", "smiles", "score", "n_rotatable", "seconds"], max_rows=15),
        _warnings(s.get("warnings")),
    ])


def _section_stage2(run_dir: Path, manifest: dict, figs: dict) -> str:
    s = _load_json(run_dir / "stage2" / "stage2_summary.json") or \
        manifest.get("stages", {}).get("stage2", {})
    cons = _load_table(run_dir / "stage2" / "consensus_ranking.parquet")
    qc = _load_table(run_dir / "stage2" / "pose_qc.parquet")
    if cons is None and not s:
        return ""
    pose = s.get("pose_qc") or {}
    target = s.get("target") or {}
    body = [
        "<h2>Stage 2 &mdash; Boltz co-folding</h2>",
        '<p class="note">Boltz does not rescore the stage-1 pose. It re-predicts '
        "the protein&ndash;ligand complex from sequence and SMILES, so it is "
        "orthogonal evidence about whether the complex is plausible at all, not a "
        "more accurate docking score. <code>dock_score</code> below is the stage-1 "
        "score carried forward unchanged.</p>",
        _kv([
            ("shortlist size", s.get("n_input")),
            ("scored", s.get("n_scored")),
            ("failed", s.get("n_failed")),
            ("rejected before submission", s.get("n_rejected")),
            ("backend", s.get("backend")),
            ("engine", s.get("engine")),
            ("engine version", s.get("engine_version")),
            ("live mode", s.get("livemode")),
            ("API runs", s.get("n_runs")),
            ("estimated cost (USD)", s.get("estimated_cost_usd")),
            ("fusion policy", s.get("fusion")),
            ("best binding confidence", s.get("best_binding_confidence")),
            ("Spearman vs stage 1", s.get("spearman_vs_stage1")),
            ("wall seconds", s.get("seconds")),
        ]),
    ]
    if s.get("livemode") is False:
        body.append(
            '<p class="note"><strong>Test-mode API key.</strong> Every number in '
            "this section is synthetic and must not be used to pick compounds.</p>"
        )
    if target.get("chains"):
        chains = ", ".join(
            f"{c.get('chain_id')}:{c.get('length')}aa" for c in target["chains"]
        )
        pockets = target.get("pocket_sizes") or {}
        body.append(
            f'<p class="note">target sent to the API: {_esc(chains)}'
            + (f'; pocket residues {_esc(str(pockets))}' if pockets else "")
            + "</p>"
        )
    body.append(_fig(figs, "stage1_vs_boltz",
                     "Docking score against binding confidence. The two readouts are "
                     "independent, so compounds in the upper-left &mdash; confident "
                     "complex, mediocre docking score &mdash; are what stage 2 adds."))
    body.append(_fig(figs, "boltz_confidence_distribution",
                     "Binding confidence across the shortlist. The dashed line is the "
                     "0.7 triage threshold the API documentation calls high confidence."))
    body.append(_fig(figs, "consensus_agreement",
                     "Pairwise rank correlation between the fused score inputs "
                     "(rank_fusion policy only)."))
    body.append("<h3>Consensus ranking</h3>")
    body.append(_table(cons, ["consensus_rank", "mol_id", "smiles", "dock_score",
                              "binding_confidence", "optimization_score",
                              "consensus_score", "ligand_in_box"],
                       max_rows=20))
    if qc is not None and len(qc):
        body.append("<h3>Pose quality control</h3>")
        body.append(
            '<p class="note">Nothing constrains the Boltz-predicted ligand to the '
            "docking box except the pocket residues that were sent with the target, "
            "so a pose outside the box means the model placed the ligand somewhere "
            "the screen was not asking about.</p>"
        )
        body.append(_table(qc, ["mol_id", "ligand_in_box", "centroid_offset_A",
                                "frac_atoms_in_box", "max_axis_excess_A",
                                "pose_n_atoms", "pose_error"], max_rows=15))
        if pose.get("fraction_in_box") is not None:
            body.append(
                f'<p class="note">{pose.get("n_in_box")} of {pose.get("n_checked")} '
                f'checked poses are fully inside the box '
                f'({float(pose["fraction_in_box"]):.0%}); median centroid offset '
                f'{pose.get("median_centroid_offset_A")} &Aring;.</p>'
            )
    body.append(_warnings(s.get("warnings")))
    return "".join(body)


def _section_stage3(run_dir: Path, manifest: dict, figs: dict) -> str:
    s = _load_json(run_dir / "stage3" / "stage3_summary.json")
    scores = _load_table(run_dir / "stage3" / "objective_scores.parquet")
    if scores is None and not s:
        return ""
    obj = s.get("objective_stats") or {}
    body = [
        "<h2>Stage 3 &mdash; virtual-cell scoring</h2>",
        _kv([
            ("compounds scored", s.get("n_compounds")),
            ("cell line", s.get("cell_line")),
            ("dose", f"{s.get('dose_um')} uM" if s.get("dose_um") is not None else None),
            ("time", f"{s.get('time_h')} h" if s.get("time_h") is not None else None),
            ("gene space", f"{s.get('gene_space')} ({s.get('n_genes')} genes)"),
            ("backends scored", ", ".join(s.get("backends_scored") or []) or None),
            ("reference-only", ", ".join(s.get("backends_reference_only") or []) or None),
            ("out of domain", s.get("n_out_of_domain")),
            ("objective median", obj.get("median")),
            ("objective max", obj.get("max")),
        ]),
        '<p class="note">Genes requested up: '
        f'{_esc(", ".join(s.get("genes_up") or []) or "none")} &middot; down: '
        f'{_esc(", ".join(s.get("genes_down") or []) or "none")}. Component weights: '
        f'{_esc(json.dumps(s.get("objective_weights") or {}))}.</p>',
        _fig(figs, "stage3_components",
             "Objective decomposition for the top compounds. A high total driven by one "
             "component only is a different claim from a balanced profile."),
        _fig(figs, "stage3_objective_vs_docking",
             "Binding score against cellular objective. Weak correlation is expected: "
             "the two axes are measuring different things."),
        "<h3>Objective ranking</h3>",
        _table(scores, ["objective_rank", "mol_id", "smiles", "objective", "gene_score",
                        "pathway_score", "target_consistency", "specificity",
                        "max_tanimoto", "out_of_domain"], max_rows=20),
    ]
    if s.get("tf_routed_requests"):
        body.append(
            f'<p class="note">requested genes scored through inferred TF activity rather '
            f'than measured expression: {_esc(", ".join(s["tf_routed_requests"]))}</p>'
        )
    body.append(_warnings(s.get("warnings")))
    body.append(
        '<div class="warn">Protein-level readouts here are <em>inferred activities</em> '
        "from a transcriptional footprint, not measured protein abundance. "
        "A compound flagged out-of-domain has no close analog in the reference "
        "perturbation set and its predicted signature is an extrapolation.</div>"
    )
    return "".join(body)


def _section_stage4(run_dir: Path, manifest: dict, figs: dict) -> str:
    s = _load_json(run_dir / "stage4" / "stage4_summary.json")
    recs = _load_table(run_dir / "stage4" / "analog_recommendations.csv")
    sar = _load_table(run_dir / "stage4" / "sar_rules.csv")
    if recs is None and not s:
        return ""
    dc = s.get("delta_cell_objective") or {}
    dd = s.get("delta_dock_score") or {}
    body = [
        "<h2>Stage 4 &mdash; how to modify the top compounds</h2>",
        _kv([
            ("parents", s.get("n_parents")),
            ("analogs proposed", s.get("n_proposed")),
            ("passed guardrails", s.get("n_guardrail_pass")),
            ("generators", ", ".join(s.get("generators_used") or []) or None),
            ("re-dock protocol", s.get("redock_mode")),
            ("gene of interest", s.get("gene_of_interest")),
            ("analogs improving cell objective", dc.get("n_improved")),
            ("analogs improving docking", dd.get("n_improved")),
            ("Pareto front 1", s.get("n_pareto_front1")),
            ("SAR rules", s.get("n_sar_rules")),
            ("wall seconds", s.get("seconds")),
        ]),
    ]
    reasons = s.get("guardrail_rejection_reasons") or {}
    if reasons:
        body.append("<h3>Guardrail funnel</h3>")
        body.append(_table(
            pd.DataFrame({"rejection reason": list(reasons), "n": list(reasons.values())}),
            max_rows=12,
        ))
    body.append(_fig(figs, "stage4_deltas",
                     "Each analog against its own parent. Bottom-left is better on both "
                     "axes; the shaded band marks docking gains beyond scoring noise."))
    body.append("<h3>Recommended modifications</h3>")
    body.append(_table(recs, ["rank", "parent_id", "analog_id", "analog_smiles",
                              "generator", "transformation", "delta_dock_score",
                              "delta_cell_objective", "design_score", "pareto_rank",
                              "similarity_to_parent"], max_rows=25))
    body.append(_fig(figs, "stage4_sar",
                     "Median effect of each transformation across parents; n is the number "
                     "of parents contributing."))
    if sar is not None:
        top = sar[sar["metric"] == "delta_cell_objective"] if "metric" in sar else sar
        body.append("<h3>Transformation rules</h3>")
        body.append(_table(top, ["transformation", "n_parents", "n_parents_improved",
                                 "median_delta", "p_sign_test", "verdict"], max_rows=15))
    body.append(_warnings(s.get("warnings")))
    return "".join(body)


def _section_methods(cfg, manifest: dict) -> str:
    cfg_json = manifest.get("config") or (cfg.resolved_dict() if cfg else {})
    return "".join([
        "<h2>Methods and reproducibility</h2>",
        '<p class="note">The resolved configuration below, together with '
        "<code>manifest.json</code> and <code>run.db</code>, is everything needed to "
        "reproduce this run. Re-running the same command resumes rather than "
        "recomputes.</p>",
        "<details><summary>resolved configuration</summary><pre>"
        + _esc(json.dumps(cfg_json, indent=2, default=str))
        + "</pre></details>",
        "<details><summary>input files and checksums</summary><pre>"
        + _esc(json.dumps(manifest.get("inputs", {}), indent=2, default=str))
        + "</pre></details>",
    ])


def _section_limits() -> str:
    return """
<h2>What this pipeline does not tell you</h2>
<ul>
<li><strong>Docking ranks, it does not measure.</strong> Empirical scoring functions
carry roughly 2-3 kcal/mol of error and were fitted on crystallographic complexes.
Score differences under ~1 kcal/mol are not meaningful.</li>
<li><strong>One rigid receptor conformation.</strong> Induced fit, water-mediated
contacts, covalent chemistry, metalloenzyme coordination and PROTAC ternary
complexes are all outside this model.</li>
<li><strong>The virtual cell is a surrogate.</strong> Predicted transcriptional
responses come from models trained on perturbation atlases; a compound unlike
anything in the training set gets an extrapolation, flagged here as
out-of-domain, not a measurement.</li>
<li><strong>Protein readouts are inferred.</strong> Transcription-factor and pathway
activities are footprint inferences from mRNA, not proteomics.</li>
<li><strong>Analog scores are surrogate scores.</strong> Stage-4 rankings optimise
the same models used to score the parents, so they inherit those models' biases.
The guardrails constrain how far a generator can exploit them; they do not make
the underlying prediction correct.</li>
<li><strong>Nothing here replaces an assay.</strong> The output is a prioritised,
auditable list of what to make and test next.</li>
</ul>
"""


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def render_html(run_dir: str | Path, cfg=None) -> str:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    figs = build_all(run_dir)
    name = (getattr(cfg, "name", None)
            or manifest.get("config", {}).get("name") or run_dir.name)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        _section_overview(run_dir, cfg, manifest),
        _section_inputs(run_dir, manifest),
        _section_stage1(run_dir, manifest, figs),
        _section_stage2(run_dir, manifest, figs),
        _section_stage3(run_dir, manifest, figs),
        _section_stage4(run_dir, manifest, figs),
        _section_limits(),
        _section_methods(cfg, manifest),
    ]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vcdd report - {_esc(name)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(name)}</h1>
<p class="sub">virtual-cell-guided drug design &middot; vcdd {__version__} &middot;
generated {stamp}</p>
{"".join(s for s in sections if s)}
<footer>Generated by vcdd {__version__}. Figures are also written as standalone
SVG and PNG under <code>report/figures/</code>; every table shown here has a
full-length CSV or Parquet counterpart in the stage directories.</footer>
</div></body></html>
"""


def build_report(run_dir: str | Path, cfg=None) -> Path:
    """Write ``report/report.html`` and the figure directory. Returns the path."""
    run_dir = Path(run_dir)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(out / "report.html", render_html(run_dir, cfg))
