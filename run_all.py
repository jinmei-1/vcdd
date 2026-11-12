#!/usr/bin/env python
"""vcdd workflow v2 - one file, every operation, in the right order.

    python run_all.py --list        # show the plan, gates and cost, run nothing
    python run_all.py --dry-run     # print the exact commands, run nothing
    python run_all.py               # run everything that is not done yet
    python run_all.py --only bench_aff
    python run_all.py --from decision
    python run_all.py --report      # rebuild summary.md / INDEX.md from existing outputs

Everything lands under one root (``workflow_v2/``) so the paper can find data and
the code can be published as-is:

    workflow_v2/
      configs/        every generated stage config (cite these in Methods)
      logs/           one log per stage plus workflow.log
      provenance.json hashes of receptor, MSA, configs, library + tool versions
      gates.json      every pre-registered gate and its measured value
      summary.md      the numbers, with the artifact each one comes from
      INDEX.md        artifact index: file -> what it is -> which claim it backs
      <stage>/        the stage's own run directory (vcdd layout inside)

The order follows paper/workflow_redesign.md: build the measuring instruments
(noise floor, calibration benchmarks, receptor gate) *before* ranking molecules.
Gates are pre-registered in GATES below and are evaluated automatically; a failed
gate stops the run unless --keep-going is given, and always downgrades the claim
in summary.md.

Only the GPU steps need a GPU; nothing here writes outside the repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("VCDD_PYTHON", Path.home() / "miniconda3/envs/vcdd/bin/python"))


def _prepare_env() -> None:
    """Export the paths the stage-3 cell model needs, if they are present."""
    root = Path(os.environ.get("VCDD_XPERT_DIR") or CONFIG["xpert_dir"]).expanduser()
    if root.is_dir():
        os.environ.setdefault("VCDD_XPERT_DIR", str(root))
    else:
        log(f"VCDD_XPERT_DIR is not set and {root} does not exist; the xpert cell "
            "backend will be unavailable and stage 3 will fail", level="WARN")


def _ensure_interpreter() -> None:
    """Re-exec under the environment that has vcdd's dependencies.

    ``python run_all.py`` with a different interpreter (a system python, say)
    imports pandas but not pyarrow, so the failure would surface deep inside the
    first parquet read - after stages had already started. Re-execing here means
    any python that can parse this file will run the workflow correctly.
    """
    if os.environ.get("VCDD_NO_REEXEC") or not PYTHON.exists():
        return
    try:
        same = Path(sys.executable).resolve() == PYTHON.resolve()
    except OSError:
        same = False
    if same:
        return
    print(f"[run_all] re-executing under {PYTHON}", flush=True)
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_interpreter()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Callable  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. settings - everything a reader of the Methods needs is in this block
# --------------------------------------------------------------------------- #

RUN = ROOT / "workflow_v2"
CONFIGS = RUN / "configs"
LOGS = RUN / "logs"
ENV_BIN = PYTHON.parent
TEMPLATE = ROOT / "run_dnapkcs.yaml"          # the campaign config, used as the base
CAMPAIGN = ROOT / "dnapkcs_run"               # the original run, read-only input here
RECEPTOR_SRC = "DNA-PKcs/7OTW_pocket3701_4123.pdb"
REFERENCE_LIGAND = "DNA-PKcs/MBW_ligand.pdb"  # enables the native-ligand redock gate
MSA = "msa_out/receptor_chain_A_vcdd.a3m"

CONFIG = {
    # instruments
    "noise_replicates": 3,          # identical inputs, identical receptor
    "noise_subset": 100,
    "offset_subset": 2000,
    "recover_unscored": True,       # re-dock the molecules the campaign could not dock
    "decision_region": 10_000,      # top-N of the campaign ranking re-docked 3x
    "decision_replicates": 3,
    # production
    "shortlist": 500,
    "cofold_affinity": True,
    # The 500-molecule affinity co-folding is launched by hand on this run dir
    # (it is a 14 h GPU job). Everything downstream reads from here.
    "cofold_run": "dnapkcs_affinity500",
    "affinity_config": "run_dnapkcs_affinity500.yaml",   # drives stage 3 and 4 on that run
    # Stage 3's XPert backend reads its checkpoint from this directory; the
    # variable is set here so `python run_all.py --only stage3` works in a shell
    # that has not exported it (the model itself is ~6 GB and not in the repo).
    "xpert_dir": str(Path.home() / "cell_models/xpert"),
    # Stage-2 calibration re-run on a receptor whose sequence is verified against
    # its MSA. The earlier benchmark is kept as evidence of the P10 defect.
    "benchmark_fixed_run": "dnapkcs_benchmark_fixed",        # only honoured when the affinity gate passes
    "stage3_top": 200,
    "stage4_parents": 10,
    # engine
    "devices": [0, 1],
    "boltz_concurrency": 2,
    "affinity_samples": 5,
    "affinity_sampling_steps": 200,
    # Boltz-2's affinity heads correlate with molecular weight (measured here:
    # r = -0.45 over 70 benchmark molecules). This flag is the model's own
    # correction for it; the benchmark must be re-run with the same setting.
    "affinity_mw_correction": True,
}

# Pre-registered gates (paper/workflow_redesign.md section 5). Do not tune these
# after seeing a result; change them here and say so in the paper.
GATES = {
    "redock_rmsd_max": 2.0,
    "dock_auc_min": 0.70,
    "dock_rho_min": 0.40,
    "aff_auc_min": 0.70,
    "aff_rho_min": 0.40,
    "aff_control_top_frac": 0.25,   # AZD7648 must land in the top quartile
    "delta_sigma_multiple": 2.0,    # a claimed difference must exceed 2 sigma
    "ambiguity_frac_max": 0.10,     # near-cut ambiguity allowed in the shortlist
    "offset_sd_max": 0.5,           # above this the campaign ranking is not portable
    "min_coverage": 0.95,           # scored fraction of every docking/co-folding run
    "mw_gap_max": 50.0,             # Da: actives and negatives must be size-matched
}

ACTIVE_TIERS = ("potent", "mid", "weak")

# Run directories produced outside the workflow that should be adopted rather
# than recomputed (symlinked into RUN/<stage>). Point a stage at an existing run
# to save its cost; delete the symlink to force a fresh run.
REUSE = {
    "benchmark": "dnapkcs_benchmark",
}


# --------------------------------------------------------------------------- #
# 2. plumbing
# --------------------------------------------------------------------------- #

def log(msg: str, *, level: str = "INFO") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {level:<5} {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "workflow.log").open("a") as fh:
        fh.write(line + "\n")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numbacache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vcdd(*args: str) -> list[str]:
    return [str(PYTHON), "-m", "vcdd.cli", *args]


def write_config(name: str, body: dict[str, Any], note: str) -> Path:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    path = CONFIGS / f"{name}.yaml"
    path.write_text(f"# {note}\n# generated by run_all.py - do not edit by hand\n"
                    + yaml.safe_dump(body, sort_keys=False, allow_unicode=True))
    return path


def base_config() -> dict[str, Any]:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["receptor"]["path"] = RECEPTOR_SRC
    cfg["receptor"]["reference_ligand_path"] = REFERENCE_LIGAND
    cfg["receptor"]["redock_validation"] = True
    cfg["receptor"]["redock_fail_hard"] = False
    cfg["stage2"]["local_boltz"].update({
        "devices": CONFIG["devices"],
        "max_concurrency": CONFIG["boltz_concurrency"],
        "use_msa_server": False,
        "protein_msa_files": {"A": MSA},
    })
    return cfg


def stage_dir(name: str) -> Path:
    return RUN / name


def done(marker: Path) -> bool:
    return marker.exists() and marker.stat().st_size > 0


def link_reused() -> dict[str, str]:
    """Adopt existing run directories instead of recomputing them."""
    adopted: dict[str, str] = {}
    for stage, external in REUSE.items():
        src, dst = ROOT / external, stage_dir(stage)
        if not src.exists() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src, target_is_directory=True)
            adopted[stage] = external
            log(f"{stage}: reusing existing run {external}/ (symlinked)")
        except OSError as exc:  # noqa: BLE001 - filesystems without symlinks
            log(f"{stage}: could not link {external}/ ({exc}); it will be recomputed", level="WARN")
    return adopted


def run_step(name: str, cmd: list[str], marker: Path, *, force: bool = False,
             dry_run: bool = False, timeout: int | None = None) -> bool:
    """Run one command, log it, return True if it ran (False if skipped)."""
    if done(marker) and not force:
        log(f"{name}: already done ({marker.relative_to(ROOT)}) - skipping")
        return False
    # never write into an adopted run directory: an external job may still be
    # filling it, and two boltz/docking processes on one directory corrupt each other
    if marker.parent.parent.is_symlink():
        log(f"{name}: {marker.parent.parent.name}/ is an adopted external run with no "
            f"{marker.name} yet - leaving it alone (wait for the external job, or delete the "
            f"symlink to recompute here)", level="WARN")
        return False
    log(f"{name}: {' '.join(cmd)}")
    if dry_run:
        return False
    LOGS.mkdir(parents=True, exist_ok=True)
    running = RUN / f"{name}.running"
    running.write_text(json.dumps({"stage": name, "pid": os.getpid(),
                                   "started": time.strftime("%F %T"),
                                   "cmd": " ".join(cmd)}, indent=2))
    try:
        return _run_subprocess(name, cmd, timeout)
    finally:
        running.unlink(missing_ok=True)


def _run_subprocess(name: str, cmd: list[str], timeout: int | None) -> bool:
    with (LOGS / f"{name}.log").open("a") as fh:
        fh.write(f"\n===== {time.strftime('%F %T')} =====\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=child_env(), stdout=fh,
                              stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}; "
                           f"see {LOGS / f'{name}.log'}")
    log(f"{name}: ok")
    return True


# --------------------------------------------------------------------------- #
# 3. metrics used by the gates (inlined so this file stands alone)
# --------------------------------------------------------------------------- #

def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rho and a two-sided p-value.

    Computed as the Pearson correlation of ranks: docking scores are quantised
    to 0.1 kcal/mol, so ties are common and the textbook rank-difference
    formula (``1 - 6*sum(d^2)/n(n^2-1)``) is wrong on this data.
    """
    import math
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rho = float(x.rank().corr(y.rank()))
    if rho != rho or abs(rho) >= 1:
        return rho, 0.0
    try:  # exact t-distribution p-value when scipy is present
        from scipy import stats
        return rho, float(stats.spearmanr(x, y).pvalue)
    except Exception:  # noqa: BLE001 - normal approximation is close enough
        t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
        return rho, math.erfc(abs(t) / math.sqrt(2))


def roc_auc(strength_pos: pd.Series, strength_neg: pd.Series) -> float:
    """P(a random positive scores stronger than a random negative); ties = 0.5."""
    pos, neg = list(strength_pos), list(strength_neg)
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def score_strength(df: pd.DataFrame, column: str) -> pd.Series:
    """Orient every score so that larger = stronger binder."""
    return df[column] if column == "affinity_probability_binary" else -df[column]


def benchmark_balance(truth: pd.DataFrame, strength: pd.Series, tier: str) -> dict[str, Any]:
    """Is the benchmark size-matched, and does the score track size?

    A calibration set whose actives and negatives differ in molecular weight
    cannot separate "ranks potency" from "likes big molecules": Boltz-2's
    affinity heads are known to track size, and we measured r = -0.45 against
    MW on this set with the model's own correction switched off.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    out: dict[str, Any] = {}
    mw = {}
    for mol_id, row in truth.iterrows():          # mol_id is the index here
        mol = Chem.MolFromSmiles(row["smiles"])
        mw[mol_id] = Descriptors.MolWt(mol) if mol else float("nan")
    mws = pd.Series(mw)
    act = truth.index[truth.tier.isin(ACTIVE_TIERS)]
    inact = truth.index[truth.tier == "inactive"]
    out["mw_actives_median"] = round(float(mws.reindex(act).median()), 1)
    out["mw_negatives_median"] = round(float(mws.reindex(inact).median()), 1)
    out["mw_gap"] = round(abs(out["mw_actives_median"] - out["mw_negatives_median"]), 1)
    s = strength.dropna()
    common = s.index.intersection(mws.dropna().index)
    if len(common) > 5:
        try:
            from scipy import stats
            r, p = stats.pearsonr(mws.reindex(common), s.reindex(common))
            out["r_score_vs_mw"] = round(float(r), 3)
            out["p_score_vs_mw"] = round(float(p), 5)
        except Exception:  # noqa: BLE001
            pass
    out["balanced"] = bool(out["mw_gap"] <= GATES["mw_gap_max"])
    return out


def benchmark_metrics(run: Path, label: str) -> dict[str, Any]:
    """ROC-AUC / Spearman / control placement for one scoring function."""
    truth = pd.read_csv(ROOT / "benchmark" / "known_inhibitors.csv")
    truth["mol_id"] = truth["mol_id"].astype(str)
    out: dict[str, Any] = {"label": label}
    frames = {"truth": truth.set_index("mol_id")}
    for key, path in (("dock", run / "stage1" / "ranking.parquet"),
                      ("aff", run / "stage2" / "scores.parquet")):
        if path.exists():
            frames[key] = pd.read_parquet(path).set_index("mol_id")
    if "dock" not in frames:
        return out | {"error": "no docking output"}
    data = frames["truth"].join(frames["dock"][["score"]].rename(columns={"score": "dock"}), how="left")
    if "aff" in frames and "affinity_pred_value" in frames["aff"].columns:
        for col in ("affinity_pred_value", "affinity_probability_binary", "binding_confidence", "iptm"):
            data[col] = frames["aff"][col] if col in frames["aff"].columns else float("nan")

    act = data[data.tier.isin(ACTIVE_TIERS)]
    inact = data[data.tier == "inactive"]
    for column in ("dock", "affinity_pred_value", "affinity_probability_binary"):
        if column not in data.columns or data[column].isna().all():
            continue
        strength = score_strength(data, column)
        sa, si = strength.loc[act.index].dropna(), strength.loc[inact.index].dropna()
        auc = roc_auc(sa, si)
        rho, p = spearman(act["pIC50"], strength.loc[act.index])
        entry: dict[str, Any] = {"roc_auc": round(auc, 3), "spearman": round(rho, 3),
                                 "p": None if p != p else round(p, 4),
                                 "n_actives": len(sa), "n_inactives": len(si)}
        if "AZD7648" in strength.index and strength.loc["AZD7648"] == strength.loc["AZD7648"]:
            rest = strength.drop("AZD7648").dropna()
            entry["control_stronger_than"] = int((rest > strength.loc["AZD7648"]).sum())
            entry["control_frac"] = round(float((rest > strength.loc["AZD7648"]).mean()), 3)
        entry["median_potent"] = round(float(data[data.tier == "potent"][column].median()), 3)
        entry["median_inactive"] = round(float(data[data.tier == "inactive"][column].median()), 3)
        entry["balance"] = benchmark_balance(data, strength, column)
        out[column] = entry
    return out


def replicates_mean(run_dirs: list[Path], column: str = "score") -> pd.DataFrame:
    """Per-molecule mean and SD across independent docking replicates."""
    frames = []
    for i, d in enumerate(run_dirs):
        p = d / "stage1" / "ranking.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p)[["mol_id", column]].rename(columns={column: f"r{i}"}))
    if not frames:
        return pd.DataFrame(columns=["mol_id", "mean", "sd", "n"])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="mol_id", how="outer")
    cols = [c for c in merged.columns if c.startswith("r")]
    merged["mean"] = merged[cols].mean(axis=1)
    merged["sd"] = merged[cols].std(axis=1)
    merged["n"] = merged[cols].notna().sum(axis=1)
    return merged


# --------------------------------------------------------------------------- #
# 4. stages
# --------------------------------------------------------------------------- #

@dataclass
class Stage:
    name: str
    title: str
    cost: str
    gate: str
    run: Callable[[bool, bool], None]
    marker: Callable[[], Path]
    outputs: list[str] = field(default_factory=list)
    optional: bool = False   # never runs unless asked for by name


def _prep(force: bool = False, dry: bool = False) -> None:
    """0.1-0.2 receptor frozen + native-ligand redock gate."""
    cfg = write_config("receptor_gate", base_config(),
                       "Phase 0.1/0.2: receptor preparation and the native-ligand redock gate.")
    run_step("prep_receptor",
             vcdd("prep-receptor", "--config", str(cfg), "--out", str(stage_dir("receptor_gate"))),
             stage_dir("receptor_gate") / "receptor" / "redock_validation.json",
             force=force, dry_run=dry)


def _instrument_cfg(name: str, smi: Path, note: str) -> Path:
    body = base_config()
    body["name"] = f"wfv2-{name.replace('_', '-')}"
    body["library"] = dict(body["library"], path=str(smi.relative_to(ROOT)))
    return write_config(name, body, note)


def _noise(force: bool = False, dry: bool = False) -> None:
    """0.3 engine noise floor: identical inputs, identical receptor, N times."""
    smi = ROOT / "instrument" / "noise_floor_100.smi"
    if not smi.exists():
        raise RuntimeError("run scripts/build_instrument_sets.py first "
                           "(missing instrument/noise_floor_100.smi)")
    cfg = _instrument_cfg("noise_floor", smi,
                          f"Phase 0.3: {CONFIG['noise_subset']} molecules docked "
                          f"{CONFIG['noise_replicates']}x for the noise floor.")
    for i in range(1, CONFIG["noise_replicates"] + 1):
        run_step(f"noise_r{i}",
                 vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir(f"noise_r{i}"))),
                 stage_dir(f"noise_r{i}") / "stage1" / "ranking.parquet",
                 force=force, dry_run=dry)


def _bench_dock(force: bool = False, dry: bool = False) -> None:
    """0.4 docking calibration on known inhibitors + same-assay negatives."""
    cfg = _instrument_cfg("benchmark", ROOT / "benchmark" / "known_inhibitors.smi",
                          "Phase 0.4/0.5: known DNA-PKcs inhibitors (0.1 nM - 1.2 uM),\n"
                          "# same-assay negatives (IC50 > 10 uM) and AZD7648.\n"
                          "# Calibrates stage 1 and stage 2.")
    run_step("bench_dock",
             vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir("benchmark"))),
             stage_dir("benchmark") / "stage1" / "ranking.parquet",
             force=force, dry_run=dry)


def _bench_aff(force: bool = False, dry: bool = False) -> None:
    """0.5 affinity calibration (Boltz-2 structure pass with the affinity head)."""
    cfg = CONFIGS / "benchmark.yaml"
    if not cfg.exists():
        _bench_dock(force=False, dry=True)
    run_step("bench_aff",
             vcdd("stage2", "--config", str(cfg), "--out", str(stage_dir("benchmark"))),
             stage_dir("benchmark") / "stage2" / "scores.parquet",
             force=force, dry_run=dry)


def _recover(force: bool = False, dry: bool = False) -> None:
    """3.0 re-dock the molecules the original campaign could not dock.

    The campaign scored 167,342 of 172,954 molecules. The missing 5,612 (3.2%)
    were lost to a ligand-preparation defect - meeko opens macrocycles and marks
    the opened bond with glue pseudo-atoms (types G0/CG0) that Vina-GPU cannot
    parse - so they were never screened and cannot appear in any shortlist.
    Re-docking only them restores full coverage in ~2 h instead of 62 h.
    """
    if not CONFIG["recover_unscored"]:
        log("recover: disabled by CONFIG['recover_unscored']")
        return
    frames = [pd.read_parquet(p) for p in sorted((CAMPAIGN / "stage1" / "scores").glob("shard_*.parquet"))]
    scored = pd.concat(frames, ignore_index=True)
    missing = scored[scored.score.isna()][["mol_id", "smiles"]].drop_duplicates("mol_id")
    lib = ROOT / "instrument" / "unscored_campaign.smi"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_text("\n".join(f"{r.smiles} {r.mol_id}" for r in missing.itertuples()) + "\n")
    log(f"recover: {len(missing)} molecules were never scored by the campaign "
        f"({len(missing) / max(1, len(scored)):.1%} of the library)")
    cfg = _instrument_cfg("recover", lib,
                          "Phase 3.0: the molecules the original campaign could not dock\n"
                          "# (meeko glue-atom defect), re-docked with the frozen receptor.")
    run_step("recover",
             vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir("recover"))),
             stage_dir("recover") / "stage1" / "ranking.parquet",
             force=force, dry_run=dry)


def _offset(force: bool = False, dry: bool = False) -> None:
    """3.1 is the receptor-preparation shift a constant offset?"""
    cfg = _instrument_cfg("offset_test", ROOT / "instrument" / "offset_test_2000.smi",
                          "Phase 3.1: paired against the campaign scores to test whether the\n"
                          "# receptor-preparation shift is a constant offset (2 h path) or\n"
                          "# molecule-dependent (62 h full re-dock path).")
    run_step("offset",
             vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir("offset"))),
             stage_dir("offset") / "stage1" / "ranking.parquet",
             force=force, dry_run=dry)


def _decision(force: bool = False, dry: bool = False) -> None:
    """3.2 decision region re-docked N times with the frozen receptor."""
    ranking = pd.read_parquet(CAMPAIGN / "stage1" / "ranking.parquet")
    top = ranking[ranking.score.notna()].nsmallest(CONFIG["decision_region"], "score")
    lib = ROOT / "instrument" / f"decision_{CONFIG['decision_region']}.smi"
    lib.parent.mkdir(exist_ok=True)
    lib.write_text("\n".join(f"{r.smiles} {r.mol_id}" for r in top.itertuples()) + "\n")
    cfg = _instrument_cfg("decision", lib,
                          f"Phase 3.2: top {CONFIG['decision_region']} of the campaign ranking,\n"
                          f"# docked {CONFIG['decision_replicates']}x with the frozen receptor.")
    for i in range(1, CONFIG["decision_replicates"] + 1):
        run_step(f"decision_r{i}",
                 vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir(f"decision_r{i}"))),
                 stage_dir(f"decision_r{i}") / "stage1" / "ranking.parquet",
                 force=force, dry_run=dry)


def build_shortlist() -> pd.DataFrame:
    """3.3 replicate-averaged ranking + the near-cut ambiguity set."""
    dirs = [stage_dir(f"decision_r{i}") for i in range(1, CONFIG["decision_replicates"] + 1)]
    reps = replicates_mean(dirs)
    if reps.empty:
        raise RuntimeError("no decision-region replicates found; run the decision stage first")
    # molecules recovered by the 3.0 stage carry a single score on the same
    # frozen receptor; give them the pooled noise estimate so the ambiguity rule
    # can still speak about them
    rec = stage_dir("recover") / "stage1" / "ranking.parquet"
    if rec.exists():
        r = pd.read_parquet(rec)[["mol_id", "score"]].rename(columns={"score": "mean"})
        r = r[~r.mol_id.isin(reps.mol_id)]
        r["sd"] = float(reps["sd"].median()) if reps["sd"].notna().any() else 0.0
        r["n"] = 1
        reps = pd.concat([reps, r], ignore_index=True)
        log(f"shortlist: pooled {len(r)} molecules recovered by the 3.0 stage "
            f"(single replicate each, SD set to the pooled {r['sd'].iloc[0]:.3f})")
    reps = reps.sort_values("mean").reset_index(drop=True)
    reps["rank"] = reps.index + 1
    cut = CONFIG["shortlist"]
    sd = reps["sd"].fillna(0.0)
    reps["ambiguous"] = (reps["mean"] - GATES["delta_sigma_multiple"] * sd) <= reps["mean"].iloc[cut - 1]
    short = reps.head(cut)
    extra = reps[(reps["rank"] > cut) & reps["ambiguous"]]
    out_dir = RUN / "shortlist"
    out_dir.mkdir(parents=True, exist_ok=True)
    reps.to_csv(out_dir / "decision_ranking.csv", index=False)
    keep = pd.concat([short, extra])
    lib = ROOT / "instrument" / f"decision_{CONFIG['decision_region']}.smi"
    smiles = {line.split()[-1]: line.split()[0] for line in lib.read_text().splitlines() if line.strip()}
    keep = keep[keep.mol_id.isin(smiles)]
    (out_dir / "shortlist.smi").write_text("\n".join(f"{smiles[m]} {m}" for m in keep.mol_id) + "\n")
    frac = len(extra) / max(1, cut)
    log(f"shortlist: {len(short)} + {len(extra)} ambiguous = {len(keep)} molecules "
        f"(ambiguity {frac:.1%} of the cut, limit {GATES['ambiguity_frac_max']:.0%})")
    return reps


def _shortlist(force: bool = False, dry: bool = False) -> None:
    if dry:
        log("shortlist: would average the decision replicates and build the ambiguity set")
        return
    build_shortlist()


def cofold_run() -> Path:
    """The run directory holding the hand-launched 500-molecule affinity screen."""
    path = Path(CONFIG["cofold_run"])
    return path if path.is_absolute() else ROOT / path


def _structure_qc(force: bool = False, dry: bool = False) -> None:
    """Per-complex structural metrics, then inter-model consistency.

    Both are CPU-only and read the complexes the affinity run already produced, so
    they cost nothing but make every structural number in the paper traceable to
    the same run as the affinity numbers.
    """
    run = cofold_run()
    pattern = str(run / "stage2" / "boltz" / "local" / "*" / "out" /
                  "boltz_results_input" / "predictions" / "input" / "input_model_0.cif")
    run_step("structure_qc", [str(PYTHON), "scripts/analyze_stage2_complexes.py",
                              "--pattern", pattern,
                              "--out", str(run / "stage2" / "complex_analysis.csv")],
             run / "stage2" / "complex_analysis.csv", force=force, dry_run=dry)
    run_step("structure_qc_intermodel", [str(PYTHON), "scripts/stage2_intermodel.py",
                                         "--run", str(CONFIG["cofold_run"])],
             run / "stage2" / "intermodel_consistency.json", force=force, dry_run=dry)


def _select(force: bool = False, dry: bool = False) -> None:
    """Apply the pre-registered rule: structural gate, then affinity ranking.

    ``--apply`` reorders consensus_ranking.parquet so stage 3 consumes the same
    shortlist the report describes; the previous ordering is kept alongside it.
    """
    run = cofold_run()
    run_step("select", [str(PYTHON), "scripts/select_stage2_shortlist.py",
                        "--run", str(CONFIG["cofold_run"]),
                        "--top-n", str(CONFIG["stage3_top"]),
                        "--apply"],
             run / "stage2" / "selection_report.json", force=force, dry_run=dry)


def _bench_fixed(force: bool = False, dry: bool = False) -> None:
    """Re-run the affinity calibration on a receptor verified against its MSA."""
    run = stage_dir(CONFIG["benchmark_fixed_run"])
    run_step("bench_fixed", vcdd("stage2", "-c", "run_benchmark_relative.yaml",
                                 "-o", CONFIG["benchmark_fixed_run"]),
             run / "stage2" / "scores.parquet", force=force, dry_run=dry,
             timeout=6 * 3600)


def _cofold(force: bool = False, dry: bool = False) -> None:
    """3'.1 co-folding of the shortlist (affinity head only if its gate passed)."""
    if not dry and not (RUN / "shortlist" / "shortlist.smi").exists():
        raise RuntimeError("run the shortlist stage first "
                           "(workflow_v2/shortlist/shortlist.smi is missing)")
    gates = load_gates()
    use_affinity = bool(CONFIG["cofold_affinity"])
    if use_affinity and gates.get("aff_benchmark") and not gates["aff_benchmark"].get("passed"):
        log("cofold: affinity head ON as requested. The calibration gate did not pass "
            f"({json.dumps(gates['aff_benchmark'].get('metrics', {}), ensure_ascii=False)[:180]}), "
            "so affinity values are reported as an in-pipeline ranking signal, NOT as "
            "potency. Read `benchmark_balance` in gates.json before making any affinity "
            "claim: the benchmark's actives and negatives differ in molecular weight.",
            level="WARN")
    cfg_body = base_config()
    cfg_body["name"] = "wfv2-cofold"
    cfg_body["library"] = dict(cfg_body["library"], path="workflow_v2/shortlist/shortlist.smi")
    cfg_body["stage2"].update({"top_n": CONFIG["shortlist"] * 2, "chunk_size": CONFIG["shortlist"] * 2})
    cfg_body["stage2"]["local_boltz"].update({
        "affinity": use_affinity,
        "affinity_samples": CONFIG["affinity_samples"],
        "affinity_sampling_steps": CONFIG["affinity_sampling_steps"],
        "affinity_mw_correction": CONFIG["affinity_mw_correction"],
        "timeout_s": 7200,
    })
    cfg = write_config("cofold", cfg_body,
                       f"Phase 3'.1: co-fold the shortlist; affinity head "
                       f"{'ON' if use_affinity else 'OFF'} per the pre-registered gate.")
    run_step("cofold",
             vcdd("stage2", "--config", str(cfg), "--out", str(stage_dir("cofold"))),
             stage_dir("cofold") / "stage2" / "scores.parquet",
             force=force, dry_run=dry)


def _stage3(force: bool = False, dry: bool = False) -> None:
    """Virtual-cell scoring of the selected shortlist.

    Stage 3 reads ``<run>/stage2/consensus_ranking.parquet`` and takes the first
    ``cell.top_n`` rows, so ``--out`` must be the affinity run itself: that is what
    makes it consume the structure-gated, affinity-ranked shortlist rather than the
    earlier confidence ordering.
    """
    run = cofold_run()
    run_step("stage3", vcdd("stage3", "-c", CONFIG["affinity_config"], "--out", str(run)),
             run / "stage3" / "stage3_summary.json", force=force, dry_run=dry)


def _stage4(force: bool = False, dry: bool = False) -> None:
    """Analog design from the stage-3 winners, redocked against the same receptor."""
    run = cofold_run()
    run_step("stage4", vcdd("stage4", "-c", CONFIG["affinity_config"], "--out", str(run)),
             run / "stage4" / "stage4_summary.json", force=force, dry_run=dry)


def _redock_full(force: bool = False, dry: bool = False) -> None:
    """Optional: re-dock the whole library (62 h) when the offset test says the
    campaign ranking is not portable."""
    cfg_body = base_config()
    cfg_body["name"] = "wfv2-full-redock"
    cfg = write_config("full_redock", cfg_body,
                       "Optional Phase 3.1b: whole-library re-dock with the frozen receptor.")
    run_step("full_redock",
             vcdd("stage1", "--config", str(cfg), "--out", str(stage_dir("full_redock"))),
             stage_dir("full_redock") / "stage1" / "ranking.parquet",
             force=force, dry_run=dry, timeout=60 * 60 * 96)


def _report(force: bool = False, dry: bool = False) -> None:
    """Phase 6: gates, provenance, summary.md, INDEX.md."""
    if dry:
        log("report: would rebuild gates.json, provenance.json, summary.md and INDEX.md")
        return
    if not (stage_dir("benchmark") / "stage1").exists():
        log("report: no benchmark output yet, summary will be partial", level="WARN")
    evaluate_gates()
    write_provenance()
    write_summary()


STAGES: list[Stage] = [
    Stage("prep", "0.1/0.2 freeze receptor + native-ligand redock gate", "seconds",
          "redock RMSD < 2.0 A", lambda f, d: _prep(f, d),
          lambda: stage_dir("receptor_gate") / "receptor" / "redock_validation.json"),
    Stage("noise", f"0.3 engine noise floor ({CONFIG['noise_subset']} x {CONFIG['noise_replicates']})", "~6 min",
          "report sigma_dock", lambda f, d: _noise(f, d),
          lambda: stage_dir(f"noise_r{CONFIG['noise_replicates']}") / "stage1" / "ranking.parquet"),
    Stage("bench_dock", "0.4 docking calibration (known inhibitors vs negatives)", "~2 min",
          "AUC >= 0.70 and rho >= 0.40", lambda f, d: _bench_dock(f, d),
          lambda: stage_dir("benchmark") / "stage1" / "ranking.parquet"),
    Stage("bench_aff", "0.5 affinity calibration (Boltz-2, 76 molecules)", "~2.2 h",
          "AUC >= 0.70, rho >= 0.40, control in top 25%", lambda f, d: _bench_aff(f, d),
          lambda: stage_dir("benchmark") / "stage2" / "scores.parquet"),
    Stage("recover", "3.0 re-dock the campaign's unscored molecules", "~2 h",
          f"coverage >= {GATES['min_coverage']:.0%} of the recovered set",
          lambda f, d: _recover(f, d),
          lambda: stage_dir("recover") / "stage1" / "ranking.parquet"),
    Stage("offset", f"3.1 receptor-shift test ({CONFIG['offset_subset']} molecules)", "~42 min",
          "SD of the paired delta <= 0.5 kcal/mol", lambda f, d: _offset(f, d),
          lambda: stage_dir("offset") / "stage1" / "ranking.parquet"),
    Stage("redock_full", "3.1b OPTIONAL whole-library re-dock", "~62 h",
          "only if the offset test fails", lambda f, d: _redock_full(f, d),
          lambda: stage_dir("full_redock") / "stage1" / "ranking.parquet", optional=True),
    Stage("decision", f"3.2 decision region {CONFIG['decision_region']} x {CONFIG['decision_replicates']}",
          "~10.5 h", "report near-cut ambiguity",
          lambda f, d: _decision(f, d),
          lambda: stage_dir(f"decision_r{CONFIG['decision_replicates']}") / "stage1" / "ranking.parquet"),
    Stage("shortlist", "3.3 replicate-averaged shortlist + ambiguity set", "seconds",
          f"ambiguity <= {GATES['ambiguity_frac_max']:.0%} of the cut",
          _shortlist, lambda: RUN / "shortlist" / "shortlist.smi"),
    Stage("cofold", "3'.1 co-fold the shortlist", "~14 h with affinity / ~8 h without",
          "structural QC only", lambda f, d: _cofold(f, d),
          lambda: stage_dir("cofold") / "stage2" / "scores.parquet"),
    Stage("structure_qc", "3'.1b structural QC of the co-folded complexes", "~5 min",
          "ATP-site RMSD < 2.0 A for the majority", _structure_qc,
          lambda: cofold_run() / "stage2" / "intermodel_consistency.json"),
    Stage("select", "3'.2 structure gate, then affinity ranking -> top 200", "seconds",
          "positive control survives the rule", _select,
          lambda: cofold_run() / "stage2" / "selection_report.json"),
    Stage("bench_fixed", "3'.3 re-run the affinity calibration on a verified receptor", "~2.2 h",
          "sequence == MSA, then the calibration gates", lambda f, d: _bench_fixed(f, d),
          lambda: stage_dir(CONFIG["benchmark_fixed_run"]) / "stage2" / "scores.parquet"),
    Stage("stage3", f"3'.4 virtual-cell scoring (top {CONFIG['stage3_top']})", "~9 min",
          "none (orthogonal signal)", lambda f, d: _stage3(f, d),
          lambda: cofold_run() / "stage3" / "stage3_summary.json"),
    Stage("stage4", "3'.3 analog design", "~10 min", "none (circularity disclosed)",
          lambda f, d: _stage4(f, d),
          lambda: cofold_run() / "stage4" / "stage4_summary.json"),
    Stage("report", "6 gates + provenance + summary.md + INDEX.md", "seconds", "-",
          lambda f, d: _report(f, d), lambda: RUN / "summary.md"),
]


# --------------------------------------------------------------------------- #
# 5. gates
# --------------------------------------------------------------------------- #

def run_coverage(run: Path) -> dict[str, Any] | None:
    """Scored fraction of one run, per stage, from the scores files.

    The campaign silently lost 3.2% of its library this way, and a single
    unparsable ligand once cost a whole 200-ligand batch (35% of that run), so
    every run reports its own coverage and a shortfall fails a gate.
    """
    out: dict[str, Any] = {}
    for stage, name in (("stage1", "shard_*.parquet"), ("stage2", "scores.parquet")):
        files = sorted((run / stage / ("scores" if stage == "stage1" else "")).glob(name))
        if not files:
            continue
        frames = [pd.read_parquet(f) for f in files]
        d = pd.concat(frames, ignore_index=True)
        col = "score" if stage == "stage1" else "iptm"
        if col not in d.columns:
            continue
        failed = d[d[col].isna()]
        reasons = failed["error"].value_counts().to_dict() if "error" in failed.columns else {}
        out[stage] = {"input": int(len(d)), "scored": int(d[col].notna().sum()),
                      "coverage": round(float(d[col].notna().mean()), 4),
                      "reasons": {str(k): int(v) for k, v in list(reasons.items())[:5]}}
    return out or None


def _safe_evaluate() -> None:
    """Evaluate the gates, but never let a reporting bug stop the workflow."""
    try:
        evaluate_gates()
    except Exception as exc:  # noqa: BLE001
        log(f"gate evaluation failed ({type(exc).__name__}: {exc}); "
            f"stages are unaffected, fix and re-run with --report", level="WARN")


def check_dependencies() -> list[str]:
    """Dependencies the *driver* needs (the subprocesses have their own env)."""
    missing = []
    for module in ("pandas", "yaml", "pyarrow"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def load_gates() -> dict[str, Any]:
    path = RUN / "gates.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_gates(gates: dict[str, Any]) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False))


def evaluate_gates() -> dict[str, Any]:
    gates = load_gates()

    # redock
    redock_file = stage_dir("receptor_gate") / "receptor" / "redock_validation.json"
    if redock_file.exists():
        r = json.loads(redock_file.read_text())
        rmsd = r.get("all_atom_rmsd") or r.get("rmsd")
        gates["redock"] = {"measured_rmsd": rmsd, "threshold": GATES["redock_rmsd_max"],
                           "passed": bool(rmsd is not None and rmsd < GATES["redock_rmsd_max"]),
                           "verdict": r.get("verdict")}

    # noise floor
    dirs = [stage_dir(f"noise_r{i}") for i in range(1, CONFIG["noise_replicates"] + 1)]
    reps = replicates_mean(dirs)
    if not reps.empty and reps["n"].max() > 1:
        d = reps.dropna(subset=["sd"])
        gates["noise"] = {"n_molecules": int(len(d)), "sigma_dock_median": round(float(d.sd.median()), 3),
                          "sigma_dock_p90": round(float(d.sd.quantile(0.9)), 3),
                          "max_abs_delta": round(float((reps[[c for c in reps if c.startswith('r')]].max(axis=1)
                                                        - reps[[c for c in reps if c.startswith('r')]].min(axis=1)).max()), 3),
                          "resolution_2sigma": round(2 * float(d.sd.median()), 3)}

    # docking calibration
    bdir = stage_dir("benchmark")
    if (bdir / "stage1" / "ranking.parquet").exists():
        m = benchmark_metrics(bdir, "stage1+stage2")
        dock = m.get("dock", {})
        gates["dock_benchmark"] = m | {"thresholds": {"auc": GATES["dock_auc_min"], "rho": GATES["dock_rho_min"]},
                                       "passed": bool(dock.get("roc_auc", 0) >= GATES["dock_auc_min"]
                                                      and dock.get("spearman", 0) >= GATES["dock_rho_min"])}
        aff = m.get("affinity_pred_value", {})
        if aff:
            gates["aff_benchmark"] = {
                "metrics": aff,
                "thresholds": {"auc": GATES["aff_auc_min"], "rho": GATES["aff_rho_min"],
                               "control_top_frac": GATES["aff_control_top_frac"]},
                "passed": bool(aff.get("roc_auc", 0) >= GATES["aff_auc_min"]
                               and aff.get("spearman", 0) >= GATES["aff_rho_min"]
                               and aff.get("control_frac", 1) <= GATES["aff_control_top_frac"]),
            }

    # receptor-shift test
    opath = stage_dir("offset") / "stage1" / "ranking.parquet"
    if opath.exists():
        new = pd.read_parquet(opath)[["mol_id", "score"]].rename(columns={"score": "new"})
        old = pd.read_parquet(ROOT / "instrument" / "offset_test_2000_campaign_scores.csv")
        paired = old.merge(new, on="mol_id")
        delta = paired["new"] - paired["score"]
        gates["offset"] = {"n": int(len(paired)), "mean_delta": round(float(delta.mean()), 3),
                           "sd_delta": round(float(delta.std()), 3),
                           "constant_offset": bool(delta.std() <= GATES["offset_sd_max"]),
                           "threshold_sd": GATES["offset_sd_max"]}
        if not gates["offset"]["constant_offset"]:
            log("offset test: the shift is molecule-dependent - the campaign ranking is NOT "
                "portable; re-dock the whole library (stage redock_full) before using it", level="WARN")

    # shortlist ambiguity
    dec = RUN / "shortlist" / "decision_ranking.csv"
    if dec.exists():
        d = pd.read_csv(dec)
        cut = CONFIG["shortlist"]
        extra = int(((d["rank"] > cut) & (d["ambiguous"])).sum())
        gates["shortlist"] = {"cut": cut, "ambiguous_beyond_cut": extra,
                              "frac": round(extra / cut, 4),
                              "passed": bool(extra / cut <= GATES["ambiguity_frac_max"])}

    cov: dict[str, Any] = {}
    for run in sorted(RUN.iterdir()) if RUN.exists() else []:
        if run.is_dir() and ((run / "stage1").exists() or (run / "stage2").exists()):
            c = run_coverage(run)
            if c:
                cov[run.name] = c
    campaign_cov = run_coverage(CAMPAIGN)
    if cov or campaign_cov:
        worst = min((v["coverage"] for per in cov.values() for v in per.values()), default=1.0)
        gates["coverage"] = {"per_run": cov, "worst": worst,
                             "threshold": GATES["min_coverage"],
                             # historical, reported but not gated: the campaign ran
                             # before this workflow existed
                             "campaign_reference": campaign_cov,
                             "passed": bool(worst >= GATES["min_coverage"])}
        if worst < GATES["min_coverage"]:
            log(f"coverage gate: worst scored fraction {worst:.1%} is below "
                f"{GATES['min_coverage']:.0%} - molecules were dropped silently, "
                f"see gates.json", level="WARN")

    save_gates(gates)
    return gates


# --------------------------------------------------------------------------- #
# 6. provenance, summary, index
# --------------------------------------------------------------------------- #

def write_provenance() -> None:
    files = {
        "receptor_pdbqt": stage_dir("receptor_gate") / "receptor" / "receptor.pdbqt",
        "receptor_clean": stage_dir("receptor_gate") / "receptor" / "receptor_clean.pdb",
        "box": stage_dir("receptor_gate") / "receptor" / "box.json",
        "msa": ROOT / MSA,
        "library": ROOT / "benchmark" / "known_inhibitors.smi",
        "template_config": TEMPLATE,
        "campaign_receptor_pdbqt": CAMPAIGN / "receptor" / "receptor.pdbqt",
    }
    prov: dict[str, Any] = {"generated": time.strftime("%F %T"), "files": {}, "tools": {},
                            "reused_stages": {k: v for k, v in REUSE.items()
                                              if stage_dir(k).is_symlink()}}
    for name, path in files.items():
        prov["files"][name] = {"path": str(path.relative_to(ROOT)), "md5": md5(path),
                               "bytes": path.stat().st_size} if path.exists() else {"path": str(path), "missing": True}
    for cfg in sorted(CONFIGS.glob("*.yaml")):
        prov["files"][f"config:{cfg.stem}"] = {"path": str(cfg.relative_to(ROOT)), "md5": md5(cfg)}
    for label, cmd in (("python", [str(PYTHON), "-V"]), ("boltz", ["boltz", "--version"]),
                       ("obabel", ["obabel", "-V"])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, env=child_env(), timeout=60)
            prov["tools"][label] = (out.stdout or out.stderr).strip().splitlines()[0][:120]
        except Exception as exc:  # noqa: BLE001
            prov["tools"][label] = f"unavailable: {exc}"
    # count hydrogens in the frozen receptor: the campaign's had none (P1)
    rec = files["receptor_pdbqt"]
    if rec.exists():
        types: dict[str, int] = {}
        for line in rec.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                types[line[77:79].strip()] = types.get(line[77:79].strip(), 0) + 1
        prov["receptor"] = {"n_atoms": sum(types.values()),
                            "n_polar_hydrogens": sum(v for k, v in types.items() if k in ("H", "HD", "HS")),
                            "atom_types": types}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))


def write_summary() -> None:
    gates = load_gates()
    prov = json.loads((RUN / "provenance.json").read_text()) if (RUN / "provenance.json").exists() else {}
    rows = []
    for name, g in gates.items():
        passed = g.get("passed") if isinstance(g, dict) else None
        rows.append((name, {True: "PASS", False: "FAIL", None: "measured"}[passed],
                     json.dumps({k: v for k, v in g.items() if k != "passed"}, ensure_ascii=False)[:400]))
    lines = ["# vcdd workflow v2 - measured results", "",
             f"generated {time.strftime('%F %T')} by `run_all.py`", "",
             "## Pre-registered gates", "",
             "| gate | verdict | measured |", "|---|---|---|"]
    lines += [f"| {n} | **{v}** | `{m}` |" for n, v, m in rows]

    lines += ["", "## Claim boundaries (paper/workflow_redesign.md section 6)", ""]
    dock_pass = gates.get("dock_benchmark", {}).get("passed", False)
    aff_pass = gates.get("aff_benchmark", {}).get("passed", False)
    lines += [
        f"* docking ranking claims: **{'allowed' if dock_pass else 'NOT allowed'}** "
        f"(gate `dock_benchmark`)",
        f"* affinity ranking claims: **{'allowed' if aff_pass else 'NOT allowed'}** "
        f"(gate `aff_benchmark`)",
        "* potency claims: **never allowed without experimental data**",
        "",
    ]

    noise = gates.get("noise", {})
    if noise:
        lines += ["## Score resolution", "",
                  f"* median per-molecule SD over {CONFIG['noise_replicates']} identical runs: "
                  f"**{noise.get('sigma_dock_median')} kcal/mol**",
                  f"* largest single-molecule swing: **{noise.get('max_abs_delta')} kcal/mol**",
                  f"* differences below **{noise.get('resolution_2sigma')} kcal/mol** are not "
                  f"interpretable (2 sigma)", ""]

    off = gates.get("offset", {})
    if off:
        lines += ["## Receptor-preparation shift", "",
                  f"* paired delta over {off.get('n')} molecules: mean **{off.get('mean_delta')}**, "
                  f"SD **{off.get('sd_delta')}** kcal/mol",
                  f"* constant offset: **{off.get('constant_offset')}** "
                  f"({'campaign ranking is portable' if off.get('constant_offset') else 'whole-library re-dock required'})",
                  ""]

    for key, title in (("dock", "Stage 1 docking calibration"),
                       ("affinity_pred_value", "Stage 2 affinity calibration")):
        entry = gates.get("dock_benchmark", {}).get(key)
        if entry:
            lines += [f"## {title}", "",
                      f"* ROC-AUC (actives vs >10 uM negatives): **{entry['roc_auc']}** "
                      f"(n={entry['n_actives']} vs {entry['n_inactives']})",
                      f"* Spearman vs measured pIC50: **{entry['spearman']}** (p={entry.get('p')})",
                      f"* median potent tier {entry.get('median_potent')} vs negatives "
                      f"{entry.get('median_inactive')}",
                      f"* AZD7648 out-scored by {entry.get('control_frac')} of the benchmark"]
            bal = entry.get("balance") or {}
            if bal:
                lines.append(
                    f"* benchmark size match: actives {bal.get('mw_actives_median')} Da vs "
                    f"negatives {bal.get('mw_negatives_median')} Da "
                    f"(gap {bal.get('mw_gap')} Da, limit {GATES['mw_gap_max']:.0f}) - "
                    f"**{'matched' if bal.get('balanced') else 'NOT MATCHED'}**")
                if bal.get("r_score_vs_mw") is not None:
                    lines.append(
                        f"* score vs molecular weight: r = {bal['r_score_vs_mw']:+.3f} "
                        f"(p={bal.get('p_score_vs_mw')}) - "
                        + ("the AUC above is confounded by size; treat it as unmeasured, "
                           "not as a failure" if not bal.get("balanced")
                           else "no material size dependence"))
            lines.append("")

    cov = gates.get("coverage", {})
    if cov:
        lines += ["## Library coverage (molecules that actually got a score)", "",
                  f"* threshold: {cov['threshold']:.0%} of every run's input",
                  f"* worst run in this workflow: **{cov['worst']:.1%}** "
                  f"({'PASS' if cov['passed'] else 'FAIL'})", ""]
        for run, per in sorted(cov.get("per_run", {}).items()):
            for stg, v in per.items():
                lines.append(f"* `{run}` {stg}: {v['scored']}/{v['input']} "
                             f"({v['coverage']:.1%}) {list(v['reasons'])[:2]}")
        ref = cov.get("campaign_reference") or {}
        for stg, v in ref.items():
            lines.append(f"* original campaign {stg} (reference): {v['scored']}/{v['input']} "
                         f"({v['coverage']:.1%}) {list(v['reasons'])[:2]}")
        lines.append("")

    short = gates.get("shortlist", {})
    if short:
        lines += ["## Shortlist stability", "",
                  f"* cut at {short['cut']}: {short['ambiguous_beyond_cut']} near-cut molecules "
                  f"({short['frac']:.1%}) inside the 2-sigma band", ""]

    lines += ["## Provenance", "",
              f"* receptor: {prov.get('receptor', {})}",
              f"* tools: {json.dumps(prov.get('tools', {}), ensure_ascii=False)}", "",
              "Full artifact list: `INDEX.md`. Raw gates: `gates.json`. "
              "Every number above is reproducible from the files listed there."]
    (RUN / "summary.md").write_text("\n".join(lines) + "\n")

    index = ["# Artifact index", "", "| artifact | what it is | backs which claim |", "|---|---|---|"]
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.suffix in (".parquet", ".csv", ".json", ".smi", ".md"):
            rel = path.relative_to(ROOT)
            what = {
                "gates.json": "every gate and its measured value",
                "summary.md": "the numbers quoted in the paper",
                "provenance.json": "hashes and tool versions",
                "decision_ranking.csv": "replicate-averaged docking ranks + SD",
                "shortlist.smi": "the molecules carried into co-folding",
                "scores.parquet": "per-molecule stage-2 confidence/affinity",
                "ranking.parquet": "per-molecule docking scores",
                "pose_qc.parquet": "per-complex structural QC",
            }.get(path.name, "stage output")
            index.append(f"| `{rel}` | {what} | {path.parent.name} |")
    (RUN / "INDEX.md").write_text("\n".join(index) + "\n")
    log(f"report: wrote {(RUN / 'summary.md').relative_to(ROOT)} and INDEX.md")


# --------------------------------------------------------------------------- #
# 7. command line
# --------------------------------------------------------------------------- #

def stage_state(stage: Stage) -> tuple[str, str]:
    """(state, detail) for one stage: done / running / pending / opt-in."""
    marker = stage.marker()
    if done(marker):
        return "done", str(marker.parent.relative_to(ROOT)) if marker.parent != ROOT else ""
    run_dir = marker.parents[1] if marker.parent.name in ("stage1", "stage2", "stage3", "stage4") else marker.parent
    if (RUN / f"{stage.name}.running").exists():
        return "RUNNING", progress_detail(stage, run_dir)
    # A stage whose marker lives outside this workflow's root reads a run launched
    # by hand (the 500-molecule affinity screen). That directory is busy for hours
    # while the stage itself has not run yet, so mtime must not decide here.
    if not str(marker.resolve()).startswith(str(RUN.resolve())):
        return "pending", "reads " + str(run_dir.relative_to(ROOT))
    if run_dir.exists():
        newest = max((f.stat().st_mtime for f in run_dir.rglob("*") if f.is_file()), default=0)
        silent = (time.time() - newest) / 60
        # a docking stage writes nothing between its pilot and its final shard,
        # so silence is normal; call it running while it is inside the projection
        proj = projected_hours(run_dir)
        if proj is not None and silent / 60 < proj + 0.5:
            return "RUNNING", progress_detail(stage, run_dir)
        return (("partial", progress_detail(stage, run_dir)) if silent > 30
                else ("RUNNING", progress_detail(stage, run_dir)))
    if stage.optional:
        return "opt-in", "not part of a plain run"
    return "pending", ""


def projected_hours(run_dir: Path) -> float | None:
    """The run's own throughput projection, if it has written one."""
    path = run_dir / "stage1" / "throughput.json"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text()).get("projected_hours"))
    except Exception:  # noqa: BLE001
        return None


def progress_detail(stage: Stage, run_dir: Path) -> str:
    """Scored/total and an ETA for a stage that is producing output now."""
    cfg_path = CONFIGS / f"{CONFIG_NAME.get(stage.name, stage.name)}.yaml"
    total = None
    try:
        body = yaml.safe_load(cfg_path.read_text())
        lib = ROOT / str(body["library"]["path"])
        total = sum(1 for _ in lib.open()) if lib.exists() else None
    except Exception:  # noqa: BLE001
        pass
    scored = 0
    for scores in (run_dir / "stage1" / "scores").glob("shard_*.parquet") if (run_dir / "stage1" / "scores").exists() else []:
        try:
            d = pd.read_parquet(scores)
            scored += int(d.score.notna().sum())
        except Exception:  # noqa: BLE001
            pass
    if not total:
        return "started"
    proj = projected_hours(run_dir)
    if proj is not None:
        tp = run_dir / "stage1" / "throughput.json"
        started = time.mktime(time.strptime(
            json.loads(tp.read_text())["measured_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        left = max(0.0, proj - (time.time() - started) / 3600)
        return f"{total} molecules, run projects {proj:.2f} h, ~{left:.2f} h left"
    rate = 2844.0  # campaign throughput, ligands/hour
    return f"{scored}/{total} scored, ETA ~{max(0.0, (total - scored) / rate):.1f} h"


CONFIG_NAME = {"prep": "receptor_gate", "noise": "noise_floor", "bench_dock": "benchmark",
               "bench_aff": "benchmark", "recover": "recover", "offset": "offset_test",
               "redock_full": "full_redock", "decision": "decision", "cofold": "cofold",
               "stage3": "stage3", "stage4": "stage3", "shortlist": "shortlist"}


def print_status() -> None:
    """What is done, what is running, what is next."""
    gates = load_gates()
    gate_for = {"prep": "redock", "bench_dock": "dock_benchmark", "bench_aff": "aff_benchmark",
                "offset": "offset", "shortlist": "shortlist"}
    print(f"{'stage':<12} {'state':<9} {'gate':<26} detail")
    print("-" * 110)
    for stage in STAGES:
        state, detail = stage_state(stage)
        g = gates.get(gate_for.get(stage.name, ""), {})
        verdict = ""
        if isinstance(g, dict) and "passed" in g:
            verdict = ("PASS " if g["passed"] else "FAIL ") + str(
                g.get("verdict") or g.get("measured_rmsd") or g.get("worst") or "")[:18]
        print(f"{stage.name:<12} {state:<9} {verdict:<26} {detail}")

    cov = gates.get("coverage", {})
    if cov:
        ref = (cov.get("campaign_reference") or {}).get("stage1", {})
        print(f"\ncoverage: worst run in this workflow {cov.get('worst', 1):.1%} "
              f"(threshold {cov.get('threshold', 0):.0%})")
        if ref:
            print(f"          original campaign {ref.get('scored')}/{ref.get('input')} "
                  f"({ref.get('coverage', 0):.1%}) - the 3.2% gap is the glue-atom defect")

    print("\nnext:")
    for stage in STAGES:
        state, _ = stage_state(stage)
        if state == "RUNNING":
            print(f"  wait for `{stage.name}` (see above), then: python run_all.py --report")
            break
    else:
        todo = [s for s in STAGES if not s.optional and stage_state(s)[0] in ("pending", "partial")]
        if todo:
            names = " ".join(f"--only {s.name}" for s in todo[:3])
            print(f"  python run_all.py {names}")
        else:
            print("  nothing left in the default plan; see --list for opt-in stages")


def print_plan() -> None:
    print(f"{'stage':<12} {'cost':<34} {'gate':<52} title")
    print("-" * 150)
    for s in STAGES:
        flag = "  [opt-in]" if s.optional else ""
        print(f"{s.name:<12} {s.cost:<34} {s.gate:<52} {s.title}{flag}")
    print()
    print("gates (pre-registered, edit GATES in this file to change):")
    for k, v in GATES.items():
        print(f"  {k:<24} {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the plan and exit")
    ap.add_argument("--status", action="store_true", help="what is done, running and next")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--only", action="append", default=[], help="run just this stage (repeatable)")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--skip", action="append", default=[], help="skip this stage (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-run stages whose output exists")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed gate")
    ap.add_argument("--report", action="store_true", help="only rebuild gates/summary/INDEX")
    ap.add_argument("--include-optional", action="store_true",
                    help="also run opt-in stages such as the 62 h whole-library re-dock")
    args = ap.parse_args()

    if args.list:
        print_plan()
        return 0
    if args.status:
        print_status()
        return 0

    if not PYTHON.exists():
        log(f"python not found: {PYTHON} (set VCDD_PYTHON)", level="ERROR")
        return 2
    missing = check_dependencies()
    if missing:
        log(f"missing dependencies in {sys.executable}: {', '.join(missing)}. "
            f"Install them, or point VCDD_PYTHON at an environment that has them "
            f"(currently {PYTHON}).", level="ERROR")
        return 2
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    _prepare_env()
    link_reused()

    if args.report:
        _safe_evaluate()
        write_provenance()
        write_summary()
        return 0

    names = [s.name for s in STAGES]
    # optional (expensive) stages are never part of a plain run: they are only
    # reached by naming them, so a default invocation cannot spend 62 h
    selected = [s for s in STAGES if not s.optional]
    if args.include_optional:
        selected = list(STAGES)
    if args.only:
        unknown = [n for n in args.only if n not in names]
        if unknown:
            log(f"unknown stage(s): {unknown}; see --list", level="ERROR")
            return 2
        selected = [s for s in STAGES if s.name in args.only]
    elif args.from_stage:
        if args.from_stage not in names:
            log(f"unknown stage: {args.from_stage}", level="ERROR")
            return 2
        selected = STAGES[names.index(args.from_stage):]
    selected = [s for s in selected if s.name not in args.skip]

    log(f"workflow v2: {len(selected)} stage(s) selected, root {RUN.relative_to(ROOT)}")
    failures: list[str] = []
    for stage in selected:
        log(f"--- {stage.name}: {stage.title} ({stage.cost})")
        try:
            stage.run(args.force, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            log(f"{stage.name} failed: {exc}", level="ERROR")
            failures.append(stage.name)
            if not args.keep_going:
                log("stopping (use --keep-going to continue past failures)", level="ERROR")
                break
            continue
        if not args.dry_run:
            _safe_evaluate()
            gate = load_gates().get({"prep": "redock", "noise": "noise", "bench_dock": "dock_benchmark",
                                     "bench_aff": "aff_benchmark", "offset": "offset",
                                     "shortlist": "shortlist"}.get(stage.name, ""), {})
            if isinstance(gate, dict) and gate.get("passed") is False:
                log(f"GATE FAILED for {stage.name}: {json.dumps(gate, ensure_ascii=False)[:300]}", level="WARN")
                if not args.keep_going:
                    log("stopping at the failed gate (this is the pre-registered rule; "
                        "--keep-going overrides)", level="ERROR")
                    break

    if not args.dry_run:
        _safe_evaluate()
        write_provenance()
        write_summary()
    log(f"done. summary: {(RUN / 'summary.md').relative_to(ROOT)} | "
        f"gates: {(RUN / 'gates.json').relative_to(ROOT)} | "
        f"failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
