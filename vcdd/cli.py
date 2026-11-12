"""``vcdd`` command line interface.

One command per pipeline stage plus ``run`` (all four), ``resume`` (re-enter a
run directory), ``doctor`` (what is actually installed here) and ``report``.

Every stage command is *independently runnable* against an existing run
directory, and every stage is resumable, because the whole point of a
million-compound screen is that it will be interrupted.

    vcdd doctor
    vcdd config > my_run.yaml
    vcdd run -c my_run.yaml
    vcdd stage3 -c my_run.yaml --force
    vcdd resume -c my_run.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from vcdd.config.loader import load_config, parse_overrides, save_resolved, template_text
from vcdd.config.schema import CellConfig, RunConfig, Stage4Config
from vcdd.runtime.ledger import RunLedger
from vcdd.runtime.logging import close_logs, setup_logging
from vcdd.runtime.manifest import Manifest, tool_versions
from vcdd.runtime.paths import ensure_dir
from vcdd.version import __version__

app = typer.Typer(
    name="vcdd",
    help="Virtual-cell-guided drug design: fast screen -> accurate re-dock -> "
    "cell-model scoring -> analog design.",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

# --------------------------------------------------------------------------- #
# Shared options
# --------------------------------------------------------------------------- #

_CONFIG = typer.Option(None, "--config", "-c", help="YAML config file.")
_SET = typer.Option(
    None, "--set", "-s",
    help="Override a config key, e.g. -s stage1.variant=quickvina2_gpu (repeatable).",
)
_OUT = typer.Option(None, "--out", "-o", help="Run directory (overrides config out_dir).")
_FORCE = typer.Option(False, "--force", help="Ignore the resume ledger and recompute.")
_QUIET = typer.Option(False, "--quiet", "-q", help="Errors only on the console.")


def _bootstrap(
    config: Optional[Path],
    overrides: Optional[list[str]],
    out: Optional[Path],
    quiet: bool = False,
    stage: str | None = None,
) -> tuple[RunConfig, Path]:
    """Load config, create the run directory, start logging."""
    merged: dict[str, Any] = parse_overrides(list(overrides or []))
    if out is not None:
        merged.setdefault("out_dir", str(out))
    try:
        cfg = load_config(config, merged)
    except Exception as exc:  # noqa: BLE001 - config errors must be readable
        console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    run_dir = ensure_dir(cfg.run_dir)
    setup_logging(run_dir, level=cfg.log_level, quiet=quiet)
    save_resolved(cfg, run_dir)
    if stage:
        console.print(f"[bold cyan]vcdd {stage}[/bold cyan]  run_dir={escape(str(run_dir))}")
    return cfg, run_dir


def _ledger(run_dir: Path) -> RunLedger:
    return RunLedger(run_dir / "run.db")


def _need(path: Path, what: str, hint: str) -> Path:
    if not path.exists():
        console.print(f"[red]missing {what}:[/red] {escape(str(path))}\n  -> {escape(hint)}")
        raise typer.Exit(code=1)
    return path


def _receptor_inputs(cfg: RunConfig, run_dir: Path):
    """Prepared receptor + box, preparing them on demand if absent."""
    from vcdd.receptor.box import Box

    rec_dir = run_dir / "receptor"
    pdbqt, box_json = rec_dir / "receptor.pdbqt", rec_dir / "box.json"
    if not (pdbqt.exists() and box_json.exists()):
        if not cfg.receptor.path:
            console.print(
                "[red]no prepared receptor and no receptor.path in the config[/red]\n"
                "  -> run `vcdd prep-receptor -c <config>` first"
            )
            raise typer.Exit(code=1)
        _prep_receptor(cfg, run_dir)
    return pdbqt, Box.load(box_json)


def _print_kv(title: str, data: dict) -> None:
    table = Table(title=title, show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for k, v in data.items():
        if isinstance(v, float):
            v = f"{v:,.4g}"
        elif isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
            if len(v) > 110:
                v = v[:107] + "..."
        # Rich reads `[nH]` in a SMILES as a style tag and deletes it, which
        # would silently print a different molecule. Never interpret markup
        # in values that came from the data.
        table.add_row(Text(str(k)), Text(str(v)))
    console.print(table)


# --------------------------------------------------------------------------- #
# doctor / config
# --------------------------------------------------------------------------- #


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Report what is installed here and what each missing piece would buy."""
    from vcdd.cell.assets import bundle_status
    from vcdd.cell.base import list_backends
    from vcdd.docking.base import available_engines, engine_names, get_engine
    from vcdd.optimize.generators import get_generator_class, list_generators

    report: dict[str, Any] = {"vcdd": __version__, "python": sys.version.split()[0]}

    # ---- docking engines
    avail = available_engines()
    engines = {}
    for name in engine_names():
        try:
            eng = get_engine(name)
            ok = name in avail
            caps = eng.capabilities()
            detail = caps.notes or f"{caps.speed_class}, scoring={list(caps.supports_scoring_functions)}"
            if ok:
                detail = f"{eng.version() or 'version unknown'}; {detail}"
            else:
                detail = f"not runnable here ({detail})"
            engines[name] = {"available": ok, "detail": detail}
        except Exception as exc:  # noqa: BLE001
            engines[name] = {"available": False, "detail": str(exc)}
    report["docking_engines"] = engines

    # ---- GPU: what stage 1 needs, checked in the order it fails
    report["gpu"] = _gpu_report()

    # ---- Boltz: what stage 2 needs
    report["boltz"] = _boltz_report()

    # ---- cell backends
    from vcdd.cell.base import get_backend_class

    backends = {}
    for name in list_backends():
        cls = get_backend_class(name)
        caps = cls.capabilities()
        try:
            model = cls(CellConfig())
            ok, why = model.is_available()
            model.close()
        except Exception as exc:  # noqa: BLE001
            ok, why = False, str(exc)
        backends[name] = {
            "tier": caps.tier,
            "available": ok,
            "detail": why,
            "novel_smiles": caps.accepts_novel_smiles,
            "reference_only": caps.reference_only,
            "install_hint": caps.install_hint,
        }
    report["cell_backends"] = backends

    # ---- analog generators
    gens = {}

    for name in list_generators():
        try:
            g = get_generator_class(name)(Stage4Config())
            ok, why = g.is_available()
        except Exception as exc:  # noqa: BLE001
            ok, why = False, str(exc)
        gens[name] = {"available": ok, "detail": why}
    report["analog_generators"] = gens

    report["cell_resources"] = bundle_status()
    report["tools"] = tool_versions()

    if json_out:
        console.print_json(json.dumps(report, default=str))
        raise typer.Exit(code=0)

    console.print(f"[bold]vcdd[/bold] {__version__}   python {report['python']}")
    for section, key in (
        ("docking_engines", "docking engines"),
        ("gpu", "stage 1: GPU + Vina-GPU 2.1"),
        ("boltz", "stage 2: Boltz API / local boltz"),
        ("cell_backends", "virtual-cell backends"),
        ("analog_generators", "analog generators"),
    ):
        t = Table(title=key, title_justify="left", header_style="bold")
        t.add_column("name"); t.add_column("ok", justify="center"); t.add_column("detail")
        for name, info in report[section].items():
            mark = "[green]yes[/green]" if info.get("available") else "[yellow]no[/yellow]"
            detail = str(info.get("detail") or "")
            if not info.get("available") and info.get("install_hint"):
                detail = f"{detail}  ({info['install_hint']})"
            # Details carry paths, error text and pip extras like `vcdd[boltz]`.
            # Rich would eat `[boltz]` as a style tag and silently print an
            # install hint that installs the wrong thing, so escape the cell and
            # let the table wrap instead of truncating the instruction mid-word.
            t.add_row(escape(name), mark, escape(detail))
        console.print(t)

    t = Table(title="cell resource bundle", title_justify="left", header_style="bold")
    t.add_column("resource"); t.add_column("ok", justify="center"); t.add_column("rows", justify="right")
    for name, info in report["cell_resources"].items():
        mark = "[green]yes[/green]" if info.get("present") else "[red]no[/red]"
        t.add_row(name, mark, f"{info.get('rows', '-'):,}" if info.get("rows") else "-")
    console.print(t)

    missing = [k for k, v in report["cell_resources"].items() if not v.get("present")]
    if missing:
        console.print(
            f"[yellow]missing cell resources: {', '.join(missing)}[/yellow]\n"
            "  -> python scripts/build_cell_resources.py"
        )
    if not avail:
        console.print(
            "[red]no docking engine available; stage 1 cannot run[/red]\n"
            "  -> bash scripts/build_vina_gpu.sh --help"
        )
    if not report["boltz"]["api key"]["available"] and not report["boltz"]["local boltz"]["available"]:
        console.print(
            f"[yellow]no Boltz route configured: neither an API key nor a local boltz install[/yellow]\n"
            f"  -> export BOLTZ_API_KEY='sk_bc_...'   (a test-mode key rehearses it for free)\n"
            f"  -> or: pip install boltz and set stage2.backend: local (free, your GPU)"
        )


def _gpu_report() -> dict[str, dict[str, Any]]:
    """Everything stage 1 needs from the machine, in the order it breaks.

    Split out because "no docking engine available" is a useless diagnosis when
    the real answer is one of: no GPU, no driver, no OpenCL ICD, an unbuilt
    binary, or missing kernel caches.
    """
    import shutil
    import subprocess

    from vcdd.docking.base import gpu_available
    from vcdd.docking.engines.vina_gpu import VARIANTS, VinaGPUEngine

    out: dict[str, dict[str, Any]] = {}

    driver = "no nvidia-smi on PATH"
    has_gpu = gpu_available()
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            driver = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else proc.stderr.strip()
        except Exception as exc:  # noqa: BLE001
            driver = str(exc)
    out["gpu"] = {
        "available": has_gpu, "detail": driver,
        "install_hint": None if has_gpu else "Vina-GPU needs an OpenCL-capable GPU; "
                                             "set VCDD_ASSUME_GPU=1 only to test argv building",
    }

    found = VinaGPUEngine.variant_binaries()
    for key, variant in VARIANTS.items():
        path = found.get(key)
        out[key] = {
            "available": bool(path),
            "detail": (
                f"{path} (box edge < {variant.max_box_edge:.0f} A, "
                f"default thread {variant.default_thread})"
                if path else f"{variant.binary} not on PATH - {variant.notes}"
            ),
            "install_hint": None if path else "bash scripts/build_vina_gpu.sh",
        }

    icd = Path("/etc/OpenCL/vendors")
    vendors = sorted(p.name for p in icd.glob("*.icd")) if icd.is_dir() else []
    out["opencl icd"] = {
        "available": bool(vendors),
        "detail": ", ".join(vendors) if vendors else f"{icd} is empty or missing",
        "install_hint": None if vendors else "install the CUDA toolkit or an OpenCL ICD loader",
    }

    # `build_vina_gpu.sh` installs the kernel cache to
    # $PREFIX/share/vcdd/vina-gpu-kernels - a sibling of $PREFIX/bin, not the
    # binary's own directory. Looking only next to the binary reported "no
    # cached kernels" after a correct build and sent people back to `make
    # source` for nothing.
    kernel_dirs: list[Path] = []
    for p in found.values():
        if not p:
            continue
        bindir = Path(p).parent
        kernel_dirs += [bindir, bindir.parent / "share" / "vcdd" / "vina-gpu-kernels"]
    kernel_dirs += [Path.home() / ".local" / "share" / "vcdd" / "vina-gpu-kernels",
                    Path.cwd()]
    seen: set[Path] = set()
    kernels: list[str] = []
    for base in kernel_dirs:
        if base in seen:
            continue
        seen.add(base)
        kernels += [str(base / k) for k in ("Kernel1_Opt.bin", "Kernel2_Opt.bin")
                    if (base / k).exists()]
    if kernels:
        detail = (f"{', '.join(kernels)}  "
                  f"-> stage1.opencl_binary_path: {Path(kernels[0]).parent}")
    else:
        detail = "no Kernel*_Opt.bin found; the first run recompiles them (slow, once)"
    out["cached kernels"] = {
        "available": bool(kernels),
        "detail": detail,
        "install_hint": None if kernels else
                        "make source in the Vina-GPU tree, or set stage1.opencl_binary_path",
    }
    return out


def _boltz_report() -> dict[str, dict[str, Any]]:
    """Whether stage 2 can talk to the Boltz API, and whether local boltz is installed."""
    from vcdd.refine.boltz import local as bloc
    from vcdd.refine.boltz.client import boltz_available, resolve_credentials, sdk_version

    sdk_ok = boltz_available()
    creds = resolve_credentials()
    mode = "live" if creds.livemode else "test (synthetic results, no billing)"
    local_binary = bloc.find_binary()
    local_ver = bloc.version() if local_binary else None
    return {
        "sdk": {
            "available": sdk_ok,
            "detail": f"boltz-api {sdk_version()}" if sdk_ok else "boltz-api not importable",
            "install_hint": None if sdk_ok else "pip install 'vcdd[boltz]'",
        },
        "api key": {
            "available": creds.present,
            "detail": (
                f"$BOLTZ_API_KEY set (...{creds.suffix}), mode: {mode}"
                if creds.present else "$BOLTZ_API_KEY not set"
            ),
            "install_hint": None if creds.present else
                            "export BOLTZ_API_KEY='sk_bc_...' - keys at https://api.boltz.bio",
        },
        "local boltz": {
            "available": local_binary is not None,
            "detail": (
                f"boltz {local_ver} at {local_binary}" if local_binary
                else "boltz executable not found on PATH"
            ),
            "install_hint": None if local_binary else
                            "pip install boltz   (enables stage2.backend: local, no API key, no billing)",
        },
    }


@app.command()
def config(
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write here instead of stdout."),
) -> None:
    """Print the annotated configuration template."""
    text = template_text()
    if out:
        Path(out).write_text(text)
        console.print(f"wrote {escape(str(out))}")
    else:
        sys.stdout.write(text)


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #


def _prep_library(cfg: RunConfig, run_dir: Path) -> dict:
    from vcdd.io.library import prepare_library

    if not cfg.library.path:
        console.print("[red]library.path is not set[/red]")
        raise typer.Exit(code=2)
    t0 = time.time()
    stats = prepare_library(cfg.library.path, run_dir / "library", cfg.library)
    d = stats.as_dict()
    d["seconds"] = round(time.time() - t0, 2)
    _print_kv("library", d)
    if d.get("n_accepted", 0) == 0:
        console.print("[red]no molecules survived preparation; check filters[/red]")
        raise typer.Exit(code=1)
    return d


def _prep_receptor(cfg: RunConfig, run_dir: Path) -> dict:
    from vcdd.docking.base import DockParams, select_engine
    from vcdd.receptor.box import resolve_box
    from vcdd.receptor.prepare import prepare_receptor
    from vcdd.receptor.redock import run_redock_gate

    if not cfg.receptor.path:
        console.print("[red]receptor.path is not set[/red]")
        raise typer.Exit(code=2)
    rec_dir = ensure_dir(run_dir / "receptor")
    rec = prepare_receptor(
        cfg.receptor.path, rec_dir,
        chain=cfg.receptor.chain,
        keep_waters=cfg.receptor.keep_waters,
        keep_metals=cfg.receptor.keep_metals,
    )
    for w in rec.warnings:
        console.print(f"[yellow]receptor:[/yellow] {escape(str(w))}")
    box = resolve_box(cfg.receptor, cfg.receptor.reference_ligand_path)
    box.save(rec_dir / "box.json")
    for w in box.warnings:
        console.print(f"[yellow]box:[/yellow] {escape(str(w))}")
    out = {
        "method": rec.method, "n_atoms": rec.n_atoms,
        "box_center": list(box.center), "box_size": list(box.size), "box_source": box.source,
    }

    if cfg.receptor.redock_validation and cfg.receptor.reference_ligand_path:
        # The gate uses stage 1's engine and protocol: it is the only docking
        # protocol left, and validating a protocol other than the one that will
        # screen the library would tell you nothing about the screen.
        from vcdd.screening.stage1 import build_dock_params

        engine = select_engine(cfg.stage1.engine, cfg.stage1.engine_selection)
        params = DockParams(**build_dock_params(cfg.stage1))
        res = run_redock_gate(
            engine, rec.pdbqt_path, cfg.receptor.reference_ligand_path, box, params,
            threshold=cfg.receptor.redock_rmsd_threshold,
            reference_resname=cfg.receptor.reference_ligand,
        )
        res.save(rec_dir / "redock_validation.json")
        out.update(
            redock_passed=res.passed, redock_rmsd=res.all_atom_rmsd,
            redock_core_rmsd=res.core_rmsd, redock_verdict=res.verdict,
        )
        colour = "green" if res.passed else "red"
        console.print(
            f"[{colour}]redock gate: {res.verdict}[/{colour}] "
            f"(all-atom RMSD {res.all_atom_rmsd}, core {res.core_rmsd}, "
            f"threshold {res.threshold} A)"
        )
        if not res.passed and cfg.receptor.redock_fail_hard:
            console.print(
                "[red]redock gate failed and receptor.redock_fail_hard is set[/red]\n"
                "  -> the pose geometry is not reproduced; docking scores on this "
                "receptor are not trustworthy"
            )
            raise typer.Exit(code=1)
    else:
        out["redock_passed"] = None
        console.print("[yellow]redock validation skipped[/yellow] (no reference ligand or disabled)")
    _print_kv("receptor", out)
    return out


@app.command("prep-library")
def prep_library_cmd(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
) -> None:
    """Read, standardize, filter, deduplicate and shard the compound library."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="prep-library")
    _prep_library(cfg, run_dir)


@app.command("prep-receptor")
def prep_receptor_cmd(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
) -> None:
    """Clean the receptor, build the docking box, run the native-ligand redock gate."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="prep-receptor")
    _prep_receptor(cfg, run_dir)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


@app.command()
def stage1(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, force: bool = _FORCE, quiet: bool = _QUIET,
) -> None:
    """Stage 1: fast docking over the whole library."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="stage1")
    _stage1(cfg, run_dir, force)


def _stage1(cfg: RunConfig, run_dir: Path, force: bool) -> dict:
    from vcdd.screening.stage1 import run_stage1

    lib = run_dir / "library"
    if not (lib / "index.parquet").exists():
        _prep_library(cfg, run_dir)
    pdbqt, box = _receptor_inputs(cfg, run_dir)
    with _ledger(run_dir) as ledger:
        summary = run_stage1(cfg, pdbqt, box, lib, run_dir / "stage1", ledger, force=force)
    _print_kv("stage 1", summary)
    return summary


_YES_SPEND = typer.Option(
    False, "--yes-spend",
    help="Proceed with stage 2 even when the Boltz cost estimate is unavailable "
         "or above stage2.max_spend_usd. This spends real money.",
)


@app.command()
def stage2(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, force: bool = _FORCE, quiet: bool = _QUIET,
    yes_spend: bool = _YES_SPEND,
) -> None:
    """Stage 2: Boltz co-folding + binding confidence on the stage-1 shortlist.

    ``stage2.backend`` picks the route: ``library_screen`` / ``structure_and_binding``
    call the hosted API and are billed per molecule or per sample - the cost is
    estimated first and the run aborts above ``stage2.max_spend_usd`` unless
    ``--yes-spend``. ``local`` runs boltz-2 on this machine via ``boltz
    predict``: no API key, no billing, no spend gate.
    """
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="stage2")
    _stage2(cfg, run_dir, force, allow_spend=yes_spend)


def _stage2(cfg: RunConfig, run_dir: Path, force: bool, allow_spend: bool = False) -> dict:
    from vcdd.refine.stage2 import run_stage2

    _need(
        run_dir / "stage1" / "ranking.parquet", "stage-1 ranking",
        "run `vcdd stage1` first",
    )
    pdbqt, box = _receptor_inputs(cfg, run_dir)
    ensemble = [str(p) for p in cfg.receptor.ensemble_paths] or None
    with _ledger(run_dir) as ledger:
        summary = run_stage2(
            cfg, pdbqt, box, run_dir / "stage1", run_dir / "stage2", ledger,
            force=force, ensemble_pdbqt=ensemble, allow_spend=allow_spend,
        )
    _print_kv("stage 2", summary)
    return summary


@app.command()
def stage3(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, force: bool = _FORCE, quiet: bool = _QUIET,
) -> None:
    """Stage 3: virtual-cell perturbation prediction and objective scoring."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="stage3")
    _stage3(cfg, run_dir, force)


def _stage3(cfg: RunConfig, run_dir: Path, force: bool) -> dict:
    from vcdd.cell.stage3 import run_stage3

    if not (run_dir / "stage2").exists() and not (run_dir / "stage1").exists():
        console.print("[red]neither stage 1 nor stage 2 has run[/red]")
        raise typer.Exit(code=1)
    with _ledger(run_dir) as ledger:
        res = run_stage3(cfg, run_dir / "stage2", run_dir / "stage3", ledger, force=force)
    _print_kv("stage 3", res.summary)
    for w in res.summary.get("warnings", []):
        console.print(f"[yellow]stage3:[/yellow] {escape(str(w))}")
    if not res.scores.empty:
        cols = [c for c in ["mol_id", "objective", "gene_score", "pathway_score",
                            "target_consistency", "specificity", "out_of_domain"]
                if c in res.scores.columns]
        console.print(Text(res.scores.head(10)[cols].to_string(index=False)))
    return res.summary


@app.command()
def stage4(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, force: bool = _FORCE, quiet: bool = _QUIET,
) -> None:
    """Stage 4: propose and rank analogs of the stage-3 winners."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="stage4")
    _stage4(cfg, run_dir, force)


def _stage4(cfg: RunConfig, run_dir: Path, force: bool) -> dict:
    from vcdd.optimize import run_stage4

    _need(
        run_dir / "stage3" / "objective_scores.parquet", "stage-3 scores",
        "run `vcdd stage3` first",
    )
    pdbqt: Path | None = None
    box = None
    if cfg.stage4.redock_mode != "none":
        pdbqt, box = _receptor_inputs(cfg, run_dir)
    with _ledger(run_dir) as ledger:
        res = run_stage4(cfg, pdbqt, box, run_dir / "stage3", run_dir / "stage4",
                         ledger, force=force)
    _print_kv("stage 4", {k: v for k, v in res.summary.items()
                          if k not in ("parent_baseline", "generators", "cell_backends")})
    for w in res.summary.get("warnings", []):
        console.print(f"[yellow]stage4:[/yellow] {escape(str(w))}")
    if not res.recommendations.empty:
        cols = [c for c in ["rank", "analog_id", "parent_id", "generator", "transformation",
                            "delta_dock_score", "delta_cell_objective", "design_score"]
                if c in res.recommendations.columns]
        console.print(Text(res.recommendations.head(10)[cols].to_string(index=False)))
    return res.summary


# --------------------------------------------------------------------------- #
# report / run / resume
# --------------------------------------------------------------------------- #


@app.command()
def report(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
) -> None:
    """Build the standalone HTML report for a run directory."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="report")
    path = _report(cfg, run_dir)
    console.print(f"report: {escape(str(path))}")


def _report(cfg: RunConfig, run_dir: Path) -> Path:
    from vcdd.report.html import build_report

    return build_report(run_dir, cfg)


@app.command()
def run(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, force: bool = _FORCE, quiet: bool = _QUIET,
    stages: str = typer.Option("1,2,3,4", "--stages", help="Subset of stages, e.g. '1,2'."),
    skip_report: bool = typer.Option(False, "--skip-report"),
    yes_spend: bool = _YES_SPEND,
) -> None:
    """Run the whole pipeline: prep -> stage 1 -> 2 -> 3 -> 4 -> report."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="run")
    wanted = {s.strip() for s in stages.split(",") if s.strip()}
    t0 = time.time()

    man = Manifest(run_dir, cfg.resolved_dict())
    if cfg.library.path:
        man.add_input("library", cfg.library.path)
    if cfg.receptor.path:
        man.add_input("receptor", cfg.receptor.path)
    if cfg.receptor.reference_ligand_path:
        man.add_input("reference_ligand", cfg.receptor.reference_ligand_path)

    if "1" in wanted:
        man.add_stage("library", _prep_library(cfg, run_dir))
    man.add_stage("receptor", _prep_receptor(cfg, run_dir))
    if "1" in wanted:
        man.add_stage("stage1", _stage1(cfg, run_dir, force))
    if "2" in wanted:
        man.add_stage("stage2", _stage2(cfg, run_dir, force, allow_spend=yes_spend))
    if "3" in wanted:
        man.add_stage("stage3", _stage3(cfg, run_dir, force))
    if "4" in wanted:
        man.add_stage("stage4", _stage4(cfg, run_dir, force))
    if not skip_report:
        try:
            man.add_stage("report", {"path": str(_report(cfg, run_dir))})
        except Exception as exc:  # noqa: BLE001 - a report failure must not void the run
            man.warn(f"report generation failed: {exc}")
            console.print(f"[yellow]report generation failed:[/yellow] {escape(str(exc))}")
    man.add_stage("run", {"seconds": round(time.time() - t0, 2)})
    man.save()
    close_logs()
    console.print(
        f"[bold green]done[/bold green] in {time.time() - t0:,.1f}s -> {run_dir}"
    )


@app.command()
def resume(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
    stages: str = typer.Option("1,2,3,4", "--stages"),
    show: bool = typer.Option(False, "--show", help="Only print ledger state, do not run."),
    yes_spend: bool = _YES_SPEND,
) -> None:
    """Continue an interrupted run; completed work is reused, not repeated."""
    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="resume")
    with _ledger(run_dir) as ledger:
        state = ledger.summary()
    t = Table(title="resume ledger", header_style="bold", title_justify="left")
    t.add_column("stage"); t.add_column("done", justify="right")
    t.add_column("running", justify="right"); t.add_column("failed", justify="right")
    for stage, counts in sorted(state.items()):
        t.add_row(stage, str(counts.get("done", 0)), str(counts.get("running", 0)),
                  str(counts.get("failed", 0)))
    console.print(t)
    if show:
        return
    _resume_run(cfg, run_dir, stages, allow_spend=yes_spend)


def _resume_run(
    cfg: RunConfig, run_dir: Path, stages: str, allow_spend: bool = False
) -> None:
    wanted = {s.strip() for s in stages.split(",") if s.strip()}
    if "1" in wanted:
        _stage1(cfg, run_dir, force=False)
    if "2" in wanted:
        _stage2(cfg, run_dir, force=False, allow_spend=allow_spend)
    if "3" in wanted:
        _stage3(cfg, run_dir, force=False)
    if "4" in wanted:
        _stage4(cfg, run_dir, force=False)
    try:
        _report(cfg, run_dir)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]report generation failed:[/yellow] {escape(str(exc))}")


# --------------------------------------------------------------------------- #
# boltz: price / inspect / recover stage-2 API work without re-running stage 2
# --------------------------------------------------------------------------- #

boltz_app = typer.Typer(
    help="Inspect and price the hosted Boltz work that stage 2 does.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(boltz_app, name="boltz")


def _boltz_client(cfg: RunConfig):
    """Build an SDK client or exit with the reason it could not be built."""
    from vcdd.refine.boltz import client as bclient

    try:
        creds = bclient.resolve_credentials(cfg.stage2.api_key_env)
        if not creds.present:
            console.print(
                f"[red]{cfg.stage2.api_key_env} is not set[/red]  "
                "export a Boltz API key (a `_test_` key costs nothing)"
            )
            raise typer.Exit(code=2)
        if not creds.livemode:
            console.print(
                "[yellow]test-mode key: results are synthetic and are not real "
                "predictions[/yellow]"
            )
        return bclient.make_client(
            cfg.stage2.api_key_env, request_timeout_s=float(cfg.stage2.request_timeout_s)
        ), creds
    except bclient.BoltzUnavailable as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc


@boltz_app.command("estimate")
def boltz_estimate(
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
) -> None:
    """Price stage 2 without submitting anything.

    ``estimate_cost`` allocates no GPU and is not billed, so this is the safe
    way to find out what the shortlist will cost before you commit to it.
    """
    from vcdd.refine.boltz import target as btarget
    from vcdd.refine.stage2 import _estimate, _shortlist
    from vcdd.runtime.parallel import batched

    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="boltz-estimate")
    if cfg.stage2.backend == "local":
        console.print(
            "backend 'local' runs boltz predict on this machine: there is no API to "
            "estimate and nothing to pay - `vcdd stage2` skips the spend gate."
        )
        raise typer.Exit(code=0)
    _need(run_dir / "stage1" / "ranking.parquet", "stage-1 ranking",
          "run `vcdd stage1` first")
    pdbqt, box = _receptor_inputs(cfg, run_dir)
    s2 = cfg.stage2
    shortlist = _shortlist(run_dir / "stage1", int(s2.top_n))
    mol_ids = shortlist["mol_id"].astype(str).tolist()
    smiles = shortlist["smiles"].astype(str).tolist()
    target, prov = btarget.build_target(
        pdbqt, box=box, pocket_from_box=bool(s2.pocket_from_box),
        pocket_margin=float(s2.pocket_margin_a),
    )
    filters = (
        {"boltz_smarts_catalog_filter_level": s2.molecule_filter_level}
        if s2.molecule_filter_level != "disabled" else None
    )
    client, _creds = _boltz_client(cfg)
    chunks = [(ci, list(idx)) for ci, idx in
              enumerate(batched(range(len(mol_ids)), int(s2.chunk_size)))]
    estimates = _estimate(client, s2, target, filters, mol_ids, smiles, chunks)

    total = 0.0
    unknown = False
    for est in estimates:
        try:
            total += float(est.get("estimated_cost_usd"))
        except (TypeError, ValueError):
            unknown = True
    _print_kv("boltz estimate", {
        "backend": s2.backend,
        "n_molecules": len(mol_ids),
        "n_chunks": len(chunks),
        "chunk_size": s2.chunk_size,
        "num_samples": s2.num_samples if s2.backend == "structure_and_binding" else "n/a",
        "estimated_cost_usd": "unknown" if unknown else round(total, 2),
        "max_spend_usd": s2.max_spend_usd,
        "pocket_sizes": prov.get("pocket_sizes"),
    })
    if unknown:
        console.print("[yellow]at least one chunk returned no price; "
                      "`vcdd stage2` will refuse to run without --yes-spend[/yellow]")
    elif total > float(s2.max_spend_usd):
        console.print(f"[yellow]above stage2.max_spend_usd "
                      f"({s2.max_spend_usd}); raise it or pass --yes-spend[/yellow]")


@boltz_app.command("status")
def boltz_status(
    run_id: str = typer.Argument(..., help="Boltz run / prediction id."),
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
) -> None:
    """Retrieve the live status of one Boltz run submitted by stage 2."""
    from vcdd.refine.boltz import sab, screen

    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="boltz-status")
    if cfg.stage2.backend == "local":
        from vcdd.refine.boltz.screen import load_runs

        runs_path = run_dir / "stage2" / "boltz" / "runs.json"
        runs = load_runs(runs_path)
        match = [r for r in runs if r.run_id == run_id]
        if not match:
            console.print(
                f"[yellow]no local run {run_id!r} recorded in {escape(str(runs_path))}; "
                f"stage 2 may not have reached this molecule[/yellow]"
            )
            raise typer.Exit(code=1)
        _print_kv("boltz run", match[0].as_dict())
        return
    client, _creds = _boltz_client(cfg)
    if cfg.stage2.backend == "structure_and_binding":
        resp = sab.retrieve(client, run_id, workspace_id=cfg.stage2.workspace_id)
        info = {
            "run_id": run_id, "backend": "structure_and_binding",
            "status": screen._get(resp, "status"),
            "model": screen._get(resp, "model"),
            "version": screen._get(resp, "version"),
            "livemode": screen._get(resp, "livemode"),
        }
    else:
        kwargs = ({"workspace_id": cfg.stage2.workspace_id}
                  if cfg.stage2.workspace_id else {})
        resp = client.small_molecule.library_screen.retrieve(run_id, **kwargs)
        progress = screen._get(resp, "progress")
        info = {
            "run_id": run_id, "backend": "library_screen",
            "status": screen._get(resp, "status"),
            "engine": screen._get(resp, "engine"),
            "engine_version": screen._get(resp, "engine_version"),
            "livemode": screen._get(resp, "livemode"),
            "screened": screen._get(progress, "num_molecules_screened"),
            "failed": screen._get(progress, "num_molecules_failed"),
            "total": screen._get(progress, "total_molecules_to_screen"),
        }
    err = screen._get(resp, "error")
    if err is not None:
        info["error"] = f"{screen._get(err, 'code')}: {screen._get(err, 'message')}"
    _print_kv("boltz run", info)


@boltz_app.command("fetch")
def boltz_fetch(
    run_id: str = typer.Argument(..., help="Boltz library_screen run id."),
    config: Optional[Path] = _CONFIG, set_: Optional[list[str]] = _SET,
    out: Optional[Path] = _OUT, quiet: bool = _QUIET,
    to: Optional[Path] = typer.Option(
        None, "--to", help="JSONL to write (default: <run>/stage2/boltz/results.jsonl)."),
) -> None:
    """Re-download results for one ``library_screen`` run into the run directory.

    Use this when stage 2 was interrupted after submission: the work is already
    paid for, and the results sit on the server until they expire.
    """
    from vcdd.refine.boltz import screen

    cfg, run_dir = _bootstrap(config, set_, out, quiet, stage="boltz-fetch")
    if cfg.stage2.backend == "local":
        console.print(
            "[red]fetch applies to API runs only; local predictions are already on disk "
            "under <run>/stage2/boltz/local/[/red]"
        )
        raise typer.Exit(code=2)
    if cfg.stage2.backend != "library_screen":
        console.print("[red]fetch only applies to the library_screen backend; "
                      "structure_and_binding results are per-molecule[/red]")
        raise typer.Exit(code=2)
    client, _creds = _boltz_client(cfg)
    rows = screen.fetch_results(client, run_id, workspace_id=cfg.stage2.workspace_id)
    dest = Path(to) if to else run_dir / "stage2" / "boltz" / "results.jsonl"
    ensure_dir(dest.parent)
    total = screen.append_results_jsonl(dest, rows)
    _print_kv("boltz fetch", {
        "run_id": run_id, "n_fetched": len(rows), "n_in_file": total, "path": str(dest),
    })


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


def main() -> None:  # pragma: no cover - console-script entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
