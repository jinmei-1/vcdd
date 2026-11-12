"""Vina-GPU 2.1 engine - the only production docking engine in 0.2.0.

Upstream: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1

Three binaries are built from that one source tree and share a command line, so
they are one engine here with a ``variant`` switch rather than three classes:

===========================  ===========================  ==============  =========
variant key                  binary                       max box edge    default --thread
===========================  ===========================  ==============  =========
``autodock_vina_gpu``        ``AutoDock-Vina-GPU-2-1``     < 100 A         8000
``quickvina2_gpu``           ``QuickVina2-GPU-2-1``        < 70 A          5000
``quickvina_w_gpu``          ``QuickVina-W-GPU-2-1``       < 70 A          8000
===========================  ===========================  ==============  =========

Why this file exists at all
---------------------------
0.1.0 shipped ``VinaGPUEngine`` as a two-line subclass of the CPU
``VinaLikeEngine``, which meant it emitted ``--ligand --out --exhaustiveness
--num_modes --energy_range --cpu --seed``. **Vina-GPU 2.1 accepts none of those
flags.** That engine could never have produced a single docking score; it was a
name with no implementation behind it. Everything below is written against the
actual upstream command line.

Consequences the caller has to live with
----------------------------------------
* **Batch, not per-ligand.** The binary takes ``--ligand_directory`` and writes
  to ``--output_directory``. It uploads the receptor grid and the OpenCL kernels
  once and then keeps the device busy. Calling it once per molecule pays that
  setup per molecule and is *slower* than CPU Vina. :meth:`VinaGPUEngine.dock`
  exists only so the abstract contract is satisfied; the real entry point is
  :meth:`VinaGPUEngine.dock_batch`.
* **No ``--seed``.** There is no way to fix the Monte-Carlo seed, so stage 1 is
  reproducible in *protocol* but not bit-for-bit in *score*. The engine declares
  ``deterministic=False`` and stage 1 warns once. Rankings should be treated as
  stable only to the resolution the score noise allows.
* **Compiled-in box limit.** The upstream Makefile bakes ``-DSMALL_BOX`` or
  ``-DLARGE_BOX`` into the binary. A box wider than the compiled limit does not
  error inside the binary - it produces garbage - so it is rejected here first.
* **Kernel cache.** Without ``--opencl_binary_path`` pointing at the directory
  holding ``Kernel1_Opt.bin`` and ``Kernel2_Opt.bin``, the binary recompiles the
  OpenCL kernels on every invocation, which can dominate a small batch.
* **Stack size.** Upstream requires ``ulimit -s 8192``; the default 8 MB on most
  Linux distributions is already enough, but containers frequently set less and
  the binary then segfaults. The soft limit is raised in the child process.

Run ``scripts/build_vina_gpu.sh`` to compile the binaries, or ``vcdd doctor`` to
see which of them this machine can find.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vcdd.docking.base import (
    DockParams,
    DockResult,
    DockingEngine,
    EngineCapabilities,
    gpu_available,
    register_engine,
)
from vcdd.docking.engines.pdbqt_parse import best_score, parse_pdbqt_scores
from vcdd.receptor.box import Box
from vcdd.runtime.logging import get_logger

LOG = get_logger()


@dataclass(frozen=True)
class VinaGPUVariant:
    """One compiled binary from the Vina-GPU 2.1 source tree."""

    key: str
    binary: str
    #: Compiled-in ceiling on the box edge, in Angstrom (exclusive).
    max_box_edge: float
    #: Upstream's default ``--thread`` for this binary.
    default_thread: int
    #: Rough speedup versus one CPU Vina core, used only to order ``auto``.
    relative_speed: float
    notes: str


VARIANTS: dict[str, VinaGPUVariant] = {
    "autodock_vina_gpu": VinaGPUVariant(
        key="autodock_vina_gpu",
        binary="AutoDock-Vina-GPU-2-1",
        max_box_edge=100.0,
        default_thread=8000,
        relative_speed=15.0,
        notes="Full AutoDock Vina scoring on GPU. Most accurate of the three; "
              "accepts the largest box.",
    ),
    "quickvina2_gpu": VinaGPUVariant(
        key="quickvina2_gpu",
        binary="QuickVina2-GPU-2-1",
        max_box_edge=70.0,
        default_thread=5000,
        relative_speed=20.0,
        notes="QuickVina 2 heuristics on GPU. Fastest; the search is more "
              "aggressive, so scores are noisier than AutoDock-Vina-GPU.",
    ),
    "quickvina_w_gpu": VinaGPUVariant(
        key="quickvina_w_gpu",
        binary="QuickVina-W-GPU-2-1",
        max_box_edge=70.0,
        default_thread=8000,
        relative_speed=18.0,
        notes="QuickVina-W, tuned for blind/wide-box docking. Use when the "
              "pocket is uncertain rather than for a defined site.",
    ),
}

DEFAULT_VARIANT = "autodock_vina_gpu"

#: Vina-GPU 2.1 writes ``<ligand-stem>_out.pdbqt`` into --output_directory.
_OUT_SUFFIX = "_out.pdbqt"

#: Upstream requires ``ulimit -s 8192`` (kB) before running.
_STACK_SOFT_BYTES = 8192 * 1024


def resolve_variant(name: str | None) -> VinaGPUVariant:
    """Look up a variant by key, with an error that lists the valid ones."""
    key = name or DEFAULT_VARIANT
    if key not in VARIANTS:
        raise ValueError(
            f"unknown Vina-GPU variant {key!r}. Valid: {sorted(VARIANTS)}"
        )
    return VARIANTS[key]


def check_box(box: Box, variant: VinaGPUVariant) -> None:
    """Reject a box the compiled binary cannot handle.

    The binary does not validate this itself - it silently returns nonsense - so
    catching it here is the difference between a failed run and a plausible-
    looking but meaningless ranking.
    """
    edge = max(box.size)
    if edge >= variant.max_box_edge:
        raise ValueError(
            f"box edge {edge:.1f} A exceeds the {variant.max_box_edge:.0f} A limit compiled "
            f"into {variant.binary}. Either shrink the box, or rebuild the binary with "
            f"-DLARGE_BOX instead of -DSMALL_BOX in the upstream Makefile "
            f"(scripts/build_vina_gpu.sh --large-box)."
        )


def _raise_stack_limit() -> None:  # pragma: no cover - exercised only in a child
    """``preexec_fn`` raising RLIMIT_STACK to upstream's required 8 MB."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        want = _STACK_SOFT_BYTES if hard == resource.RLIM_INFINITY else min(hard, _STACK_SOFT_BYTES)
        if soft == resource.RLIM_INFINITY or soft >= want:
            return
        resource.setrlimit(resource.RLIMIT_STACK, (want, hard))
    except Exception:  # noqa: BLE001 - never block the exec over a soft limit
        pass


def build_cmd(
    binary_path: str,
    receptor_pdbqt_path: str,
    ligand_dir: str | Path,
    output_dir: str | Path,
    box: Box,
    params: DockParams,
    variant: VinaGPUVariant,
) -> list[str]:
    """Assemble the Vina-GPU 2.1 command line.

    Only flags the binary actually accepts appear here. A unit test asserts
    that none of the CPU-Vina flags 0.1.0 used to emit (``--ligand``, ``--out``,
    ``--exhaustiveness``, ``--num_modes``, ``--energy_range``, ``--cpu``,
    ``--seed``) can ever be produced.
    """
    cx, cy, cz = box.center
    sx, sy, sz = box.size
    cmd = [
        binary_path,
        "--receptor", str(receptor_pdbqt_path),
        "--ligand_directory", str(ligand_dir),
        "--output_directory", str(output_dir),
        "--center_x", f"{cx:.3f}",
        "--center_y", f"{cy:.3f}",
        "--center_z", f"{cz:.3f}",
        "--size_x", f"{sx:.3f}",
        "--size_y", f"{sy:.3f}",
        "--size_z", f"{sz:.3f}",
        "--thread", str(int(params.thread)),
    ]
    if params.search_depth is not None:
        cmd += ["--search_depth", str(int(params.search_depth))]
    # Upstream default is 1; only emit the flag when switching it off, so the
    # command line stays the documented one in the common case.
    if not params.rilc_bfgs:
        cmd += ["--rilc_bfgs", "0"]
    if params.opencl_binary_path:
        cmd += ["--opencl_binary_path", str(params.opencl_binary_path)]
    cmd += [str(a) for a in params.extra_args]
    return cmd


class VinaGPUEngine(DockingEngine):
    """Batch-native GPU docking via the Vina-GPU 2.1 binary family."""

    name = "vina_gpu"

    # ------------------------------------------------------------ discovery
    @staticmethod
    def variant_binaries() -> dict[str, str | None]:
        """Map every variant key to its resolved binary path, or ``None``."""
        return {key: shutil.which(v.binary) for key, v in VARIANTS.items()}

    def available(self) -> bool:
        """Needs both a compiled binary *and* a visible GPU.

        Reporting available on a GPU-less machine would let
        ``engine_selection: auto`` pick an engine that fails on the first
        molecule of a million-molecule run.
        """
        return any(self.variant_binaries().values()) and gpu_available()

    def capabilities(self) -> EngineCapabilities:
        found = {k: p for k, p in self.variant_binaries().items() if p}
        if found:
            speed = max(VARIANTS[k].relative_speed for k in found)
            edge = min(VARIANTS[k].max_box_edge for k in found)
            detail = f"found: {', '.join(sorted(found))}"
        else:
            speed, edge = VARIANTS[DEFAULT_VARIANT].relative_speed, VARIANTS[DEFAULT_VARIANT].max_box_edge
            detail = "no variant binary on PATH; run scripts/build_vina_gpu.sh"
        return EngineCapabilities(
            name=self.name,
            needs_gpu=True,
            supports_flexible_residues=False,
            supports_scoring_functions=("vina",),
            supports_local_only=False,
            speed_class="fast",
            relative_speed=speed,
            batch_native=True,
            preferred_batch_size=1000,
            deterministic=False,
            max_box_edge=edge,
            notes=(
                "Vina-GPU 2.1 (DeltaGroupNJUPT). Batch-native: dock whole directories, "
                "never one ligand per process. No --seed flag exists, so scores are not "
                "bit-reproducible between runs. " + detail
            ),
        )

    def version(self) -> str | None:
        for key, path in self.variant_binaries().items():
            if path:
                return f"{VARIANTS[key].binary} (Vina-GPU 2.1)"
        return None

    # ----------------------------------------------------------------- dock
    def dock_batch(
        self,
        items: Sequence[tuple[str, str]],
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
    ) -> list[DockResult]:
        """Dock a whole batch, isolating ligands that break the binary.

        The binary takes a directory and aborts on the first ligand it cannot
        parse, which would otherwise mark every ligand in the batch as failed.
        A failed invocation is therefore retried in halves until the offending
        ligand is alone, so one bad molecule costs one molecule instead of
        ``gpu_batch_size`` of them.
        """
        variant = resolve_variant(params.variant)
        check_box(box, variant)
        binary_path = shutil.which(variant.binary)
        if not binary_path:
            return [
                DockResult(mol_id=mid, engine=self.name,
                           error=f"binary_not_found:{variant.binary}")
                for mid, _ in items
            ]
        if not items:
            return []

        order = [mid for mid, _ in items]
        by_id: dict[str, DockResult] = {}
        dockable: list[tuple[str, str]] = []
        for mol_id, pdbqt in items:
            if pdbqt and pdbqt.strip():
                dockable.append((mol_id, pdbqt))
            else:
                # Preparation failed upstream; an empty file also aborts the run.
                by_id[mol_id] = DockResult(mol_id=mol_id, engine=self.name,
                                           error="empty_ligand_pdbqt")
        for result in self._dock_isolated(dockable, binary_path, receptor_pdbqt_path,
                                          box, params, variant, depth=0):
            by_id[result.mol_id] = result
        return [by_id[mol_id] for mol_id in order]

    def _dock_isolated(
        self,
        items: list[tuple[str, str]],
        binary_path: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        variant: VinaGPUVariant,
        depth: int,
    ) -> list[DockResult]:
        """Run one invocation; on failure, bisect the failed ligands and retry."""
        if not items:
            return []
        results = self._invoke_once(items, binary_path, receptor_pdbqt_path,
                                    box, params, variant)
        failed_ids = {r.mol_id for r in results if r.error}
        if not failed_ids or len(items) == 1 or depth >= 16:
            return results
        LOG.warning(
            "dock batch of %d failed for %d ligand(s) (%s); splitting to isolate",
            len(items), len(failed_ids),
            next(r.error for r in results if r.error),
        )
        retry = [(mid, pdbqt) for mid, pdbqt in items if mid in failed_ids]
        half = len(retry) // 2
        return (
            [r for r in results if not r.error]
            + self._dock_isolated(retry[:half], binary_path, receptor_pdbqt_path,
                                  box, params, variant, depth + 1)
            + self._dock_isolated(retry[half:], binary_path, receptor_pdbqt_path,
                                  box, params, variant, depth + 1)
        )

    def _invoke_once(
        self,
        items: Sequence[tuple[str, str]],
        binary_path: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        variant: VinaGPUVariant,
    ) -> list[DockResult]:
        """One invocation of the binary over ``items``."""
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="vcdd_vinagpu_") as tmp:
            lig_dir = Path(tmp) / "ligands"
            out_dir = Path(tmp) / "out"
            lig_dir.mkdir()
            out_dir.mkdir()

            # Index-based filenames, deliberately not mol_id-based: library
            # identifiers routinely contain '/', spaces or non-ASCII, and the
            # binary derives the output name from the input name.
            stems: list[str] = []
            for i, (_mol_id, pdbqt) in enumerate(items):
                stem = f"lig_{i:06d}"
                stems.append(stem)
                (lig_dir / f"{stem}.pdbqt").write_text(pdbqt or "")

            cmd = build_cmd(binary_path, receptor_pdbqt_path, lig_dir, out_dir,
                            box, params, variant)
            env = dict(os.environ)
            batch_error: str | None = None
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, env=env,
                    timeout=params.batch_timeout_s, check=False,
                    preexec_fn=_raise_stack_limit,
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                    batch_error = f"rc{proc.returncode}:{tail[-1][:200]}" if tail else f"rc{proc.returncode}"
            except subprocess.TimeoutExpired:
                batch_error = "batch_timeout"
            except Exception as exc:  # noqa: BLE001
                batch_error = f"{type(exc).__name__}:{exc}"[:200]

            elapsed = time.time() - t0
            per_ligand = elapsed / max(1, len(items))
            results: list[DockResult] = []
            for (mol_id, _pdbqt), stem in zip(items, stems):
                results.append(
                    self._collect_one(mol_id, out_dir, stem, variant, len(items),
                                      per_ligand, batch_error)
                )
        return results

    def _collect_one(
        self,
        mol_id: str,
        out_dir: Path,
        stem: str,
        variant: VinaGPUVariant,
        batch_size: int,
        seconds: float,
        batch_error: str | None,
    ) -> DockResult:
        """Read one ligand's output file back into a :class:`DockResult`."""
        from vcdd.docking.ligprep import split_pdbqt_models

        extra = {"variant": variant.key, "batch_size": batch_size}
        # Upstream writes '<stem>_out.pdbqt'; some builds keep the input name.
        path = out_dir / f"{stem}{_OUT_SUFFIX}"
        if not path.exists():
            path = out_dir / f"{stem}.pdbqt"
        if not path.exists():
            return DockResult(mol_id=mol_id, engine=self.name, seconds=seconds,
                              error=batch_error or "no_output_file", extra=extra)
        text = path.read_text(errors="replace")
        scores = parse_pdbqt_scores(text)
        top = best_score(scores)
        if top is None:
            return DockResult(mol_id=mol_id, engine=self.name, seconds=seconds,
                              error=batch_error or "no_score_in_output", extra=extra)
        models = split_pdbqt_models(text)
        return DockResult(
            mol_id=mol_id,
            score=float(top),
            all_scores=[float(s) for s in scores],
            pose_pdbqt=models[0] if models else text,
            all_poses_pdbqt=text,
            engine=self.name,
            seconds=seconds,
            error=None,
            extra=extra,
        )

    def dock(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        mol_id: str = "",
    ) -> DockResult:
        """Single-ligand docking.

        Present only to satisfy the abstract contract and to let stage 4 re-dock
        a handful of analogs without special-casing the engine. It launches a
        whole GPU process for one molecule and is roughly an order of magnitude
        less efficient per ligand than :meth:`dock_batch`; drivers handling more
        than a few molecules must use the batch path.
        """
        return self.dock_batch([(mol_id, ligand_pdbqt)], receptor_pdbqt_path, box, params)[0]


register_engine("vina_gpu", VinaGPUEngine)
