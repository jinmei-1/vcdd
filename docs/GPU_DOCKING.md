# GPU docking engine

Stage 1 and the stage-4 re-dock both need a docking engine. The only production engine is
**Vina-GPU 2.1**, which is a compiled binary with its own OpenCL kernels and therefore cannot be
installed with pip. Everything else in vcdd is pip-installable.

## Check first

```bash
vcdd doctor
```

`doctor` reports what is installed here and what each missing piece would buy, so a partial
environment is visible before it wastes GPU time.

## Variants

| variant | notes |
|---|---|
| `autodock_vina_gpu` | upstream default |
| `quickvina2_gpu` | faster search; use `thread: 5000` (keep below 10000) |
| `quickvina_w_gpu` | widest search of the three |

Box edge limits are **compiled into the binary**: below 100 A for `AutoDock-Vina-GPU-2.1` and below
70 A for the QuickVina variants. A larger box needs `-DLARGE_BOX` at build time.

## Failure behaviour

If no engine is available, stage 1 and the stage-4 re-dock **raise** rather than substituting a
slower method:

```
RuntimeError: docking engine 'vina_gpu' is not available on this machine.
Available: []. Run 'vcdd doctor' for details.
```

This is intentional. The engine variant is a configuration-level setting while the artifact schema
is fixed by the stage contract, so an explicit failure keeps the semantics of a run unambiguous. It
also means stage 4 can be resumed after an engine is fixed: the cached analog proposals are reused
rather than regenerated, because generation does not depend on the docking engine.

## Building Vina-GPU 2.1

See `scripts/build_vina_gpu.sh` and `scripts/build_vina_README`. An OpenCL runtime and the kernel
files (`Kernel1_Opt.bin`, `Kernel2_Opt.bin`) must be present; point
`stage1.opencl_binary_path` at the directory holding them if they are not found automatically.

## Mock engine

`stage1.engine: mock` exercises the pipeline shape without docking. It requires
`VCDD_ALLOW_MOCK=1` in the environment, so a mock run cannot be mistaken for a real one by
accident.
