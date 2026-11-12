# Contributing to vcdd

## Development setup

```bash
cd vcdd
conda env create -f environment.yml
conda activate vcdd
pip install -e ".[dev]"
```

The chemistry stack (RDKit) and the GPU docking engine are best installed
through conda; see `environment.yml`. Everything else is pip-installable, and
`pip install -e ".[dev]"` is enough for the parts of the test suite that do not
touch a real docking engine.

## Running the tests

```bash
pytest                      # fast suite
pytest -m "not slow"        # skip end-to-end tests that need a docking engine
pytest -m "not network"     # skip tests that need outbound network access
```

Tests marked `slow` invoke a real docking engine and are skipped automatically
when none is available; tests marked `network` fetch remote resources.

## Stage contract

Each stage is an independent unit with a versioned artifact schema. If you add
a stage or change an existing one:

1. Keep the stage runnable on its own — a stage reads its inputs from disk and
   writes its outputs to disk, without requiring a prior stage to be re-run.
2. Keep the artifact schema stable, or bump it explicitly and record the change
   in `CHANGELOG.md`.
3. Record every run in the SQLite ledger so results stay traceable.
4. Never let a stage fail silently. If a required engine or model is missing,
   raise with an actionable message instead of degrading to a different method.

## Adding a virtual-cell backend

Backends live in `vcdd/cell/backends/` and implement the interface in
`vcdd/cell/base.py`. A backend must declare whether it predicts absolute
expression or a perturbation signature, and must not be silently substituted for
a different backend: `cell.engine_selection` controls that explicitly.

## Reporting issues

Please include the resolved configuration (`config.resolved.yaml`), the stage
log, and the Python and dependency versions. A run is only reproducible if the
resolved configuration is available, so it is attached to every run directory
by design.

## Code style

- Type-annotate public functions.
- Prefer explicit failure over implicit fallback.
- Keep numerical definitions in one place and let reports read them from there,
  so a re-run cannot leave stale numbers behind in a manuscript or a figure.
