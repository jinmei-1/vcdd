"""Packaging contract.

These tests guard the two failure modes that actually bit this repository:

1. Package **data** (the configuration template and the virtual-cell reference
   resources) silently missing from the installed distribution. A wheel that
   imports fine but raises `FileNotFoundError` on first use is worse than one
   that fails to build.
2. The **version drifted** between `pyproject.toml` and `vcdd/version.py`.
   `pyproject.toml` reads the version from `vcdd/version.py`, so these tests
   assert the single-source-of-truth wiring still holds.

They read from the source tree and, when a built wheel is present, from the
wheel as well, so they pass both before and after a release build.
"""
from __future__ import annotations

import re
import zipfile
from importlib import resources
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]

# Files that must ship with the package: the pipeline cannot run without them.
REQUIRED_RESOURCES = [
    "vcdd/config/template.yaml",
    "vcdd/cell/resources/MANIFEST.json",
    "vcdd/cell/resources/lincs_chem_consensus.parquet",
    "vcdd/cell/resources/lincs_crispr_ko_consensus.parquet",
    "vcdd/cell/resources/l1000_genes.tsv",
    "vcdd/cell/resources/collectri.tsv",
    "vcdd/cell/resources/progeny.tsv",
    "vcdd/cell/resources/hallmark.tsv",
]


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------- version wiring
def test_version_is_single_sourced():
    """pyproject must not hard-code a version; it must read vcdd/version.py."""
    proj = _pyproject()["project"]
    assert "version" not in proj, "version is hard-coded in pyproject; use dynamic"
    assert "version" in proj["dynamic"], "version must be listed under dynamic"
    hatch = _pyproject()["tool"]["hatch"]["version"]
    assert hatch["path"] == "vcdd/version.py"


def test_version_string_matches_dunder():
    """The version declared for the build and the imported one must agree."""
    import vcdd

    py = (ROOT / "vcdd" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', py, re.M)
    assert m, "vcdd/version.py has no __version__ assignment"
    assert m.group(1) == vcdd.__version__


def test_version_is_pep440():
    import vcdd

    assert re.fullmatch(r"\d+\.\d+\.\d+([.-].+)?", vcdd.__version__), vcdd.__version__


# ------------------------------------------------------------------- resources
@pytest.mark.parametrize("rel", REQUIRED_RESOURCES)
def test_resource_present_on_disk(rel):
    assert (ROOT / rel).is_file(), f"{rel} is missing from the source tree"


@pytest.mark.parametrize("rel", REQUIRED_RESOURCES)
def test_resource_reachable_via_importlib(rel):
    """Reachable the way the pipeline actually reads it, not just present on disk."""
    parts = rel.split("/")
    node = resources.files("vcdd")
    for part in parts[1:]:
        node = node.joinpath(part)
    assert node.is_file(), f"{rel} is not reachable from the installed package"


def test_lincs_resource_is_readable():
    import pandas as pd

    with resources.as_file(
        resources.files("vcdd").joinpath("cell/resources/lincs_chem_consensus.parquet")
    ) as f:
        df = pd.read_parquet(f)
    assert df.shape[0] > 1000, df.shape
    assert df.shape[1] > 3, df.shape


def test_resource_manifest_agrees_with_directory():
    """The shipped MANIFEST.json must describe the resources that are actually there."""
    import json

    base = ROOT / "vcdd" / "cell" / "resources"
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    listed = json.dumps(manifest)
    for name in ("collectri", "hallmark", "progeny", "l1000_genes"):
        assert name in listed, f"{name} is not recorded in MANIFEST.json"


def test_manifest_file_hashes_actually_match():
    """`file_sha256` must verify the shipped file, or it is a false assurance.

    Regression: the manifest used to carry a bare `sha256` that was the digest of
    the *downloaded source*, sitting next to the *written* file's size. Nothing
    could be verified from it, yet it read like an integrity check.
    """
    import hashlib
    import json

    base = ROOT / "vcdd" / "cell" / "resources"
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, entry in manifest["resources"].items():
        f = base / entry["file"]
        assert f.is_file(), f"{name}: {entry['file']} missing"
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        assert digest == entry["file_sha256"], f"{name}: file_sha256 does not match"
        assert f.stat().st_size == entry["file_bytes"], f"{name}: file_bytes does not match"
        assert "source_sha256" in entry, f"{name}: source digest must be labelled distinctly"


def test_shipped_files_contain_no_internal_absolute_paths():
    """No machine-specific absolute path may ship in the package.

    Regression: MANIFEST.json recorded `/mnt/datalake/...` as the source of the
    compound structures — meaningless to any user, and a disclosure of internal
    infrastructure.
    """
    import re

    # Paths that are *platform conventions* the code legitimately detects or
    # documents, not references to this project's own infrastructure. A bare
    # "is there a /mnt/... here" rule flags these and is useless.
    PLATFORM_PATHS = {
        b"/mnt/results",          # vcdd/runtime/paths.py: NETWORK_FS_PREFIXES
        b"/mnt/shared-workspace",
        b"/mnt/user-uploads",
        b"/mnt/user-results",
        b"/data/parse/schema.py",  # comments citing boltz's own package layout
        b"/data/types.py",
    }
    # A *machine-specific* path: a deep, project-named absolute path or a home
    # directory. These are what must never ship.
    patterns = [
        re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+"),          # someone's home
        re.compile(rb"/(?:mnt|srv|opt|data)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"),  # deep project dir
    ]
    offenders = []
    for p in (ROOT / "vcdd").rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        blob = p.read_bytes()
        hits = {h for pat in patterns for h in pat.findall(blob)}
        for h in hits:
            if any(h.startswith(a) or a.startswith(h) for a in PLATFORM_PATHS):
                continue
            offenders.append((p.relative_to(ROOT).as_posix(), h.decode("utf-8", "replace")))
    assert not offenders, f"machine-specific absolute paths shipped: {offenders}"


def test_manifest_records_public_structure_source():
    """The LINCS structure provenance must point at a public URL, not a local file."""
    import json

    base = ROOT / "vcdd" / "cell" / "resources"
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    entry = manifest["resources"]["lincs_chem_consensus"]
    src = entry.get("structure_source")
    if src is not None:
        assert src.startswith(("http://", "https://")), src


# ----------------------------------------------------------------- wheel (opt.)
def _first_wheel() -> Path | None:
    dist = ROOT / "dist"
    if not dist.is_dir():
        return None
    wheels = sorted(dist.glob("vcdd-*.whl"))
    return wheels[0] if wheels else None


def test_built_wheel_ships_every_resource():
    wheel = _first_wheel()
    if wheel is None:
        pytest.skip("no built wheel in dist/; run `python -m hatchling build`")
    names = set(zipfile.ZipFile(wheel).namelist())
    missing = [r for r in REQUIRED_RESOURCES if r not in names]
    assert not missing, f"wheel is missing {missing}"


def test_built_wheel_has_no_bytecode_or_caches():
    wheel = _first_wheel()
    if wheel is None:
        pytest.skip("no built wheel in dist/")
    junk = [
        n for n in zipfile.ZipFile(wheel).namelist()
        if "__pycache__" in n or n.endswith((".pyc", ".pyo"))
    ]
    assert not junk, f"wheel contains caches: {junk[:5]}"


def test_built_sdist_carries_every_top_level_file():
    """The sdist is the canonical source distribution: nothing may be left out.

    Regression: the sdist `include` list was written for the wheel's needs and
    silently omitted `run_all.py` (the campaign driver), `DESIGN.md`,
    `MIGRATION-0.2.0.md` and `requirements.txt`.
    """
    import tarfile

    dist = ROOT / "dist"
    sdists = sorted(dist.glob("vcdd-*.tar.gz")) if dist.is_dir() else []
    if not sdists:
        pytest.skip("no built sdist in dist/; run `python -m hatchling build`")
    names = tarfile.open(sdists[0]).getnames()
    wanted = [
        "README.md", "DESIGN.md", "MIGRATION-0.2.0.md", "CHANGELOG.md",
        "CONTRIBUTING.md", "CITATION.cff", "LICENSE", "LICENSE-NOTICES.md",
        "run_all.py", "pyproject.toml", "environment.yml", "requirements.txt",
    ]
    missing = [f for f in wanted if not any(n.endswith("/" + f) for n in names)]
    assert not missing, f"sdist is missing {missing}"


def test_built_wheel_version_matches():
    import vcdd

    wheel = _first_wheel()
    if wheel is None:
        pytest.skip("no built wheel in dist/")
    assert vcdd.__version__ in wheel.name, (wheel.name, vcdd.__version__)


# ------------------------------------------------------------------ example cfg
def test_example_config_validates_against_schema():
    """The shipped example must load, so users do not start from a broken file."""
    from vcdd.config.loader import load_config

    example = ROOT / "examples" / "minimal.yaml"
    if not example.is_file():
        pytest.skip("no examples/minimal.yaml")
    cfg = load_config(example)
    assert cfg.name
    assert cfg.stage2 is not None and cfg.cell is not None and cfg.stage4 is not None


def test_example_config_has_no_fabricated_gate():
    """Regression: the selection gate is post-hoc, and must not appear as config.

    An earlier version of the example invented a `selection:` block. Nothing in
    the schema implements it, so a user setting those keys would have silently
    configured nothing.
    """
    import yaml

    example = ROOT / "examples" / "minimal.yaml"
    if not example.is_file():
        pytest.skip("no examples/minimal.yaml")
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert "selection" not in data, (
        "examples/minimal.yaml declares a `selection:` block, but selection is "
        "applied post-hoc by scripts/select_stage2_shortlist.py, not by the pipeline"
    )


# ---------------------------------------------------------------------- hygiene
def test_no_stray_scratch_modules_at_top_level():
    """Importable scratch files at the top level make the package look untidy and
    can shadow real modules."""
    # run_all.py is the single-file campaign driver shipped with the paper:
    # `python run_all.py --list | --dry-run | --only <stage> | --from <step>`.
    allowed = {"run_all.py"}
    stray = [
        p.name for p in ROOT.glob("*.py")
        if p.name not in allowed and not p.name.startswith("_")
    ]
    assert not stray, f"unexpected top-level modules: {stray}"
