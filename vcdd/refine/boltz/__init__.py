"""Boltz support for stage 2.

Stage 2 in 0.2.0 does not dock. It sends the stage-1 shortlist to Boltz,
which co-folds each ligand with the receptor sequence and returns its own
confidence metrics. Read :mod:`vcdd.refine.stage2` for what that does and does
not buy you.

Layout:

``client``
    Credentials, SDK construction, live/test mode detection. The API key is read
    from an environment variable, never from the config file, and never appears
    in a log line, a manifest or an exception message.
``target``
    Turns the prepared receptor PDBQT and the stage-1 docking box into the
    ``Target`` payload: per-chain sequences and 0-indexed pocket residues.
``screen``
    The ``library_screen`` (BoltzMol-1) backend: cost estimate, spend gate,
    submit, poll, paginate results, re-attach.
``sab``
    The ``structure_and_binding`` (boltz-2.1) backend.
``local``
    The ``local`` backend: boltz-2 run on this machine via ``boltz predict``
    subprocesses. No API key, no billing, no spend gate; pocket residues and
    reference ligands are not sent (the input YAML has no such fields).
``poseqc``
    Ligand-in-box check on the predicted complex (shared by all backends; the
    local backend injects a file-reading fetch).
"""

from vcdd.refine.boltz.client import (  # noqa: F401
    BoltzCredentials,
    BoltzUnavailable,
    boltz_available,
    make_client,
    resolve_credentials,
)

__all__ = [
    "BoltzCredentials",
    "BoltzUnavailable",
    "boltz_available",
    "make_client",
    "resolve_credentials",
]
