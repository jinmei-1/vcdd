"""Tier-3 cell backends: interface + documented skip.

These models are published and interesting, but each one needs something this
pipeline cannot assume: a checkpoint behind an acceptable-use agreement, a
single-cell reference AnnData, or a training panel the query compound must
belong to.  Rather than pretend, each class declares its contract, states the
precise blocker, and skips.

Filling one in is deliberately small work: implement ``_extra_availability``
and ``predict``, and the rest of the pipeline picks it up unchanged.

``prnet``
    PRnet, a perturbation-conditioned generative model for out-of-distribution
    drug response [1].
``perturbnet``
    PerturbNet, which couples a chemical latent space to a cell-state
    generative model [12].
``state_tahoe``
    Arc Institute STATE (``ST-Tahoe``), a transformer trained on the Tahoe-100M
    compound-perturbation atlas [20-25].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ..base import (
    CellModel,
    CellModelCapabilities,
    PerturbationRequest,
    Signature,
    register_backend,
)

__all__ = ["PRnetModel", "PerturbNetModel", "StateTahoeModel"]


class _StubBackend(CellModel):
    """Shared skip behaviour for backends without a runnable local install."""

    ENV_VAR = ""

    def _root(self) -> Path | None:
        val = os.environ.get(self.ENV_VAR) if self.ENV_VAR else None
        return Path(val).expanduser() if val and Path(val).expanduser().exists() else None

    def _extra_availability(self) -> tuple[bool, str]:
        if self._root() is None:
            return False, (
                f"set {self.ENV_VAR} to a prepared model directory; "
                + self.capabilities().install_hint
            )
        return False, (
            f"{self.capabilities().name}: adapter is an interface stub in this "
            "release -- implement predict() against your local checkpoint"
        )

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        reason = (
            f"{self.capabilities().name} is not implemented in this release; "
            f"{self.capabilities().install_hint}"
        )
        return [Signature.failed(r, self.capabilities().name, reason) for r in requests]


@register_backend
class PRnetModel(_StubBackend):
    """PRnet: perturbation-conditioned generative model for drug response."""

    ENV_VAR = "VCDD_PRNET_DIR"

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="prnet",
            tier=3,
            description=(
                "PRnet perturbation-conditioned generative model; predicts "
                "transcriptional response to unseen compounds."
            ),
            accepts_novel_smiles=True,
            gene_space="checkpoint",
            value_scale="logfc",
            python_requires=("torch",),
            env_requires=("VCDD_PRNET_DIR",),
            install_hint=(
                "git clone https://github.com/Perturbation-Response-Prediction/PRnet "
                "and place the trained checkpoint plus its reference AnnData in "
                "VCDD_PRNET_DIR"
            ),
            citation="Qi et al., Nature Communications 2024 (PRnet)",
            cell_line_aware=True,
            dose_aware=True,
        )


@register_backend
class PerturbNetModel(_StubBackend):
    """PerturbNet: chemical latent space coupled to a cell-state generator."""

    ENV_VAR = "VCDD_PERTURBNET_DIR"

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="perturbnet",
            tier=3,
            description=(
                "PerturbNet conditional generative model linking a chemical "
                "latent space to single-cell state."
            ),
            accepts_novel_smiles=True,
            gene_space="checkpoint",
            value_scale="logfc",
            python_requires=("torch",),
            env_requires=("VCDD_PERTURBNET_DIR",),
            install_hint=(
                "git clone https://github.com/welch-lab/PerturbNet; a trained "
                "ChemicalVAE + cell-state model and its reference AnnData are required"
            ),
            citation="Yu & Welch (PerturbNet)",
            cell_line_aware=True,
            dose_aware=True,
        )


@register_backend
class StateTahoeModel(_StubBackend):
    """Arc Institute STATE, ``ST-Tahoe`` checkpoint.

    Two blockers, both intentional:

    1. The checkpoints are distributed under an acceptable-use agreement, so
       they cannot be fetched unattended.
    2. STATE conditions on *perturbation identity* drawn from its training
       panel. Scoring a molecule outside the Tahoe-100M drug set needs an
       embedding-mapping step (nearest training drug in chemical space, or a
       learned projection) that is not part of the published model. Stage 4
       designs novel analogs by construction, so this backend is unsuitable
       there unless that mapping is built and validated first.
    """

    ENV_VAR = "VCDD_STATE_DIR"

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="state_tahoe",
            tier=3,
            description=(
                "Arc Institute STATE (ST-Tahoe) transformer trained on the "
                "Tahoe-100M compound-perturbation atlas."
            ),
            accepts_novel_smiles=False,
            gene_space="checkpoint",
            value_scale="logfc",
            python_requires=("torch",),
            env_requires=("VCDD_STATE_DIR",),
            install_hint=(
                "pip install arc-state; accept the AUP and download "
                "arcinstitute/ST-Tahoe from Hugging Face, then set VCDD_STATE_DIR. "
                "Run inference with `state tx infer`."
            ),
            citation="Arc Institute STATE / Tahoe-100M",
            cell_line_aware=True,
            dose_aware=True,
        )
