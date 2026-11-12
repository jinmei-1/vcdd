"""Tier-1 cell backends: usable immediately from the bundled reference data.

``mock``
    Deterministic pseudo-signature. Exists so the pipeline, tests and the smoke
    run never depend on network access or third-party weights.
``lincs_readacross``
    Chemical-similarity read-across over LINCS L1000 consensus signatures.
``target_knockdown``
    The consensus CRISPR-knockout signature of the docking target, used as the
    mechanistic reference for the ``target_consistency`` objective component.

Read-across is an honest baseline, not a virtual cell: it assumes similar
structures produce similar transcriptional responses.  That assumption fails
exactly where medicinal chemistry is most interesting (activity cliffs), which
is why every prediction carries its maximum-Tanimoto applicability-domain flag.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .. import assets
from ..base import (
    CellModel,
    CellModelCapabilities,
    PerturbationRequest,
    Signature,
    register_backend,
)

__all__ = ["MockCellModel", "LincsReadAcrossModel", "TargetKnockdownModel", "reference_knockdown_signature"]


# --------------------------------------------------------------------------- #
# Fingerprint helpers
# --------------------------------------------------------------------------- #


def _fp_generator(radius: int = 2, n_bits: int = 2048):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _fingerprints(smiles_list: Sequence[str]):
    """Morgan fingerprints; ``None`` for structures RDKit cannot parse."""
    from rdkit import Chem

    gen = _fp_generator()
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        out.append(gen.GetFingerprint(mol) if mol is not None else None)
    return out


# --------------------------------------------------------------------------- #
# mock
# --------------------------------------------------------------------------- #


@register_backend
class MockCellModel(CellModel):
    """Deterministic synthetic signatures derived from the SMILES hash.

    Values are reproducible for a given structure and gene space, so tests can
    assert exact numbers, but they carry no biological meaning whatsoever.
    """

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="mock",
            tier=1,
            description="Deterministic synthetic signature for testing (no biology).",
            accepts_novel_smiles=True,
            gene_space="config",
            value_scale="zscore",
            resources_requires=("l1000_genes",),
            citation="",
        )

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        from ...runtime.hashing import hash_text

        genes = self.genes()
        out: list[Signature] = []
        for req in requests:
            seed = int(hash_text(f"{req.smiles}|{req.cell_line}")[:8], 16)
            rng = np.random.default_rng(seed)
            vals = rng.normal(0.0, 1.0, size=len(genes)).astype(np.float32)
            # Dose enters as a saturating multiplier so dose sweeps are ordered.
            vals *= float(np.log10(max(req.dose_um, 1e-3)) + 2.0) / 3.0
            out.append(
                Signature(
                    mol_id=req.mol_id,
                    smiles=req.smiles,
                    genes=genes,
                    values=vals,
                    backend="mock",
                    cell_line=req.cell_line,
                    dose_um=req.dose_um,
                    time_h=req.time_h,
                    meta={"synthetic": True},
                )
            )
        return out


# --------------------------------------------------------------------------- #
# lincs_readacross
# --------------------------------------------------------------------------- #


@register_backend
class LincsReadAcrossModel(CellModel):
    """Tanimoto-weighted k-nearest-neighbour read-across over LINCS L1000.

    For a query structure the k most similar reference compounds contribute
    their consensus up/down membership vectors, weighted by ``similarity ** p``.
    The result is a signed consensus in [-1, 1]: +1 means every close neighbour
    up-regulates the gene.

    The bundled consensus signatures are aggregated across the LINCS cell-line
    panel, so this backend is *not* cell-line aware.  Requesting MCF7 does not
    make the prediction MCF7-specific; the ``cell_line`` field is carried
    through for provenance only.
    """

    K_NEIGHBOURS = 10
    SIM_FLOOR = 0.2
    POWER = 3.0

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="lincs_readacross",
            tier=1,
            description=(
                "Tanimoto-weighted kNN read-across over LINCS L1000 consensus "
                "compound signatures (cell-line agnostic)."
            ),
            accepts_novel_smiles=True,
            gene_space="config",
            value_scale="membership",
            python_requires=("rdkit",),
            resources_requires=("lincs_chem_consensus", "l1000_genes"),
            install_hint=assets.BUILD_HINT,
            citation="Subramanian et al., Cell 2017 (LINCS L1000); Enrichr consensus signatures",
            cell_line_aware=False,
            dose_aware=False,
        )

    def _extra_availability(self) -> tuple[bool, str]:
        try:
            ref = self._reference()
        except assets.ResourceMissing as exc:
            return False, str(exc)
        if not ref[1]:
            return False, "no reference compounds with parsable structures"
        return True, f"{len(ref[1])} reference compounds with structures"

    @lru_cache(maxsize=1)  # noqa: B019 - one instance per run, bounded by design
    def _reference(self):
        """(dataframe, fingerprints) for reference compounds having SMILES."""
        df = assets.load_lincs_chem()
        df = df[df["smiles"].notna() & (df["smiles"].astype(str).str.len() > 0)]
        df = df.reset_index(drop=True)
        fps = _fingerprints(df["smiles"].tolist())
        keep = [i for i, fp in enumerate(fps) if fp is not None]
        return df.iloc[keep].reset_index(drop=True), [fps[i] for i in keep]

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        from rdkit import DataStructs

        ref_df, ref_fps = self._reference()
        genes = self.genes()
        gene_index = {g: i for i, g in enumerate(genes)}

        # Pre-project every reference signature once.
        ref_vecs = np.zeros((len(ref_df), len(genes)), dtype=np.float32)
        for i, (up, down) in enumerate(zip(ref_df["up_genes"], ref_df["down_genes"])):
            for g in up:
                j = gene_index.get(g)
                if j is not None:
                    ref_vecs[i, j] = 1.0
            for g in down:
                j = gene_index.get(g)
                if j is not None:
                    ref_vecs[i, j] = -1.0

        query_fps = _fingerprints([r.smiles for r in requests])
        names = ref_df["name"].tolist()
        out: list[Signature] = []
        for req, fp in zip(requests, query_fps):
            if fp is None:
                out.append(Signature.failed(req, self.name, "unparsable SMILES"))
                continue
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=np.float32)
            order = np.argsort(sims)[::-1]
            # An exact structural match is the compound itself; keep it, but
            # record it so the caller can see the prediction is a lookup.
            picked = [i for i in order[: self.K_NEIGHBOURS] if sims[i] >= self.SIM_FLOOR]
            max_sim = float(sims[order[0]]) if sims.size else 0.0
            if not picked:
                out.append(
                    Signature(
                        mol_id=req.mol_id,
                        smiles=req.smiles,
                        genes=genes,
                        values=np.zeros(len(genes), dtype=np.float32),
                        backend=self.name,
                        cell_line=req.cell_line,
                        dose_um=req.dose_um,
                        time_h=req.time_h,
                        value_scale="membership",
                        meta={
                            "max_tanimoto": max_sim,
                            "n_neighbors": 0,
                            "out_of_domain": True,
                            "note": "no reference compound above the similarity floor",
                        },
                    )
                )
                continue
            w = sims[picked] ** self.POWER
            vals = (w[:, None] * ref_vecs[picked]).sum(axis=0) / w.sum()
            out.append(
                Signature(
                    mol_id=req.mol_id,
                    smiles=req.smiles,
                    genes=genes,
                    values=vals.astype(np.float32),
                    backend=self.name,
                    cell_line=req.cell_line,
                    dose_um=req.dose_um,
                    time_h=req.time_h,
                    value_scale="membership",
                    meta={
                        "max_tanimoto": max_sim,
                        "n_neighbors": len(picked),
                        "neighbors": [names[i] for i in picked[:5]],
                        "neighbor_sims": [round(float(sims[i]), 3) for i in picked[:5]],
                        "exact_match": max_sim >= 0.999,
                    },
                )
            )
        return out


# --------------------------------------------------------------------------- #
# target_knockdown
# --------------------------------------------------------------------------- #


def reference_knockdown_signature(gene_symbol: str, genes: Sequence[str]) -> np.ndarray | None:
    """Consensus CRISPR-KO signature of ``gene_symbol`` projected onto ``genes``."""
    df = assets.load_lincs_crispr()
    hit = df[df["gene"] == gene_symbol.upper()]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return assets.signature_from_updown(row["up_genes"], row["down_genes"], list(genes))


@register_backend
class TargetKnockdownModel(CellModel):
    """The consensus CRISPR-knockout signature of the configured target gene.

    This is a *mechanistic reference*, not a compound predictor: it returns the
    same signature for every compound, namely the transcriptional state of a
    cell in which the target has been knocked out.  Its purpose is to define
    "what on-target inhibition should look like" for the
    ``target_consistency`` component.  Knockout is a genetic proxy for complete
    pharmacological inhibition and will diverge from a partial or biased
    inhibitor, and from any agonist.
    """

    @classmethod
    def capabilities(cls) -> CellModelCapabilities:
        return CellModelCapabilities(
            name="target_knockdown",
            tier=1,
            description=(
                "Consensus CRISPR-KO signature of the target gene; mechanistic "
                "reference for target-consistency scoring."
            ),
            accepts_novel_smiles=False,
            gene_space="config",
            value_scale="membership",
            resources_requires=("lincs_crispr_ko_consensus", "l1000_genes"),
            install_hint=assets.BUILD_HINT,
            citation="LINCS L1000 CRISPR knockout consensus signatures (Enrichr)",
            reference_only=True,
        )

    def _extra_availability(self) -> tuple[bool, str]:
        gene = (self.cfg.target_gene or "").strip()
        if not gene:
            return False, "cell.target_gene is not set"
        df = assets.load_lincs_crispr()
        if gene.upper() not in set(df["gene"]):
            return False, f"no consensus knockout signature for '{gene}'"
        return True, f"knockout reference available for {gene.upper()}"

    def predict(self, requests: Sequence[PerturbationRequest]) -> list[Signature]:
        genes = self.genes()
        vec = reference_knockdown_signature(self.cfg.target_gene or "", genes)
        if vec is None:
            return [
                Signature.failed(r, self.name, f"no KO signature for {self.cfg.target_gene}")
                for r in requests
            ]
        return [
            Signature(
                mol_id=r.mol_id,
                smiles=r.smiles,
                genes=genes,
                values=vec.copy(),
                backend=self.name,
                cell_line=r.cell_line,
                dose_um=r.dose_um,
                time_h=r.time_h,
                value_scale="membership",
                meta={
                    "target_gene": (self.cfg.target_gene or "").upper(),
                    "compound_independent": True,
                },
            )
            for r in requests
        ]
