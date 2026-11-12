"""Stage-3 multi-objective scoring of predicted cellular responses.

The user's question -- "which compounds drive these genes or proteins up?" --
is not a single number, so the objective is decomposed into four scored
components plus one reported-only confidence flag:

``gene_score``
    Do the requested genes move in the requested direction?  A requested name
    that is not in the measured gene space but *is* a transcription factor in
    CollecTRI is scored from its inferred activity instead, which is how a
    "protein should be high" request gets answered at all.
``pathway_score``
    Do the requested pathways move?  PROGENy footprints for the 14 canonical
    signalling pathways, Hallmark sets by competitive set-mean z.
``target_consistency``
    Does the response resemble knocking the docking target out?  A compound
    that binds the target but produces an unrelated transcriptional response is
    either not engaging it in cells or is dominated by off-target activity.
``specificity``
    Is the response focused, or does the compound move everything and switch on
    the stress programme?  Penalises the generic cytotoxic signature that
    otherwise wins every transcriptional objective.
``applicability_domain``
    Reported, never scored.  A prediction is not more correct because the
    compound resembles the training set -- but you should trust it less when it
    does not.

Every component is mapped to [0, 1] and combined by a **weighted geometric
mean**, so weights act as exponents and a component at zero drags the product
down instead of being averaged away.  Set a weight to 0 to report a component
without letting it influence the ranking.

All transforms are absolute, never batch-relative: adding compounds to a run
must not change the score of compounds already in it, or stage-4 analogs could
not be compared with their parents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..config.schema import ObjectiveConfig
from . import assets, protein_activity
from .base import Signature

__all__ = [
    "ComponentScores",
    "ObjectiveContext",
    "squash",
    "score_signature",
    "score_signatures",
    "aggregate_backends",
]

#: Value at which a component's raw statistic maps to ~0.76 after squashing,
#: per signature value scale. Membership vectors live in [-1, 1]; z-scores do
#: not, so they need a wider scale.
SQUASH_SCALE = {"membership": 0.5, "zscore": 2.0, "logfc": 1.0}

#: Fallback |value| above which a gene counts as "moved", used by the
#: specificity term. Overridden per run by ``ObjectiveConfig.moved_threshold``.
MOVED_THRESHOLD = {"membership": 0.1, "zscore": 2.0, "logfc": 1.0}

#: Activity t-values are on their own scale, independent of the signature's.
ACTIVITY_SQUASH_SCALE = 3.0

COMPONENT_NAMES = ("gene_score", "pathway_score", "target_consistency", "specificity")


def squash(x: float | np.ndarray, scale: float) -> float | np.ndarray:
    """Map a signed statistic to [0, 1] with 0 -> 0.5, monotonically."""
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=float) / max(scale, 1e-9)))


@dataclass
class ComponentScores:
    """Per-compound objective breakdown."""

    mol_id: str
    backend: str
    gene_score: float = np.nan
    pathway_score: float = np.nan
    target_consistency: float = np.nan
    specificity: float = np.nan
    objective: float = np.nan
    # Diagnostics
    gene_detail: dict = field(default_factory=dict)
    pathway_detail: dict = field(default_factory=dict)
    max_tanimoto: float = np.nan
    out_of_domain: bool = False
    frac_genes_moved: float = np.nan
    stress_score: float = np.nan
    n_components_used: int = 0
    error: str | None = None

    def to_row(self) -> dict:
        return {
            "mol_id": self.mol_id,
            "backend": self.backend,
            "gene_score": self.gene_score,
            "pathway_score": self.pathway_score,
            "target_consistency": self.target_consistency,
            "specificity": self.specificity,
            "objective": self.objective,
            "max_tanimoto": self.max_tanimoto,
            "out_of_domain": self.out_of_domain,
            "frac_genes_moved": self.frac_genes_moved,
            "stress_score": self.stress_score,
            "n_components_used": self.n_components_used,
            "error": self.error,
        }


@dataclass
class ObjectiveContext:
    """Reference data resolved once and reused across every compound."""

    cfg: ObjectiveConfig
    genes: list[str]
    target_reference: np.ndarray | None = None
    target_gene: str | None = None
    activity_method: str = "ulm"
    #: Requested names that are TFs rather than measured genes.
    tf_requests: dict[str, int] = field(default_factory=dict)  # name -> +1/-1
    hallmark_sets: dict[str, set[str]] = field(default_factory=dict)
    progeny_pathways: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        cfg: ObjectiveConfig,
        genes: Sequence[str],
        target_gene: str | None = None,
        activity_method: str = "ulm",
    ) -> "ObjectiveContext":
        genes = list(genes)
        gene_set = set(genes)
        warnings: list[str] = []

        # Requested names that are not measured: try to score them as TFs.
        tf_requests: dict[str, int] = {}
        collectri_tfs: set[str] = set()
        try:
            collectri_tfs = set(assets.load_collectri()["tf"].unique())
        except assets.ResourceMissing:
            warnings.append("CollecTRI unavailable: protein-level requests cannot be scored")
        for name, sign in [(g, 1) for g in cfg.genes_up] + [(g, -1) for g in cfg.genes_down]:
            if name in gene_set:
                continue
            if name in collectri_tfs:
                tf_requests[name] = sign
            else:
                warnings.append(
                    f"requested gene '{name}' is neither in the measured gene space "
                    "nor a CollecTRI transcription factor; it will be ignored"
                )

        # Pathway requests split between PROGENy and Hallmark.
        progeny_pathways: set[str] = set()
        try:
            progeny_pathways = set(assets.load_progeny()["pathway"].unique())
        except assets.ResourceMissing:
            warnings.append("PROGENy unavailable: pathway scoring limited to Hallmark")
        hallmark_sets: dict[str, set[str]] = {}
        requested_paths = list(cfg.pathways_up) + list(cfg.pathways_down)
        if requested_paths:
            try:
                hm = assets.load_hallmark()
                by_set = hm.groupby("gene_set")["gene_symbol"].apply(set).to_dict()
            except assets.ResourceMissing:
                by_set = {}
            lookup = {k.lower(): k for k in by_set}
            for p in requested_paths:
                if p in progeny_pathways:
                    continue
                key = lookup.get(p.lower())
                if key is not None:
                    hallmark_sets[p] = by_set[key]
                else:
                    warnings.append(
                        f"pathway '{p}' matched neither a PROGENy pathway "
                        f"({sorted(progeny_pathways)}) nor a Hallmark set"
                    )

        target_reference = None
        if target_gene:
            try:
                from .backends.tier1 import reference_knockdown_signature

                target_reference = reference_knockdown_signature(target_gene, genes)
                if target_reference is None:
                    warnings.append(
                        f"no consensus knockout signature for target '{target_gene}': "
                        "target_consistency will be skipped"
                    )
            except assets.ResourceMissing as exc:
                warnings.append(f"target_consistency unavailable: {exc}")

        return cls(
            cfg=cfg,
            genes=genes,
            target_reference=target_reference,
            target_gene=target_gene,
            activity_method=activity_method,
            tf_requests=tf_requests,
            hallmark_sets=hallmark_sets,
            progeny_pathways=progeny_pathways,
            warnings=warnings,
        )


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #


def _gene_component(
    sig: Signature,
    ctx: ObjectiveContext,
    activity: pd.Series | None,
) -> tuple[float, dict]:
    cfg = ctx.cfg
    scale = SQUASH_SCALE.get(sig.value_scale, 1.0)
    index = {g: i for i, g in enumerate(sig.genes)}
    per_gene: dict[str, dict] = {}
    scores: list[float] = []
    weights: list[float] = []

    requests = [(g, 1.0) for g in cfg.genes_up] + [(g, -1.0) for g in cfg.genes_down]
    for name, desired in requests:
        w = float(cfg.gene_weights.get(name, 1.0))
        if w <= 0:
            continue
        i = index.get(name)
        if i is not None:
            raw = float(sig.values[i])
            s = float(squash(desired * raw, scale))
            per_gene[name] = {"route": "measured", "value": raw, "score": s, "desired": desired}
        elif name in ctx.tf_requests and activity is not None and name in activity.index:
            raw = float(activity[name])
            s = float(squash(desired * raw, ACTIVITY_SQUASH_SCALE))
            per_gene[name] = {
                "route": "inferred_tf_activity",
                "value": raw,
                "score": s,
                "desired": desired,
            }
        else:
            continue
        scores.append(s)
        weights.append(w)

    if not scores:
        return np.nan, {"n_scored": 0, "per_gene": per_gene}
    arr, wts = np.asarray(scores), np.asarray(weights)
    if cfg.gene_mode == "all":
        # Strict: the objective is only as good as the worst requested gene.
        value = float(arr.min())
    else:
        value = float((arr * wts).sum() / wts.sum())
    return value, {"n_scored": len(scores), "per_gene": per_gene, "mode": cfg.gene_mode}


def _pathway_component(
    sig: Signature,
    ctx: ObjectiveContext,
    pathway_act: pd.Series | None,
) -> tuple[float, dict]:
    cfg = ctx.cfg
    requests = [(p, 1.0) for p in cfg.pathways_up] + [(p, -1.0) for p in cfg.pathways_down]
    if not requests:
        return np.nan, {"n_scored": 0}

    detail: dict[str, dict] = {}
    scores: list[float] = []
    vals = sig.values
    mu = float(vals.mean()) if vals.size else 0.0
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 1.0
    sd = sd if sd > 1e-9 else 1.0
    index = {g: i for i, g in enumerate(sig.genes)}

    for name, desired in requests:
        if pathway_act is not None and name in pathway_act.index:
            raw = float(pathway_act[name])
            s = float(squash(desired * raw, ACTIVITY_SQUASH_SCALE))
            detail[name] = {"route": "progeny", "value": raw, "score": s}
        elif name in ctx.hallmark_sets:
            members = [index[g] for g in ctx.hallmark_sets[name] if g in index]
            if len(members) < 5:
                detail[name] = {"route": "hallmark", "error": "fewer than 5 measured members"}
                continue
            sub = vals[members]
            # Competitive set-mean z: how far the set mean sits from the
            # transcriptome-wide mean, in standard errors.
            raw = float((sub.mean() - mu) / (sd / np.sqrt(len(members))))
            s = float(squash(desired * raw, ACTIVITY_SQUASH_SCALE))
            detail[name] = {"route": "hallmark", "value": raw, "score": s, "n": len(members)}
        else:
            detail[name] = {"route": "unmatched"}
            continue
        scores.append(s)

    if not scores:
        return np.nan, {"n_scored": 0, "per_pathway": detail}
    return float(np.mean(scores)), {"n_scored": len(scores), "per_pathway": detail}


def _target_component(sig: Signature, ctx: ObjectiveContext) -> float:
    """Cosine similarity to the target-knockout reference, mapped to [0, 1]."""
    if ctx.target_reference is None or sig.values.size == 0:
        return np.nan
    ref = ctx.target_reference
    if ref.shape[0] != sig.values.shape[0]:
        return np.nan
    mask = ref != 0.0
    if mask.sum() < 5:
        return np.nan
    a, b = sig.values[mask].astype(float), ref[mask].astype(float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float((float(a @ b / (na * nb)) + 1.0) / 2.0)


def _specificity_component(sig: Signature, ctx: ObjectiveContext) -> tuple[float, float, float]:
    """Return ``(specificity, frac_moved, stress)``.

    Two absolute sub-terms combined by geometric mean: how much of the
    transcriptome moved, and how strongly the generic stress programme fired.
    """
    if sig.values.size == 0:
        return np.nan, np.nan, np.nan
    tau = float(
        ctx.cfg.moved_threshold.get(
            sig.value_scale, MOVED_THRESHOLD.get(sig.value_scale, 1.0)
        )
    )
    frac_moved = float((np.abs(sig.values) >= tau).mean())
    breadth_term = float(1.0 - frac_moved)

    index = {g: i for i, g in enumerate(sig.genes)}
    idx = [index[g] for g in ctx.cfg.stress_genes if g in index]
    if idx:
        stress = float(sig.values[idx].mean())
        # Only induction is penalised; suppressing stress genes is not a crime.
        # squash(x>=0) is in [0.5, 1); rescale that half-range onto [1, 0].
        excess = float(squash(max(stress, 0.0), SQUASH_SCALE.get(sig.value_scale, 1.0))) - 0.5
        stress_term = float(np.clip(1.0 - 2.0 * excess, 0.0, 1.0))
    else:
        stress, stress_term = np.nan, 1.0
    spec = float(np.sqrt(max(breadth_term, 0.0) * max(stress_term, 0.0)))
    return spec, frac_moved, stress


def _aggregate(components: dict[str, float], cfg: ObjectiveConfig) -> tuple[float, int]:
    """Weighted geometric mean over the components with positive weight."""
    logs, wts = [], []
    for name in COMPONENT_NAMES:
        w = float(cfg.weights.get(name, 0.0))
        v = components.get(name, np.nan)
        if w <= 0 or v is None or not np.isfinite(v):
            continue
        logs.append(np.log(max(float(v), max(cfg.component_floor, 1e-12))))
        wts.append(w)
    if not logs:
        return np.nan, 0
    return float(np.exp(np.average(logs, weights=wts))), len(logs)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def score_signature(
    sig: Signature,
    ctx: ObjectiveContext,
    tf_act: pd.Series | None = None,
    pathway_act: pd.Series | None = None,
) -> ComponentScores:
    """Score one predicted signature against the objective."""
    out = ComponentScores(mol_id=sig.mol_id, backend=sig.backend)
    if not sig.ok:
        out.error = sig.error or "empty signature"
        return out

    gene_val, gene_detail = _gene_component(sig, ctx, tf_act)
    path_val, path_detail = _pathway_component(sig, ctx, pathway_act)
    target_val = _target_component(sig, ctx)
    spec, frac_moved, stress = _specificity_component(sig, ctx)

    out.gene_score = gene_val
    out.pathway_score = path_val
    out.target_consistency = target_val
    out.specificity = spec
    out.gene_detail = gene_detail
    out.pathway_detail = path_detail
    out.frac_genes_moved = frac_moved
    out.stress_score = stress
    out.max_tanimoto = float(sig.meta.get("max_tanimoto", np.nan))
    if np.isfinite(out.max_tanimoto):
        out.out_of_domain = bool(out.max_tanimoto < ctx.cfg.ad_tanimoto_warn)
    else:
        out.out_of_domain = bool(sig.meta.get("out_of_domain", False))

    components = {
        "gene_score": gene_val,
        "pathway_score": path_val,
        "target_consistency": target_val,
        "specificity": spec,
    }
    out.objective, out.n_components_used = _aggregate(components, ctx.cfg)
    return out


def score_signatures(
    signatures: Sequence[Signature],
    ctx: ObjectiveContext,
    compute_activity: bool = True,
) -> tuple[list[ComponentScores], dict]:
    """Score a batch, computing TF/pathway activity once for the whole batch."""
    usable = [s for s in signatures if s.ok]
    tf_frame: pd.DataFrame | None = None
    path_frame: pd.DataFrame | None = None
    diagnostics: dict = {"n_signatures": len(signatures), "n_usable": len(usable)}

    need_tf = compute_activity and bool(ctx.tf_requests)
    need_path = compute_activity and bool(
        set(ctx.cfg.pathways_up) | set(ctx.cfg.pathways_down)
    ) and bool(ctx.progeny_pathways)

    if usable and (need_tf or need_path):
        genes = usable[0].genes
        same = all(s.genes == genes for s in usable)
        if not same:
            diagnostics["activity_skipped"] = "backends emit different gene spaces"
        else:
            mat = np.column_stack([s.values for s in usable])
            ids = [s.mol_id for s in usable]
            if need_tf:
                try:
                    res = protein_activity.tf_activity(
                        mat, genes, ids, method=ctx.activity_method
                    )
                    tf_frame = res.scores
                    diagnostics["tf_activity"] = {
                        "n_regulators": res.n_regulators,
                        "coverage": round(res.coverage, 3),
                        "method": res.method,
                    }
                except assets.ResourceMissing as exc:
                    diagnostics["tf_activity_error"] = str(exc)
            if need_path:
                try:
                    res = protein_activity.pathway_activity(
                        mat, genes, ids, method=ctx.activity_method
                    )
                    path_frame = res.scores
                    diagnostics["pathway_activity"] = {
                        "n_regulators": res.n_regulators,
                        "coverage": round(res.coverage, 3),
                        "method": res.method,
                    }
                except assets.ResourceMissing as exc:
                    diagnostics["pathway_activity_error"] = str(exc)

    scored: list[ComponentScores] = []
    for sig in signatures:
        tf_col = (
            tf_frame[sig.mol_id]
            if tf_frame is not None and sig.mol_id in tf_frame.columns
            else None
        )
        path_col = (
            path_frame[sig.mol_id]
            if path_frame is not None and sig.mol_id in path_frame.columns
            else None
        )
        scored.append(score_signature(sig, ctx, tf_col, path_col))
    return scored, diagnostics


def aggregate_backends(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-backend objective rows to one row per compound.

    Backends are averaged rather than voted: with two or three predictors there
    is no majority to take.  ``backend_spread`` (the standard deviation of the
    aggregate) is the honest disagreement signal -- a high value means the
    backends do not agree about the compound and the ranking is soft there.
    """
    if rows.empty:
        return rows.assign(objective_mean=[], backend_spread=[], n_backends=[])
    numeric = ["gene_score", "pathway_score", "target_consistency", "specificity", "objective"]
    present = [c for c in numeric if c in rows.columns]
    grouped = rows.groupby("mol_id", as_index=False).agg(
        **{c: (c, "mean") for c in present},
        backend_spread=("objective", "std"),
        n_backends=("backend", "nunique"),
        max_tanimoto=("max_tanimoto", "max"),
        out_of_domain=("out_of_domain", "any"),
        frac_genes_moved=("frac_genes_moved", "mean"),
        stress_score=("stress_score", "mean"),
    )
    grouped["backend_spread"] = grouped["backend_spread"].fillna(0.0)
    return grouped
