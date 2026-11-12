"""Deterministic mock docking engine.

Used by the unit and contract tests, and by ``--engine mock`` when someone wants
to exercise the whole cascade without a docking binary. The "score" is a
deterministic function of the ligand's heavy-atom count and a hash of its
coordinates, shaped to look like a Vina affinity distribution (roughly -4 to -11
kcal/mol). It is **not** a prediction of anything and is labelled as such in the
capabilities notes so it can never be mistaken for a real result.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

from vcdd.docking.base import (
    DockParams,
    DockResult,
    DockingEngine,
    EngineCapabilities,
    register_engine,
)
from vcdd.receptor.box import Box


class MockEngine(DockingEngine):
    name = "mock"

    def available(self) -> bool:
        """Only under pytest, or with an explicit opt-in.

        0.2.0 makes stage 1 GPU-only. If ``mock`` reported itself available
        everywhere, ``engine_selection: auto`` on a GPU-less machine would
        silently fall back to fake scores rather than failing loudly - which is
        the worst possible outcome for a screening campaign.
        """
        if os.environ.get("VCDD_ALLOW_MOCK") == "1":
            return True
        return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="mock",
            supports_scoring_functions=("mock",),
            supports_local_only=True,
            speed_class="fast",
            relative_speed=1000.0,
            notes="NOT A DOCKING PROGRAM. Deterministic pseudo-scores for testing only.",
        )

    def dock(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt_path: str,
        box: Box,
        params: DockParams,
        mol_id: str = "",
    ) -> DockResult:
        t0 = time.time()
        atom_lines = [l for l in ligand_pdbqt.splitlines() if l.startswith(("ATOM", "HETATM"))]
        n_atoms = len(atom_lines)
        if n_atoms == 0:
            return DockResult(mol_id=mol_id, engine=self.name, error="empty_ligand")

        digest = hashlib.sha256((mol_id + ligand_pdbqt[:2000]).encode()).digest()
        jitter = int.from_bytes(digest[:4], "big") / 2**32  # [0,1)

        # Loose size-dependence mimics the well-known Vina heavy-atom bias.
        base = -3.5 - 0.11 * min(n_atoms, 45) - 2.5 * jitter
        scores = [round(base + 0.35 * i, 2) for i in range(max(1, params.num_modes))]

        pose_lines = []
        for line in atom_lines[:n_atoms]:
            pose_lines.append(line)
        pose = "REMARK VINA RESULT:  {:.2f}      0.000      0.000\n".format(scores[0]) + "\n".join(pose_lines)
        all_poses = "\n".join(f"MODEL {i+1}\n{pose}\nENDMDL" for i in range(len(scores)))

        return DockResult(
            mol_id=mol_id,
            score=scores[0],
            all_scores=scores,
            pose_pdbqt=pose,
            all_poses_pdbqt=all_poses,
            engine=self.name,
            seconds=time.time() - t0,
            extra={"mock": True, "n_atoms": n_atoms},
        )

    def score_only(self, ligand_pdbqt, receptor_pdbqt_path, box, params, mol_id=""):  # noqa: D102
        result = self.dock(ligand_pdbqt, receptor_pdbqt_path, box, params, mol_id)
        result.extra["local_only"] = True
        return result

    def version(self) -> str:
        return "mock-1.0"


register_engine("mock", MockEngine)
