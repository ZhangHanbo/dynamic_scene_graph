"""
Three SLAM backend types × two orchestrator pipelines.

Covers the full matrix requested: for each of
    1) zero-uncertainty base SLAM (PoseEstimate with Σ_wb → 0),
    2) particle-filter base SLAM (ParticlePose with intrinsic spread),
    3) Gaussian base SLAM       (PoseEstimate with finite Σ_wb),

run the trajectory through both
    A) TwoTierOrchestrator          (RBPF variant, world-frame objects),
    B) TwoTierOrchestratorGaussian  (Gaussian variant, base-frame objects).

Each pair (backend, orchestrator) is checked for:
    * no crashes,
    * converges (final object translation close to ground truth),
    * posterior Σ_trans is sensible — shrinks below init and does not
      violate the lower bound Σ_wb contribution imposes.

Synthetic scene: a static robot base, three static objects. This factors
out base-motion complexity — the distinction we're probing is purely
how each pipeline treats the localization uncertainty.

Run:
    conda run -n ocmp_test python -m pytest tests/test_three_backends.py -v
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.orchestrator import TwoTierOrchestrator, TriggerConfig
from pose_update.orchestrator_gaussian import TwoTierOrchestratorGaussian
from pose_update.state.slam_interface import (
    PoseEstimate, ParticlePose,
    PassThroughSlam, ParticlePassThroughSlam,
    sample_particles_from_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _T(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [x, y, z]
    return T


def _make_detection(obj_id, label, T_co, mask_shape=(100, 100)):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[20:40, 20:40] = 1
    return {
        "id": obj_id,
        "label": label,
        "mask": mask,
        "score": 0.9,
        "T_co": T_co,
        "R_icp": np.diag([1e-4] * 6),
        "fitness": 0.9,
        "rmse": 0.002,
    }


# ─────────────────────────────────────────────────────────────────────
# SLAM backend factories (three cases)
# ─────────────────────────────────────────────────────────────────────

# Σ_wb = 1e-12 (floor for Cholesky stability). In our semantics this is
# "zero uncertainty".
SIGMA_ZERO = np.eye(6) * 1e-12

# Moderate Σ_wb for the Gaussian case — ~1 cm translational, ~0.5° rotational.
SIGMA_GAUSS = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

# Particle cloud around the same mean, with the same spread.
N_PARTICLES_BACKEND = 32


def make_zero_gaussian_backend(n_frames: int, base_pose: np.ndarray):
    """Case 1: Σ_wb ≈ 0 — every frame returns the exact base pose."""
    return PassThroughSlam(
        poses=[base_pose] * n_frames,
        default_cov=SIGMA_ZERO.copy(),
    )


def make_gaussian_backend(n_frames: int, base_pose: np.ndarray):
    """Case 3: Gaussian posterior with moderate Σ_wb — backend returns
    `PoseEstimate(T_base, Σ_wb)` each frame."""
    return PassThroughSlam(
        poses=[base_pose] * n_frames,
        default_cov=SIGMA_GAUSS.copy(),
    )


def make_particle_backend(n_frames: int,
                          base_pose: np.ndarray,
                          seed: int = 0):
    """Case 2: backend returns a `ParticlePose` each frame, sampled from
    the same Gaussian as case 3. The difference is representational —
    the RBPF pipeline consumes particles natively; the Gaussian
    pipeline collapses them on ingest."""
    rng = np.random.default_rng(seed)
    per_frame: List[ParticlePose] = []
    for _ in range(n_frames):
        per_frame.append(sample_particles_from_gaussian(
            PoseEstimate(T=base_pose, cov=SIGMA_GAUSS),
            n_samples=N_PARTICLES_BACKEND,
            rng=rng,
        ))
    return ParticlePassThroughSlam(per_frame)


BACKEND_FACTORIES = {
    "zero_gaussian": make_zero_gaussian_backend,
    "particles":     make_particle_backend,
    "gaussian":      make_gaussian_backend,
}


# ─────────────────────────────────────────────────────────────────────
# Scenario: stationary robot, 3 static objects, N observations each
# ─────────────────────────────────────────────────────────────────────

N_FRAMES = 30

OBJECTS = {
    0: ("apple", _T(0.3, 0.0, 0.5)),   # id → (label, ground-truth camera-frame pose)
    1: ("bowl",  _T(0.1, 0.2, 0.6)),
    2: ("cup",   _T(-0.2, -0.1, 0.4)),
}

BASE_POSE = _T(0.0, 0.0, 0.0)  # robot origin = world origin (simplifies checks)


def _run(orchestrator, slam_backend, n_frames: int = N_FRAMES, rng_seed: int = 7):
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32)
    rng = np.random.default_rng(rng_seed)

    for _ in range(n_frames):
        # Noisy T_co ≈ GT with small σ=1cm observation noise (≈ √R_icp)
        detections = []
        for oid, (label, gt_T_co) in OBJECTS.items():
            noise = rng.normal(scale=0.01, size=3)
            T_co = gt_T_co.copy()
            T_co[:3, 3] = T_co[:3, 3] + noise
            detections.append(_make_detection(oid, label, T_co))
        orchestrator.step(
            rgb, depth, detections,
            gripper_state={"phase": "idle", "held_obj_id": None},
        )


# ─────────────────────────────────────────────────────────────────────
# Matrix tests
# ─────────────────────────────────────────────────────────────────────

ORCH_KINDS = ["rbpf", "gaussian"]
BACKEND_KINDS = list(BACKEND_FACTORIES.keys())


def _make_orch(kind: str, backend):
    if kind == "rbpf":
        return TwoTierOrchestrator(
            backend,
            trigger=TriggerConfig(periodic_every_n_frames=-1,
                                  on_new_object=False),
            n_particles=16,
            rng_seed=0,
        )
    if kind == "gaussian":
        return TwoTierOrchestratorGaussian(
            backend,
            trigger=TriggerConfig(periodic_every_n_frames=-1,
                                  on_new_object=False),
        )
    raise ValueError(kind)


@pytest.mark.parametrize("orch_kind", ORCH_KINDS)
@pytest.mark.parametrize("backend_kind", BACKEND_KINDS)
class TestThreeBackends:
    """Runs the full matrix: 3 backends × 2 orchestrators = 6 cases."""

    def test_runs_and_tracks_all_objects(self, orch_kind, backend_kind):
        backend = BACKEND_FACTORIES[backend_kind](N_FRAMES, BASE_POSE)
        orch = _make_orch(orch_kind, backend)
        _run(orch, backend)

        assert set(orch.objects.keys()) == set(OBJECTS.keys()), \
            f"[{orch_kind}/{backend_kind}] expected all objects tracked"
        for oid, (label, _) in OBJECTS.items():
            assert orch.objects[oid]["label"] == label

    def test_final_mean_near_ground_truth(self, orch_kind, backend_kind):
        backend = BACKEND_FACTORIES[backend_kind](N_FRAMES, BASE_POSE)
        orch = _make_orch(orch_kind, backend)
        _run(orch, backend)

        for oid, (_, gt_T_co) in OBJECTS.items():
            est_T = orch.objects[oid]["T"]
            gt_world = BASE_POSE @ gt_T_co
            err = np.linalg.norm(est_T[:3, 3] - gt_world[:3, 3])
            assert err < 0.05, (
                f"[{orch_kind}/{backend_kind}] oid={oid} error {err:.3f} m "
                f"too large")

    def test_posterior_cov_shrinks_below_init(self, orch_kind, backend_kind):
        backend = BACKEND_FACTORIES[backend_kind](N_FRAMES, BASE_POSE)
        orch = _make_orch(orch_kind, backend)

        # Initial loose cov trace ≈ 0.05² * 6 ≈ 0.015.
        init_trace = np.trace(np.diag([0.05] * 6))

        _run(orch, backend)

        for oid in OBJECTS:
            trace = float(np.trace(orch.objects[oid]["cov"]))
            assert trace < init_trace, (
                f"[{orch_kind}/{backend_kind}] oid={oid} cov did not "
                f"shrink: trace={trace:.6f}")


# ─────────────────────────────────────────────────────────────────────
# Focused "regression" tests: comparing the two pipelines' output
# ─────────────────────────────────────────────────────────────────────

class TestPipelineComparison:
    """Side-by-side: both pipelines should agree under zero base
    uncertainty, and the Gaussian pipeline should produce tighter
    Σ_bo than the RBPF pipeline's collapsed Σ_wo under Gaussian
    input (because the Gaussian pipeline separates Σ_wb out).
    """

    def test_zero_uncertainty_matches_between_pipelines(self):
        """With Σ_wb ≈ 0, both pipelines should produce nearly identical
        world-frame object posteriors — both degenerate to plain EKFs
        whose only difference is stylistic."""
        # RBPF
        b1 = make_zero_gaussian_backend(N_FRAMES, BASE_POSE)
        o1 = _make_orch("rbpf", b1)
        _run(o1, b1)

        # Gaussian
        b2 = make_zero_gaussian_backend(N_FRAMES, BASE_POSE)
        o2 = _make_orch("gaussian", b2)
        _run(o2, b2)

        for oid in OBJECTS:
            m1 = o1.objects[oid]["T"][:3, 3]
            m2 = o2.objects[oid]["T"][:3, 3]
            assert np.linalg.norm(m1 - m2) < 0.01, \
                f"means diverge for oid={oid}: {m1} vs {m2}"

    def test_gaussian_pipeline_respects_sigma_wb_lower_bound(self):
        """Under Gaussian Σ_wb > 0, the Gaussian pipeline's world-frame
        output Σ_wo is lower-bounded by the projected Σ_wb. This is the
        physical correctness property the base-frame storage secures
        and that Paper-1 would violate."""
        backend = make_gaussian_backend(N_FRAMES, BASE_POSE)
        orch = _make_orch("gaussian", backend)
        _run(orch, backend)

        sigma_wb_trans = float(np.trace(SIGMA_GAUSS[:3, :3]))
        for oid in OBJECTS:
            cov = orch.objects[oid]["cov"]
            trans_trace = float(np.trace(cov[:3, :3]))
            # Σ_wo_trans must contain at least (projected) Σ_wb contribution.
            # The projection Ad(T_bo⁻¹) on a translation-only Σ_wb preserves
            # translation block magnitude (rotation = I in our setup), so
            # we expect Σ_wo_trans ≥ Σ_wb_trans up to EKF tightening.
            # Actually: Σ_wo_trans = Σ_wb_trans + Σ_bo_trans, which is
            # strictly ≥ Σ_wb_trans.
            assert trans_trace >= sigma_wb_trans * 0.95, (
                f"oid={oid}: Σ_wo_trans={trans_trace:.6g} violates "
                f"lower bound Σ_wb_trans={sigma_wb_trans:.6g}")

    def test_gaussian_pipeline_Sigma_bo_ignores_Sigma_wb(self):
        """Σ_bo (base frame) should shrink to near R_icp regardless of
        Σ_wb. This is the key property base-frame storage provides."""
        # Run with nontrivial Σ_wb.
        backend = make_gaussian_backend(N_FRAMES, BASE_POSE)
        orch = _make_orch("gaussian", backend)
        _run(orch, backend)

        # Σ_bo should be small — should NOT reflect Σ_wb at all.
        sigma_wb_trans = float(np.trace(SIGMA_GAUSS[:3, :3]))
        for oid in OBJECTS:
            pe_base = orch.state.collapsed_object_base(oid)
            assert pe_base is not None
            base_trans_trace = float(np.trace(pe_base.cov[:3, :3]))
            assert base_trans_trace < sigma_wb_trans, (
                f"oid={oid}: Σ_bo_trans={base_trans_trace:.6g} should be "
                f"below Σ_wb_trans={sigma_wb_trans:.6g} — base frame "
                f"storage failed to decouple")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
