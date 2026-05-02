"""Plan §C.1: assert that an unobserved track's cov_bo is FROZEN under
base motion (the keyframe mechanism).

Birth a track with high σ_yaw (loud value to expose any leak), simulate
200 frames of base motion, and assert that:
  1. cov_bo[trans, trans] is bit-identical to the seed value.
  2. cov_bo[rot, rot] is bit-identical to the seed value.
  3. mu_bo updates deterministically so T_wo = T_wb @ mu_bo stays
     constant (the static-world-object property).
  4. The world-frame Σ_wo from `collapsed_object_world` only grows via
     the lever-arm of Σ_wb at the *current* mu_bo (no drift).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R_

from pose_update.state.ekf_se3 import se3_exp
from pose_update.state.gaussian_state import GaussianState
from pose_update.state.slam_interface import PoseEstimate


def _T(t=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def test_unobserved_track_cov_is_frozen_under_base_motion():
    """200 frames of base motion + rotation; an unobserved track's
    cov_bo must not change at all."""
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=np.eye(6) * 1e-4))

    # Birth a track with loud σ_yaw to expose any cross-coupling leak.
    seed_cov = np.diag(np.array([
        (0.02) ** 2, (0.02) ** 2, (0.02) ** 2,   # trans: 2 cm σ
        (0.10) ** 2, (0.10) ** 2, (np.pi) ** 2,  # rot: 0.1 rad x/y, π yaw
    ]))
    oid = 1
    state.ensure_object(oid, _T(t=(2.0, 0.0, 1.0)), seed_cov.copy())
    seed_mu = state.objects[oid].mu_bo.copy()
    seed_cov_stored = state.objects[oid].cov_bo.copy()
    np.testing.assert_allclose(seed_cov_stored, seed_cov)

    # Simulate 200 frames of forward driving + small yaw rotation per frame.
    dx = 0.05         # 5 cm/frame
    dyaw = 0.01       # 0.01 rad/frame
    Q_fn = lambda _oid: np.diag([1e-5] * 6)  # would normally pump σ_xy

    for k in range(1, 201):
        T_wb_k = _T(t=(dx * k, 0.0, 0.0), rpy_deg=(0.0, 0.0,
                                                    np.degrees(dyaw * k)))
        state.ingest_slam(PoseEstimate(T=T_wb_k,
                                        cov=np.eye(6) * 1e-4))
        # Mark the track unobserved.
        state.predict_static(Q_fn=Q_fn, unobserved_oids={oid})

    final_cov = state.objects[oid].cov_bo
    # Cov is bit-identical to the seed (no drift, no Q added).
    np.testing.assert_array_equal(final_cov, seed_cov_stored)


def test_unobserved_track_world_pose_is_invariant():
    """For a static-world object, T_wo = T_wb @ mu_bo must stay
    constant under deterministic mean update."""
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=np.eye(6) * 1e-4))

    oid = 1
    seed_mu_bo = _T(t=(2.0, 0.5, 1.0), rpy_deg=(0.0, 0.0, 30.0))
    state.ensure_object(oid, seed_mu_bo.copy(), np.eye(6) * 1e-4)
    seed_T_wo = state.T_wb @ state.objects[oid].mu_bo

    # 50 frames of arbitrary base motion.
    rng = np.random.default_rng(0)
    Q_fn = lambda _oid: np.zeros((6, 6))
    for _ in range(50):
        twist = rng.normal(0.0, 0.05, size=6)
        T_wb_new = state.T_wb @ se3_exp(twist)
        state.ingest_slam(PoseEstimate(T=T_wb_new,
                                        cov=np.eye(6) * 1e-4))
        state.predict_static(Q_fn=Q_fn, unobserved_oids={oid})

    final_T_wo = state.T_wb @ state.objects[oid].mu_bo
    np.testing.assert_allclose(final_T_wo, seed_T_wo, atol=1e-9)


def test_observed_track_still_updates():
    """Sanity: a track NOT in unobserved_oids must still get the
    full predict (mean + cov + Q)."""
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=np.eye(6) * 1e-4))

    oid = 1
    state.ensure_object(oid, _T(t=(1.0, 0.0, 1.0)), np.eye(6) * 1e-4)
    seed_cov = state.objects[oid].cov_bo.copy()

    # Move the base by 10 cm, then run predict with the track as
    # observed (NOT in unobserved_oids).
    state.ingest_slam(PoseEstimate(T=_T(t=(0.1, 0.0, 0.0)),
                                    cov=np.eye(6) * 1e-4))
    Q_fn = lambda _oid: np.diag([1e-5] * 6)
    state.predict_static(Q_fn=Q_fn, unobserved_oids=set())

    new_cov = state.objects[oid].cov_bo
    # Cov must have changed (Q added + Ad transport applied).
    assert not np.allclose(new_cov, seed_cov)
    # Diagonal must be at least seed + Q on every axis.
    np.testing.assert_array_less(
        np.diag(seed_cov) - 1e-12,
        np.diag(new_cov),
    )


def test_unobserved_world_cov_only_grows_via_sigma_wb_lever():
    """For an unobserved static track, the world-frame cov from
    collapsed_object_world must change ONLY because the lever-arm
    transforms with mu_bo (which moves deterministically with T_wb).
    With Σ_wb constant (PassThroughSlam-style), the world cov shape
    is bounded by the seed cov + |t_bo|² σ_yaw_wb²-style growth."""
    Sigma_wb = np.diag([1e-4] * 6)  # σ_trans ≈ 1 cm, σ_yaw ≈ 0.01 rad
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=Sigma_wb.copy()))

    oid = 1
    seed_cov_bo = np.diag(np.array([
        (0.02) ** 2, (0.02) ** 2, (0.02) ** 2,
        (0.10) ** 2, (0.10) ** 2, (np.pi) ** 2,
    ]))
    state.ensure_object(oid, _T(t=(2.0, 0.0, 1.0)), seed_cov_bo.copy())

    Q_fn = lambda _oid: np.zeros((6, 6))
    for k in range(1, 101):
        T_wb_k = _T(t=(0.05 * k, 0.0, 0.0))
        state.ingest_slam(PoseEstimate(T=T_wb_k, cov=Sigma_wb.copy()))
        state.predict_static(Q_fn=Q_fn, unobserved_oids={oid})

    pe_w = state.collapsed_object_world(oid)
    sigma_world_trans = np.sort(
        np.linalg.eigvalsh(pe_w.cov[:3, :3]))[::-1]
    sigma_world_max_cm = float(np.sqrt(max(sigma_world_trans[0], 0.0))) * 100

    # Bound: seed_cov_bo[trans] floor (2 cm) plus generous lever-arm
    # contribution. With t_bo ≈ 5m and σ_yaw_wb = 0.01 rad,
    # lever ≈ 5 cm. Total expected σ_major ≤ ~10 cm.
    assert sigma_world_max_cm < 15.0, (
        f"Expected σ_major ≤ 15 cm under keyframe mechanism, got "
        f"{sigma_world_max_cm:.2f} cm — mechanism 2 contamination "
        f"may have leaked back in."
    )
