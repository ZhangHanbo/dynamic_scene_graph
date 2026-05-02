"""Plan §C.2: tests for the new GaussianState.overwrite_object_pose
hook used by the Gaussian orchestrator's gravity_predict integration.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R_

from pose_update.state.gaussian_state import GaussianState
from pose_update.state.slam_interface import PoseEstimate


def _T(t=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _isclose_T(a, b, atol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol)


def test_overwrite_object_pose_round_trip_world():
    """Writing (T_wo, P_wo) and reading back via collapsed_object_world
    must reproduce the input (modulo the Σ_wb-lever-arm contribution
    from the existing world-frame composition formula)."""
    state = GaussianState(T_bc=_T())
    # Use a non-trivial T_wb to exercise the inverse lift.
    T_wb = _T(t=(1.5, 0.7, 0.0), rpy_deg=(0.0, 0.0, 30.0))
    # Make Σ_wb tiny so the round-trip is effectively exact.
    Sigma_wb = np.eye(6) * 1e-12
    state.ingest_slam(PoseEstimate(T=T_wb, cov=Sigma_wb))

    oid = 1
    state.ensure_object(oid, _T(), np.eye(6) * 1e-4)

    T_wo = _T(t=(2.5, 1.2, 0.3), rpy_deg=(0.0, 0.0, 90.0))
    P_wo = np.diag(np.array([
        (0.05) ** 2, (0.05) ** 2, (0.10) ** 2,
        (0.04) ** 2, (0.04) ** 2, (np.pi) ** 2,  # σ_yaw=π for the bottle case
    ]))
    ok = state.overwrite_object_pose(oid, T_wo, P_wo)
    assert ok

    pe = state.collapsed_object_world(oid)
    assert pe is not None
    _isclose_T(pe.T, T_wo)
    # cov is P_wo + tiny Σ_wb-lever; expect close to P_wo.
    np.testing.assert_allclose(pe.cov, P_wo, atol=1e-6)


def test_overwrite_object_pose_unknown_oid_returns_false():
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=np.eye(6) * 1e-4))
    ok = state.overwrite_object_pose(99, _T(), np.eye(6) * 1e-3)
    assert ok is False


def test_overwrite_object_pose_rejects_bad_shapes():
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=np.eye(6) * 1e-4))
    state.ensure_object(1, _T(), np.eye(6) * 1e-4)
    with pytest.raises(ValueError):
        state.overwrite_object_pose(1, np.eye(3), np.eye(6))
    with pytest.raises(ValueError):
        state.overwrite_object_pose(1, _T(), np.eye(3))


def test_overwrite_then_unobserved_freeze_keeps_world_pose():
    """End-to-end: overwrite at release, then 60 frames of unobserved
    motion. The world cov should remain dominated by P_wo (frozen by
    §C.1) plus a small Σ_wb-lever contribution."""
    Sigma_wb = np.eye(6) * 1e-4  # PassThroughSlam-style
    state = GaussianState(T_bc=_T())
    state.ingest_slam(PoseEstimate(T=_T(), cov=Sigma_wb.copy()))

    oid = 1
    state.ensure_object(oid, _T(), np.eye(6) * 1e-4)

    # Simulate gravity_predict: install a world-frame (T, P) with σ_yaw=π
    # in the rotation block and ~5 cm σ_xy in the translation block.
    T_wo = _T(t=(2.0, 0.0, 0.2))
    P_wo = np.diag(np.array([
        (0.05) ** 2, (0.05) ** 2, (0.40) ** 2,
        (0.05) ** 2, (0.05) ** 2, (np.pi) ** 2,
    ]))
    state.overwrite_object_pose(oid, T_wo, P_wo)

    # 60 frames of forward driving with the bottle unobserved.
    Q_fn = lambda _oid: np.diag([1e-5] * 6)
    for k in range(1, 61):
        T_wb_k = _T(t=(0.05 * k, 0.0, 0.0))
        state.ingest_slam(PoseEstimate(T=T_wb_k, cov=Sigma_wb.copy()))
        state.predict_static(Q_fn=Q_fn, unobserved_oids={oid})

    pe = state.collapsed_object_world(oid)
    # World pose should still be at T_wo (no observations to revise).
    _isclose_T(pe.T, T_wo, atol=1e-6)

    # σ_xy (top-down, the user-visible ellipse) must be bounded by
    # gravity_predict's σ_xy + small Σ_wb-lever. NOT 1.2 m (the
    # pre-fix bottle artifact).
    eigs_xy = np.sort(np.linalg.eigvalsh(pe.cov[:2, :2]))[::-1]
    sigma_major_xy_cm = float(np.sqrt(max(eigs_xy[0], 0.0))) * 100
    assert sigma_major_xy_cm < 30.0, (
        f"Top-down σ_major = {sigma_major_xy_cm:.1f} cm; expected "
        f"~5 cm (gravity_predict σ_xy) + small Σ_wb-lever."
    )

    # σ_yaw should remain ≈ π (gravity_predict's uniform-yaw prior).
    sigma_yaw = float(np.sqrt(max(pe.cov[5, 5], 0.0)))
    assert abs(sigma_yaw - np.pi) < 0.01, (
        f"σ_yaw = {sigma_yaw:.3f}, expected π. The yaw uncertainty "
        f"must NOT be eroded by predict propagation."
    )
