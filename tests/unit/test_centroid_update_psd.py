"""PSD invariant for ``GaussianState.update_observation_centroid``.

After many sequential centroid updates, the EKF state covariance
must stay positive semi-definite (eigenvalues ≥ −1e-12). This was a
real bug in an earlier round of the code; we now zero the cross-cov
between translation and rotation in the centroid Joseph step.
"""
from __future__ import annotations

import numpy as np

from pose_update.state.gaussian_state import GaussianState, GaussianObjectBelief


def _make_state(oid=1):
    state = GaussianState(T_bc=np.eye(4))
    P = np.eye(6) * 1e-3
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.1, 0.4]
    state.objects[oid] = GaussianObjectBelief(mu_bo=T, cov_bo=P)
    state.T_wb = np.eye(4)
    return state


def test_psd_preserved_under_200_centroid_updates():
    rng = np.random.default_rng(0)
    state = _make_state()

    R_cam = np.diag([(0.02) ** 2] * 3)
    for _ in range(200):
        # Random centroid jiggle in camera frame.
        c_cam = np.array([0.3 + rng.normal(0, 0.01),
                            0.0 + rng.normal(0, 0.01),
                            1.0 + rng.normal(0, 0.01)])
        state.update_observation_centroid(
            oid=1, centroid_cam=c_cam, R_cam=R_cam, huber_w=1.0)
        cov = state.objects[1].cov_bo
        evals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        assert evals.min() >= -1e-12, f"min eigenvalue {evals.min()} after update"


def test_translation_is_pulled_toward_measurement():
    """Sanity: after one update, mu_b moves toward the measurement
    (positive Kalman gain in the translation direction)."""
    state = _make_state()
    mu_b_pre = np.asarray(state.objects[1].mu_bo[:3, 3]).copy()
    # T_bc is identity, so centroid_cam → t_bo_meas = centroid_cam.
    # Pick a measurement clearly to the right of the prior.
    target = np.array([0.6, 0.1, 0.4])
    state.update_observation_centroid(
        oid=1, centroid_cam=target,
        R_cam=np.diag([(0.02) ** 2] * 3), huber_w=1.0)
    mu_b_post = np.asarray(state.objects[1].mu_bo[:3, 3])
    # X moved toward the target.
    assert mu_b_post[0] > mu_b_pre[0]
    assert mu_b_post[0] < target[0]   # didn't snap all the way (Kalman gain < 1)


def test_rotation_block_unchanged_by_centroid_update():
    """Centroid measurement carries no rotation info; the rotation
    block of mu_bo must not move (translation-only update)."""
    state = _make_state()
    R_pre = np.asarray(state.objects[1].mu_bo[:3, :3]).copy()
    state.update_observation_centroid(
        oid=1, centroid_cam=np.array([0.55, 0.1, 0.4]),
        R_cam=np.diag([(0.02) ** 2] * 3), huber_w=1.0)
    R_post = np.asarray(state.objects[1].mu_bo[:3, :3])
    np.testing.assert_allclose(R_pre, R_post, atol=1e-12)
