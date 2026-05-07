"""Unit tests that pin down GaussianState behaviour before the SceneTracker refactor.

These tests are the regression safety net: every `GaussianState` method that
will be touched during the refactor is exercised here with synthetic SE(3)
inputs whose expected outputs are hand-derivable. A passing suite on both
the current code and the refactored code guarantees equivalence.

Test organization matches the per-frame pipeline order:
    1. Construction & setters.
    2. Object lifecycle (ensure_object, delete_object, merge_tracks).
    3. SLAM ingest (prev_T_wb caching).
    4. Predict (static, rigid_attachment).
    5. Innovation statistics.
    6. Observation update.
    7. Covariance floor & saturation (via their action on predict/update).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R_

from ekf_tracker.state.gaussian_state import GaussianState
from utils.ekf_se3 import se3_exp, se3_log, se3_adjoint
from utils.slam_interface import PoseEstimate


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _T(t=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Convenience SE(3) builder."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _is_psd(P: np.ndarray, tol: float = 1e-9) -> bool:
    """Symmetric positive semi-definite test."""
    sym = np.allclose(P, P.T, atol=1e-10)
    if not sym:
        return False
    eigs = np.linalg.eigvalsh(0.5 * (P + P.T))
    return bool(np.all(eigs >= -tol))


def _simple_state(T_bc=None, P_min_diag=None) -> GaussianState:
    """Fresh GaussianState with an identity T_bc unless overridden."""
    return GaussianState(T_bc=T_bc, P_min_diag=P_min_diag)


# ─────────────────────────────────────────────────────────────────────
# 1. Construction & setters
# ─────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_init_defaults(self):
        s = _simple_state()
        assert s.T_bc.shape == (4, 4)
        np.testing.assert_allclose(s.T_bc, np.eye(4))
        np.testing.assert_allclose(s._Ad_bc, np.eye(6))
        assert s.T_wb is None
        assert s.prev_T_wb is None
        assert s.objects == {}
        assert s.P_min_diag is None

    def test_init_with_T_bc_caches_Ad(self):
        T_bc = _T(t=(0.1, 0.0, 1.0), rpy_deg=(0.0, 30.0, 0.0))
        s = _simple_state(T_bc=T_bc)
        np.testing.assert_allclose(s.T_bc, T_bc)
        np.testing.assert_allclose(s._Ad_bc, se3_adjoint(T_bc))

    def test_init_with_P_min_diag_stored(self):
        floor = np.array([1e-4] * 3 + [1e-2] * 3)
        s = _simple_state(P_min_diag=floor)
        np.testing.assert_allclose(s.P_min_diag, floor)

    def test_set_camera_extrinsic_updates_both_T_bc_and_Ad(self):
        s = _simple_state()
        new = _T(t=(0.2, 0.1, 0.8), rpy_deg=(0.0, -20.0, 45.0))
        s.set_camera_extrinsic(new)
        np.testing.assert_allclose(s.T_bc, new)
        np.testing.assert_allclose(s._Ad_bc, se3_adjoint(new))

    def test_set_camera_extrinsic_makes_a_copy(self):
        s = _simple_state()
        new = _T(t=(0.2, 0.1, 0.8))
        s.set_camera_extrinsic(new)
        # mutating the input must not change the stored T_bc
        new[0, 3] = 99.0
        assert s.T_bc[0, 3] == pytest.approx(0.2)

    def test_set_camera_extrinsic_rejects_bad_shape(self):
        s = _simple_state()
        with pytest.raises(ValueError):
            s.set_camera_extrinsic(np.eye(3))


# ─────────────────────────────────────────────────────────────────────
# 2. Object lifecycle
# ─────────────────────────────────────────────────────────────────────

class TestObjectLifecycle:
    def test_ensure_object_lift_through_T_bc(self):
        """Mean: T_bo = T_bc @ T_co.
        Cov:  stored verbatim (NOT Ad-lifted) -- see the
              `ensure_object` docstring for why this differs from the
              `update_observation` behaviour.
        """
        T_bc = _T(t=(0.15, 0.0, 1.1), rpy_deg=(0.0, 30.0, 0.0))
        s = _simple_state(T_bc=T_bc)
        T_co = _T(t=(0.5, 0.0, 0.8))
        R = np.diag([1e-3] * 3 + [1e-2] * 3)
        assert s.ensure_object(42, T_co, R) is True
        belief = s.objects[42]
        np.testing.assert_allclose(belief.mu_bo, T_bc @ T_co, atol=1e-12)
        # Cov is the symmetrised raw input (no Ad-lift at birth).
        np.testing.assert_allclose(belief.cov_bo, R, atol=1e-12)

    def test_ensure_object_duplicate_returns_false(self):
        s = _simple_state()
        T_co = _T(t=(0.5, 0.0, 0.8))
        R = np.diag([1e-3] * 6)
        s.ensure_object(1, T_co, R)
        assert s.ensure_object(1, T_co, R) is False

    def test_ensure_object_copies_init_cov(self):
        s = _simple_state()
        R = np.diag([1e-3] * 6)
        s.ensure_object(1, _T(t=(0.5,)*3), R)
        R[0, 0] = 99.0
        assert s.objects[1].cov_bo[0, 0] == pytest.approx(1e-3)

    def test_delete_object_returns_true_if_existed(self):
        s = _simple_state()
        s.ensure_object(7, _T(t=(0.5,)*3), np.diag([1e-3] * 6))
        assert s.delete_object(7) is True
        assert 7 not in s.objects
        assert s.delete_object(7) is False

    def test_merge_tracks_info_sum(self):
        """P_inv_new = P_inv_a + P_inv_b (Bayesian information fusion)."""
        s = _simple_state()
        T = _T(t=(1.0, 0.0, 0.5))
        cov_a = np.diag([0.01] * 3 + [0.1] * 3)
        cov_b = np.diag([0.02] * 3 + [0.2] * 3)
        s.ensure_object(1, T, cov_a)
        s.ensure_object(2, T, cov_b)        # same mean, different cov
        assert s.merge_tracks(1, 2) is True
        # keeper (oid=1) survives, oid=2 removed
        assert 1 in s.objects and 2 not in s.objects
        # Identical means -> merged mean equals either; cov is the info sum.
        expected_P_inv = np.linalg.inv(cov_a) + np.linalg.inv(cov_b)
        expected_P = np.linalg.inv(expected_P_inv)
        # Floor may have been applied; ignore it if the raw cov is above the floor.
        np.testing.assert_allclose(
            s.objects[1].cov_bo, expected_P, atol=1e-9)

    def test_merge_tracks_result_is_psd(self):
        s = _simple_state()
        T = _T(t=(1.0, 0.0, 0.5))
        cov_a = np.diag([0.05] * 3 + [0.3] * 3)
        cov_b = np.diag([0.02] * 3 + [0.1] * 3)
        s.ensure_object(1, T, cov_a)
        s.ensure_object(2, _T(t=(1.01, 0.0, 0.5)), cov_b)
        s.merge_tracks(1, 2)
        assert _is_psd(s.objects[1].cov_bo)


# ─────────────────────────────────────────────────────────────────────
# 3. SLAM ingest
# ─────────────────────────────────────────────────────────────────────

class TestSLAMIngest:
    def test_ingest_slam_first_call_sets_only_T_wb(self):
        s = _simple_state()
        pe = PoseEstimate(T=_T(t=(1.0, 2.0, 0.0)), cov=np.diag([1e-6]*6))
        s.ingest_slam(pe)
        np.testing.assert_allclose(s.T_wb, pe.T)
        assert s.prev_T_wb is None        # first call: nothing cached yet

    def test_ingest_slam_second_call_caches_prev(self):
        s = _simple_state()
        pe1 = PoseEstimate(T=_T(t=(1.0, 2.0, 0.0)), cov=np.diag([1e-6]*6))
        pe2 = PoseEstimate(T=_T(t=(1.5, 2.0, 0.0)), cov=np.diag([1e-6]*6))
        s.ingest_slam(pe1)
        s.ingest_slam(pe2)
        np.testing.assert_allclose(s.T_wb, pe2.T)
        np.testing.assert_allclose(s.prev_T_wb, pe1.T)


# ─────────────────────────────────────────────────────────────────────
# 4. Predict
# ─────────────────────────────────────────────────────────────────────

class TestPredictStatic:
    @staticmethod
    def _seed_state_with_motion(t_prev=(0., 0., 0.), t_now=(0.3, 0., 0.),
                                 rpy_prev_deg=(0., 0., 0.),
                                 rpy_now_deg=(0., 0., 45.),
                                 T_co_init=None, P0=None, T_bc=None):
        s = _simple_state(T_bc=T_bc)
        s.ingest_slam(PoseEstimate(
            T=_T(t=t_prev, rpy_deg=rpy_prev_deg),
            cov=np.diag([1e-6]*6)))
        s.ingest_slam(PoseEstimate(
            T=_T(t=t_now, rpy_deg=rpy_now_deg),
            cov=np.diag([1e-6]*6)))
        T_co = T_co_init if T_co_init is not None else _T(t=(0.5, 0., 0.8))
        P = P0 if P0 is not None else np.diag([1e-3]*3 + [1e-2]*3)
        s.ensure_object(1, T_co, P)
        return s

    def test_predict_static_identity_motion_preserves_mean(self):
        """u_k = I -> μ unchanged, cov = cov + Q."""
        s = self._seed_state_with_motion(
            t_prev=(0.3, 0., 0.), t_now=(0.3, 0., 0.),      # same pose
            rpy_prev_deg=(0., 0., 0.), rpy_now_deg=(0., 0., 0.))
        mu0 = s.objects[1].mu_bo.copy()
        cov0 = s.objects[1].cov_bo.copy()
        Q = np.diag([1e-5] * 6)
        s.predict_static(lambda oid: Q)
        np.testing.assert_allclose(s.objects[1].mu_bo, mu0, atol=1e-12)
        np.testing.assert_allclose(
            s.objects[1].cov_bo, cov0 + Q, atol=1e-12)

    def test_predict_static_body_form_u_k_matches_derivation(self):
        """μ_new = u_k @ μ_old where u_k = T_wb,now^-1 @ T_wb,prev."""
        t_prev = (0.3, 0., 0.); t_now = (1.0, 0., 0.)
        rpy_prev = (0., 0., 0.); rpy_now = (0., 0., 0.)     # pure translation
        s = self._seed_state_with_motion(
            t_prev=t_prev, t_now=t_now,
            rpy_prev_deg=rpy_prev, rpy_now_deg=rpy_now,
            T_co_init=_T(t=(0.5, 0., 0.8)))
        mu0 = s.objects[1].mu_bo.copy()
        s.predict_static(lambda oid: np.zeros((6, 6)))
        u_k = np.linalg.inv(_T(t_now, rpy_now)) @ _T(t_prev, rpy_prev)
        expected_mu = u_k @ mu0
        np.testing.assert_allclose(
            s.objects[1].mu_bo, expected_mu, atol=1e-12)

    def test_predict_static_rotation_regression_body_vs_world_form(self):
        """Critical: non-trivial base rotation reveals the body-vs-world-form bug.

        World-static object at world (2, 0, 0). Base rotates +90 deg about z
        from (0,1,0) to (0,1,0,R_z(90)). Predicted T_bo must be the body-form
        result (u_k = T_wb,now^-1 T_wb,prev), NOT the world-form
        (T_wb,prev @ T_wb,now^-1).
        """
        t_prev = (0.0, 1.0, 0.0); rpy_prev = (0., 0., 0.)
        t_now  = (0.0, 1.0, 0.0); rpy_now  = (0., 0., 90.)

        # True world-static object at (1, 1, 0) (from base-prev at (0,1,0)+x=1).
        true_T_wo = _T(t=(1.0, 1.0, 0.0))

        # Seed belief from T_bo at t_prev.
        T_wb_prev = _T(t_prev, rpy_prev)
        T_bo_prev = np.linalg.inv(T_wb_prev) @ true_T_wo
        s = _simple_state()
        s.ingest_slam(PoseEstimate(T=T_wb_prev, cov=np.diag([1e-6]*6)))
        s.ingest_slam(PoseEstimate(T=_T(t_now, rpy_now),
                                    cov=np.diag([1e-6]*6)))
        s.ensure_object(1, T_bo_prev, np.diag([1e-3]*3 + [1e-2]*3))
        # Note: ensure_object lifts by T_bc=I, so μ_bo = T_bo_prev.
        s.predict_static(lambda oid: np.zeros((6, 6)))

        expected_T_bo_now = np.linalg.inv(_T(t_now, rpy_now)) @ true_T_wo
        # body-form should recover the true base-frame pose exactly.
        np.testing.assert_allclose(
            s.objects[1].mu_bo, expected_T_bo_now, atol=1e-12)

        # Sanity: the world-form alternative would give a DIFFERENT answer.
        wrong_u_k_world = T_wb_prev @ np.linalg.inv(_T(t_now, rpy_now))
        wrong_mu = wrong_u_k_world @ T_bo_prev
        # They should differ materially (this is the bug regression).
        assert not np.allclose(s.objects[1].mu_bo, wrong_mu, atol=1e-3)

    def test_predict_static_adjoint_conjugates_covariance(self):
        s = self._seed_state_with_motion(
            t_prev=(0., 0., 0.), t_now=(0., 0., 0.),
            rpy_prev_deg=(0., 0., 0.), rpy_now_deg=(0., 0., 30.))
        cov0 = s.objects[1].cov_bo.copy()
        s.predict_static(lambda oid: np.zeros((6, 6)))
        u_k = np.linalg.inv(_T(rpy_deg=(0., 0., 30.))) @ _T()
        Ad = se3_adjoint(u_k)
        expected_cov = Ad @ cov0 @ Ad.T
        # allow tiny numerical noise from symmetrization.
        np.testing.assert_allclose(
            s.objects[1].cov_bo, expected_cov, atol=1e-10)

    def test_predict_static_applies_P_max(self):
        s = self._seed_state_with_motion(
            t_prev=(0.,)*3, t_now=(0.,)*3,
            P0=np.diag([1.0]*6))           # huge prior
        # Make Q trivial but request a tight cap.
        Q = np.zeros((6, 6))
        P_max = np.diag([0.1]*6)
        s.predict_static(lambda oid: Q, P_max=P_max)
        # After saturation, trace(P) <= trace(P_max).
        assert np.trace(s.objects[1].cov_bo) <= np.trace(P_max) + 1e-9

    def test_predict_static_applies_P_min_diag_floor(self):
        floor = np.array([0.01] * 3 + [0.1] * 3)
        s = _simple_state(P_min_diag=floor)
        s.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-6]*6)))
        s.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-6]*6)))
        s.ensure_object(1, _T(t=(0.5,)*3), np.diag([1e-6]*6))  # way below floor
        s.predict_static(lambda oid: np.zeros((6, 6)))
        diag = np.diag(s.objects[1].cov_bo)
        np.testing.assert_array_less(floor - 1e-12, diag)


class TestRigidAttachmentPredict:
    def test_identity_delta_bg_preserves_mean(self):
        s = _simple_state()
        s.ensure_object(1, _T(t=(0.5, 0., 0.3)), np.diag([1e-3]*6))
        mu0 = s.objects[1].mu_bo.copy()
        cov0 = s.objects[1].cov_bo.copy()
        Q = np.diag([1e-6]*6)
        s.rigid_attachment_predict(1, np.eye(4), Q)
        np.testing.assert_allclose(s.objects[1].mu_bo, mu0, atol=1e-12)
        np.testing.assert_allclose(
            s.objects[1].cov_bo, cov0 + Q, atol=1e-12)

    def test_translation_applied_left(self):
        s = _simple_state()
        s.ensure_object(1, _T(t=(0.5, 0., 0.3)), np.diag([1e-3]*6))
        dT = _T(t=(0.1, 0.0, 0.0))
        s.rigid_attachment_predict(1, dT, np.zeros((6, 6)))
        expected = dT @ _T(t=(0.5, 0., 0.3))
        np.testing.assert_allclose(s.objects[1].mu_bo, expected, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 5. Innovation statistics
# ─────────────────────────────────────────────────────────────────────

class TestInnovationStats:
    def test_zero_innovation_when_measurement_matches_prior(self):
        """Measurement = T_bc^-1 @ μ_bo  =>  ν = 0."""
        T_bc = _T(t=(0.15, 0.0, 1.1))
        s = _simple_state(T_bc=T_bc)
        T_bo = _T(t=(1.0, 0.2, 1.5), rpy_deg=(10., 0., 0.))
        s.ensure_object(1, np.linalg.inv(T_bc) @ T_bo,
                        np.diag([1e-3]*6))
        # Recompute: μ_bo should equal T_bo now.
        np.testing.assert_allclose(s.objects[1].mu_bo, T_bo, atol=1e-12)

        # A measurement that re-produces μ_bo exactly.
        T_co_meas = np.linalg.inv(T_bc) @ T_bo
        R_icp = np.diag([1e-4]*3 + [1e-3]*3)
        stats = s.innovation_stats(1, T_co_meas, R_icp)
        assert stats is not None
        nu, S, d2, log_lik = stats
        np.testing.assert_allclose(nu, np.zeros(6), atol=1e-12)
        assert d2 == pytest.approx(0.0, abs=1e-12)

    def test_known_innovation(self):
        """Hand-computed ν, S, d^2 for a fixed small translation offset."""
        s = _simple_state()           # T_bc = I
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-4]*6))
        # Translate measurement by 2 cm along x.
        T_co_meas = _T(t=(0.52, 0., 0.5))
        R_icp = np.diag([1e-4]*6)
        stats = s.innovation_stats(1, T_co_meas, R_icp)
        assert stats is not None
        nu, S, d2, log_lik = stats
        # ν = Log(μ^{-1} @ T_meas); for a pure translation this is (Δt, 0).
        np.testing.assert_allclose(
            nu, np.array([0.02, 0, 0, 0, 0, 0]), atol=1e-9)
        # d² = Δt² / (P_tt + R_tt) = 0.0004 / 0.0002 = 2.
        assert d2 == pytest.approx(2.0, rel=1e-9)

    def test_innovation_stats_returns_none_for_missing_oid(self):
        s = _simple_state()
        assert s.innovation_stats(999, _T(), np.diag([1e-3]*6)) is None


# ─────────────────────────────────────────────────────────────────────
# 5b. Centroid-only innovation (Phase C coarse association)
# ─────────────────────────────────────────────────────────────────────

class TestCentroidInnovationStats:
    def test_zero_innovation_when_centroid_matches_prior(self):
        """With T_bc=I: measured camera-frame centroid == μ_bo[:3,3] -> ν=0."""
        s = _simple_state()           # T_bc = I so μ_bo == T_co
        T_bo = _T(t=(0.5, 0., 0.5))
        s.ensure_object(1, T_bo, np.diag([1e-3]*6))
        stats = s.centroid_innovation_stats(
            1, np.array([0.5, 0.0, 0.5]),
            R_cam=np.diag([(0.02)**2]*3))
        assert stats is not None
        nu, S, d2, _ = stats
        np.testing.assert_allclose(nu, np.zeros(3), atol=1e-12)
        assert d2 == pytest.approx(0.0, abs=1e-12)

    def test_known_centroid_innovation(self):
        """2 cm x-offset with P_tt = R_tt = diag(1e-4) -> d² = (0.02)²/(2·1e-4) = 2."""
        s = _simple_state()
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-4]*6))
        stats = s.centroid_innovation_stats(
            1, np.array([0.52, 0.0, 0.5]),
            R_cam=np.diag([1e-4]*3))
        assert stats is not None
        nu, S, d2, _ = stats
        np.testing.assert_allclose(nu, np.array([0.02, 0., 0.]), atol=1e-12)
        assert d2 == pytest.approx(2.0, rel=1e-9)

    def test_lifts_through_T_bc(self):
        """With T_bc != I, a centroid at camera origin maps to T_bc translation."""
        T_bc = _T(t=(0.15, 0.0, 1.1))
        s = _simple_state(T_bc=T_bc)
        # Prior at T_bc · T_co = T_bc (since T_co = I).
        s.ensure_object(1, np.eye(4), np.diag([1e-4]*6))
        # Zero centroid in camera -> measured position in base = T_bc · 0 = t_bc.
        stats = s.centroid_innovation_stats(
            1, np.zeros(3), R_cam=np.diag([1e-4]*3))
        nu, _, _, _ = stats
        # μ_bo = T_bc; t_bo_prior = (0.15, 0, 1.1).
        # t_bo_meas = (0.15, 0, 1.1).
        # So ν should be 0.
        np.testing.assert_allclose(nu, np.zeros(3), atol=1e-12)

    def test_returns_none_for_missing_oid(self):
        s = _simple_state()
        assert s.centroid_innovation_stats(999, np.zeros(3)) is None


# ─────────────────────────────────────────────────────────────────────
# 6. Observation update
# ─────────────────────────────────────────────────────────────────────

class TestUpdateObservation:
    def test_perfect_match_posterior_mean_unchanged(self):
        T_bc = _T(t=(0.15, 0., 1.1))
        s = _simple_state(T_bc=T_bc)
        T_bo = _T(t=(1.0, 0.0, 1.5))
        s.ensure_object(1, np.linalg.inv(T_bc) @ T_bo, np.diag([1e-3]*6))
        mu0 = s.objects[1].mu_bo.copy()
        T_co_meas = np.linalg.inv(T_bc) @ T_bo
        R = np.diag([1e-4]*6)
        s.update_observation(1, T_co_meas, R, iekf_iters=2)
        np.testing.assert_allclose(
            s.objects[1].mu_bo, mu0, atol=1e-9)
        # Posterior covariance must be smaller (or equal) on every axis
        # than the prior.
        new = np.diag(s.objects[1].cov_bo)
        old = np.full(6, 1e-3)
        assert np.all(new <= old + 1e-12)

    def test_update_produces_psd_covariance(self):
        s = _simple_state()
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        T_co_meas = _T(t=(0.55, 0.01, 0.49))
        R = np.diag([1e-4]*6)
        s.update_observation(1, T_co_meas, R, iekf_iters=3)
        assert _is_psd(s.objects[1].cov_bo)

    def test_huber_w_scales_effective_R(self):
        """Smaller w -> larger R/w -> smaller gain -> less shift."""
        R = np.diag([1e-4]*6)
        results = []
        for w in (1.0, 0.5, 0.1):
            s = _simple_state()
            s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
            T_co_meas = _T(t=(0.55, 0., 0.5))
            s.update_observation(1, T_co_meas, R, iekf_iters=1, huber_w=w)
            shift = s.objects[1].mu_bo[0, 3] - 0.5
            results.append(shift)
        # Tighter Huber (smaller w) -> smaller shift.
        assert results[0] > results[1] > results[2] > 0

    def test_update_floor_diag_applied(self):
        floor = np.array([0.01] * 3 + [0.1] * 3)
        s = _simple_state(P_min_diag=floor)
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-6]*6))
        T_co_meas = _T(t=(0.5, 0., 0.5))        # perfect match
        R = np.diag([1e-6]*6)
        s.update_observation(1, T_co_meas, R, iekf_iters=1)
        diag = np.diag(s.objects[1].cov_bo)
        np.testing.assert_array_less(floor - 1e-12, diag)

    def test_update_with_huber_zero_is_noop(self):
        """huber_w = 0 should not mutate state (outer-gate reject)."""
        s = _simple_state()
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        mu0 = s.objects[1].mu_bo.copy()
        cov0 = s.objects[1].cov_bo.copy()
        s.update_observation(1, _T(t=(0.6, 0., 0.5)),
                              np.diag([1e-4]*6), huber_w=0.0)
        np.testing.assert_allclose(s.objects[1].mu_bo, mu0)
        np.testing.assert_allclose(s.objects[1].cov_bo, cov0)
