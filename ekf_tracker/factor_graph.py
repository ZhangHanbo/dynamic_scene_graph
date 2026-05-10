""":class:`PoseGraphOptimizer` — joint GTSAM pose graph over movable objects with EKF priors, ICP betweens, relation factors, and adaptive robust loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import gtsam

from utils.slam_interface import PoseEstimate
from perception.adaptive_kernel import AdaptiveKernel, irls_weight


# ─────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """ICP observation of an object relative to the camera."""
    obj_id: int
    T_co: np.ndarray                # (4, 4) camera-to-object
    R_icp: np.ndarray = field(      # (6, 6) intrinsic ICP noise
        default_factory=lambda: np.eye(6) * 1e-3)
    fitness: float = 1.0
    rmse: float = 0.0


@dataclass
class RelationEdge:
    """A scene graph relation between two objects."""
    parent: int
    child: int
    relation_type: str              # 'on' | 'in'
    score: float                    # [0, 1] detection confidence
    # Optional geometric hints (axis-aligned bbox sizes in object frame)
    parent_size: Optional[np.ndarray] = None     # (3,) [sx, sy, sz]
    child_size: Optional[np.ndarray] = None


@dataclass
class OptimizationResult:
    """Output of PoseGraphOptimizer.run()."""
    posteriors: Dict[int, PoseEstimate]
    residuals: Dict[str, np.ndarray]    # by factor type
    alpha: float                        # adaptive kernel shape used
    num_iterations: int


# ─────────────────────────────────────────────────────────────────────
# Noise model helpers
# ─────────────────────────────────────────────────────────────────────

def _gtsam_noise_from_cov(cov: np.ndarray) -> gtsam.noiseModel.Base:
    """GTSAM noise model from a 6x6 covariance (symmetric PSD)."""
    # Symmetrize and add tiny diagonal regularization for numerical stability
    cov_sym = 0.5 * (cov + cov.T) + np.eye(6) * 1e-12
    return gtsam.noiseModel.Gaussian.Covariance(cov_sym)


def _scale_noise_by_weight(cov: np.ndarray, weight: float,
                            min_weight: float = 1e-6) -> np.ndarray:
    r"""Scale a noise model's standard deviations by :math:`1/\sqrt{w}` for IRLS."""
    w = max(weight, min_weight)
    return cov / w


def _gtsam_pose3(T: np.ndarray) -> gtsam.Pose3:
    """Convert (4, 4) numpy to gtsam.Pose3."""
    return gtsam.Pose3(np.asarray(T, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────
# Relation residual helpers (used both as custom factors and for residual
# inspection / adaptive kernel fitting)
# ─────────────────────────────────────────────────────────────────────

def relation_residual(T_parent: np.ndarray, T_child: np.ndarray,
                      relation_type: str,
                      parent_size: Optional[np.ndarray] = None,
                      child_size: Optional[np.ndarray] = None) -> float:
    r"""Residual for an :math:`\text{on}` / :math:`\text{in}` relation factor (relative-pose constraint between parent and child)."""
    parent_pos = T_parent[:3, 3]
    child_pos = T_child[:3, 3]

    # Default sizes if missing (10cm cube-ish)
    ps = parent_size if parent_size is not None else np.array([0.1, 0.1, 0.1])
    cs = child_size if child_size is not None else np.array([0.05, 0.05, 0.05])

    if relation_type == "on":
        parent_top_z = parent_pos[2] + 0.5 * ps[2]
        child_bottom_z = child_pos[2] - 0.5 * cs[2]
        # Penalize gap (child floating) and overlap (child sunk into parent)
        return abs(child_bottom_z - parent_top_z)

    elif relation_type == "in":
        # Child center should lie inside parent's bbox
        rel_pos = child_pos - parent_pos
        excess = np.maximum(0.0, np.abs(rel_pos) - 0.5 * ps)
        return float(np.linalg.norm(excess))

    return 0.0


def _between_t(T_child_in_parent: np.ndarray,
                relation_type: str,
                parent_size: np.ndarray,
                child_size: np.ndarray) -> np.ndarray:
    """SE(3) between-factor whose error penalises only the translation block."""
    if relation_type == "on":
        expected = np.zeros(3)
        expected[2] = 0.5 * (parent_size[2] + child_size[2])
        return expected
    # "in": child near parent center
    return np.zeros(3)


# ─────────────────────────────────────────────────────────────────────
# PoseGraphOptimizer
# ─────────────────────────────────────────────────────────────────────

class PoseGraphOptimizer:
    """GTSAM pose graph over movable objects: EKF priors, ICP betweens, relation factors, adaptive robust loss."""

    def __init__(self,
                 adaptive_c: float = 0.01,
                 manip_noise_sigma: float = 0.015,   # base-loc uncertainty magnitude
                 relation_base_sigma: float = 0.03,  # per-relation base stdev
                 lm_max_iterations: int = 30,
                 verbose: bool = False):
        self.kernel = AdaptiveKernel(c=adaptive_c)
        self.manip_noise_sigma = manip_noise_sigma
        self.relation_base_sigma = relation_base_sigma
        self.lm_max_iterations = lm_max_iterations
        self.verbose = verbose

    # --------------------------------------------------------------- #

    def run(self,
            slam_pose: PoseEstimate,
            priors: Dict[int, PoseEstimate],
            observations: List[Observation],
            relations: Optional[List[RelationEdge]] = None,
            held_obj_id: Optional[int] = None,
            T_ew: Optional[np.ndarray] = None,
            T_oe: Optional[np.ndarray] = None,
            active_set: Optional[Set[int]] = None,
            ) -> OptimizationResult:
        """Build and optimize the graph; returns updated :math:`(T_{wo}, P_{wo})` per object."""
        if active_set is None:
            active_set = set(priors.keys())
        # Add any observation or relation target into the active set
        for obs in observations:
            active_set.add(obs.obj_id)
        if relations:
            for r in relations:
                active_set.add(r.parent)
                active_set.add(r.child)
        if held_obj_id is not None:
            active_set.add(held_obj_id)

        # Fit adaptive kernel from a first pass of residuals
        alpha = self._fit_alpha(priors, observations, relations, slam_pose)

        graph, initial, key_by_id = self._build_graph(
            slam_pose=slam_pose,
            priors=priors,
            observations=observations,
            relations=relations,
            held_obj_id=held_obj_id,
            T_ew=T_ew,
            T_oe=T_oe,
            alpha=alpha,
            active_set=active_set,
        )

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.lm_max_iterations)
        if self.verbose:
            params.setVerbosity("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result_values = optimizer.optimize()

        # Recover posteriors with marginal covariance
        try:
            marginals = gtsam.Marginals(graph, result_values)
        except Exception:
            marginals = None

        posteriors: Dict[int, PoseEstimate] = {}
        for oid, key in key_by_id.items():
            T_opt = result_values.atPose3(key).matrix()
            if marginals is not None:
                try:
                    cov = marginals.marginalCovariance(key)
                except Exception:
                    cov = priors[oid].cov if oid in priors else np.eye(6) * 0.01
            else:
                cov = priors[oid].cov if oid in priors else np.eye(6) * 0.01
            posteriors[oid] = PoseEstimate(T=T_opt, cov=cov)

        residuals = self._collect_residuals(
            result_values, key_by_id, observations, relations,
            held_obj_id, T_ew, T_oe, slam_pose)

        return OptimizationResult(
            posteriors=posteriors,
            residuals=residuals,
            alpha=alpha,
            num_iterations=optimizer.iterations(),
        )

    # --------------------------------------------------------------- #
    #  Graph construction
    # --------------------------------------------------------------- #

    def _build_graph(self,
                     slam_pose: PoseEstimate,
                     priors: Dict[int, PoseEstimate],
                     observations: List[Observation],
                     relations: Optional[List[RelationEdge]],
                     held_obj_id: Optional[int],
                     T_ew: Optional[np.ndarray],
                     T_oe: Optional[np.ndarray],
                     alpha: float,
                     active_set: Set[int],
                     ) -> Tuple[gtsam.NonlinearFactorGraph,
                                gtsam.Values,
                                Dict[int, int]]:
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key_by_id: Dict[int, int] = {}

        # Variables and priors
        for oid in active_set:
            key = gtsam.symbol('o', oid)
            key_by_id[oid] = key
            if oid in priors:
                prior = priors[oid]
                initial.insert(key, _gtsam_pose3(prior.T))
                prior_noise = _gtsam_noise_from_cov(prior.cov)
                graph.add(gtsam.PriorFactorPose3(
                    key, _gtsam_pose3(prior.T), prior_noise))
            else:
                # No prior — rely on observation to anchor. Use a loose init.
                initial.insert(key, gtsam.Pose3())

        # Observation factors (camera → object)
        T_wb = slam_pose.T
        Sigma_wb = slam_pose.cov
        for obs in observations:
            if obs.obj_id not in key_by_id:
                continue
            # Observed world-frame pose via the fixed camera
            T_wo_obs = T_wb @ obs.T_co
            # Effective noise composed with SLAM uncertainty (Paper 1):
            # R_eff = R_icp + Σ_wb (here J = I for a world-frame pose prior-ish factor)
            R_icp_scaled = self._scale_by_icp_quality(obs.R_icp, obs.fitness, obs.rmse)
            R_eff = R_icp_scaled + Sigma_wb

            # Adaptive downweighting based on residual size (one-shot — α already fitted)
            residual_mag = self._observation_residual(T_wb, obs, priors)
            w = float(irls_weight(np.array([residual_mag]), alpha, self.kernel.c)[0])
            R_weighted = _scale_noise_by_weight(R_eff, w)

            graph.add(gtsam.PriorFactorPose3(
                key_by_id[obs.obj_id],
                _gtsam_pose3(T_wo_obs),
                _gtsam_noise_from_cov(R_weighted),
            ))

        # Scene graph relation factors
        if relations:
            for rel in relations:
                if rel.parent not in key_by_id or rel.child not in key_by_id:
                    continue
                self._add_relation_factor(graph, key_by_id, priors, rel, alpha)

        # Manipulation factor (rigid attachment to EE during HOLDING)
        if held_obj_id is not None and T_ew is not None and T_oe is not None:
            if held_obj_id in key_by_id:
                # The held object's world pose should equal T_ew @ T_oe
                # Noise reflects base-localization uncertainty, NOT kinematic precision
                T_wo_held = T_ew @ T_oe
                manip_cov = np.diag(
                    [self.manip_noise_sigma ** 2] * 3 +
                    [self.manip_noise_sigma ** 2] * 3
                )
                graph.add(gtsam.PriorFactorPose3(
                    key_by_id[held_obj_id],
                    _gtsam_pose3(T_wo_held),
                    _gtsam_noise_from_cov(manip_cov),
                ))

        return graph, initial, key_by_id

    # --------------------------------------------------------------- #

    def _add_relation_factor(self,
                             graph: gtsam.NonlinearFactorGraph,
                             key_by_id: Dict[int, int],
                             priors: Dict[int, PoseEstimate],
                             rel: RelationEdge,
                             alpha: float) -> None:
        """Add a single relation factor (``on`` / ``in``) wrapping the residual in the adaptive Barron kernel."""
        parent_key = key_by_id[rel.parent]
        child_key = key_by_id[rel.child]

        # Pull sizes from prior covariance diag as a rough proxy if not given
        ps = (rel.parent_size if rel.parent_size is not None
              else np.array([0.1, 0.1, 0.1]))
        cs = (rel.child_size if rel.child_size is not None
              else np.array([0.05, 0.05, 0.05]))

        # Expected translation from parent to child under the relation
        t_expected = _between_t(None, rel.relation_type, ps, cs)
        # Between pose (parent->child): translation only, identity rotation
        T_between = np.eye(4)
        T_between[:3, 3] = t_expected

        # Noise: base σ scaled by (1 / score) and inflated by pose uncertainties
        base_sigma = self.relation_base_sigma / max(rel.score, 0.05)
        sigmas = np.array([
            10.0, 10.0, 10.0,      # rotation: essentially free
            base_sigma, base_sigma, base_sigma,  # translation
        ])
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

        # Check current residual for this relation and apply adaptive weight
        if rel.parent in priors and rel.child in priors:
            res_mag = relation_residual(
                priors[rel.parent].T, priors[rel.child].T,
                rel.relation_type, ps, cs)
            w = float(irls_weight(np.array([res_mag]), alpha, self.kernel.c)[0])
            sigmas = sigmas / np.sqrt(max(w, 1e-6))
            noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

        graph.add(gtsam.BetweenFactorPose3(
            parent_key, child_key,
            _gtsam_pose3(T_between),
            noise,
        ))

    # --------------------------------------------------------------- #
    #  Residual helpers
    # --------------------------------------------------------------- #

    @staticmethod
    def _scale_by_icp_quality(R_icp: np.ndarray,
                              fitness: float, rmse: float) -> np.ndarray:
        """Inflate ICP noise when fitness is low or RMSE is high."""
        f = max(fitness, 0.05)
        scale = (max(rmse, 1e-3)) / f
        # Normalize by expected baseline (fitness=0.8, rmse=5mm → scale≈0.006)
        normalized = scale / 0.006
        return R_icp * max(normalized, 1.0)

    @staticmethod
    def _observation_residual(T_wb: np.ndarray,
                              obs: Observation,
                              priors: Dict[int, PoseEstimate]) -> float:
        """Scalar residual magnitude: ||T_wo_obs − T_wo_prior||."""
        if obs.obj_id not in priors:
            return 0.0
        T_wo_obs = T_wb @ obs.T_co
        T_wo_prior = priors[obs.obj_id].T
        diff = np.linalg.inv(T_wo_prior) @ T_wo_obs
        # Tangent-space norm
        from utils.ekf_se3 import se3_log
        xi = se3_log(diff)
        return float(np.linalg.norm(xi))

    def _fit_alpha(self,
                   priors: Dict[int, PoseEstimate],
                   observations: List[Observation],
                   relations: Optional[List[RelationEdge]],
                   slam_pose: PoseEstimate) -> float:
        """Build residual distribution from ALL factors and fit Barron α."""
        residuals: List[float] = []

        T_wb = slam_pose.T
        for obs in observations:
            residuals.append(self._observation_residual(T_wb, obs, priors))

        if relations:
            for rel in relations:
                if rel.parent in priors and rel.child in priors:
                    ps = rel.parent_size if rel.parent_size is not None else np.array([0.1]*3)
                    cs = rel.child_size if rel.child_size is not None else np.array([0.05]*3)
                    residuals.append(relation_residual(
                        priors[rel.parent].T, priors[rel.child].T,
                        rel.relation_type, ps, cs))

        if not residuals:
            return self.kernel.alpha_max
        return self.kernel.fit_alpha(np.asarray(residuals))

    def _collect_residuals(self,
                           result: gtsam.Values,
                           key_by_id: Dict[int, int],
                           observations: List[Observation],
                           relations: Optional[List[RelationEdge]],
                           held_obj_id: Optional[int],
                           T_ew: Optional[np.ndarray],
                           T_oe: Optional[np.ndarray],
                           slam_pose: PoseEstimate) -> Dict[str, np.ndarray]:
        """Compute post-optimization residuals for diagnostics."""
        from utils.ekf_se3 import se3_log

        out: Dict[str, List[float]] = {
            "observation": [], "relation": [], "manipulation": []
        }

        T_wb = slam_pose.T

        for obs in observations:
            if obs.obj_id not in key_by_id:
                continue
            T_opt = result.atPose3(key_by_id[obs.obj_id]).matrix()
            T_wo_obs = T_wb @ obs.T_co
            xi = se3_log(np.linalg.inv(T_wo_obs) @ T_opt)
            out["observation"].append(np.linalg.norm(xi))

        if relations:
            for rel in relations:
                if rel.parent in key_by_id and rel.child in key_by_id:
                    Tp = result.atPose3(key_by_id[rel.parent]).matrix()
                    Tc = result.atPose3(key_by_id[rel.child]).matrix()
                    ps = rel.parent_size if rel.parent_size is not None else np.array([0.1]*3)
                    cs = rel.child_size if rel.child_size is not None else np.array([0.05]*3)
                    out["relation"].append(
                        relation_residual(Tp, Tc, rel.relation_type, ps, cs))

        if held_obj_id is not None and T_ew is not None and T_oe is not None:
            if held_obj_id in key_by_id:
                T_opt = result.atPose3(key_by_id[held_obj_id]).matrix()
                T_wo_expected = T_ew @ T_oe
                xi = se3_log(np.linalg.inv(T_wo_expected) @ T_opt)
                out["manipulation"].append(np.linalg.norm(xi))

        return {k: np.asarray(v) for k, v in out.items()}
