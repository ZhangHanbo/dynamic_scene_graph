"""Two-tier Gaussian-EKF tracker = `GaussianEkfTracker` + slow-tier smoothing.

The fast tier (per-frame predict / associate / EKF update / Bernoulli /
self-merge) is inherited unchanged from
:class:`ekf_tracker.gaussian_ekf_tracker.GaussianEkfTracker`. The only
addition is a slow-tier pass that calls
:class:`ekf_tracker.factor_graph.PoseGraphOptimizer`.run on the current
priors + relation graph, gated by :class:`TriggerConfig`.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ekf_tracker.gaussian_ekf_tracker import GaussianEkfTracker
from ekf_tracker.factor_graph import (
    PoseGraphOptimizer,
    Observation,
    OptimizationResult,
    RelationEdge,
)
from ekf_tracker.config import TriggerConfig
from utils.slam_interface import PoseEstimate


_TINY_COV = np.diag([1e-6] * 6)


class TwoTierOrchestratorGaussian(GaussianEkfTracker):
    """`GaussianEkfTracker` + slow-tier pose-graph smoothing.

    The fast tier runs every frame in `step()` (parent's `step`).
    A slow-tier `PoseGraphOptimizer.run` fires only when
    :meth:`_should_trigger` says so, or unconditionally via
    :meth:`smooth`. The slow tier reads world-frame priors from
    `state.collapsed_objects_world()`, optionally folds in relation
    factors, optionally adds a held-object manipulation factor, and
    writes posteriors back via `state.inject_posterior_world`.
    """

    def __init__(self,
                 K: np.ndarray,
                 bernoulli_cfg,
                 *,
                 trigger: Optional[TriggerConfig] = None,
                 optimizer: Optional[PoseGraphOptimizer] = None,
                 pose_method: str = "icp_chain",
                 T_bc: Optional[np.ndarray] = None,
                 verbose: bool = False):
        super().__init__(K, bernoulli_cfg,
                         pose_method=pose_method, T_bc=T_bc)
        self.trigger = trigger or TriggerConfig()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.last_opt_frame: int = -1
        self.verbose = bool(verbose)
        # Track the last gripper phase across frames (used by
        # _should_trigger's on_grasp / on_release edges).
        self._last_phase: str = "idle"

    # --------------------------------------------------------------- #
    #  Public step (extends parent's step with optional slow tier)
    # --------------------------------------------------------------- #

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             T_wb: np.ndarray,
             detections: List[Dict[str, Any]],
             phase: str = "idle",
             T_bg: Optional[np.ndarray] = None,
             held_oids: Optional[set] = None,
             held_seed: Optional[int] = None,
             relation_edges: Optional[Iterable] = None,
             T_bc: Optional[np.ndarray] = None,
             ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Snapshot known oids BEFORE the fast tier so the trigger's
        # on_new_object check sees only newly-born tracks.
        known_before = set(self.object_labels.keys())

        dbg, dets_with_pose = super().step(
            rgb=rgb, depth=depth, T_wb=T_wb, detections=detections,
            phase=phase, T_bg=T_bg, held_oids=held_oids,
            held_seed=held_seed, relation_edges=relation_edges,
            T_bc=T_bc,
        )

        if self._should_trigger(phase, dets_with_pose, known_before):
            opt = self._slow_tier(
                T_wb=T_wb,
                relation_edges=list(relation_edges or []),
                held_oid=held_seed,
                T_bg=T_bg,
                T_bc=T_bc,
            )
            dbg["triggered"] = True
            dbg["slow_tier"] = {
                "alpha": float(opt.alpha),
                "n_iters": int(opt.num_iterations),
                "n_priors": len(opt.posteriors),
            }
            self.last_opt_frame = self._frame_count - 1
        else:
            dbg["triggered"] = False

        self._last_phase = phase
        return dbg, dets_with_pose

    # --------------------------------------------------------------- #
    #  Public smooth (unconditional slow tier)
    # --------------------------------------------------------------- #

    def smooth(self,
               T_wb: np.ndarray,
               *,
               relation_edges: Optional[Iterable] = None,
               held_oid: Optional[int] = None,
               T_bg: Optional[np.ndarray] = None,
               T_bc: Optional[np.ndarray] = None,
               ) -> OptimizationResult:
        """Force-run the slow tier and write posteriors back into state."""
        return self._slow_tier(
            T_wb=T_wb,
            relation_edges=list(relation_edges or []),
            held_oid=held_oid,
            T_bg=T_bg,
            T_bc=T_bc,
        )

    # --------------------------------------------------------------- #
    #  Trigger policy
    # --------------------------------------------------------------- #

    def _should_trigger(self,
                        phase: str,
                        detections: List[Dict[str, Any]],
                        known_before: set) -> bool:
        last_phase = self._last_phase
        if (self.trigger.on_grasp
                and last_phase != "grasping"
                and phase == "grasping"):
            return True
        if (self.trigger.on_release
                and last_phase == "releasing"
                and phase != "releasing"):
            return True
        if self.trigger.on_new_object:
            seen = {d.get("id") for d in detections
                    if d.get("id") is not None}
            if not seen.issubset(known_before):
                return True
        if self.trigger.periodic_every_n_frames > 0:
            if (self._frame_count - self.last_opt_frame
                    >= self.trigger.periodic_every_n_frames):
                return True
        return False

    # --------------------------------------------------------------- #
    #  Slow tier
    # --------------------------------------------------------------- #

    def _slow_tier(self,
                   T_wb: np.ndarray,
                   relation_edges: List[RelationEdge],
                   held_oid: Optional[int],
                   T_bg: Optional[np.ndarray],
                   T_bc: Optional[np.ndarray]) -> OptimizationResult:
        """Optimise priors + relation factors + (optional) held-object factor.

        Inputs:
          * priors: world-frame collapse of the fast-tier posterior.
          * relations: passed-in `relation_edges` (typed as RelationEdge,
            or any object exposing `.parent`, `.child`, `.relation_type`,
            `.score`).
          * manipulation: only when `held_oid is not None`, both `T_bc`
            and `T_bg` are supplied, and the held oid has a stored
            chain entry that pins T_oe.

        Posteriors are written back via `state.inject_posterior_world`.
        """
        T_wb = np.asarray(T_wb, dtype=np.float64)
        slam_pose = PoseEstimate(T=T_wb, cov=_TINY_COV)
        priors: Dict[int, PoseEstimate] = dict(
            self.state.collapsed_objects_world() or {})

        if not priors:
            return OptimizationResult(
                posteriors={}, residuals={}, alpha=1.0, num_iterations=0)

        # Restrict relations to live tracks.
        rels: List[RelationEdge] = []
        for e in (relation_edges or []):
            if not (hasattr(e, "parent") and hasattr(e, "child")
                    and hasattr(e, "relation_type")):
                continue
            if int(e.parent) in priors and int(e.child) in priors:
                rels.append(e)

        # Held-object manipulation factor: needs T_ew and T_oe (locked).
        T_ew: Optional[np.ndarray] = None
        T_oe: Optional[np.ndarray] = None
        if (held_oid is not None
                and int(held_oid) in priors
                and T_bg is not None and T_bc is not None):
            T_bc_arr = np.asarray(T_bc, dtype=np.float64)
            T_bg_arr = np.asarray(T_bg, dtype=np.float64)
            T_ec = np.linalg.inv(T_bc_arr) @ T_bg_arr
            T_ew = T_wb @ T_ec
            # T_oe = inv(T_ew) @ μ_world (the locked offset at the time
            # of this slow-tier invocation; the fast tier already pinned
            # the held object via rigid_attachment_predict, so we can
            # safely freeze the current world pose as the lock).
            T_oe = np.linalg.inv(T_ew) @ priors[int(held_oid)].T

        # Slow-tier optimization. Observations are intentionally empty:
        # the chain's information is already absorbed into `priors` by
        # the fast tier's IEKF update, so re-feeding chain entries as
        # Observations here would double-count. The slow tier's value
        # is the relation + manipulation constraints applied on top.
        observations: List[Observation] = []
        result = self.optimizer.run(
            slam_pose=slam_pose,
            priors=priors,
            observations=observations,
            relations=rels if rels else None,
            held_obj_id=int(held_oid) if held_oid is not None else None,
            T_ew=T_ew,
            T_oe=T_oe,
        )

        for oid, pe in result.posteriors.items():
            self.state.inject_posterior_world(int(oid), pe)

        if self.verbose:
            print(f"[slow tier] α={result.alpha:.2f}, "
                  f"iters={result.num_iterations}, "
                  f"n_priors={len(priors)}, n_relations={len(rels)}")

        return result
