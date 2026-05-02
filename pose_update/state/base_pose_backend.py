"""Abstract `BasePoseBackend` interface.

The backend owns everything that depends on the SLAM uncertainty
representation (single Gaussian vs particle cloud) and the per-object
belief storage that follows from that choice:

    Gaussian backend (GaussianState)
        - single (T_wb, Σ_wb) pair.
        - object beliefs in BASE frame, one per oid.
        - no weight update (no-op `absorb_likelihoods`).

    RBPF backend (RBPFState)
        - N particles, each with its own T_wb^k and log_weight.
        - object beliefs in WORLD frame, per-particle per-oid.
        - vision log-likelihoods add to particle log-weights.

The `SceneTracker.step()` pipeline (predict → ICP-with-prior →
Hungarian → refine → update → miss → birth → prune → self-merge →
emit) uses only this interface. It doesn't know which backend is live.

Split of responsibilities
─────────────────────────
  base-pose          : T_wb, Σ_wb, prev_T_wb, u_k (body-form control input).
  camera extrinsic   : T_bc(t_k) and Ad(T_bc), installed per-frame.
  object store       : ensure_object / delete_object / merge_tracks,
                       camera_frame_prior(oid) for ICP seeding,
                       innovation_stats / update_observation,
                       predict_static_all / rigid_attachment_predict.
  weight update      : absorb_likelihoods(log_liks) -- RBPF only.

`innovation_stats` and `update_observation` each take a
`T_co_meas, R_icp` pair (camera frame, as delivered by ICP). Backends
apply their own frame lift (base for Gaussian, world for RBPF).

A backend need not inherit from `BasePoseBackend` statically -- Python
duck-typing will do -- but declaring `register(BasePoseBackend)` is
useful for `isinstance` checks and for IDE navigation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class BasePoseBackend(ABC):
    """Contract for an uncertainty-rep-specific base-pose + object-store
    back-end plugged into `SceneTracker`."""

    # ────────────── base pose + camera extrinsic ──────────────

    T_bc: np.ndarray        # base-to-camera-optical (set per-frame)

    @abstractmethod
    def set_camera_extrinsic(self, T_bc: np.ndarray) -> None: ...

    @abstractmethod
    def ingest_slam(self, slam_result) -> None: ...

    @abstractmethod
    def collapsed_T_wb(self) -> np.ndarray:
        """Return a single representative T_wb (mean / weighted mean)."""
        ...

    # `prev_T_wb` is a stored attribute (not an abstract method) so that
    # simple attribute access `backend.prev_T_wb` works regardless of
    # backend. Concrete backends must initialise it to `None` in
    # `__init__` and update it in `ingest_slam`.
    prev_T_wb: Optional[np.ndarray]

    # ────────────── object lifecycle ──────────────

    @abstractmethod
    def ensure_object(self,
                      oid: int,
                      T_co_meas: np.ndarray,
                      init_cov: np.ndarray) -> bool: ...

    @abstractmethod
    def delete_object(self, oid: int) -> bool: ...

    @abstractmethod
    def merge_tracks(self, oid_keep: int, oid_drop: int) -> bool: ...

    @abstractmethod
    def known_oids(self) -> List[int]:
        """All tracked oids currently held in the object store."""
        ...

    # ────────────── camera-frame prior for ICP seeding ──────────────

    @abstractmethod
    def camera_frame_prior(self, oid: int) -> Optional[np.ndarray]:
        """Return T_co^pred for this track (camera-frame prior mean).

        For base-frame-stored backends: T_bc^{-1} · μ_bo.
        For world-frame-stored backends: (T_wb · T_bc)^{-1} · collapsed μ_wo.
        `None` if the oid is unknown.
        """
        ...

    # ────────────── predict ──────────────

    @abstractmethod
    def predict_static_all(self,
                            Q_fn: Callable[[int], np.ndarray],
                            skip_oids: Optional[set] = None,
                            P_max: Optional[np.ndarray] = None) -> None:
        """Static-object predict for every oid not in `skip_oids`."""
        ...

    @abstractmethod
    def rigid_attachment_predict(self,
                                  oid: int,
                                  delta_T_bg: np.ndarray,
                                  Q_manip: np.ndarray,
                                  P_max: Optional[np.ndarray] = None
                                  ) -> None: ...

    # ────────────── measurement ──────────────

    @abstractmethod
    def innovation_stats(self,
                          oid: int,
                          T_co_meas: np.ndarray,
                          R_icp: np.ndarray
                          ) -> Optional[Tuple[np.ndarray, np.ndarray,
                                              float, float]]:
        """Return (ν, S, d², log_lik). None if oid is unknown."""
        ...

    @abstractmethod
    def update_observation(self,
                            oid: int,
                            T_co_meas: np.ndarray,
                            R_icp: np.ndarray,
                            iekf_iters: int = 2,
                            huber_w: float = 1.0,
                            P_max: Optional[np.ndarray] = None) -> None: ...

    # ────────────── weight update (RBPF only) ──────────────

    def absorb_likelihoods(self, oid: int, log_liks: List[float]) -> None:
        """Absorb per-sample log-likelihoods into whatever weights the
        backend maintains. Default is no-op (single-Gaussian backend has
        no weights). RBPF overrides to add `log_liks[k]` to
        particle `k`'s log weight.
        """
        return None

    # ────────────── emit ──────────────

    @abstractmethod
    def collapsed_object_base(self, oid: int) -> Optional[object]:
        """Return a PoseEstimate-shaped (T, cov) snapshot of the object's
        base-frame posterior. Used for ICP priors, visibility, and the
        per-frame dbg dump.
        """
        ...
