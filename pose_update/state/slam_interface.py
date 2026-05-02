"""
Layer 1 interface — fixed protocol between SLAM (any backend) and the
movable-object tracker (Layer 2).

Design (Task 1):

* Movable-object detections are masked OUT of the depth image before it
  reaches Layer 1. This prevents Layer 1 from using movable objects as
  landmarks, which would poison the camera pose when an object is moved
  (Paper 3's dynamic masking insight).

* Two representations for localization uncertainty:
    - Gaussian:  `PoseEstimate(T, cov)`          — natural for EKF/FG/iSAM.
    - Particles: `ParticlePose(particles, weights)` — natural for AMCL-like
                   particle filters.
  Both implement `to_gaussian()`; downstream code (EKF, factor graph)
  internally consumes the Gaussian form. A backend can return either.

Usage:

    slam = MySlamBackend(...)
    movable_mask = collect_movable_masks(detections, depth.shape)
    masked_depth = mask_out_movable(depth, movable_mask)
    result = slam.step(rgb, masked_depth, odom_prior)
    pose = as_gaussian(result)   # accepts PoseEstimate or ParticlePose
    # pose.T, pose.cov are now the SLAM-forwarded quantities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Shared types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PoseEstimate:
    """Pose with covariance, the contract between Layer 1 and Layer 2.

    Attributes:
        T:   (4, 4) SE(3) mean.
        cov: (6, 6) covariance in se(3) tangent space, ordering [v, ω].
    """
    T: np.ndarray
    cov: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.01)

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=np.float64)
        self.cov = np.asarray(self.cov, dtype=np.float64)
        assert self.T.shape == (4, 4), f"T must be (4,4), got {self.T.shape}"
        assert self.cov.shape == (6, 6), f"cov must be (6,6), got {self.cov.shape}"

    def to_gaussian(self) -> "PoseEstimate":
        """Trivial — already Gaussian."""
        return self


@dataclass
class ParticlePose:
    """Particle-filter representation of a pose distribution.

    Natural fit for AMCL and other Monte Carlo localization backends.
    Downstream consumers call `.to_gaussian()` to get the moment-matched
    representation for EKF / factor-graph use.

    Attributes:
        particles: (N, 4, 4) array of SE(3) pose samples.
        weights:   (N,) non-negative weights summing to 1 (normalized
                   internally if they don't).
    """
    particles: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.particles = np.asarray(self.particles, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        assert self.particles.ndim == 3 and self.particles.shape[1:] == (4, 4), \
            f"particles must be (N,4,4), got {self.particles.shape}"
        assert self.weights.ndim == 1 and self.weights.shape[0] == self.particles.shape[0], \
            f"weights shape {self.weights.shape} inconsistent with particles {self.particles.shape}"
        assert np.all(self.weights >= 0), "weights must be non-negative"
        w_sum = self.weights.sum()
        if w_sum <= 0:
            # Degenerate → uniform
            self.weights = np.full_like(self.weights, 1.0 / len(self.weights))
        else:
            self.weights = self.weights / w_sum

    @property
    def n(self) -> int:
        return self.particles.shape[0]

    def effective_sample_size(self) -> float:
        """1 / Σ w² — number of 'effective' particles after weighting."""
        return float(1.0 / np.sum(self.weights ** 2))

    def map_pose(self) -> np.ndarray:
        """(4, 4) particle with the highest weight."""
        return self.particles[int(np.argmax(self.weights))].copy()

    def to_gaussian(self, max_iter: int = 10,
                    tol: float = 1e-6) -> "PoseEstimate":
        """Moment-match to a Gaussian on SE(3).

        Uses the weighted Karcher mean for the mean pose, then the weighted
        second moment in its tangent space for the covariance. This is the
        standard approach for projecting a pose particle cloud to a Gaussian.

        The iteration converges in 2–3 steps for sensible distributions;
        `max_iter` is a safety cap.
        """
        from pose_update.state.ekf_se3 import se3_log, se3_exp

        # Iteratively refine the mean via weighted tangent-space updates
        T_ref = self.map_pose()
        for _ in range(max_iter):
            # Tangent vectors at T_ref
            xis = np.stack([
                se3_log(np.linalg.inv(T_ref) @ self.particles[i])
                for i in range(self.n)
            ], axis=0)                        # (N, 6)
            mean_xi = np.sum(self.weights[:, None] * xis, axis=0)  # (6,)
            T_ref = T_ref @ se3_exp(mean_xi)
            if np.linalg.norm(mean_xi) < tol:
                break

        # Final covariance in the tangent at T_ref
        xis = np.stack([
            se3_log(np.linalg.inv(T_ref) @ self.particles[i])
            for i in range(self.n)
        ], axis=0)
        # Centered — mean_xi should be ≈ 0 at convergence; use explicit
        # to be safe for capped iterations
        mean_xi = np.sum(self.weights[:, None] * xis, axis=0)
        centered = xis - mean_xi[None, :]
        # Weighted outer-product sum
        cov = (centered.T * self.weights) @ centered  # (6, 6)
        # Regularize
        cov = 0.5 * (cov + cov.T) + np.eye(6) * 1e-12

        return PoseEstimate(T=T_ref, cov=cov)


def as_gaussian(pose) -> PoseEstimate:
    """Normalize either representation to a PoseEstimate.

    Lets consumer code be agnostic: `pose = as_gaussian(slam.step(...))`.
    """
    if isinstance(pose, PoseEstimate):
        return pose
    if isinstance(pose, ParticlePose):
        return pose.to_gaussian()
    raise TypeError(
        f"Expected PoseEstimate or ParticlePose, got {type(pose).__name__}")


def sample_particles_from_gaussian(pose: PoseEstimate,
                                    n_samples: int,
                                    rng: Optional[np.random.Generator] = None
                                    ) -> ParticlePose:
    """Reverse direction: draw N particles from a Gaussian pose estimate.

    Useful when a downstream consumer needs particles and the backend
    only provides a Gaussian (e.g., wrapping an EKF-SLAM for a PF-based
    planner).
    """
    from pose_update.state.ekf_se3 import se3_exp

    if rng is None:
        rng = np.random.default_rng()
    L = np.linalg.cholesky(pose.cov + np.eye(6) * 1e-12)
    xis = rng.standard_normal(size=(n_samples, 6)) @ L.T  # (N, 6)
    particles = np.stack([pose.T @ se3_exp(xi) for xi in xis], axis=0)
    weights = np.full(n_samples, 1.0 / n_samples)
    return ParticlePose(particles=particles, weights=weights)


# ─────────────────────────────────────────────────────────────────────
# Movable-mask pre-processing
# ─────────────────────────────────────────────────────────────────────

def collect_movable_masks(detections: List[dict],
                          img_shape: tuple,
                          dilate_px: int = 5) -> np.ndarray:
    """Union all movable-object masks into a single (H, W) bool image.

    Args:
        detections: list of dicts, each with a "mask" field (H, W) uint8/bool.
                    Callers decide which detections count as "movable";
                    typically: all detected foreground objects.
        img_shape:  (H, W) target shape.
        dilate_px:  optional morphological dilation to grow masks and cover
                    mask-edge depth pixels that would otherwise leak into SLAM.

    Returns:
        (H, W) bool array: True where movable. Safe to pass an empty list
        (returns all-False).
    """
    H, W = img_shape[:2]
    union = np.zeros((H, W), dtype=bool)
    if not detections:
        return union

    for det in detections:
        m = det.get("mask")
        if m is None:
            continue
        m_bool = np.asarray(m).astype(bool)
        if m_bool.shape != (H, W):
            # Resize if mask dimensions differ (caller error, but be robust)
            import cv2
            m_bool = cv2.resize(m_bool.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        union |= m_bool

    if dilate_px > 0:
        import cv2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        dilated = cv2.dilate(union.astype(np.uint8), kernel, iterations=1)
        union = dilated.astype(bool)

    return union


def mask_out_movable(depth: np.ndarray,
                     movable_mask: np.ndarray,
                     fill_value: float = 0.0) -> np.ndarray:
    """Return a depth image copy with movable-object pixels zeroed.

    Does NOT modify the input. SLAM backends should treat zero-depth as
    "no measurement" (standard convention).
    """
    assert depth.shape == movable_mask.shape, \
        f"shape mismatch: depth {depth.shape} vs mask {movable_mask.shape}"
    out = depth.copy()
    out[movable_mask] = fill_value
    return out


# ─────────────────────────────────────────────────────────────────────
# SLAM backend protocol
# ─────────────────────────────────────────────────────────────────────

class SlamBackend(Protocol):
    """Structural type that any SLAM backend should satisfy.

    Implementations can be: ORB-SLAM3 wrapper, RTAB-Map, custom ICP pipeline,
    AMCL with particle output, or a pass-through that simply reports the
    ground-truth pose from a rosbag.

    A backend may return either a `PoseEstimate` (Gaussian — natural for
    EKF-SLAM / smoothing backends) or a `ParticlePose` (natural for Monte
    Carlo localization). Callers should pipe the result through
    `as_gaussian()` before feeding it to the EKF or factor graph.
    """

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,       # already movable-masked
             odom_prior: Optional[PoseEstimate] = None,
             ):
        """Process one frame and return the current base-to-world pose.

        Args:
            rgb: (H, W, 3) RGB image.
            depth: (H, W) depth in meters, with movable pixels zeroed.
            odom_prior: optional pose estimate from wheel odometry or IMU.

        Returns:
            PoseEstimate or ParticlePose for the current frame
            (base-to-world).
        """
        ...


# ─────────────────────────────────────────────────────────────────────
# Reference implementation: pass-through (reads pre-computed poses)
# ─────────────────────────────────────────────────────────────────────

class PassThroughSlam:
    """Trivial SLAM backend that returns pre-loaded poses.

    Useful for testing Layer 2 against a known trajectory (e.g.,
    apple_bowl_2 with its camera_pose.txt) without running a real SLAM.

    Attaches a constant covariance to every returned pose unless the
    caller supplies per-frame covariances.
    """

    def __init__(self,
                 poses: List[np.ndarray],
                 default_cov: Optional[np.ndarray] = None,
                 per_frame_cov: Optional[List[np.ndarray]] = None):
        self._poses = [np.asarray(p, dtype=np.float64) for p in poses]
        self._frame_idx = 0
        if default_cov is None:
            # Conservative: ~1cm translation, ~0.5° rotation
            default_cov = np.diag([1e-4] * 3 + [1e-4] * 3)
        self._default_cov = np.asarray(default_cov, dtype=np.float64)
        self._per_frame_cov = per_frame_cov

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             odom_prior: Optional[PoseEstimate] = None,
             ) -> PoseEstimate:
        T = self._poses[self._frame_idx]
        cov = (self._per_frame_cov[self._frame_idx]
               if self._per_frame_cov is not None
               else self._default_cov)
        self._frame_idx += 1
        return PoseEstimate(T=T.copy(), cov=cov.copy())

    def reset(self, frame_idx: int = 0):
        self._frame_idx = frame_idx


class ParticlePassThroughSlam:
    """Particle-filter SLAM backend (reference implementation).

    Takes a pre-computed sequence of `ParticlePose` objects — e.g., direct
    AMCL particle dumps — and returns them in order. Useful for testing
    the Layer 2 pipeline under a particle representation.

    The companion `PassThroughSlam` does the same for Gaussians.
    """

    def __init__(self, particle_poses: List[ParticlePose]):
        self._particle_poses = list(particle_poses)
        self._frame_idx = 0

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             odom_prior: Optional[PoseEstimate] = None,
             ) -> ParticlePose:
        pp = self._particle_poses[self._frame_idx]
        self._frame_idx += 1
        return pp

    def reset(self, frame_idx: int = 0):
        self._frame_idx = frame_idx
