"""Load per-frame AMCL particle clouds produced by rosbag2dataset_5hz.py.

On disk each frame has one file:

    <dataset>/particles/particles_NNNNNN.npy    # (N, 4) float32

where each row is ``[x, y, yaw, weight]`` in the ``map`` frame. Weights are
uniform (AMCL's PoseArray strips them) unless a newer ParticleCloud source
was used during extraction.

The RBPF orchestrator in ``pose_update.state.slam_interface.ParticlePose`` wants
particles as (N, 4, 4) SE(3) matrices + (N,) weights. This module converts.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def particles_path(dataset_root: str, frame_idx: int) -> str:
    return os.path.join(
        dataset_root, "particles", f"particles_{int(frame_idx):06d}.npy")


def xyyaw_to_SE3(xyyaw: np.ndarray) -> np.ndarray:
    """(N, 3) [x, y, yaw] → (N, 4, 4) SE(3) with z=0, roll=pitch=0."""
    n = xyyaw.shape[0]
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c = np.cos(xyyaw[:, 2])
    s = np.sin(xyyaw[:, 2])
    T[:, 0, 0] = c
    T[:, 0, 1] = -s
    T[:, 1, 0] = s
    T[:, 1, 1] = c
    T[:, 0, 3] = xyyaw[:, 0]
    T[:, 1, 3] = xyyaw[:, 1]
    return T


def load_particle_pose(dataset_root: str, frame_idx: int):
    """Return a ``ParticlePose`` for the given frame, or ``None`` if missing.

    Kept as a lazy import of ``ParticlePose`` so this helper is usable from
    scripts that don't otherwise depend on the pose_update package.
    """
    p = particles_path(dataset_root, frame_idx)
    if not os.path.isfile(p):
        return None
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
        return None

    from pose_update.state.slam_interface import ParticlePose  # lazy
    particles = xyyaw_to_SE3(arr[:, :3])
    weights = arr[:, 3].astype(np.float64)
    return ParticlePose(particles=particles, weights=weights)


def load_particles_raw(dataset_root: str,
                       frame_idx: int) -> Optional[np.ndarray]:
    """Return the raw (N, 4) [x, y, yaw, weight] array, or None."""
    p = particles_path(dataset_root, frame_idx)
    if not os.path.isfile(p):
        return None
    return np.load(p)
