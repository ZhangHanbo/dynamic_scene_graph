"""Birth-gate proximity check.

Suppress duplicate births of an object that is already tracked: when a
new perception detection's world-frame centroid lies within
``birth_min_dist_m`` of a SAME-LABEL live track, reject the birth. For
the currently-held oid (if any) the gate switches to a wider
``held_birth_radius_m`` and adds a SECONDARY anchor (the proprio-
derived gripper position ``T_we``) so a held track whose ``μ_w`` has
drifted from the EE doesn't admit ghost duplicates.

The check is non-fatal — the caller decides what to do (typically
append a ``birth_rejects`` record and continue). Returning a dict of
diagnostic fields lets the rejection record carry the offending oid,
distance, gate, and which anchor matched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np


@dataclass
class BirthGateConfig:
    """Tuning knobs for :func:`is_near_live_track`.

    Set both radii to <= 0 to disable the gate entirely.
    """
    birth_min_dist_m: float = 0.05    # default same-label proximity
    held_birth_radius_m: float = 0.25  # wider radius for the held oid


class BirthGateTrackerLike(Protocol):
    """Subset of tracker state the gate consults.

    The tracker must expose ``object_labels`` (oid → label dict) and a
    ``state`` attribute with ``collapsed_object_base(oid) → object with
    .T (4x4 base-frame mean)``. This matches the existing
    ``InstrumentedTracker``/orchestrator structure where the EKF state
    lives on a substate object.
    """
    object_labels: Dict[int, str]
    state: Any  # has .collapsed_object_base(oid)


def is_near_live_track(
    det: Dict[str, Any],
    *,
    tracker: BirthGateTrackerLike,
    T_wb: np.ndarray,
    T_bc: np.ndarray,
    held_oid_now: Optional[int],
    held_T_we_now: Optional[np.ndarray],
    cfg: BirthGateConfig,
) -> Optional[Dict[str, Any]]:
    """Return ``{nearest_oid, dist_m, gate_m, anchor}`` if the
    detection lies within the proximity gate of a same-label live
    track; otherwise ``None``.

    ``det`` must carry a ``_centroid_cam`` field (camera-frame
    back-projected centroid) and a ``label`` field.

    For the held oid the comparison runs in two anchors (the EKF
    mean ``mu_w`` and the gripper proprio anchor ``T_we``) and
    returns the first hit.
    """
    default_gate = float(getattr(cfg, "birth_min_dist_m", 0.0))
    held_gate = float(getattr(cfg, "held_birth_radius_m", default_gate))
    if ((default_gate <= 0.0 and held_gate <= 0.0)
            or not tracker.object_labels):
        return None
    c_cam = det.get("_centroid_cam")
    if c_cam is None:
        return None
    if T_wb is None or T_bc is None:
        return None
    c_h = np.array([float(c_cam[0]), float(c_cam[1]),
                     float(c_cam[2]), 1.0], dtype=np.float64)
    c_world = (T_wb @ T_bc @ c_h)[:3]
    cand_label = det.get("label")

    for oid, lbl in tracker.object_labels.items():
        if cand_label is not None and lbl != cand_label:
            continue
        pe = tracker.state.collapsed_object_base(oid)
        if pe is None:
            continue
        mu_b = np.asarray(pe.T, dtype=np.float64)[:3, 3]
        mu_w = (T_wb @ np.append(mu_b, 1.0))[:3]
        if oid == held_oid_now:
            if held_gate <= 0.0:
                continue
            d_mu = float(np.linalg.norm(c_world - mu_w))
            if d_mu <= held_gate:
                return {"nearest_oid": int(oid), "dist_m": d_mu,
                         "gate_m": held_gate, "anchor": "mu_w"}
            if held_T_we_now is not None:
                we_t = held_T_we_now[:3, 3]
                d_we = float(np.linalg.norm(c_world - we_t))
                if d_we <= held_gate:
                    return {"nearest_oid": int(oid), "dist_m": d_we,
                             "gate_m": held_gate, "anchor": "T_we"}
        else:
            if default_gate <= 0.0:
                continue
            d_mu = float(np.linalg.norm(c_world - mu_w))
            if d_mu <= default_gate:
                return {"nearest_oid": int(oid), "dist_m": d_mu,
                         "gate_m": default_gate, "anchor": "mu_w"}
    return None
