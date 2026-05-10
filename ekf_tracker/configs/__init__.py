""":func:`load_config` (with ``_extends:`` deep-merge) plus ``to_*_config`` and subsystem-kwargs builders."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent

# ekf_tracker/configs/__init__.py  ->  ekf_tracker/configs/  ->  ekf_tracker/  ->  repo root
_REPO_ROOT = _THIS_FILE.parent.parent.parent

#: Path to the canonical default YAML.
DEFAULT_PATH: Path = _THIS_DIR / "default.yaml"


def _resolve(p: Union[str, Path]) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    rooted = (_REPO_ROOT / pp).resolve()
    if rooted.exists():
        return rooted
    return pp.resolve()


# ─────────────────────────────────────────────────────────────────────
# Strict key access
# ─────────────────────────────────────────────────────────────────────

def _strict_get(d: Mapping[str, Any], *path: str) -> Any:
    """Fetch ``key`` from a dict, raising :class:`KeyError` if missing (no defaults)."""
    cur: Any = d
    walked: List[str] = []
    for seg in path:
        if not isinstance(cur, Mapping):
            raise KeyError(
                f"missing config key: {'.'.join(walked + [seg])} "
                f"(parent at {'.'.join(walked) or '<root>'} is not a mapping)")
        if seg not in cur:
            raise KeyError(
                f"missing config key: {'.'.join(walked + [seg])}")
        cur = cur[seg]
        walked.append(seg)
    return cur


class _Section:
    """Wraps a dict subsection so ``_strict_get`` raises with a helpful path."""
    __slots__ = ("_cfg", "_root")

    def __init__(self, cfg: Mapping[str, Any], *root: str) -> None:
        self._cfg = cfg
        self._root = root
        # Validate root path eagerly so missing root raises a precise error.
        _strict_get(cfg, *root)

    def __call__(self, *sub: str) -> Any:
        return _strict_get(self._cfg, *self._root, *sub)

    def section(self, *sub: str) -> "_Section":
        return _Section(self._cfg, *self._root, *sub)


def _diag_from_list(path_label: str, lst: Any, n: int = 6) -> np.ndarray:
    """Validate a length-``n`` list and lift to ``np.diag``."""
    arr = np.asarray(lst, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(
            f"{path_label}: expected a length-{n} vector, got shape {arr.shape}")
    return np.diag(arr)


def _vec_from_list(path_label: str, lst: Any, n: Optional[int] = None) -> np.ndarray:
    """Validate a vector list. If ``n`` supplied, also enforces length."""
    arr = np.asarray(lst, dtype=np.float64)
    if arr.ndim != 1 or (n is not None and arr.size != n):
        raise ValueError(
            f"{path_label}: expected a length-{n} vector, got shape {arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────────────
# Deep-merge with `_extends:` support
# ─────────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load a YAML config, recursively applying ``_extends:`` deep-merge."""
    target = _resolve(path) if path is not None else DEFAULT_PATH
    with open(target, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(
            f"YAML at {target} must be a mapping at the top level, got "
            f"{type(cfg).__name__}.")
    parent_path = cfg.pop("_extends", None)
    if parent_path is not None:
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


# ─────────────────────────────────────────────────────────────────────
# BernoulliConfig + TriggerConfig (Tier 1)
# ─────────────────────────────────────────────────────────────────────

# These BernoulliConfig fields are 6-vector lists in YAML and must be
# lifted into specific ndarray shapes.
_BERNOULLI_DIAG_FIELDS = ("P_max",)        # 6-vector -> (6,6) diag
_BERNOULLI_VECTOR_FIELDS = ("P_min_diag",)  # 6-vector -> (6,) vec


def to_bernoulli_config(
    cfg: Dict[str, Any],
    *,
    K: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    T_bc: Optional[np.ndarray] = None,
):
    """Build a :class:`BernoulliConfig` from a loaded dict, ``K``, and ``image_shape``."""
    from ekf_tracker.config import BernoulliConfig

    bcfg = dict(_strict_get(cfg, "bernoulli"))

    p_max_list = bcfg.pop("P_max", None)
    p_min_diag_list = bcfg.pop("P_min_diag", None)

    P_max = (_diag_from_list("bernoulli.P_max", p_max_list)
             if p_max_list is not None else None)
    P_min_diag = (_vec_from_list("bernoulli.P_min_diag", p_min_diag_list, n=6)
                  if p_min_diag_list is not None else None)

    return BernoulliConfig(
        K=K, image_shape=image_shape, T_bc=T_bc,
        P_max=P_max, P_min_diag=P_min_diag,
        **bcfg,
    )


def to_trigger_config(cfg: Dict[str, Any]):
    """Build a :class:`TriggerConfig` from the ``trigger:`` section."""
    from ekf_tracker.config import TriggerConfig
    return TriggerConfig(**dict(_strict_get(cfg, "trigger")))


# ─────────────────────────────────────────────────────────────────────
# Birth gate (BirthGateConfig from perception/birth_gating.py)
# ─────────────────────────────────────────────────────────────────────

def build_birth_gate_config(cfg: Dict[str, Any]):
    """Build the perception-side :class:`BirthGateConfig`."""
    from perception.birth_gating import BirthGateConfig
    s = _Section(cfg, "bernoulli")
    return BirthGateConfig(
        birth_min_dist_m=float(s("birth_min_dist_m")),
        held_birth_radius_m=float(s("held_birth_radius_m")),
    )


# ─────────────────────────────────────────────────────────────────────
# Process-noise schedule (utils/ekf_se3.py)
# Returned as a dict keyed by phase name; caller indexes into it.
# ─────────────────────────────────────────────────────────────────────

def build_process_noise_schedule(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-phase :math:`Q` schedule from the YAML ``process_noise`` section."""
    s = _Section(cfg, "process_noise")
    return {
        "Q_static_stable": _diag_from_list(
            "process_noise.Q_static_stable_diag", s("Q_static_stable_diag")),
        "Q_static_unstable": _diag_from_list(
            "process_noise.Q_static_unstable_diag", s("Q_static_unstable_diag")),
        "Q_idle": _diag_from_list(
            "process_noise.Q_idle_diag", s("Q_idle_diag")),
        "Q_just_released": _diag_from_list(
            "process_noise.Q_just_released_diag", s("Q_just_released_diag")),
        "Q_grasping_releasing": _diag_from_list(
            "process_noise.Q_grasping_releasing_diag",
            s("Q_grasping_releasing_diag")),
        "Q_holding_base_frame": _diag_from_list(
            "process_noise.Q_holding_base_frame_diag",
            s("Q_holding_base_frame_diag")),
        "Q_held_world_frame": _diag_from_list(
            "process_noise.Q_held_world_frame_diag",
            s("Q_held_world_frame_diag")),
        "frames_unstable_threshold": int(s("frames_unstable_threshold")),
        "frames_stable_threshold":   int(s("frames_stable_threshold")),
    }


# ─────────────────────────────────────────────────────────────────────
# Fast-tier internal noise (gaussian_ekf_tracker.py + orchestrator_gaussian.py)
# ─────────────────────────────────────────────────────────────────────

def build_fast_tier_noise_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build fast-tier centroid/rotation/tiny-cov noise constants."""
    s = _Section(cfg, "fast_tier_noise")
    std_m = float(s("centroid_r_cam_std_m"))
    return {
        "centroid_r_cam_std_m": std_m,
        "centroid_r_cam_3d":    np.diag([std_m ** 2] * 3),
        "tiny_cov": _diag_from_list(
            "fast_tier_noise.tiny_cov_diag", s("tiny_cov_diag")),
    }


# ─────────────────────────────────────────────────────────────────────
# PoseEstimator (perception/icp_pose.py)
# ─────────────────────────────────────────────────────────────────────

def build_pose_estimator_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs for :class:`perception.icp_pose.PoseEstimator`."""
    pe = _Section(cfg, "pose_estimator")
    centroid_r_diag = _vec_from_list(
        "pose_estimator.centroid_r_diag", pe("centroid_r_diag"), n=6)
    mc = pe.section("mask_clean")
    bp = pe.section("back_project")
    return {
        "voxel_size_m":           float(pe("voxel_size_m")),
        "icp_threshold_m":        float(pe("icp_threshold_m")),
        "icp_max_iter":           int(pe("icp_max_iter")),
        "min_fitness":            float(pe("min_fitness")),
        "max_rmse":               float(pe("max_rmse")),
        "trans_var_floor":        float(pe("trans_var_floor")),
        "rot_var_floor":          float(pe("rot_var_floor")),
        "centroid_r_diag":        centroid_r_diag,
        "centroid_r":             np.diag(centroid_r_diag),
        "ref_update_min_fitness": float(pe("ref_update_min_fitness")),
        "max_ref_points":         int(pe("max_ref_points")),
        "mask_clean": {
            "erosion_iter":        int(mc("erosion_iter")),
            "depth_edge_max_jump": float(mc("depth_edge_max_jump")),
            "min_depth":           float(mc("min_depth")),
            "max_depth":           float(mc("max_depth")),
            "min_points":          int(mc("min_points")),
        },
        "back_project": {
            "min_depth":  float(bp("min_depth")),
            "max_depth":  float(bp("max_depth")),
            "min_points": int(bp("min_points")),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Voxel observability (perception/voxel_observability.py)
# ─────────────────────────────────────────────────────────────────────

def build_voxel_observability_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``VoxelObservability.__init__`` kwargs."""
    vo = _Section(cfg, "voxel_observability")
    aabb = vo.section("workspace_aabb")
    return {
        "voxel_size_m": float(vo("voxel_size_m")),
        "workspace_aabb": (
            tuple(float(x) for x in aabb("min")),
            tuple(float(x) for x in aabb("max")),
        ),
        "n_min_hit":  int(vo("n_min_hit")),
        "n_min_pass": int(vo("n_min_pass")),
    }


def build_voxel_integrate_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``VoxelObservability.integrate_depth(...)`` kwargs."""
    integ = _Section(cfg, "voxel_observability", "integrate")
    return {
        "max_range_m": float(integ("max_range_m")),
        "subsample":   int(integ("subsample")),
        "min_depth_m": float(integ("min_depth_m")),
    }


# ─────────────────────────────────────────────────────────────────────
# Per-call perception kwargs
# ─────────────────────────────────────────────────────────────────────

def build_visibility_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``visibility_p_v(...)`` kwargs."""
    vis = _Section(cfg, "perception", "visibility")
    return {
        "max_samples_per_track":  int(vis("max_samples_per_track")),
        "fallback_sphere_samples": int(vis("fallback_sphere_samples")),
        "fallback_obj_radius":    float(vis("fallback_obj_radius")),
        "z_tol_abs":              float(vis("z_tol_abs")),
        "z_tol_rel":              float(vis("z_tol_rel")),
        "min_depth":              float(vis("min_depth")),
        "max_depth":              float(vis("max_depth")),
    }


def build_det_dedup_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``voxelize_mask`` (and ``suppress_subpart_detections``) kwargs."""
    dd = _Section(cfg, "perception", "det_dedup")
    return {
        "voxel_size": float(dd("voxel_size")),
        "min_depth":  float(dd("min_depth")),
        "max_depth":  float(dd("max_depth")),
    }


# ─────────────────────────────────────────────────────────────────────
# Gripper FSM + Grasp owner
# ─────────────────────────────────────────────────────────────────────

def build_gripper_phase_tracker_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``GripperPhaseTracker.__init__`` kwargs (no ``detector``)."""
    gp = _Section(cfg, "gripper_phase")
    return {
        "closed_width_m":        float(gp("closed_width_m")),
        "open_width_m":          float(gp("open_width_m")),
        "close_delta_m":         float(gp("close_delta_m")),
        "grasp_radius_m":        float(gp("grasp_radius_m")),
        "history_size":          int(gp("history_size")),
        "motion_threshold_m":    float(gp("motion_threshold_m")),
        "min_transition_frames": int(gp("min_transition_frames")),
        "min_inside_count":      int(gp("min_inside_count")),
    }


def build_grasp_owner_detector_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``GraspOwnerDetector.__init__`` kwargs (no ``gripper``)."""
    go = _Section(cfg, "grasp_owner")
    return {
        "min_inside_count":  int(go("min_inside_count")),
        "fallback_radius_m": float(go("fallback_radius_m")),
        "perception_keys":   tuple(str(s) for s in go("perception_keys")),
    }


# ─────────────────────────────────────────────────────────────────────
# Gravity-aware release predict
# ─────────────────────────────────────────────────────────────────────

def build_gravity_predict_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``predict_landing_pose`` kwargs."""
    gp = _Section(cfg, "gravity_predict")
    return {
        "gravity":           float(gp("gravity")),
        "workspace_floor_z": float(gp("workspace_floor_z")),
        "eps_roughness":     float(gp("eps_roughness")),
        "max_drop_m":        float(gp("max_drop_m")),
        # Fix A — neighbourhood-median for all_unseen.
        "r_neighbourhood_m":                  float(gp("r_neighbourhood_m")),
        "n_neighbourhood_samples":            int(gp("n_neighbourhood_samples")),
        "min_neighbour_surfaces_for_median":  int(gp("min_neighbour_surfaces_for_median")),
        # Fix B — visibility-based override.
        "tol_release_visible_m": float(gp("tol_release_visible_m")),
        "min_depth_m":           float(gp("min_depth_m")),
        "max_depth_m":           float(gp("max_depth_m")),
    }


# ─────────────────────────────────────────────────────────────────────
# Object dynamics (utils/object_dynamics.py)
# ─────────────────────────────────────────────────────────────────────

def _build_dynamics_property(cfg: Mapping[str, Any], *path: str):
    """Resolve one per-label dynamics property (restitution / friction / shape) with default fallback."""
    from utils.object_dynamics import ObjectDynamicsProperty
    s = _Section(cfg, *path)
    return ObjectDynamicsProperty(
        label=str(s("label")),
        e=float(s("e")),
        mu=float(s("mu")),
        shape=str(s("shape")),
        radius_m=float(s("radius_m")),
        mass_kg=float(s("mass_kg")),
    )


def build_object_dynamics_table(cfg: Dict[str, Any]
                                ) -> Tuple[Any, Mapping[str, Any], Mapping[str, float]]:
    """Return ``(default_dynamics, label_table, shape_footprint_factor)``."""
    od = _Section(cfg, "object_dynamics")
    default = _build_dynamics_property(cfg, "object_dynamics", "default")

    table_raw = od("table")
    if not isinstance(table_raw, Mapping):
        raise TypeError("object_dynamics.table must be a mapping")
    table: Dict[str, Any] = {}
    for label in table_raw.keys():
        table[str(label)] = _build_dynamics_property(
            cfg, "object_dynamics", "table", label)

    sff_raw = od("shape_footprint_factor")
    if not isinstance(sff_raw, Mapping):
        raise TypeError("object_dynamics.shape_footprint_factor must be a mapping")
    sff: Dict[str, float] = {str(k): float(v) for k, v in sff_raw.items()}

    return default, table, sff


# ─────────────────────────────────────────────────────────────────────
# Relations
# ─────────────────────────────────────────────────────────────────────

def build_relation_filter_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs for :class:`RelationFilter`."""
    rf = _Section(cfg, "relation", "filter")
    return {
        "alpha":           float(rf("alpha")),
        "threshold":       float(rf("threshold")),
        "prune_threshold": float(rf("prune_threshold")),
    }


def build_relation_trigger_config(cfg: Dict[str, Any]):
    """Build a :class:`RelationTriggerConfig`."""
    from ekf_tracker.relations.relation_utils import RelationTriggerConfig
    rt = _Section(cfg, "relation", "trigger")
    return RelationTriggerConfig(
        relation_every_n_frames=int(rt("relation_every_n_frames")),
        relation_on_grasp=bool(rt("relation_on_grasp")),
        relation_on_release=bool(rt("relation_on_release")),
        relation_on_new_object=bool(rt("relation_on_new_object")),
    )


def build_relation_orchestrator_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``RelationOrchestrator.__init__`` kwargs (no ``cache_dir``)."""
    rel = _Section(cfg, "relation")
    flt = rel.section("filter")
    llm = rel.section("llm")
    rest = rel.section("rest")
    return {
        "backend":             str(rel("backend")),
        "llm_model":           str(llm("model_name")),
        "llm_temperature":     float(llm("temperature")),
        "ema_alpha":           float(flt("alpha")),
        "ema_threshold":       float(flt("threshold")),
        "ema_prune_threshold": float(flt("prune_threshold")),
        "score_threshold":     float(rel("score_threshold")),
        "rest_server_url":     rest("server_url"),
        "trigger_cfg":         build_relation_trigger_config(cfg),
    }


def build_held_set_expansion_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``expand_held_with_relations`` kwargs."""
    return {
        "max_iters": int(_strict_get(cfg, "relation", "expand", "max_iters")),
    }


# ─────────────────────────────────────────────────────────────────────
# EkfTracker.__init__ knobs that were defaults
# ─────────────────────────────────────────────────────────────────────

def build_ekf_tracker_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Bundle every runtime kwarg dict :class:`EkfTracker` needs at construction time."""
    et = _Section(cfg, "ekf_tracker")
    img = et("image_shape")
    if not (isinstance(img, (list, tuple)) and len(img) == 2):
        raise TypeError("ekf_tracker.image_shape must be a length-2 list")
    return {
        "robot_type":          str(et("robot_type")),
        "pose_method":         str(et("pose_method")),
        "image_shape":         (int(img[0]), int(img[1])),
        "default_owl_server":  et("default_owl_server"),
        "default_sam2_server": et("default_sam2_server"),
    }


# ─────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    "DEFAULT_PATH",
    "load_config",
    # Tier-1 dataclass builders
    "to_bernoulli_config",
    "to_trigger_config",
    # Tier-3 builders
    "build_birth_gate_config",
    "build_process_noise_schedule",
    "build_fast_tier_noise_config",
    "build_pose_estimator_kwargs",
    "build_voxel_observability_kwargs",
    "build_voxel_integrate_kwargs",
    "build_visibility_kwargs",
    "build_det_dedup_kwargs",
    "build_gripper_phase_tracker_kwargs",
    "build_grasp_owner_detector_kwargs",
    "build_gravity_predict_kwargs",
    "build_object_dynamics_table",
    "build_relation_filter_kwargs",
    "build_relation_trigger_config",
    "build_relation_orchestrator_kwargs",
    "build_held_set_expansion_kwargs",
    "build_ekf_tracker_runtime",
]
