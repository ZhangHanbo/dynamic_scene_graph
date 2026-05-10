# Configuration

Every knob lives in one of two YAML files:

| File | Role |
|---|---|
| `ekf_tracker/configs/default.yaml` | Canonical defaults — read-only. |
| `configs/ekf_tracker/customization.yaml` | Your overrides; `_extends:` the default and deep-merges. |

Missing keys hard-error at load time. No env vars, no constructor fallbacks.

```python
from ekf_tracker.configs import load_config, to_bernoulli_config, to_trigger_config

cfg  = load_config("configs/ekf_tracker/customization.yaml")
bcfg = to_bernoulli_config(cfg, K=K, image_shape=(480, 640))
tcfg = to_trigger_config(cfg)
```

`_extends:` may chain; child values overlay parent values via deep merge
(scalars and lists replace, dicts merge recursively).

---

## Quick start — the params you'll touch

The defaults are tuned for a Fetch on tabletop manipulation. The fields below
are the ones to change first when porting to a new setup.

| Section · key | Default | What it controls |
|---|---|---|
| `ekf_tracker.robot_type` | `fetch` | URDF lookup key for gripper geometry. Set to your robot. |
| `ekf_tracker.image_shape` | `[480, 640]` | $(H, W)$ for the in-frame border birth gate. |
| `bernoulli.p_d` | `0.9` | Detection probability. Drop if your detector misses a lot. |
| `bernoulli.gate_mode` | `trans` | `trans` / `full` / `trans_and_rot` — Mahalanobis gate dimension. |
| `bernoulli.birth_score_min` | `0.20` | OWL score floor for admitting a candidate. |
| `bernoulli.birth_fitness_min` | `0.5` | ICP fitness floor for birth. |
| `bernoulli.birth_rmse_max` | `0.02` | ICP RMSE ceiling (m) for birth. |
| `bernoulli.P_min_diag` | `[2.5e-5×3, 2.5e-3×3]` | Per-axis $\Sigma_{bo}$ floor — perception-jitter floor. |
| `pose_estimator.voxel_size_m` | `0.005` | ICP voxel size. Smaller = more accurate, slower. |
| `pose_estimator.icp_threshold_m` | `0.020` | ICP correspondence radius. |
| `voxel_observability.workspace_aabb` | `[-2.5..2.5, …]` | Workspace bounding box for the observability grid. |
| `voxel_observability.voxel_size_m` | `0.05` | Cell size of the observability grid. |
| `gripper_phase.closed_width_m` | `0.025` | Gap below which the gripper FSM declares "closed". |
| `gripper_phase.open_width_m` | `0.040` | Gap above which the FSM declares "open". |
| `relation.backend` | `llm` | `llm` / `rest` / `none`. Use `none` for offline runs. |
| `trigger.periodic_every_n_frames` | `-1` (off) | Set to a positive int to fire the slow tier every $N$ frames. |
| `trigger.on_grasp` / `on_release` / `on_new_object` | `false` | Slow-tier triggers; flip to `true` to use the GTSAM smoother. |

---

## Detailed reference

### `bernoulli` — fast-tier Bernoulli existence model + EKF gates

| Key | Default | Description |
|---|---|---|
| `association_mode` | `hungarian` | Association style: `hungarian` (production) or `oracle` (GT eval). |
| `p_s` | `1.0` | Per-frame survival probability. |
| `p_d` | `0.9` | Per-frame detection probability. |
| `alpha` | `4.4` | Adaptive-kernel shape parameter. |
| `lambda_c` | `1.0` | Clutter rate per frame (Poisson). |
| `lambda_b` | `1.0` | Birth rate per frame (Poisson). |
| `r_min` | `1.0e-3` | Floor on Bernoulli $r$ to avoid total death. |
| `G_in` / `G_out` | `12.59` / `25.0` | $\chi^2_6$ inner / outer Mahalanobis gates (full 6-DOF). |
| `G_in_trans` | `7.815` | $\chi^2_3$ inner gate (translation only). |
| `G_out_trans` | `21.108` | Outer translation gate. |
| `G_out_rot` | `21.108` | Outer rotation gate. |
| `gate_mode` | `trans` | Gate dimension: `full`, `trans`, or `trans_and_rot`. |
| `cost_d2_mode` | `sum` | Hungarian cost combining: `full`, `trans`, `sum`. |
| `max_residual_m` | `0.50` | Hard cap on world-frame translation residual (m). |
| `P_max` | `[0.0625×3, 0.617×3]` | Per-axis covariance ceiling — translation $\sigma$ 0.25 m, rotation $\sigma$ $\pi/4$ rad. |
| `P_min_diag` | `[2.5e-5×3, 2.5e-3×3]` | Per-axis covariance floor — translation $\sigma$ 5 mm, rotation $\sigma$ 0.05 rad. |
| `enable_huber` | `true` | Apply the adaptive Huber kernel. |
| `init_cov_from_R` | `false` | Initialize a new track's covariance from $R$ instead of the prior. |
| `enforce_label_match` | `false` | Block Hungarian pairs whose labels disagree. |
| `hungarian_label_penalty` | `6.0` | Cost penalty for label mismatch. |
| `hungarian_score_weight` | `2.0` | Weight on detection score in the cost. |
| `dedup_voxel_size_m` | `0.02` | Voxel size for sub-part dedup pre-Hungarian. |
| `dedup_containment_thresh` | `0.8` | IoU threshold for absorbing one mask into another. |
| `dedup_require_same_label` | `false` | Restrict dedup to same-label pairs. |
| `birth_border_margin_px` | `2` | Pixel margin from image border for birth. |
| `birth_confirm_k` | `3` | Frames a candidate must be seen before admission. |
| `birth_score_min` | `0.20` | Detection-score floor for birth. |
| `birth_fitness_min` | `0.5` | ICP fitness floor for birth. |
| `birth_rmse_max` | `0.02` | ICP RMSE ceiling for birth (m). |
| `birth_pending_ttl_frames` | `30` | Frames after which an unconfirmed candidate is forgotten. |
| `birth_min_dist_m` | `0.05` | Reject birth within this radius of a same-label live track. |
| `held_birth_radius_m` | `0.25` | Reject birth within this radius of the held object. |
| `held_meas_radius_m` | `0.25` | Within-radius gate for measurements while held. |
| `held_meas_innov_max_m` | `0.20` | Innovation cap while held (m). |
| `r_held_floor` | `0.5` | Bernoulli $r$ floor while the object is held. |
| `self_merge_trans_m` | `0.05` | Merge same-label tracks whose means are within this distance (m). |

### `trigger` — slow-tier (`PoseGraphOptimizer`) scheduling

| Key | Default | Description |
|---|---|---|
| `on_grasp` | `false` | Fire the smoother when a grasp is detected. |
| `on_release` | `false` | Fire on release. |
| `on_new_object` | `false` | Fire when a new track is admitted. |
| `periodic_every_n_frames` | `-1` | Periodic firing cadence; `-1` disables. |

### `ekf_tracker` — `EkfTracker.__init__` runtime knobs

| Key | Default | Description |
|---|---|---|
| `robot_type` | `fetch` | URDF lookup key for gripper geometry. |
| `pose_method` | `icp_chain` | ICP backend: `centroid`, `icp_chain`, or `icp_anchor`. |
| `image_shape` | `[480, 640]` | $(H, W)$. |
| `default_owl_server` | `null` | OWL-ViT server URL; usually supplied at runtime. |
| `default_sam2_server` | `null` | SAM2 server URL. |

### `process_noise` — per-phase $Q$ schedule

Each `*_diag` is a 6-vector lifted to a $6 \times 6$ diagonal $Q$.

| Key | Default | Phase / regime |
|---|---|---|
| `Q_static_stable_diag` | `[1e-8]×6` | Static object, observed for ≥ `frames_stable_threshold` frames. |
| `Q_static_unstable_diag` | `[1e-5]×6` | Static object, just observed (unstable warm-up). |
| `Q_idle_diag` | `[1e-6]×6` | Idle (no manipulation in progress). |
| `Q_just_released_diag` | `[1e-3]×6` | Frame immediately after a release (ballistic uncertainty). |
| `Q_grasping_releasing_diag` | `[0.0004×3, 0.01×3]` | During grasp-onset / release transitions. |
| `Q_holding_base_frame_diag` | `[2.5e-4×3, 1e-3×3]` | While held, base-frame storage. |
| `Q_held_world_frame_diag` | `[1e-4]×6` | While held, world-frame variant (RBPF backend). |
| `frames_unstable_threshold` | `5` | Cut-off below which a track uses the unstable schedule. |
| `frames_stable_threshold` | `50` | Frames-observed threshold for the stable schedule. |

### `fast_tier_noise` — fast-tier internal noise constants

| Key | Default | Description |
|---|---|---|
| `centroid_r_cam_std_m` | `0.02` | Default isotropic centroid noise $\sigma$ (m) in camera frame. |
| `tiny_cov_diag` | `[1e-6]×6` | Numerical floor used when a 6-DOF cov collapses to ~0. |

### `pose_estimator` — ICP backend (`perception.icp_pose.PoseEstimator`)

| Key | Default | Description |
|---|---|---|
| `voxel_size_m` | `0.005` | Down-sample voxel for ICP points. |
| `icp_threshold_m` | `0.020` | Correspondence radius. |
| `icp_max_iter` | `30` | Max ICP iterations per call. |
| `min_fitness` | `0.90` | Acceptance fitness floor. |
| `max_rmse` | `0.015` | Acceptance RMSE ceiling (m). |
| `trans_var_floor` | `4.0e-4` | Translation-variance floor on the synthesized $R$. |
| `rot_var_floor` | `1.0e-2` | Rotation-variance floor on $R$. |
| `centroid_r_diag` | `[1e-4×3, 1e-3×3]` | Diagonal of $R$ for the centroid backend. |
| `ref_update_min_fitness` | `0.95` | Fitness to update the persistent reference cloud. |
| `max_ref_points` | `4000` | Cap on reference-cloud size. |
| `mask_clean.erosion_iter` | `2` | Erosion iterations for mask cleanup. |
| `mask_clean.depth_edge_max_jump` | `0.02` | Drop mask pixels at depth jumps larger than this (m). |
| `mask_clean.min_depth` | `0.1` | Drop pixels below this depth (m). |
| `mask_clean.max_depth` | `5.0` | Drop pixels above this depth (m). |
| `mask_clean.min_points` | `30` | Reject masks with fewer valid pixels. |
| `back_project.min_depth` / `max_depth` / `min_points` | `0.1` / `5.0` / `30` | Same gates for back-projection. |

### `perception` — visibility + det-dedup

| Key | Default | Description |
|---|---|---|
| `visibility.max_samples_per_track` | `256` | Surface samples per track for the z-buffer test. |
| `visibility.fallback_sphere_samples` | `64` | Samples on a fallback sphere when the cloud is empty. |
| `visibility.fallback_obj_radius` | `0.05` | Fallback sphere radius (m). |
| `visibility.z_tol_abs` / `z_tol_rel` | `0.02` / `0.02` | Depth tolerance (absolute m, relative). |
| `visibility.min_depth` / `max_depth` | `0.1` / `10.0` | Depth gates (m). |
| `det_dedup.voxel_size` | `0.02` | Voxel for sub-part overlap dedup (m). |
| `det_dedup.min_depth` / `max_depth` | `0.1` / `5.0` | Depth gates (m). |

### `voxel_observability` — workspace voxel grid

| Key | Default | Description |
|---|---|---|
| `voxel_size_m` | `0.05` | Cell size of the grid (m). |
| `workspace_aabb.min` / `max` | `[-2.5,-2.5,-1.0]` / `[2.5,2.5,2.0]` | World-frame AABB. |
| `n_min_hit` | `2` | Hits required to mark a voxel OCCUPIED. |
| `n_min_pass` | `3` | Pass-throughs required to mark a voxel EMPTY. |
| `integrate.max_range_m` | `3.0` | Maximum ray length per integration call. |
| `integrate.subsample` | `4` | Pixel stride for ray casting. |
| `integrate.min_depth_m` | `0.05` | Below this depth a sample is dropped. |

### `gripper_phase` — gripper FSM (`utils.gripper_state.GripperPhaseTracker`)

| Key | Default | Description |
|---|---|---|
| `closed_width_m` | `0.025` | Finger-gap below which the FSM reports "closed". |
| `open_width_m` | `0.040` | Finger-gap above which it reports "open". |
| `close_delta_m` | `0.025` | Rolling-baseline deadband for "any closing → grasp". |
| `grasp_radius_m` | `0.30` | World-frame radius around the gripper for candidate grasp targets. |
| `history_size` | `5` | Width samples kept for the rolling baseline. |
| `motion_threshold_m` | `0.01` | Minimum width change to count as "moving". |
| `min_transition_frames` | `5` | Frames the FSM must hold a state before transitioning. |
| `min_inside_count` | `20` | Track-points inside the jaws before a grasp is declared. |

### `grasp_owner` — grasp-ownership detector

| Key | Default | Description |
|---|---|---|
| `min_inside_count` | `20` | Inside-jaws point threshold for geometric containment. |
| `fallback_radius_m` | `0.05` | Nearest-track fallback radius (m). |
| `perception_keys` | `["grasp_owner_pid", "is_grasped"]` | Detection-dict keys read for the perception override. |

### `gravity_predict` — release-time landing prior

| Key | Default | Description |
|---|---|---|
| `gravity` | `9.81` | $g$ (m/s²). |
| `workspace_floor_z` | `-1.0` | Floor plane $z$ (m). |
| `eps_roughness` | `5.0e-3` | Roughness $\sigma$ added to landing-pose covariance (m). |
| `max_drop_m` | `2.0` | Cap on free-fall distance considered (m). |
| `r_neighbourhood_m` | `0.30` | Radius around the predicted landing for neighbourhood-median fallback. |
| `n_neighbourhood_samples` | `8` | Angular samples on the neighbourhood circle. |
| `min_neighbour_surfaces_for_median` | `3` | Minimum hits to use the neighbourhood-median fallback. |
| `tol_release_visible_m` | `0.05` | Depth-match tolerance for "release pose still visible" override (m). |
| `min_depth_m` / `max_depth_m` | `0.05` / `5.0` | Depth gates (m). |

### `object_dynamics` — per-label physics

`default` is the fallback; `table.<label>` overrides it for a known label.

| Key | Default | Description |
|---|---|---|
| `default.label` | `default` | Identifier. |
| `default.e` | `0.40` | Coefficient of restitution. |
| `default.mu` | `0.50` | Friction coefficient. |
| `default.shape` | `irregular` | Primitive: `spherical` / `cylindrical` / `box` / `irregular`. |
| `default.radius_m` | `0.05` | Bounding radius (m). |
| `default.mass_kg` | `0.1` | Mass (kg). |
| `shape_footprint_factor.<shape>` | `0.25 / 0.50 / 0.70 / 1.00` | Multiplier on the resting footprint per primitive. |
| `table.<label>.*` | (per row) | Overrides for `apple`, `milkbox`, `cola`, `cup`, `pot`, `flowerpot`. |

### `relation` — scene-graph backend

| Key | Default | Description |
|---|---|---|
| `backend` | `llm` | `llm` / `rest` / `none`. |
| `score_threshold` | `0.5` | Edge score floor for emit. |
| `llm.model_name` | `gpt-5.1` | LLM model (e.g. `gpt-5.1`, `gpt-4o`). |
| `llm.temperature` | `0.0` | Sampling temperature. |
| `rest.server_url` | `null` | REST endpoint; runtime-overridable. |
| `filter.alpha` | `0.3` | EMA weight on the new score per re-detection. |
| `filter.threshold` | `0.5` | Smoothed-score threshold to keep an edge live. |
| `filter.prune_threshold` | `0.01` | Per-edge prune floor. |
| `trigger.relation_every_n_frames` | `90` | Periodic re-detection cadence. |
| `trigger.relation_on_grasp` | `true` | Re-detect on grasp transitions. |
| `trigger.relation_on_release` | `true` | Re-detect on release. |
| `trigger.relation_on_new_object` | `true` | Re-detect when a new track is admitted. |
| `expand.max_iters` | `8` | Iteration cap for transitive held-set closure under $\text{in}$ / $\text{on}$. |
