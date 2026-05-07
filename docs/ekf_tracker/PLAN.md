# SceneRep Improvement Plan — Condensed

> **Implementation Status (2026-04-30):** Tasks 1–7 are largely landed. The
> two-tier orchestrator runs in `pose_update/orchestrator.py` (Bernoulli/RBPF
> backend) and `pose_update/orchestrator_gaussian.py` (Gaussian backend); the
> joint pose graph is `pose_update/factor_graph.py::PoseGraphOptimizer`; the
> per-object EKF lives in `pose_update/ekf_se3.py` with state on
> `scene/scene_object.py`; the adaptive kernel is in
> `pose_update/adaptive_kernel.py`; relation factors are in
> `pose_update/factor_graph.py::relation_residual` (+ `RelationEdge`); the
> SLAM protocol is `pose_update/slam_interface.py`. Bernoulli existence
> probability and birth gating (`pose_update/birth_gating.py`) cover the
> hypothesis-layer concept. The legacy `data_demo.py:530` and
> `realtime_app.py:745` paths still call
> `pose_update/camera_pose_refiner.py::refine_camera_pose` and have not yet
> migrated to the orchestrator — tracked separately, out of scope for this doc.
>
> | Task | Implemented in |
> |---|---|
> | 1. Layer separation | `pose_update/slam_interface.py` (protocol); legacy paths still on `refine_camera_pose` |
> | 2. Per-object EKF | `pose_update/ekf_se3.py`; `scene/scene_object.py` (`pose_cov`, `label_belief`, `pose_uncertain` property) |
> | 3. Base-frame fusion | `pose_update/ekf_se3.py::ekf_update_base_frame` (folded into Task 2) |
> | 4. Joint pose graph | `pose_update/factor_graph.py::PoseGraphOptimizer` (class API; the `optimize_frame()` function originally proposed below was never built — wired via the orchestrator slow tier instead) |
> | 5. Adaptive kernel | `pose_update/adaptive_kernel.py` |
> | 6. Relation factors | `pose_update/factor_graph.py::relation_residual`, `RelationEdge`; orchestrated by `pose_update/relation_orchestrator.py` and `relation_utils.py` |
> | 7. Two-tier orchestration | `pose_update/orchestrator.py::TwoTierOrchestrator` and `orchestrator_gaussian.py` |
> | 8. Verification | `tests/` — see "Verification — Actual Test Inventory" below |
>
> The narrative below is the original design proposal, preserved for context.
> Concrete claims (file paths, function signatures, line counts) reflect the
> *plan*, not the current code. Where signatures or call sites have evolved,
> consult the source.

Decoupled, actionable tasks. Each item can be implemented independently. Ordering reflects dependency, not priority.

## Scope

Transform SceneRep from a sequential pipeline into a two-tier architecture: per-object EKF (fast) + joint pose graph over movable objects (slow). Uncertainty from Layer 1 (SLAM) propagates down; movable objects do not feed back into Layer 1.

---

## What the Pose Graph Guarantees

Two robustness properties must hold end-to-end. Both are achieved by composable mechanisms spread across Tasks 2, 4, and 5; stated here explicitly so they are not implicit in the design.

### Guarantee A — Localization uncertainty softens observations

Every factor that depends on the camera pose (observation factors, manipulation factor) uses a composed noise model:

$$\Sigma_f = R_{\text{local}} + J \Sigma_{wb} J^T$$

where `R_local` is the factor's intrinsic measurement noise (e.g., ICP fitness-based), `J` is the Jacobian of the residual w.r.t. `T_wb`, and `Σ_wb` is the SLAM pose covariance from Layer 1. When `Σ_wb` is large, `Σ_f` is inflated, the factor's IRLS weight drops, and the object pose stays near its prior rather than chasing a geometrically unreliable observation. When `Σ_wb` is small, `Σ_f ≈ R_local` and observations dominate. No threshold — continuous degradation.

**Implemented by:** Task 2 (composes the noise for EKF observation update) and Task 4 (same composition inside each factor graph observation/manipulation factor).

### Guarantee B — Sparse outliers are auto-filtered

Every factor in the pose graph is wrapped in the adaptive Barron robust kernel with truncated partition function. Alternating minimization jointly estimates the kernel shape `α` from the current residual distribution and the poses that minimize the weighted least squares. When most residuals are small (inliers) and a few are large (e.g., an object was moved by a human between observations), `α` becomes strongly negative and the outlier factors receive near-zero weight, leaving the inliers to determine the solution.

**Assumption:** outliers are a minority (Paper 2 validated up to ~30-40%). For persistent disagreement (an object actually did move), the EKF prior decays via process noise across frames and consistent re-observations eventually dominate — the system converges to the new state over multiple frames rather than being fooled in a single one.

**Implemented by:** Task 5 (the kernel module) wired into Task 4 (wraps every factor in the graph).

### Composition — Orthogonal mechanisms

The two guarantees address different failure modes and compose without interference:

| Localization | Observation residual | Behavior |
| --- | --- | --- |
| Confident | Small | Observation factor dominates, pose updates toward observation. Standard case. |
| Confident | Large | Adaptive kernel downweights. Pose stays at prior. (Outlier rejected) |
| Uncertain | Small | Observation weak due to inflated `Σ_f`. Prior dominates. |
| Uncertain | Large | Already weak factor, further downweighted by kernel. Factor effectively removed. |

**What is NOT covered:** (1) systematic majority outliers (shelf rearrangement) — handled by eventual consensus switching rather than rejection; (2) slow systematic drift with small consistent residuals — invisible to both mechanisms, requires external correction (loop closure, manual re-init); (3) wrong scene graph relations — handled by the same adaptive kernel but with thin statistics if few relations are active.

---

## Module Interfaces (Original Proposal — see source for current API)

> The signatures below are the **proposal**. Several diverged in
> implementation: `optimize_frame()` became `class PoseGraphOptimizer`;
> `ObjectEKF` became free functions in `pose_update/ekf_se3.py`;
> `ObjectHypothesis` was subsumed by Bernoulli existence probability.
> See `pose_update/` for the current source of truth.

Shared types across modules:

```python
@dataclass
class PoseEstimate:
    T: np.ndarray        # (4,4) SE(3)
    cov: np.ndarray      # (6,6) in se(3) tangent, ordering [v, omega]

@dataclass
class Observation:
    obj_id: int
    T_co: np.ndarray     # (4,4) camera-to-object from ICP
    R_icp: np.ndarray    # (6,6) ICP measurement noise
    fitness: float       # [0,1]
    rmse: float
    mask: np.ndarray     # (H,W) bool, which pixels belonged to this object

@dataclass
class ManipulationState:
    phase: str           # 'idle' | 'grasping' | 'holding' | 'releasing'
    held_obj_id: Optional[int]
    T_oe: Optional[np.ndarray]   # (4,4) object-to-EE, locked at grasp

@dataclass
class SceneGraphRelation:
    type: str            # 'on' | 'in' | 'contact'
    parent: int
    child: int
    score: float         # [0,1] detection confidence
```

### Layer 1 — SLAM (Task 1)

Interface (fixed, SLAM-backend-agnostic):

```python
class SlamBackend(Protocol):
    def step(self, rgb: np.ndarray, depth: np.ndarray,
             movable_mask: np.ndarray,    # (H,W) bool, True where movable
             odom_prior: Optional[PoseEstimate]) -> PoseEstimate:
        """Returns (T_wb, Σ_wb). Movable-masked pixels are excluded from depth."""
```

Upstream dependency: none. Downstream: Tasks 2, 4, 7 consume `PoseEstimate`.

### Per-object EKF (Task 2)

```python
class ObjectEKF:
    def predict(self, dt: float, manip: ManipulationState) -> None:
        """Inflate cov by Q_t, set by manipulation phase + object's role."""

    def update(self, obs: Observation,
               slam: PoseEstimate,
               manip: ManipulationState) -> None:
        """R_eff = obs.R_icp + J·slam.cov·J^T + ρ(r)·R_robust.
        If object is the held/grasped/releasing target: fuse in base frame (Task 3)."""

    @property
    def posterior(self) -> PoseEstimate: ...

    @property
    def label_belief(self) -> Dict[str, Tuple[float, float]]:
        """Beta-Bernoulli (α, β) per label."""
```

Depends on: `PoseEstimate`, `Observation`, `ManipulationState` from Layer 1 + detection pipeline. Supported by: current `SceneObject` fields (extended with `cov`).

### Base-frame fusion (Task 3)

Not a module — a branch inside `ObjectEKF.update()`. Needs `ManipulationState` to decide fusion frame. No new interface.

### Joint pose graph (Task 4)

```python
class PoseGraphOptimizer:
    def run(self,
            slam: PoseEstimate,                        # fixed parameter
            priors: Dict[int, PoseEstimate],           # from Task 2 EKFs
            observations: List[Observation],           # accumulated since last run
            relations: List[SceneGraphRelation],      # from Task 6
            manip: ManipulationState,
            T_ew: Optional[np.ndarray],                # if manip.held_obj_id set
            active_set: Set[int],
           ) -> Dict[int, PoseEstimate]:
        """Returns optimized posteriors for objects in active_set.
        Non-active objects keep their prior unchanged."""
```

Depends on: `gtsam`, Task 2 posteriors, Task 6 relations, adaptive kernel (Task 5). Supported by: wrapper around GTSAM; current pipeline supplies all required inputs once Tasks 2+6 land.

### Adaptive robust kernel (Task 5)

```python
class AdaptiveRobustKernel:
    def fit_alpha(self, residuals: np.ndarray) -> float:
        """Alternating minimization: given residuals, find α that maximizes likelihood
        under truncated Barron loss. Range [-10, 2]."""

    def weight(self, residual: float, alpha: float) -> float:
        """Returns per-residual IRLS weight for the current α."""
```

Depends on: nothing external. Used by: Task 4's factor graph wraps each observation/relation factor in this kernel.

### Scene graph relation factors (Task 6)

```python
class RelationFactor(Protocol):
    """Implemented per relation type (on, in, contact)."""
    def residual(self, T_parent: np.ndarray, T_child: np.ndarray,
                 shape_parent: BoundingBox,
                 shape_child: BoundingBox) -> np.ndarray:
        """Residual vector. Zero when relation is satisfied."""

    def noise(self, score: float,
              cov_parent: np.ndarray, cov_child: np.ndarray) -> np.ndarray:
        """Noise covariance composed from detection score + pose uncertainties."""
```

Also modifies `object_relation_graph.compute_spatial_relations()` to return `List[SceneGraphRelation]` with soft scores instead of boolean relations. Depends on: Task 2 posteriors (for `cov_*`). Supported by: existing bbox computation in `object_relation_graph.py`.

### Two-tier orchestrator (Task 7)

Not a new module — the main loop in `data_demo.py` / `realtime_app.py`. Responsibilities:

```python
# Per frame:
slam = slam_backend.step(rgb, depth, movable_mask, odom_prior)
manip = manip_state_machine.update(finger_d, T_ec, tracker.objects)
for obj_id in observed_this_frame:
    ekfs[obj_id].predict(dt, manip)
    ekfs[obj_id].update(observations[obj_id], slam, manip)
for obj_id in unobserved_this_frame:
    ekfs[obj_id].predict(dt, manip)

# On trigger:
if trigger_condition(manip, residuals, frame_count):
    active = select_active_set(ekfs, manip, relations)
    posteriors = pose_graph.run(slam, {i: ekfs[i].posterior for i in active},
                                 pending_observations, relations, manip, T_ew, active)
    for i, p in posteriors.items():
        ekfs[i].set_posterior(p)
```

Depends on: all other tasks. Currently: main loops exist and supply most inputs; manipulation state machine already present; missing piece is the EKF and pose graph substitutions.

---

## Compatibility with existing modules

| Needed input | Source in current code | Task that exposes it cleanly |
|---|---|---|
| `T_wb`, `Σ_wb` | Currently raw `T_cw` from rosbag, no covariance | Task 1 wraps SLAM behind the interface; covariance may need a placeholder until the chosen SLAM reports it |
| movable mask | `masks` list in `associate_by_id` | Task 1 collects before SLAM call |
| Observation | Partially in `associate_by_id`'s ICP path | Task 2 formalizes the dataclass |
| Manipulation state | Gripper state machine in `data_demo.py` | Already exists, just needs exposure |
| BoundingBox per object | `object_relation_graph.py` computes it inline | Task 6 promotes to a shared type |
| `T_ew`, `T_oe` | `update_obj_pose_ee()` uses them | Already available |

No blocking gaps. Covariance from the existing SLAM (AMCL / rosbag poses) is the only missing input — can be stubbed with a reasonable constant and upgraded later.

---

## Minimality Audit Against Existing Code

Cross-checked each task against the current implementation. Summary: **the originally stated effort estimates hold**, but several tasks are even smaller than listed because the existing code already has most structural support. One item (Task 3) is subsumed by Task 2 and not a separate module.

### Task 1 — Layer separation

- `data_demo.py:678–703` already builds `depth_bg, color_bg` with movable-object masks zeroed out. This is effectively the movable-mask exclusion we need, but it happens *after* pose consumption, not before.
- **Minimal change:** move the mask-out step to before any pose computation; wrap SLAM behind a `SlamBackend` protocol; drop or restrict `refine_camera_pose()`.
- **Revised effort:** ~20 lines, not 30.

### Task 2 — Per-object EKF

- `SceneObject` at `scene/scene_object.py:73–124` already has `pose_init`, `pose_cur`, `T_oe`, `pose_uncertain`, `_score_sum`. Adding `pose_cov` is literally one new field.
- The EKF update call site is `update_obj_pose_ee()` + `update_obj_pose_icp()` + the integrate branch inside `associate_by_id()`. All three already receive enough information to compute covariance.
- **Minimal change:** new file `pose_update/ekf_se3.py`, add `pose_cov` field, call `ekf_predict`/`ekf_update` at the existing update sites. `pose_uncertain` stays as a derived property.
- **Revised effort:** ~90 new lines + ~10 modified (down from 100+15).

### Task 3 — Base-frame fusion

- This is a one-branch decision inside Task 2's `ekf_update`. It is not a separable task.
- **Revised plan:** merge into Task 2. Remove Task 3 as a standalone item.

### Task 4 — Joint pose graph

- `camera_pose_refiner.py` already does partial joint reasoning via ICP against multi-object clouds. It's 337 lines of the closest-existing approximation to what we want.
- **Decision:** replace `refine_camera_pose()` wholesale with the new `PoseGraphOptimizer`, rather than extending it. The old function's source stays as reference.
- **Revised effort:** ~150 new lines, ~15 modified (unchanged).

### Task 5 — Adaptive robust kernel

- No existing approximation in the codebase. Entirely new.
- **Unchanged:** ~60 new lines.

### Task 6 — Relation factors

- `object_relation_graph.py:29–163` (`compute_spatial_relations`) and `335–361` (`get_relation_graph`) already compute bounding boxes, overlap ratios, and return per-relation structure. Currently discards the soft scores by thresholding.
- **Minimal change:** expose the overlap ratio as a `score` field on the returned relations; add factor classes in the new `factor_graph.py`.
- **Revised effort:** ~60 new lines in factor graph + ~20 modified in relation graph (down from 80+40).

### Task 7 — Orchestration

- `data_demo.py` main loop at lines 305+ already has the state machine and per-frame structure. The integration is substitution of call sites, not restructuring.
- **Unchanged:** ~40 lines modified.

### Net revised budget

| Task | New lines | Modified lines | Notes |
|---|---|---|---|
| 1. Layer separation | 0 | ~20 | Reuses existing mask-out logic |
| 2. Per-object EKF (absorbs Task 3) | ~90 | ~10 | Reuses `SceneObject` fields |
| 4. Joint pose graph | ~150 | ~15 | Replaces `refine_camera_pose` |
| 5. Adaptive kernel | ~60 | 0 | New |
| 6. Relation factors | ~60 | ~20 | Extends existing `compute_spatial_relations` |
| 7. Orchestration | 0 | ~40 | Substitute call sites |
| 8. Verification | ~230 | 0 | Tests + visualization |

**Total:** ~590 new lines, ~105 modified (down from 640 + 140). Task 3 no longer standalone; Tasks 1 and 6 smaller than originally estimated.

---

## Verification — Actual Test Inventory

Verification grounds in the four trajectories shipped under
`tests/visualization_pipeline/`: `apple_drop`, `apple_in_the_tray`,
`apple_to_cabinate`, `apple_to_cabinate_look`. The headline driver is
`tests/visualize_ekf_tracking.py --trajectory <name>`; its outputs are PNG
panels, MP4 timelines, per-frame state JSONs, and an aggregate `REPORT.md`
under each trajectory directory.

### Module-level tests (tests/)

The originally-proposed module test files were built (with minor renames):

| Originally proposed | Actual file |
|---|---|
| `tests/test_ekf_se3.py` (Task 2) | `tests/test_ekf_se3.py` |
| `tests/test_layer1_interface.py` (Task 1) | `tests/test_slam_interface.py` |
| `tests/test_pose_graph.py` (Task 4) | `tests/test_factor_graph.py` |
| `tests/test_adaptive_kernel.py` (Task 5) | `tests/test_adaptive_kernel.py` |
| `tests/test_relation_factors.py` (Task 6) | `tests/test_relation_scores.py` + `tests/test_relation_trigger.py` |
| `tests/test_orchestration.py` (Task 7) | `tests/test_orchestrator.py` + `tests/test_orchestrator_integration.py` |

Additional test surfaces that were not foreseen in the original plan and
landed alongside the orchestrator/Bernoulli rewrite:

* `tests/test_api.py` — public consumer API smoke tests.
* `tests/test_bernoulli_degeneracy.py` — Bernoulli existence probability degenerate cases.
* `tests/test_mask_cloud_filter.py` — pre-SLAM movable-pixel mask-out.
* `tests/test_particle_pose.py` — particle-filter pose maintenance.
* `tests/test_rbpf_state.py` — Rao-Blackwellized PF factorization.
* `tests/test_three_backends.py` — Gaussian / RBPF / oracle parity sweep.

### Unit suite (tests/unit/)

Eleven focused unit tests covering algorithmic primitives:

* `test_association.py` — Hungarian gate decomposition, soft label-purity gate, `max_residual_m` Euclidean cap.
* `test_centroid_update_psd.py` — covariance PSD invariant on centroid-only updates.
* `test_gaussian_state.py` — Gaussian-backend Layer-2 state.
* `test_grasp_owner.py` — gripper-geometry-based grasp-owner inference.
* `test_gripper_state.py` — gripper FSM `{idle, grasping, holding, releasing}`.
* `test_orchestrator_release.py` — phase gate for rigid-attachment predict during `releasing`.
* `test_rbpf_state.py` — RBPF state factorization.
* `test_relation_orchestrator.py` — online relation-graph throttling/EMA.
* `test_self_merge_protected.py` — post-update Euclidean self-merge with same-label gate.
* `test_visibility.py` — depth-ray-trace `visibility_p_v`.
* `test_visualize_pid_to_oid.py` — visualizer pid→oid mapping.

Run the unit suite with:

```bash
conda run -n ocmp_test python -m pytest tests/unit -q
```

### End-to-end visualization

`tests/visualize_ekf_tracking.py` drives the full Bernoulli-EKF pipeline on a
trajectory and renders per-frame and aggregate views. Key CLI flags:

* `--trajectory {apple_drop|apple_in_the_tray|apple_to_cabinate|apple_to_cabinate_look}`
* `--max-frame N` (exclusive upper bound)
* `--start N`, `--step N` (frame range subsampling)
* `--pose-method {centroid|icp_chain|icp_anchor|icp_chain_strict|icp_anchor_strict}` (default `icp_chain`)
* `--out-subdir`, `--state-subdir`
* `--no-png`, `--no-mp4`
* `--fps` (default 10)

Per-trajectory aggregate reports are written to
`tests/visualization_pipeline/<trajectory>/REPORT.md`. A top-level
`tests/visualization_pipeline/REPORT.md` summarises match rates, PSD-min,
and self-merge counts across all four trajectories.

---

## Task 1 — Fix Layer 1/Layer 2 separation

> **Implemented in:** `pose_update/slam_interface.py` defines the
> `SlamBackend` protocol consumed by `pose_update/orchestrator.py`. The
> legacy `data_demo.py:530` and `realtime_app.py:745` paths still call
> `pose_update/camera_pose_refiner.py::refine_camera_pose` and have not
> migrated; track separately.

**Goal:** Movable objects must not poison SLAM input.

**Changes:**
- In SLAM pre-processing, mask out movable-object pixels from the depth image before passing to Layer 1.
- Remove `refine_camera_pose()`'s use of movable-object clouds as camera-pose landmarks. Either drop this refinement entirely or restrict it to static-only references.
- Define a fixed Layer 1 interface: input = masked RGBD + prior pose; output = `(T_wb, Σ_wb)`.

**Depends on:** nothing.
**Enables:** Tasks 2, 3, 5.
**Estimated effort:** ~30 lines modified in `data_demo.py` / `camera_pose_refiner.py`.

---

## Task 2 — Per-object EKF on SE(3)

> **Implemented in:** `pose_update/ekf_se3.py`. Public functions: `se3_exp`,
> `se3_log`, `se3_adjoint`, `ekf_predict`, `ekf_update`,
> `ekf_update_base_frame` (Task 3), `compose_observation_noise`,
> `pose_entropy`, `pose_is_uncertain`, `process_noise_for_phase`,
> `update_label_belief`, `huber_weight`, `saturate_covariance`. State on
> `scene/scene_object.py`: `pose_cov` (6×6), `label_belief` (Beta-Bernoulli per
> label), property-backed `pose_uncertain` derived from `pose_entropy`.

**Goal:** Replace the deterministic `pose_cur` + boolean `pose_uncertain` with a proper posterior `(T_o, Σ_o)`.

**Changes:**
- New file `pose_update/ekf_se3.py` (~80 lines): `se3_exp`, `se3_log`, `ekf_predict`, `ekf_update`, `pose_entropy`.
- Add `pose_cov` (6×6) field to `SceneObject`. Remove `pose_uncertain` boolean; replace with property derived from covariance.
- Observation noise is composed additively: `R_eff = R_ICP + J·Σ_wb·J^T + ρ(r)·R_robust`.
- Process noise `Q_t` between observations is set by manipulation state:
  - Manipulated (held/grasping/releasing): huge
  - Just released, not re-observed: moderate
  - Unobserved + stable history: near zero
  - Unobserved + unstable history: small nonzero
- Label tracking: replace raw score sum with Beta-Bernoulli update per label.

**Depends on:** nothing.
**Enables:** Tasks 3, 4, 5.
**Estimated effort:** ~100 new lines + ~15 modified across `scene_object.py`, `id_associator.py`, `object_pose_updater.py`.

---

## Task 3 — Base-frame fusion for held objects

> **Implemented in:** `pose_update/ekf_se3.py::ekf_update_base_frame`. Folded
> into Task 2 as a separate update path selected per phase.

**Goal:** Prevent Kalman overconfidence when the EE prior and the camera observation share `T_wb`.

**Changes:**
- In the EKF update, check the object's manipulation state:
  - HOLDING / GRASPING / RELEASING target: fuse in **base frame** (`T_bo^EE` vs `T_bo^cam`), then project result to world frame with proper `Σ_wb` inflation.
  - All other cases: fuse in world frame as normal.
- One branch in the EKF update call, no new files.

**Depends on:** Task 2.
**Enables:** Task 4, 5.
**Estimated effort:** ~20 lines in the EKF update path.

---

## Task 4 — Joint pose graph over movable objects

> **Implemented in:** `pose_update/factor_graph.py::PoseGraphOptimizer`
> (class API — the `optimize_frame()` function originally proposed below
> was never built). Wired in via the orchestrator slow tier:
> `pose_update/orchestrator.py:39, 624` and
> `pose_update/orchestrator_gaussian.py`. Triggered by `TriggerConfig`
> (grasp/release/new-object events + 0.10 m residual threshold + 90-frame
> periodic safety net).

**Goal:** Jointly optimize all movable object poses with scene graph relations as constraints. The camera pose is a fixed parameter, not a variable.

**Changes:**
- New file `pose_update/factor_graph.py` (~150 lines) using GTSAM.
- Variables: `{T_o^k}` for objects in the active set.
- Factors:
  - Object prior from current EKF posterior.
  - Observation factors from accumulated ICP measurements since last optimization, with `R_eff` as noise.
  - Scene graph relation factors (containment, support, contact). Wrap each in an adaptive robust kernel (Barron generalized loss).
  - Manipulation rigid-attachment factors during HOLDING, with noise matching base localization uncertainty (not kinematic precision).
- Active set policy: recently observed objects + manipulated object + their scene-graph neighbors. Others are fixed priors.
- Triggers: grasp/release event, scene-graph topology change, adaptive kernel detects large residuals, periodic safety-net.
- After optimization, write posteriors back to EKF state.

**Depends on:** Task 2.
**Enables:** Task 7.
**Estimated effort:** ~150 new lines + ~15 modified in `data_demo.py`.
**Dependency:** `pip install gtsam`.

---

## Task 5 — Adaptive robust kernel

> **Implemented in:** `pose_update/adaptive_kernel.py` (Barron generalized
> loss with truncated partition; alternating α minimization). Used inside
> `PoseGraphOptimizer`; not used in the per-object EKF (residual count too
> small for α to be meaningful).

**Goal:** Automatically downweight outlier observations and wrong scene graph relations.

**Changes:**
- New file `pose_update/adaptive_kernel.py` (~60 lines) implementing Barron generalized loss with truncated partition function.
- Alternating minimization: given current residuals, estimate `α`; given `α`, run the standard factor graph optimization.
- Used inside Task 4's factor graph (wraps observation factors and relation factors).
- Not used in the per-object EKF (too few residuals per object for `α` to be meaningful).

**Depends on:** Task 4.
**Enables:** better outlier handling end-to-end.
**Estimated effort:** ~60 new lines, integrated into `factor_graph.py`.

---

## Task 6 — Scene graph relation factors

> **Implemented in:** `pose_update/factor_graph.py::relation_residual` plus
> the `RelationEdge` dataclass. Online relation-graph orchestration
> (REST/LLM backends, throttling, EMA smoothing) is in
> `pose_update/relation_orchestrator.py`; transitive-closure expansion of
> the held set under "in"/"on" edges is in `pose_update/relation_utils.py`.

**Goal:** Turn scene graph relations from downstream annotations into first-class optimization constraints.

**Changes:**
- In `object_relation_graph.py`, replace boolean relations with soft scores derived from bounding box overlap + pose covariance.
- Each relation type gets a factor class in `factor_graph.py`:
  - Containment: A's center should lie inside B's interior volume.
  - Support: A's bottom face should be at B's top face z.
  - Contact: A and B's surfaces should be within tolerance.
- Factor noise is composed of: relation detection score, pose covariance of both objects, shape uncertainty from TSDF weight.

**Depends on:** Task 4.
**Enables:** manipulation-aware propagation falls out of optimization.
**Estimated effort:** ~80 new lines in `factor_graph.py` + ~40 modified in `object_relation_graph.py`.

---

## Task 7 — Two-tier orchestration

> **Implemented in:** `pose_update/orchestrator.py::TwoTierOrchestrator`
> (Bernoulli/RBPF backend) and `pose_update/orchestrator_gaussian.py`
> (Gaussian backend). Both share the slow tier
> (`PoseGraphOptimizer`), the trigger policy (`TriggerConfig`), and the
> SLAM protocol (`SlamBackend`). The `BernoulliConfig` dataclass exposes
> 42+ knobs covering association mode, gates, label penalties, birth
> gating, dedup, self-merge, ICP fallback, and relation triggers.
> **Not yet wired into `data_demo.py` / `realtime_app.py`** — those still
> run the legacy sequential pipeline.

**Goal:** Wire the fast EKF loop and the slow pose graph loop into a coherent pipeline.

**Changes:**
- In the main loop of `data_demo.py` / `realtime_app.py`:
  - Every frame: run per-object EKF updates for observed objects; predict-only for others.
  - On trigger: run joint pose graph optimization; write posteriors back to EKFs.
- Remove the current per-frame `refine_camera_pose()` call (handled by Task 1).
- Move ICP quality metrics (fitness, RMSE) from local variables to outputs that feed the EKF and the pose graph factors.

**Depends on:** Tasks 2, 4.
**Enables:** full system.
**Estimated effort:** ~40 lines modified in `data_demo.py`.

---

## Task 8 — Verification

**Changes:**
- Extend `tests/visualize_ekf_tracking.py` (the headline driver) to show per-object covariance ellipses in the top-down panel.
- Run on the `apple_drop`, `apple_in_the_tray`, `apple_to_cabinate`, and `apple_to_cabinate_look` trajectories with and without each task enabled. Compare:
  - Reconstruction drift during HOLDING.
  - Scene graph stability across frames.
  - Convergence time after release.
- Add unit tests for the EKF (Task 2) and the factor graph (Task 4) using the existing `tests/test_api.py` pattern. **(Done — see `tests/test_ekf_se3.py` and `tests/test_factor_graph.py`.)**

**Depends on:** all other tasks.
**Estimated effort:** ~80 lines added to visualization + ~150 lines of tests.

---

## Original Budget (Superseded by "Net revised budget" above)

| Task | New lines | Modified lines | New dep | Depends on |
|---|---|---|---|---|
| 1. Layer separation | 0 | ~30 | none | — |
| 2. Per-object EKF | ~100 | ~15 | none | — |
| 3. Base-frame fusion | ~20 | 0 | none | 2 |
| 4. Joint pose graph | ~150 | ~15 | gtsam | 2 |
| 5. Adaptive kernel | ~60 | 0 | none | 4 |
| 6. Relation factors | ~80 | ~40 | none | 4 |
| 7. Orchestration | 0 | ~40 | none | 2, 4 |
| 8. Verification | ~230 | 0 | pytest | all |

**Total:** ~640 new lines, ~140 modified, 1 new dependency (`gtsam`).

Tasks 1, 2, 3, 5, 6 are independently valuable and can land in any order respecting their dependencies. Task 4 is the largest single chunk. Task 7 is integration.
