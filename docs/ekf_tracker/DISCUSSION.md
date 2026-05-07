# System Design Discussion: Hierarchical Movable Object Tracking

This document captures the design discussion for the SceneRep system architecture, drawing on three key papers and converging on a final two-tier design.

## Reference Papers

Three papers were studied to inform the design:

1. **Popović et al. (2019)** — *Informative Path Planning for Active Field Mapping under Localization Uncertainty*. Gaussian Process mapping with uncertain inputs (UIs). Pose uncertainty is propagated into the GP kernel via expected covariance, and into the Rényi-entropy utility for view planning.

2. **Chebrolu et al. (2021)** — *Adaptive Robust Kernels for Non-Linear Least Squares Problems*. Generalized Barron loss with truncated partition function. Kernel shape parameter `α` is jointly estimated with the model state via alternating minimization, automatically adapting to the residual distribution per frame.

3. **Zimmerman et al. (2022)** — *Long-Term Localization using Semantic Cues in Floor Plan Maps*. Per-class semantic stability score from offline analysis; dynamic class detections actively mask LiDAR beams; hierarchical room-category priors initialize the particle filter.

### Distilled Contributions

Each paper distilled to its single transferable idea:

- **Paper 1:** Observations from uncertain poses should contribute less to the map. Inflate observation noise by upstream pose uncertainty.
- **Paper 2:** Don't pre-commit to a noise model; let the residual distribution decide. Adaptive kernel auto-detects and downweights outliers without manual thresholds.
- **Paper 3:** Not all detected objects are reliable references. Track per-object reliability and use it as a weight. Movable/dynamic detections must be excluded from the inputs of the lower localization layer.

---

## Architectural Decisions

### Strict Hierarchy: Layer 1 (SLAM) → Layer 2 (Movable Object Tracking)

**Layer 1:** An arbitrary SLAM algorithm. Interface is fixed: outputs `(T_wb, Σ_wb)` — base-to-world pose and covariance. The choice of SLAM is interchangeable as long as the interface is respected.

**Layer 2:** Movable object tracking. Maintains per-object state and a scene graph of relations. Consumes Layer 1's pose and covariance as a prior.

**Critical rule (from Paper 3):** Movable objects must NOT be inputs to Layer 1. Object detector masks are used to *remove* movable-object pixels from depth before SLAM processes the frame. This prevents the case where a "static" object (e.g., a cup) gets pushed and becomes a poisoned landmark dragging the camera pose. The data flow is:

```
RGBD frame
  │
  ├─→ Object detector → movable masks
  │       │
  │       ↓
  │   Mask out movable pixels from depth
  │       │
  │       ↓
  │   [Layer 1: SLAM]  ← sees only static structure
  │       │
  │       ↓
  │   (T_wb, Σ_wb)
  │       │
  └───────┴─→ [Layer 2: Movable object tracker]
```

The hierarchy is strictly one-way. Layer 2 does not feed back to Layer 1. This was an explicit design choice — adding feedback would close the loop but break the layer separation, and it's unnecessary because movable objects cannot improve the camera pose anyway (see "Why the EE factor doesn't help camera pose" below).

### Why the EE Factor Cannot Improve Camera Pose (Recap)

The held object's world pose is `T_wo = T_wb · T_be · T_eo`. Both the camera pose and the EE pose are computed from the same kinematic chain rooted at `T_wb`. Any error `ε` in `T_wb` affects both equally and cancels in any relative measurement.

Therefore: comparing the camera's observation of the held object against its EE-predicted position cannot correct localization error. The manipulation factor's value is **state maintenance during occlusion**, not camera pose correction.

What actually corrects camera pose: static landmarks (accumulated object clouds whose mean position averages out per-frame `ε`). This happens inside Layer 1's SLAM, not Layer 2.

---

## Uncertainty Propagation: Layer 1 → Layer 2

### Naive Forward Propagation

For an object observed at relative pose `T_bo` with covariance `Σ_bo`, the world-frame pose covariance is:

```
Σ_wo = Ad(T_bo)^{-T} · Σ_wb · Ad(T_bo)^{-1} + Σ_bo
```

This is correct for representing the uncertainty *of the estimate*, but using it naively in EKF updates causes a problem during HOLDING.

### The Shared-Error Trap

During HOLDING, both the EE-propagated prior and the camera observation share `T_wb`:

- EE prior: `T_wo^{EE} = T_wb · T_be · T_eo`, covariance dominated by `Σ_wb`
- Camera observation: `T_wo^{cam} = T_wb · T_bc · T_co^{ICP}`, covariance dominated by `Σ_wb`

A standard Kalman update assumes independent measurement noise. Plugging both into an EKF in world frame causes the optimizer to think they are independent observations and shrink the posterior covariance. **This is wrong.** No new information about `Σ_wb` has been gained.

### Fix: Fuse in Base Frame

Both quantities have a base-frame representation that does NOT involve `T_wb`:

- EE prior in base frame: `T_bo^{EE} = T_be · T_eo` — covariance is `Σ_be + Σ_eo`. Tiny.
- Camera observation in base frame: `T_bo^{cam} = T_bc · T_co^{ICP}` — covariance is `Σ_bc + Σ_co^{ICP}`.

These are genuinely independent. The EKF update happens in base frame. Then, only at the end, we project to world frame:

```
T_wo = T_wb · T_bo^{posterior}
Σ_wo = Ad(T_bo)^{-T} · Σ_wb · Ad(T_bo)^{-1} + Σ_bo^{posterior}
```

The world-frame covariance always has `Σ_wb` as a lower bound — exactly right.

### General Principle

> Whenever an observation and a prior share a common error source from a higher layer, fuse them in the frame where that common error vanishes, then transform back.

Applied to our hierarchy:

| Phase | Object | Fusion frame | Why |
|---|---|---|---|
| IDLE | static objects | world | Observations across base poses are decorrelated |
| HOLDING | held object | base | EE prior and camera obs share `T_wb` |
| HOLDING | other objects | world | Same as IDLE |
| GRASPING / RELEASING | target object | base | ICP and EE prior share `T_wb` |

This adds one line of logic to the EKF update — pick the fusion frame based on the object's relationship to the gripper.

---

## Per-Object Uncertainty Representation

Each tracked object holds a single posterior `(T_o, Σ_o)`. No separate stability score, no shape quality score — these all enter through composition of noise terms.

### Composed Observation Noise

When a new observation arrives:

```
R_eff = R_ICP + J · Σ_wb · J^T + ρ(r) · R_robust
        \____/   \____________/   \___________/
        Paper 2    Paper 1         Paper 2
        (raw)      (forwarded       (adaptive
                    SLAM unc.)       outlier)
```

Three terms compose additively:

1. **Raw ICP measurement noise** `R_ICP` — derived from ICP fitness and correspondence count.
2. **Forwarded SLAM uncertainty** `J · Σ_wb · J^T` — Paper 1's contribution, in one line. When SLAM is uncertain, observations contribute less.
3. **Adaptive robust term** — Paper 2's contribution. Residual size determines whether to trust the observation.

### Process Noise Encodes Manipulation State (Paper 3)

Between observations:

```
Σ_o(t+1) = Σ_o(t) + Q_t
```

`Q_t` depends on the object's manipulation state:

| State | `Q_t` | Reason |
|---|---|---|
| Manipulated (held / grasped / released) | huge | Object can move arbitrarily |
| Just released, not yet re-observed | moderate | Gravity, rolling possible |
| Long-term unobserved + previously stable | near zero | World doesn't move objects on its own |
| Long-term unobserved + previously unstable | small but nonzero | Slow forgetting |

**Paper 3's stability score is not stored separately.** It is encoded by the *history* of `Σ_o`. An object that has been observed many times without surprises has a tight `Σ_o` and is automatically "stable" because new observations need to be highly confident to displace it. The class-level prior from Paper 3 enters as the initial `Q_t` when the object is first seen.

### What This Collapses

| Was going to be | Now is |
|---|---|
| `pose_cov` + `pose_uncertain` boolean + stability `s` + shape quality `q` | Just `Σ_o` |
| Hypothesis layer (from Khronos) | High `Σ_o` + few observations *is* a hypothesis. Promotion = covariance shrinks. |
| Visibility check + ICP-after-release special case | Lost sight = process noise grows. Re-observe = standard EKF update. "Absence of evidence" is just no measurement. |
| Three-tier Khronos architecture | Two tiers (see below). |

---

## The Joint Pose Graph for Movable Objects

### Initial Position: Per-Object EKFs Only

The minimal design first proposed was: each object is an independent EKF, scene graph relations are downstream readouts (computed from EKF posteriors when needed), and there is no joint optimization. Cheap, simple, real-time.

This design was reconsidered for the following reasons:

1. **Cross-object inconsistency.** Per-object EKFs cannot resolve mutual inconsistencies. Two objects can drift into geometrically impossible configurations (apple floating beside bowl).

2. **Adaptive kernel underutilized.** Paper 2's `α` is computed from the residual distribution. Per-object, there are too few residuals for `α` to be meaningful. It needs a graph-level view.

3. **Paper 3's stability concept becomes redundant.** Stability is implicit in EKF history, but the explicit per-object reliability concept is lost.

4. **Scene graph relations are mere annotations.** They don't influence pose estimates, so structural physical priors (containment, support) are wasted information.

### Final Design: Joint Pose Graph + EKF Inner Loop

All movable foreground objects compose a single pose graph that is jointly optimized. **The camera pose is not a variable** — it is a fixed parameter with associated covariance from Layer 1. This preserves the strict hierarchy.

**Variables:** `{T_o^k}` — one SE(3) pose per tracked movable object.

**Factors:**

1. **Object observation factors.** From sequential ICP / detection observations. Each factor uses the composed noise `R_eff` (raw + forwarded SLAM unc. + adaptive robust).

2. **Object prior factors.** From the per-object EKF posteriors. These are the Bayesian priors that the joint optimization sharpens.

3. **Scene graph relation factors.** Containment, support, contact. Each factor encodes the geometric structure of the relation. **Wrapped in adaptive robust kernels** so wrong relations are auto-downweighted via residual statistics.

4. **Manipulation factors.** During HOLDING, the held object is rigidly attached to the EE pose with appropriate noise (matching base localization uncertainty per the EE-shares-error analysis, not kinematic precision). During HOLDING, scene graph children of the held object propagate via their containment/support factors automatically — no special-case code needed.

### What This Activates

| Capability | Per-object EKF | Joint pose graph |
|---|---|---|
| Cross-object consistency through camera pose | no | yes |
| Cross-object consistency through physical relations | no | yes |
| Recovery from wrong observation of one object using another | no | yes |
| Manipulation-aware propagation | hard-coded | falls out of optimization |
| Adaptive outlier rejection (Paper 2) | weak (too few residuals) | strong (graph-level residual stats) |
| Held object benefits from scene graph during transport | no | yes |

### Trade-offs Accepted

1. **Wrong scene graph relations could pull poses.** Mitigated by Paper 2's adaptive kernel — large residuals from wrong relations trigger automatic downweighting.

2. **Computational cost.** Mitigated by:
   - **Incremental optimization (iSAM2-style).** Most objects don't have new observations any given frame; their factors are unchanged. Only re-solve what's affected.
   - **Bounded active set.** An object unobserved for many frames with small covariance becomes a fixed prior, not a free variable. Active set = recently-observed objects + manipulated object + their scene-graph neighbors.

3. **No camera pose correction from objects.** This is a *feature*, not a bug. It preserves the strict hierarchy and reflects the reality that movable objects share localization error with the camera.

---

## Two-Tier Architecture

The fast EKF and the joint pose graph compose, not compete:

### Inner Loop (per frame)

Per-object EKF update:
- For each object observed in this frame:
  - Compute `R_eff` (composed noise terms)
  - For held / grasped / released objects: fuse in base frame, then project to world
  - For other observed objects: fuse in world frame
  - Update `(T_o, Σ_o)`
- For each unobserved object:
  - Inflate covariance by manipulation-state-dependent `Q_t`

Cheap. Real-time. No optimization. Output: refreshed `(T_o, Σ_o)` for every object.

### Outer Loop (triggered or periodic)

Joint pose graph optimization over the active set:
- Object pose variables for active objects
- Object prior factors from current EKF posteriors
- Observation factors accumulated since last optimization
- Scene graph relation factors with adaptive kernels
- Manipulation factors for held objects

**Triggers:**
- Manipulation event (grasp onset, release)
- Scene graph topology change (new relation appears, old one breaks)
- Adaptive kernel detects large residuals (drift or wrong relations)
- Periodic schedule as a safety net (e.g., every N seconds)

After optimization, posteriors are absorbed back into the EKFs as their new state. The EKFs then run forward from the optimized configuration.

### Why This Composition Works

- **EKF provides good priors.** Without it, the pose graph would have to reconstruct state from raw observations every time. With it, the pose graph starts from a well-conditioned configuration and converges quickly.
- **EKF handles real-time tracking.** Per-frame manipulation states (HOLDING, etc.) need millisecond-level response. The pose graph runs at slower cadence.
- **Pose graph enforces global consistency.** Scene graph constraints, cross-object information, structural priors — all activated only when needed.
- **Adaptive kernels work properly.** With many residuals from many factors, `α` is statistically meaningful.

---

## How the Three Papers Compose (Summary)

| Paper | Where it lives | What it does |
|---|---|---|
| Paper 1 (uncertain inputs) | Inside `R_eff` composition, in EKF and pose graph factors | Inflates observation noise by `J · Σ_wb · J^T`. Forwarded SLAM uncertainty in one line. |
| Paper 2 (adaptive kernels) | Inside the joint pose graph, wrapped around observation factors and scene graph relation factors | Auto-downweights outliers using residual statistics. Needs graph-level view to work properly — this is a key argument for the joint optimization. |
| Paper 3 (semantic stability + masking) | Two places: (a) at the input pre-processing stage, masking movable detections out of SLAM; (b) inside per-object process noise `Q_t`, with class-level priors as initial values | Movable objects don't poison Layer 1. Per-object reliability emerges from EKF history under action-conditioned process noise. |

The three papers all reduce to one mechanism: **a per-object weight that modulates how much each observation contributes**. Paper 1 says "weight by upstream uncertainty," Paper 2 says "weight by residual size," Paper 3 says "weight by stability." All three are weights on the same factor in the same optimization.

---

## Implementation Pointers (2026-04-30)

| Concept in this document | Source file |
|---|---|
| Layer 1 SLAM protocol | `pose_update/slam_interface.py` (`SlamBackend`) |
| Layer 2 movable-object tracker (Bernoulli/RBPF) | `pose_update/orchestrator.py::TwoTierOrchestrator`; Gaussian variant in `orchestrator_gaussian.py` |
| Per-object EKF on SE(3), composed observation noise | `pose_update/ekf_se3.py` (`ekf_predict`, `ekf_update`, `compose_observation_noise`, `huber_weight`, `saturate_covariance`, `pose_entropy`) |
| Base-frame fusion for held / grasping / releasing | `pose_update/ekf_se3.py::ekf_update_base_frame` |
| Phase-dependent process noise | `pose_update/ekf_se3.py::process_noise_for_phase`; phase tracked by `pose_update/gripper_state.py` |
| Beta-Bernoulli label belief | `pose_update/ekf_se3.py::update_label_belief`; state on `scene/scene_object.py::label_belief` |
| Joint pose graph (slow tier) | `pose_update/factor_graph.py::PoseGraphOptimizer` |
| Adaptive Barron robust kernel | `pose_update/adaptive_kernel.py` |
| Scene-graph relation factors ("in"/"on"/"contact") | `pose_update/factor_graph.py::relation_residual`, `RelationEdge` |
| Online relation-graph orchestration | `pose_update/relation_orchestrator.py`, `relation_utils.py` |
| Visibility predicate via depth ray-trace | `pose_update/visibility.py::visibility_p_v` |
| Bernoulli existence probability + birth gating | `pose_update/orchestrator.py`, `pose_update/birth_gating.py` |
| Pre-association voxel dedup | `pose_update/det_dedup.py` |
| Trigger policy (event + residual + periodic) | `pose_update/orchestrator.py::TriggerConfig` |

**Gap:** `data_demo.py:530` and `realtime_app.py:745` still call the legacy
`pose_update/camera_pose_refiner.py::refine_camera_pose` rather than
constructing a `TwoTierOrchestrator`. The migration is tracked separately.

## Open Questions

1. ~~**iSAM2 vs full re-solve.**~~ **DECIDED (2026-04-30):** full re-solve via
   Levenberg-Marquardt inside `pose_update/factor_graph.py::PoseGraphOptimizer`.
   The orchestrator's trigger gate keeps the outer loop infrequent enough that
   iSAM2 was not necessary.

2. ~~**Active set policy.**~~ **IMPLEMENTED:** the trigger gate in
   `TwoTierOrchestrator` recomputes when an event fires (grasp, release, new
   object, large residual) or on the periodic safety-net tick; the active set
   at trigger time is the recently-observed + manipulated tracks. Tuning is
   ongoing but the policy itself is in place.

3. ~~**Triggering frequency for the outer loop.**~~ **IMPLEMENTED:**
   `TriggerConfig` exposes the knobs — `on_grasp`, `on_release`,
   `on_new_object`, `residual_threshold` (default 0.10 m), and
   `periodic_every_n_frames` (default 90, ~3 s @ 30 Hz).

4. ~~**Scene graph relation factor parameterization.**~~ **IMPLEMENTED:** see
   `pose_update/factor_graph.py::relation_residual` for the residual
   functions for `"in"`, `"on"`, and `"contact"` relations. Online relation
   detection is in `pose_update/relation_orchestrator.py` (REST and LLM
   backends, throttled via `relation_utils.should_recompute_relations`).

5. **Recovery from EKF / pose graph divergence.** Still open. The pose-graph
   posteriors are absorbed unconditionally; there is no automated detector
   for "the two tiers disagree significantly" and no recovery path beyond
   accepting the optimizer's output.
