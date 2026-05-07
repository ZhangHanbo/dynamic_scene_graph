# Minimal-Effort Improvements for Tight Coupling

> **Implementation Status (2026-04-30):** Both improvements are implemented;
> the actual code is broader than the snippets below.
>
> * Improvement 1 (per-object EKF on SE(3)) → `pose_update/ekf_se3.py`
>   plus the `pose_cov` / `label_belief` / `pose_uncertain` fields on
>   `scene/scene_object.py`. Public functions: `se3_exp`, `se3_log`,
>   `se3_adjoint`, `ekf_predict`, `ekf_update`, `ekf_update_base_frame`,
>   `compose_observation_noise`, `pose_entropy`, `pose_is_uncertain`,
>   `process_noise_for_phase`, `update_label_belief`, `huber_weight`,
>   `saturate_covariance`.
> * Improvement 2 (joint factor graph) →
>   `pose_update/factor_graph.py::PoseGraphOptimizer` (class API — the
>   `optimize_frame()` function in the snippets below was never built).
>   Wired in via `pose_update/orchestrator.py:39, 624` and
>   `pose_update/orchestrator_gaussian.py`. **Not yet wired into
>   `data_demo.py:530` / `realtime_app.py:745`** — those still call the
>   legacy `pose_update/camera_pose_refiner.py::refine_camera_pose`.
>   That migration is tracked separately.
>
> The original proposal is preserved below as design rationale. Concrete
> claims (line numbers, function signatures, "add 2 lines at line 89")
> reflect the *plan*, not current code; consult the source for the
> canonical API.

Two improvements, designed to work with the existing codebase with minimal changes.

---

## Clarification: What Robot Actions Can and Cannot Do for Camera Pose

A critical design constraint that shapes both improvements:

**The camera pose and EE pose share the same localization error.** Both are computed from the robot's base frame: `T_cw = T_wb @ T_bc` and `T_ew = T_wb @ T_be`, where `T_wb` is the SLAM-estimated base-to-world transform. Any error `ε` in `T_wb` affects both equally. Therefore:

- During HOLDING, comparing the camera's observation of the held object against its EE-predicted position **cannot correct localization error** — the error cancels in the relative measurement.
- The manipulation factor (`BetweenFactor(EE, held_object, T_oe)`) does **not** make the held object a privileged fiducial. It has the same `ε` as the camera.

**What actually corrects camera pose:**

1. **Static object landmarks.** Accumulated object clouds average over many frames with different `ε` values, making them more accurate than any single-frame estimate. ICP against these clouds corrects the current frame's `ε`. This is what `refine_camera_pose()` does — and this is what the factor graph's observation factors do.

2. **Scene graph constraints.** "Apple IN bowl" is a world-frame geometric fact independent of the current `ε`. If localization error pushes the apple outside the bowl in projection, the constraint resists.

**What robot actions actually provide:**

- **Object tracking through occlusion.** During HOLDING, the held object may be invisible. EE propagation maintains its pose (relative to the base) through the occlusion gap. When SLAM drift is later corrected by observing static landmarks, the held object's world pose is indirectly corrected too.
- **A transported landmark (marginal benefit).** After placing, the object's new world position can be predicted from its pre-grasp position `p₁` (well-known, accumulated over many frames) plus the kinematic displacement `Δp` (precise from joint encoders): `p_new ≈ p₁ + Δp`. This prediction provides one additional constraint on the post-placement camera pose, beyond what static objects provide. However, this is quantitatively similar to having one more static object in the scene — helpful but not qualitatively different.

**Bottom line:** The EE constraint in the factor graph should use **moderate noise** (σ ≈ 0.01-0.02m, matching base localization uncertainty), not the tight noise (σ ≈ 0.001m) that would be appropriate if the EE and camera had independent error sources. Its primary value is occlusion bridging and state maintenance, not camera pose correction.

---

## Improvement 1: Probabilistic Pose and Label Tracking

> **Implemented (2026-04-30).** See `pose_update/ekf_se3.py` and
> `scene/scene_object.py`. The EKF API is broader than the snippet below;
> notably it includes per-phase process noise (`process_noise_for_phase`),
> base-frame fusion (`ekf_update_base_frame`, originally Task 3 in PLAN.md),
> covariance saturation (`saturate_covariance`), Huber inner-gate weighting
> (`huber_weight`), and a Beta-Bernoulli label-belief helper
> (`update_label_belief`). Phase tracking is owned by
> `pose_update/gripper_state.py`'s FSM.

### Problem

Every pose in the system is a single 4×4 matrix — a point estimate with no uncertainty. The only proxy is a boolean `pose_uncertain` flag. This means:

- When ICP returns a mediocre fitness (0.3 vs 0.9), the system treats both results identically.
- When the gripper closes on an object, we switch abruptly from ICP tracking to EE propagation with no blending.
- Label scores are accumulated via raw sum (`_score_sum[label] += score`), which has no probabilistic semantics — a low-confidence detection of "cup" in 20 frames outweighs a high-confidence "bowl" in 5 frames.
- The `pose_uncertain` flag is binary: once set, there's no mechanism to gradually recover confidence.

### Solution: EKF on SE(3) per object

Add a lightweight Extended Kalman Filter that maintains a **6D mean** (pose on SE(3)) and **6×6 covariance** (in the Lie algebra se(3)) for each object. The existing code structure barely changes — we add covariance alongside the existing `pose_cur`.

#### What changes in the code

**File: `scene/scene_object.py`** — Add 3 fields to `__init__`:

```python
# After self.pose_cur = pose.copy()
self.pose_cov = np.eye(6, dtype=np.float32) * 0.01  # initial uncertainty (6×6, in se(3))
self.label_belief = {}   # {label: (alpha, beta)} — Beta-Bernoulli per label
self.pose_log_det_cov = 0.0  # cached for fast entropy queries
```

That's it for the data structure. No other fields change. `pose_cur` remains the mean; `pose_cov` is the new covariance.

**New file: `pose_update/ekf_se3.py`** (~80 lines) — The entire EKF:

```python
import numpy as np
from scipy.spatial.transform import Rotation

def se3_exp(xi):
    """xi: (6,) twist [v, omega] -> (4,4) SE(3)."""
    v, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        T = np.eye(4); T[:3, 3] = v; return T
    K = hat(omega / theta)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    V = np.eye(3) + ((1 - np.cos(theta)) / theta) * K + ((theta - np.sin(theta)) / theta) * K @ K
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = V @ v
    return T

def se3_log(T):
    """(4,4) SE(3) -> (6,) twist."""
    R, t = T[:3, :3], T[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.concatenate([t, np.zeros(3)])
    K = (R - R.T) / (2 * np.sin(theta))
    omega = np.array([K[2,1], K[0,2], K[1,0]]) * theta
    K_n = K  # already normalized
    V_inv = np.eye(3) - 0.5 * theta * K_n + (1 - theta / (2 * np.tan(theta / 2))) * K_n @ K_n
    v = V_inv @ t
    return np.concatenate([v, omega])

def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def ekf_predict(pose, cov, process_noise):
    """Prediction step: pose unchanged (static model), covariance grows.
    process_noise: (6,6) or scalar."""
    if np.isscalar(process_noise):
        process_noise = np.eye(6) * process_noise
    return pose, cov + process_noise

def ekf_update_icp(pose_prior, cov_prior, T_observed, icp_fitness, icp_rmse):
    """Update step using ICP result as observation.
    
    T_observed: (4,4) pose from ICP.
    icp_fitness: [0,1] fraction of inlier correspondences.
    icp_rmse: RMS error of inlier correspondences.
    
    The observation noise is derived from ICP quality:
      - High fitness, low RMSE → small R (trust observation)
      - Low fitness, high RMSE → large R (don't trust)
    """
    # Innovation: difference in se(3) between prior and observation
    delta = se3_log(np.linalg.inv(pose_prior) @ T_observed)  # (6,)
    
    # Observation noise from ICP quality
    sigma_obs = max(0.001, icp_rmse) / max(0.01, icp_fitness)
    R = np.eye(6) * sigma_obs**2
    
    # Kalman gain
    S = cov_prior + R
    K = cov_prior @ np.linalg.inv(S)
    
    # Update
    correction = K @ delta
    pose_new = pose_prior @ se3_exp(correction)
    cov_new = (np.eye(6) - K) @ cov_prior
    
    return pose_new, cov_new

def ekf_update_ee(pose_prior, cov_prior, T_ee_world, T_oe):
    """Update during HOLDING: EE propagation maintains the object's pose
    through occlusion. The relative offset T_oe is kinematically precise,
    but the absolute position in world frame shares the base localization
    error (~1-2cm) with the camera. Use moderate noise, not near-zero."""
    T_observed = T_ee_world @ T_oe
    delta = se3_log(np.linalg.inv(pose_prior) @ T_observed)
    
    # Moderate noise: the EE-to-object offset is precise (~1mm),
    # but the EE world pose has the same localization error as the camera.
    # This prevents covariance collapse — the object remains correctable
    # when static landmarks later refine the camera/base pose.
    R = np.eye(6) * 0.01**2  # ~1cm, matching base localization uncertainty
    S = cov_prior + R
    K = cov_prior @ np.linalg.inv(S)
    
    correction = K @ delta
    pose_new = pose_prior @ se3_exp(correction)
    cov_new = (np.eye(6) - K) @ cov_prior
    
    return pose_new, cov_new

def pose_entropy(cov):
    """Scalar uncertainty: log-determinant of covariance."""
    return 0.5 * np.log(np.linalg.det(cov) + 1e-30)
```

**Modifications to existing files:**

1. **`object_pose_updater.py` / `update_obj_pose_ee()`** — Add 2 lines at the end:

```python
# Existing line 89:
obj.pose_cur = T_ew @ obj.T_oe

# ADD after:
from pose_update.ekf_se3 import ekf_update_ee
obj.pose_cur, obj.pose_cov = ekf_update_ee(
    obj.pose_cur, obj.pose_cov, T_ew, obj.T_oe)
```

2. **`camera_pose_refiner.py` / `refine_camera_pose()`** — Already returns `(T_cw_refined, bool)`. No change needed. But the caller in `data_demo.py` should propagate `result.fitness` and `result.inlier_rmse` to the EKF. Add after the ICP call (line 226):

```python
# Store ICP quality for downstream EKF updates
T_cw_refined._icp_fitness = result.fitness
T_cw_refined._icp_rmse = result.inlier_rmse
```

Actually, since `ndarray` can't have attributes, we return a tuple instead. Minimal change: the function already returns `(T_cw_refined, bool)`. Change to `(T_cw_refined, bool, fitness, rmse)`.

3. **`id_associator.py` / `associate_by_id()`** — Where existing objects get updated (line 227-243), add EKF update:

```python
# After line 227: objects[oid].add_detection(label, score)
# ADD:
from pose_update.ekf_se3 import ekf_predict
objects[oid].pose_cur, objects[oid].pose_cov = ekf_predict(
    objects[oid].pose_cur, objects[oid].pose_cov,
    process_noise=0.0001)  # small drift per frame for static objects
```

4. **`scene_object.py` / `add_detection()`** — Replace raw score sum with Beta-Bernoulli update:

```python
def add_detection(self, label: str, score: float) -> None:
    self.detections.append((label, float(score)))
    
    # Beta-Bernoulli update per label
    if label not in self.label_belief:
        self.label_belief[label] = (1.0, 1.0)  # uniform prior
    alpha, beta = self.label_belief[label]
    # score is treated as a Bernoulli observation probability
    alpha += score
    beta += (1.0 - score)
    self.label_belief[label] = (alpha, beta)
    
    # MAP label = argmax E[Beta] = argmax alpha/(alpha+beta)
    self._label = max(
        self.label_belief.items(),
        key=lambda kv: kv[1][0] / (kv[1][0] + kv[1][1])
    )[0]
```

5. **Replace `pose_uncertain` boolean with a threshold on covariance:**

```python
@property
def pose_uncertain(self):
    """Uncertain if covariance log-det exceeds threshold."""
    from pose_update.ekf_se3 import pose_entropy
    return pose_entropy(self.pose_cov) > -5.0  # tunable threshold

@pose_uncertain.setter
def pose_uncertain(self, value):
    if value:
        self.pose_cov = np.eye(6) * 0.1  # inflate uncertainty
    else:
        self.pose_cov = np.eye(6) * 0.001  # deflate
```

#### What stays the same

- `pose_cur` still exists and is the mean. All existing code that reads `obj.pose_cur` works unchanged.
- `pose_init` unchanged.
- `T_oe` unchanged.
- All TSDF fusion logic unchanged (it uses `pose_cur` and `pose_init`).
- All scene graph computation unchanged (it uses `pose_cur`).

#### What you get

- **Smooth phase transitions**: When grasping starts, the EKF naturally blends the ICP observation (high noise) with the prior. When EE propagation kicks in, it dominates because its noise is ~1e-6. No abrupt switch.
- **Quality-aware ICP integration**: A bad ICP fit (low fitness) inflates observation noise, so the EKF barely moves. A good fit snaps the pose to the observation.
- **Uncertainty for downstream use**: `pose_entropy(obj.pose_cov)` gives a scalar uncertainty that can drive: (a) skip fusion decisions, (b) re-observation triggers, (c) grasp planning confidence.
- **Calibrated label beliefs**: `alpha/(alpha+beta)` gives a proper posterior probability. Connects directly to alpha_robot's calibrated detection scores — if the detector outputs calibrated probabilities, the Beta update is exact.

#### Lines of code changed

| File | Lines changed | Nature |
|------|--------------|--------|
| `scene/scene_object.py` | ~15 added | 3 new fields, property replacement |
| `pose_update/ekf_se3.py` | ~80 new | New file, self-contained |
| `pose_update/object_pose_updater.py` | ~3 added | Call EKF after existing update |
| `scene/id_associator.py` | ~3 added | EKF predict for static objects |
| `pose_update/camera_pose_refiner.py` | ~2 changed | Return ICP fitness/RMSE |

Total: **~100 new lines**, **~5 lines modified**.

---

## Improvement 2: Factor Graph for Joint Camera-Object Optimization

> **Implemented (2026-04-30).** See `pose_update/factor_graph.py`. The
> shipping API is `class PoseGraphOptimizer` (not `optimize_frame()`), with
> dataclasses `Observation`, `RelationEdge`, `OptimizationResult` and a
> stand-alone `relation_residual()` for "in"/"on"/"contact" relations.
> Adaptive Barron robust kernels live in `pose_update/adaptive_kernel.py`
> and are wrapped around graph factors. The optimizer is consumed by the
> orchestrator slow tier (`pose_update/orchestrator.py`,
> `orchestrator_gaussian.py`), gated by `TriggerConfig` (grasp/release/
> new-object events + 0.10 m residual + 90-frame periodic safety net).
>
> **Migration gap:** `data_demo.py:530` and `realtime_app.py:745` still call
> `refine_camera_pose` — those call sites have not migrated to the
> orchestrator. Tracked separately.

### Problem

The current pipeline is sequential:

```
SLAM gives T_cw  →  ICP refines T_cw against objects  →  update object poses  →  recompute scene graph
                                                              ↑ no feedback ↑
```

Camera pose refinement (`refine_camera_pose`) treats the accumulated object clouds as ground truth, but those clouds were built with *previous* (potentially wrong) camera poses. Object poses don't feed back into camera pose estimation. Scene graph relations (apple IN bowl) encode strong geometric constraints but aren't used for pose estimation at all.

### Solution: Replace `refine_camera_pose` with a factor graph

Use GTSAM (or a minimal hand-rolled solver) to jointly optimize camera pose and object poses in a single optimization, with the scene graph providing constraint factors.

#### Why GTSAM is practical here

GTSAM is pip-installable (`pip install gtsam`), well-documented, and handles SE(3) poses natively. The factor graph for a single frame is small (1 camera pose + K object poses + O(K²) relation factors), so optimization takes <10ms.

#### The factor graph for each frame

Variables:
- `X(t)`: camera pose at frame t — `gtsam.Pose3`
- `O(k)`: object k's current world pose — `gtsam.Pose3`

Factors:

**1. Camera odometry prior** (from SLAM):
```
PriorFactor(X(t), T_cw_slam, noise_slam)
```
This anchors the camera pose to the SLAM estimate. `noise_slam` encodes SLAM confidence (~1cm translation, ~0.5° rotation).

**2. Object observation factors** (from depth + mask):
For each visible object k with mask m_k:
```
BetweenFactor(X(t), O(k), T_co_measured, noise_obs(k))
```
where `T_co_measured = inv(T_cw) @ T_o_k_from_icp` is the relative camera-to-object transform measured by ICP, and `noise_obs(k)` comes from the EKF's observation noise (Improvement 1).

**3. Object prior** (from accumulated tracking):
```
PriorFactor(O(k), obj.pose_cur, noise_from_ekf_cov)
```
This anchors each object to its current tracked pose, weighted by its EKF covariance.

**4. Scene graph constraint factors** (the new ingredient):

For each relation `A on B`:
```
CustomFactor(O(A), O(B)):
    z_A_bottom = O(A).translation().z - height_A / 2
    z_B_top = O(B).translation().z + height_B / 2
    error = z_A_bottom - z_B_top  # should be ≈ 0
    return [error] with noise σ = 0.01m
```

For each relation `A in B`:
```
CustomFactor(O(A), O(B)):
    # A's center should be inside B's bounding box
    A_in_B = inv(O(B)) @ O(A)  # A in B's local frame
    pos = A_in_B.translation()
    error = max(0, |pos.x| - B_half_x) + max(0, |pos.y| - B_half_y) + max(0, |pos.z| - B_half_z)
    return [error] with noise σ = 0.005m
```

**5. Manipulation constraint** (during HOLDING):
```
BetweenFactor(EE_pose, O(held_obj), T_oe, noise_very_tight)
```
This rigidly attaches the held object to the EE, same as the current `update_obj_pose_ee` but now as a factor that participates in joint optimization.

#### Implementation plan

**New file: `pose_update/factor_graph.py`** (~120 lines):

```python
import gtsam
import numpy as np

def optimize_frame(
    T_cw_slam: np.ndarray,         # (4,4) camera pose from SLAM
    objects: list,                  # list of SceneObject
    observations: list,            # list of (obj_id, T_co_icp, icp_fitness, icp_rmse)
    relations: dict,               # {obj_id: {"on": [...], "in": [...], ...}}
    held_obj_id: int = None,       # object in EE
    T_ew: np.ndarray = None,       # (4,4) EE-to-world
    T_oe: np.ndarray = None,       # (4,4) object-to-EE
    slam_noise_sigmas=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),  # (tx,ty,tz,rx,ry,rz)
) -> tuple:
    """Returns (T_cw_optimized, {obj_id: T_obj_optimized})."""
    
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    # Camera variable
    cam_key = gtsam.symbol('x', 0)
    T_cw_gtsam = gtsam.Pose3(T_cw_slam.astype(np.float64))
    initial.insert(cam_key, T_cw_gtsam)
    
    # Camera prior from SLAM
    slam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(slam_noise_sigmas))
    graph.addPriorPose3(cam_key, T_cw_gtsam, slam_noise)
    
    # Object variables + priors
    obj_keys = {}
    for obj in objects:
        key = gtsam.symbol('o', obj.id)
        obj_keys[obj.id] = key
        T_obj = gtsam.Pose3(obj.pose_cur.astype(np.float64))
        initial.insert(key, T_obj)
        
        # Prior from EKF covariance
        cov = getattr(obj, 'pose_cov', np.eye(6) * 0.01)
        sigmas = np.sqrt(np.diag(cov))
        obj_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        graph.addPriorPose3(key, T_obj, obj_noise)
    
    # Observation factors (camera-object relative pose from ICP)
    for obj_id, T_co, fitness, rmse in observations:
        if obj_id not in obj_keys:
            continue
        sigma_obs = max(0.001, rmse) / max(0.01, fitness)
        obs_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma_obs)
        T_co_gtsam = gtsam.Pose3(T_co.astype(np.float64))
        graph.add(gtsam.BetweenFactorPose3(
            cam_key, obj_keys[obj_id], T_co_gtsam, obs_noise))
    
    # Scene graph constraints
    if relations:
        for obj_id, rels in relations.items():
            if obj_id not in obj_keys:
                continue
            for rel_type, target_ids in rels.items():
                for tid in target_ids:
                    if tid not in obj_keys:
                        continue
                    if rel_type == "on":
                        # Vertical contact constraint
                        _add_on_factor(graph, obj_keys[obj_id], obj_keys[tid], objects)
                    elif rel_type == "in":
                        # Containment constraint
                        _add_in_factor(graph, obj_keys[obj_id], obj_keys[tid], objects)
    
    # Manipulation constraint
    # NOTE: The EE pose and camera pose share the same base-frame localization
    # error. The EE factor cannot correct camera pose — it only maintains the
    # held object's pose through occlusion. Use moderate noise matching the
    # base localization uncertainty (~1-2cm), not the kinematic precision (~1mm).
    if held_obj_id is not None and held_obj_id in obj_keys and T_ew is not None and T_oe is not None:
        ee_key = gtsam.symbol('e', 0)
        T_ew_gtsam = gtsam.Pose3(T_ew.astype(np.float64))
        initial.insert(ee_key, T_ew_gtsam)
        
        # EE pose noise matches base localization uncertainty, NOT kinematic precision
        ee_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.015)
        graph.addPriorPose3(ee_key, T_ew_gtsam, ee_noise)
        
        # Held object rigidly attached to EE — the relative offset T_oe IS
        # kinematically precise, so this between-factor is tight. The world-frame
        # uncertainty enters through the ee_key prior above.
        T_oe_gtsam = gtsam.Pose3(T_oe.astype(np.float64))
        manip_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.002)
        graph.add(gtsam.BetweenFactorPose3(
            ee_key, obj_keys[held_obj_id], T_oe_gtsam, manip_noise))
    
    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    
    # Extract results
    T_cw_opt = result.atPose3(cam_key).matrix()
    obj_poses_opt = {}
    for obj_id, key in obj_keys.items():
        obj_poses_opt[obj_id] = result.atPose3(key).matrix()
    
    return T_cw_opt.astype(np.float32), obj_poses_opt


def _add_on_factor(graph, key_above, key_below, objects):
    """A is ON B: A's bottom should touch B's top (z-axis constraint)."""
    # Implemented as a BetweenFactor with expected relative z offset
    noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
    # The relative pose between A and B should have specific z relationship
    # We approximate: A_center.z - B_center.z ≈ (height_A + height_B) / 2
    # This is encoded as a prior on the relative z translation
    # For simplicity, use a loose BetweenFactor
    graph.add(gtsam.BetweenFactorPose3(
        key_below, key_above, 
        gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0.05)),  # ~5cm above
        noise))

def _add_in_factor(graph, key_inside, key_container, objects):
    """A is IN B: A's center should be near B's center (containment)."""
    noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.02)
    graph.add(gtsam.BetweenFactorPose3(
        key_container, key_inside,
        gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0.02)),  # slightly above center
        noise))
```

**Modification to `data_demo.py`** — Replace the `refine_camera_pose` call (line 521-546) with:

```python
from pose_update.factor_graph import optimize_frame

# Collect observations: for each visible object, compute camera-object relative pose via ICP
observations = []
for m in masks:
    obj = find_object_for_mask(objects, m)
    if obj is None or obj.id == obj_id_in_ee:
        continue
    # Quick ICP to get relative camera-object transform
    T_co = np.linalg.inv(T_cw) @ obj.pose_cur
    observations.append((obj.id, T_co, 0.8, 0.005))  # placeholder fitness/rmse

T_cw_opt, obj_poses_opt = optimize_frame(
    T_cw_slam=T_cw,
    objects=objects,
    observations=observations,
    relations=relations,
    held_obj_id=obj_id_in_ee,
    T_ew=T_cw @ T_ec if obj_id_in_ee else None,
    T_oe=find_object_by_id(obj_id_in_ee, objects).T_oe if obj_id_in_ee else None,
)

T_cw = T_cw_opt
for obj in objects:
    if obj.id in obj_poses_opt:
        obj.pose_cur = obj_poses_opt[obj.id]
```

That's it. The `refine_camera_pose` function becomes one call to `optimize_frame`.

#### What stays the same

- All TSDF fusion logic — unchanged.
- Object detection and Hungarian matching — unchanged.
- Scene graph computation — unchanged (it's an input to the factor graph now).
- EE propagation during HOLDING — now a factor rather than a direct assignment, but same result.
- All visualization — unchanged.

#### What you get

- **Camera and object poses are jointly optimal**, not sequentially refined.
- **Scene graph relations constrain poses**: "apple IN bowl" forces the apple's position to be consistent with the bowl's. If the camera pose drifts, the graph constraints pull the objects back into physically consistent positions.
- **Manipulation constraints are first-class factors**, not special-case code. The held object is just another factor with very tight noise.
- **The ICP quality from Improvement 1 flows into factor weights**: bad ICP gets low-weight observation factors, so the optimizer relies more on the prior and scene graph constraints.
- **Extensible**: new constraint types (e.g., object A must be on a table surface, object B has a known size) are just new factors.

#### Lines of code changed

| File | Lines changed | Nature |
|------|--------------|--------|
| `pose_update/factor_graph.py` | ~120 new | New file, self-contained |
| `data_demo.py` | ~15 replaced | Replace `refine_camera_pose` call with `optimize_frame` |

Total: **~120 new lines**, **~15 lines modified**.

#### Dependency

```bash
pip install gtsam
```

---

## Interaction Between the Two Improvements

The two improvements are designed to compose:

```
Frame t arrives
  │
  ├─ EKF predict: inflate all object covariances (small process noise for static, none for held)
  │
  ├─ ICP per visible object: get T_co_icp, fitness, rmse
  │     └─ EKF update per object using ICP quality → updated pose_cur, pose_cov
  │
  ├─ Factor graph: jointly optimize camera + all object poses
  │     ├─ Camera prior from SLAM
  │     ├─ Object priors from EKF (pose_cur, pose_cov → noise model)
  │     ├─ Observation factors from ICP (quality → noise weight)
  │     ├─ Scene graph constraint factors
  │     └─ Manipulation factors (if holding)
  │     → Optimized T_cw, {obj.pose_cur}
  │
  ├─ EKF absorb: update covariances based on factor graph marginals
  │
  ├─ TSDF fusion with optimized T_cw (existing code, unchanged)
  │
  └─ Scene graph recomputation (existing code, unchanged)
```

The EKF provides the uncertainty estimates that the factor graph needs as noise models. The factor graph provides the jointly optimal poses that the EKF then absorbs. They form a natural two-level system: EKF handles temporal tracking, factor graph handles spatial consistency.

---

## Summary

| | Improvement 1 (EKF) | Improvement 2 (Factor Graph) |
|---|---|---|
| **New files** | `ekf_se3.py` (~80 lines) | `factor_graph.py` (~120 lines) |
| **Modified files** | 4 files, ~5 lines each | 1 file, ~15 lines |
| **New dependency** | None | `gtsam` |
| **Runtime overhead** | ~0.1ms per frame | ~5ms per frame |
| **What it fixes** | Binary uncertainty → continuous; abrupt phase transitions → smooth; raw scores → calibrated beliefs | Sequential optimization → joint; scene graph is decorative → scene graph constrains poses |
| **Can be done independently** | Yes | Yes (but better with Improvement 1) |

### What each improvement does NOT do

- **Neither improvement corrects base localization error from robot actions alone.** The EE and camera share the same `T_wb` error. Camera pose correction comes exclusively from static object landmarks (ICP against accumulated clouds) and scene graph constraints (world-frame geometric facts). The manipulation factor maintains the held object's pose through occlusion but does not provide a privileged camera correction signal.
- **The factor graph's manipulation factor is for state maintenance, not pose correction.** Its noise model (σ ≈ 0.015m) reflects base localization uncertainty, not kinematic precision. It prevents the held object from drifting during occlusion and ensures it benefits from later camera corrections indirectly.
