# `ekf_tracker/configs/default.yaml`

Single source of truth — every key mirrors an in-code production default
bit-exactly. Read-only; user overrides go in
`configs/ekf_tracker/customization.yaml` (which `_extends:` this file).

Sections:

| Section | Purpose |
|---|---|
| `bernoulli` | Fast-tier `BernoulliConfig` — gates, weights, birth params. |
| `trigger` | Slow-tier `TriggerConfig` — never-fires by default. |
| `ekf_tracker` | `EkfTracker.__init__` runtime knobs. |
| `process_noise` | Per-phase `Q` schedule (7 regimes). |
| `fast_tier_noise` | Centroid std, rotation decouple, tiny-cov. |
| `pose_estimator` | ICP voxel size / threshold / fitness floors. |
| `perception` | Visibility + det-dedup defaults. |
| `voxel_observability` | Workspace AABB, voxel size, integrate kwargs. |
| `gripper_phase` | FSM thresholds. |
| `grasp_owner` | Three-tier grasp-owner detector. |
| `gravity_predict` | Release-time landing prior. |
| `object_dynamics` | Per-label restitution / friction / shape. |
| `relation` | Backend + EMA filter + trigger. |

**Loading:**

```python
from ekf_tracker.configs import load_config, to_bernoulli_config
cfg  = load_config()                              # default.yaml
bcfg = to_bernoulli_config(cfg, K=K, image_shape=(480, 640))
```

**Source:** [`ekf_tracker/configs/default.yaml`](https://github.com/ZhangHanbo/dynamic_scene_graph/blob/main/ekf_tracker/configs/default.yaml).
