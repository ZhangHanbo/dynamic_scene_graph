# scenerep

## Installation
```bash
conda create -n scenerep python=3.11
conda activate scenerep
cd scenerep
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
Download the checkpoints [OWLv2 CLIP B/16 ST/FT ens](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b) to `~/scenerep/scripts/rosbag2dataset/owl`.
Modify "checkpoint_path" in `~/anaconda3/envs/scenerep/lib/python3.11/site-packages/scenic/projects/owl_vit/configs/owl_v2_clip_b16.py`.

Then download the checkpoints [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `~/scenerep/scripts/rosbag2dataset/sam`.

(Optional) Install MobileSAM for realtime application: [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

## Data Processing
1. Prepare ROS bag
Bag should include: RGB-D images, TF, End-effector pose

2. Convert ROS bag to dataset
```bash
cd ~/scenerep
python scripts/rosbag2dataset/icp_amcl.py
python scripts/rosbag2dataset/rosbag2dataset_5hz.py [dataname.bag]
```

3. Run OWL-ViT object scoring & SAM segmentation
```bash
python scripts/rosbag2dataset/owl/owl_object_scores.py [dataname]
python scripts/rosbag2dataset/sam/sam.py [dataname]
```

## Trackers

This repo provides two scene-tracking variants and one reference baseline.
The choice is exposed on the runnable scripts via a `--tracker` flag.

### 1. Heuristic tracker — `heuristic_tracker/`
TSDF + Hungarian ID + ICP. Deterministic, no probabilistic state. This is
the production path used by `robi_butler` and the offline/realtime demos.

```python
from heuristic_tracker import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
)
```

When to use: offline TSDF reconstruction, real-time on Fetch over ROS,
and any consumer that needs fast deterministic pose updates with TSDF
geometry.

### 2. EKF tracker — `ekf_tracker/`
Two-tier probabilistic tracker. SLAM (Layer 1) feeds a per-object EKF on
SE(3) (Layer 2) with a Bernoulli existence model and a slow-tier factor
graph for trigger-based pose-graph refinement. Two backends:

* `TwoTierOrchestrator` — Rao-Blackwellized particle filter over the SLAM
  posterior.
* `TwoTierOrchestratorGaussian` — single-Gaussian SLAM with a base-frame
  EKF per object (the cheaper, currently-default backend).

The consumer-facing surface is `EkfTracker`, which wraps either backend
and exposes five methods:

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, backend="gaussian", T_bc=T_bc,
                     detection_server="http://127.0.0.1:8000")

dets    = tracker.detect(rgb, vocabulary=["apple", "cup"], history=None)
objs    = tracker.step(dets, rgb, depth, slam_pose, T_bc)  # {oid: EkfObject}
points  = tracker.get_points(object_id=0)                  # (N, 3) in world frame
scene   = tracker.get_scene()                              # objects + relation graph
smooth  = tracker.smooth()                                 # slow-tier on demand
```

For low-level access, the raw orchestrators (`TwoTierOrchestrator`,
`TwoTierOrchestratorGaussian`, `TriggerConfig`) are still re-exported.

When to use: research, uncertainty-quantified tracking, scenes with
frequent occlusions, anything that needs covariances on the output pose.

### 3. Visual-only baseline — `baselines/visual_only_tracker.py`
Direct ICP composition with no filter and no proprioception. For ablation
only.

```python
from baselines import VisualOnlyTracker
```

## Running

All entry points live under `scripts/`. Run every script from the repo
root (each script bootstraps `sys.path` so top-level packages resolve).

### Offline on a recorded dataset
```bash
# Heuristic (default) — TSDF + Hungarian:
python scripts/data_demo.py --config configs/demo.yaml --tracker heuristic

# EKF — dispatches to scripts/visualize_ekf_tracking.py for the named
# trajectory; produces per-frame debug PNGs under
# tests/visualization_pipeline/<traj>/ekf_debug/ and an ekf_debug.mp4:
python scripts/data_demo.py --config configs/demo.yaml --tracker ekf \
    [--ekf-backend gaussian|rbpf] [--max-frames N]
```

### Batched evaluation across many datasets
```bash
# Heuristic across every subdir of config['dataset']['path']:
python scripts/eval_run.py --config configs/eval.yaml --tracker heuristic

# EKF (passes --tracker through to per-dataset data_demo.py invocations):
python scripts/eval_run.py --config configs/eval.yaml --tracker ekf
```

### Real-time on Fetch over ROS
```bash
# 1. Detection server:
python perception/det_pipeline/det_server.py

# 2. In a separate shell, the realtime app:
python scripts/realtime_app.py --config configs/realtime_app.yaml --tracker heuristic
```
The EKF backend is **not yet wired to live ROS topics** (the SLAM
covariances needed by the EKF are not published by the current pose
subscriber). `--tracker ekf` raises `NotImplementedError` with a pointer
to the offline EKF entry point.

### EKF debug renderer (canonical EKF entry point)
```bash
python scripts/visualize_ekf_tracking.py --trajectory <name> [--max-frame N]
```
Reads cached perception from `tests/visualization_pipeline/<name>/`,
runs the Gaussian-backend orchestrator end-to-end, dumps per-frame state
to `ekf_state/`, debug PNGs to `ekf_debug/`, and assembles an
`ekf_debug.mp4`. This is the script `data_demo.py --tracker ekf`
dispatches to under the hood for known trajectory names.

## Examples

Self-contained ~100-LOC recipes under `scripts/examples/`:

| Example | Tracker | What it shows |
|---|---|---|
| `scripts/examples/heuristic_offline.py` | Heuristic | 5 RGB-D frames + 1 object via the public `ObjectTracker` API |
| `scripts/examples/ekf_offline.py` | EKF (Gaussian) | Same 5 frames driving `TwoTierOrchestratorGaussian` |
| `scripts/examples/visual_only_baseline.py` | Visual-only | The ICP-only baseline on the same data |
| `scripts/examples/compare_trackers.py` | Both | Heuristic + EKF side-by-side; prints `T_wo` per frame |
| `scripts/examples/ekf_api_demo.py` | EKF (`EkfTracker`) | All five public methods on synthetic data |
| `scripts/examples/demo_api.py` | EKF (`EkfTracker`) | All five public methods on the cached `apple_drop` trajectory; emits a 4-panel `summary.png` + per-object point clouds |
| `scripts/examples/track_apple_in_the_tray.py` | EKF | The original cached-trajectory end-to-end example (pre-existing) |

Each new example generates a tiny synthetic dataset inline so it has no
external data dependency. They double as smoke tests for the public APIs.

## Layout

```
perception/             — all perception (shared)
  ├── icp_pose, association, visibility, det_dedup, …   pose-side
  ├── detection/          — runtime detection client + Hungarian + mask extractor
  └── det_pipeline/       — standalone detection inference server (OWL-ViT / SAM)
utils/                  — shared math + SE(3) + manipulation primitives + image utilities
heuristic_tracker/      — TSDF / Hungarian / ICP variant; api.py public surface
ekf_tracker/            — orchestrator (RBPF) + orchestrator_gaussian (Gaussian) + factor graph
baselines/              — standalone reference baselines (visual-only, …)
scripts/                — all runnable entry points
  ├── data_demo, eval_run, realtime_app, visualize_ekf_tracking, render_gripper_overlay
  ├── examples/           — minimal end-to-end recipes
  └── rosbag2dataset/     — bag → dataset conversion + perception preprocessing (OWL/SAM/SAM2)
configs/                — YAML configs for the runnable scripts (tracker-agnostic)
eval/                   — evaluation scripts + cached results
tests/                  — unit + integration + e2e + diagnostics + visualizers
docs/                   — design docs (general); docs/ekf_tracker/ holds EKF-specific docs
api.py                  — backwards-compat shim re-exporting heuristic_tracker.api for robi_butler
```

## Further Reading

General:
- `docs/survey_and_analysis.md` — comparative survey vs TSDF++/MidFusion/ConceptGraphs/etc.
- `docs/khronos_lessons.md` — design patterns borrowed from Khronos.

EKF tracker:
- `docs/ekf_tracker/DISCUSSION.md` — architectural rationale (two-tier hierarchy, base-frame fusion, composed observation noise).
- `docs/ekf_tracker/PLAN.md` — implementation roadmap with current status mapping each task to its source file.
- `docs/ekf_tracker/improvements.md` — original EKF + factor-graph design notes.
- `docs/ekf_tracker/cov_anisotropy_explained.md` — why the rendered uncertainty ellipse stretches and rotates with base motion.
- `docs/ekf_tracker/latex/bernoulli_ekf.tex` — full Bernoulli-EKF derivation.

Detection:
- `docs/det_pipeline.md` — `detect_objects_on_image()` API reference.

Examples:
- `docs/examples.md` — index of `scripts/examples/` (heuristic_offline, ekf_offline, visual_only_baseline, compare_trackers, track_apple_in_the_tray).