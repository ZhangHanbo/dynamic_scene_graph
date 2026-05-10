# Dynamic Scene Graph

Object-centric, dynamic 3D scene-graph tracker for robotic manipulation.
RGB-D + SLAM + gripper proprioception → per-object pose, covariance,
Bernoulli existence, and `on / in / under / contain` relations across
grasp / lift / drop interactions.

📄 **Algorithm derivation:** [bernoulli_ekf.pdf](ekf_tracker/latex/bernoulli_ekf.pdf)
— Bernoulli-EKF on SE(3), two-tier orchestration, factor graph, adaptive
robust kernel. Part III maps every component to the file that
implements it.

This site is **file-by-file**: every page documents exactly one source
file, in `ekf_tracker/` or `scripts/`.

## Installation

```bash
git clone git@github.com:ZhangHanbo/dynamic_scene_graph.git
cd dynamic_scene_graph

conda create -n dynamic_scene_graph python=3.11
conda activate dynamic_scene_graph
pip install -r requirements.txt
pip install -e .
```

## Quickstart — bundled demo trajectory

A 37-frame Fetch slice is shipped at `demo/apple_in_the_tray.zip`.
The runner extracts it on first call, then drives the EKF tracker:

```bash
python demo/run_demo.py
```

Expected output — one line per frame listing each tracked object's
world-frame position and Bernoulli existence:

```
frame 0488  objects: 0:apple@[-0.12, 0.41, 0.78] r=0.97, 1:tray@[-0.05, 0.39, 0.74] r=0.99
…
done.
```

For the rendered debug video (covariance ellipses, top-down state,
event log) run the canonical visualizer instead:

```bash
python scripts/visualize_ekf_tracking.py \
    --trajectory apple_in_the_tray \
    --config-path configs/ekf_tracker/customization.yaml
```

→ outputs `tests/visualization_pipeline/apple_in_the_tray/ekf_debug.mp4`.
See [`scripts/visualize_ekf_tracking.py`](scripts/visualize_ekf_tracking.md)
for every flag.

## File-by-file reference

```{toctree}
:caption: ekf_tracker/
:maxdepth: 1

ekf_tracker/__init__
ekf_tracker/api
ekf_tracker/gaussian_ekf_tracker
ekf_tracker/orchestrator_gaussian
ekf_tracker/factor_graph
ekf_tracker/perception_pipeline
ekf_tracker/birth_gate
ekf_tracker/config
```

```{toctree}
:caption: ekf_tracker/configs/
:maxdepth: 1

ekf_tracker/configs/__init__
ekf_tracker/configs/default_yaml
```

```{toctree}
:caption: ekf_tracker/state/
:maxdepth: 1

ekf_tracker/state/__init__
ekf_tracker/state/gaussian_state
ekf_tracker/state/rbpf_state
ekf_tracker/state/obs_chain
ekf_tracker/state/bernoulli
```

```{toctree}
:caption: ekf_tracker/manipulation/
:maxdepth: 1

ekf_tracker/manipulation/__init__
ekf_tracker/manipulation/grasp_owner_detector
ekf_tracker/manipulation/gripper_state_inferrer
ekf_tracker/manipulation/gravity_predict
```

```{toctree}
:caption: ekf_tracker/relations/
:maxdepth: 1

ekf_tracker/relations/__init__
ekf_tracker/relations/relation_orchestrator
ekf_tracker/relations/relation_filter
ekf_tracker/relations/relation_client
ekf_tracker/relations/relation_utils
```

```{toctree}
:caption: scripts/
:maxdepth: 1

scripts/visualize_ekf_tracking
scripts/visualize_voxel_obs
scripts/render_gripper_overlay
scripts/visualize_arm_mask
```

## Build this site

```bash
pip install -r requirements-docs.txt
cd docs && make html
```
