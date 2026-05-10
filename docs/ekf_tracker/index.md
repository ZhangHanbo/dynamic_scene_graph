# `ekf_tracker/`

The Bernoulli-EKF on $\SE(3)$ that drives the public [`EkfTracker`](../api.md)
facade. Eight top-level modules plus three subpackages: state
representation (`state/`), gripper / grasp / release dynamics
(`manipulation/`), and scene-graph edges (`relations/`).

```{toctree}
:caption: Top-level modules
:maxdepth: 1

gaussian_ekf_tracker
orchestrator_gaussian
factor_graph
perception_pipeline
birth_gate
```

```{toctree}
:caption: Subpackages
:maxdepth: 1

state/index
manipulation/index
relations/index
```
