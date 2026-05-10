# Dynamic Scene Graph

Object-centric, dynamic 3D scene-graph tracker for robotic manipulation.
RGB-D + SLAM + gripper proprioception → per-object pose, covariance,
Bernoulli existence, and `on / in / under / contain` relations across
grasp / lift / drop interactions.

📄 **Algorithm derivation:** [bernoulli_ekf.pdf](ekf_tracker/latex/bernoulli_ekf.pdf)
— Bernoulli-EKF on $\SE(3)$, two-tier orchestration, factor graph,
adaptive robust kernel.

```{toctree}
:caption: Get started
:maxdepth: 1

installation
quickstart
api
config
```

```{toctree}
:caption: Reference
:maxdepth: 1

ekf_tracker/index
scripts/index
```
