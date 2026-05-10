# `ekf_tracker/state/`

Per-track state representation. The EKF stores object beliefs in the robot
**base frame** so that the SLAM covariance $\Sigma_{wb}$ never enters the
per-frame recursion; world poses are composed in only at output time
(see `gaussian_state.py`). `bernoulli.py` carries the existence
probability, `obs_chain.py` keeps an append-only log of camera-frame
observations for retroactive SLAM corrections, and `rbpf_state.py` is the
RBPF research backend.

```{toctree}
:maxdepth: 1

gaussian_state
rbpf_state
obs_chain
bernoulli
```
