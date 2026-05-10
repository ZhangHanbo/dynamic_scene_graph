# `ekf_tracker/manipulation/`

Gripper-driven dynamics. `gripper_state_inferrer.py` runs the FSM
(idle → grasping → holding → releasing). `grasp_owner_detector.py`
decides which tracked object the gripper is holding from a three-tier
signal: perception override → URDF inside-jaws containment → fallback.
`gravity_predict.py` produces a parametric free-fall + bounce + roll
prior that updates the EKF mean and covariance the moment a held object
is released.

```{toctree}
:maxdepth: 1

grasp_owner_detector
gripper_state_inferrer
gravity_predict
```
