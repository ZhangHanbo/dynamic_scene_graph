# Public API

```python
from ekf_tracker import EkfTracker, EkfObject, SceneView
```

Five methods, one facade.

## `EkfTracker`

```{eval-rst}
.. autoclass:: ekf_tracker.api.EkfTracker
   :members: detect, step, get_scene, get_points, smooth
   :show-inheritance:
```

## Snapshots

```{eval-rst}
.. autoclass:: ekf_tracker.api.EkfObject
   :members:

.. autoclass:: ekf_tracker.api.SceneView
   :members:
```

## End-to-end shape

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc)

for rgb, depth, T_wb, T_bg, width, joints in stream:
    dets, hist = tracker.detect(rgb, ["apple", "bowl"])    # or use cached dets
    scene = tracker.step(
        dets, rgb, depth,
        slam_pose=T_wb, T_bg=T_bg,
        gripper_width=width, joints=joints,
    )
    for oid, obj in scene.objects.items():
        print(oid, obj.label, obj.pose[:3, 3], "r =", obj.r)
```

* `obj.pose` — $4 \times 4$ world-frame mean $T_{wo}$.
* `obj.cov`  — $6 \times 6$ tangent covariance, ordering $[v, \omega]$.
* `obj.r`    — Bernoulli existence probability in $[0, 1]$.

## Configuration knobs

`EkfTracker.__init__` accepts every public knob; the defaults are loaded
from `ekf_tracker/configs/default.yaml`. See [Configuration](config.md).
