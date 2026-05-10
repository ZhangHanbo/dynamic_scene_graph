# Quickstart

The repo ships a 37-frame Fetch trajectory at `demo/apple_in_the_tray.zip`.
The runner extracts it on first call, then drives the EKF tracker with
the bundled default config.

## One-liner

```bash
python demo/run_demo.py
```

You should see one block per frame:

```
frame 0488  objects: 0:apple@[-0.12, 0.41, 0.78] r=0.97, 1:tray@[-0.05, 0.39, 0.74] r=0.99
…
done.
```

Each entry shows the tracked object's id, label, world-frame translation,
and Bernoulli existence probability $r \in [0, 1]$.

## Inside the demo

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc, relation_backend="none")

for rgb, depth, T_wb, T_bg, width, joints, dets in stream:
    scene = tracker.step(
        dets, rgb, depth,
        slam_pose=T_wb, T_bg=T_bg,
        gripper_width=width, joints=joints,
    )
    for oid, obj in scene.objects.items():
        print(oid, obj.label, obj.pose[:3, 3], obj.r)
```

`relation_backend="none"` skips the LLM/REST relation detector so the
demo runs offline. Drop that argument once you have an LLM key or a REST
relation server (see [Configuration → relation](config.md#relation)).

## Visualize

For the rendered debug video (covariance ellipses, top-down state,
event log) use the canonical visualizer instead:

```bash
python scripts/visualize_ekf_tracking.py \
    --trajectory apple_in_the_tray \
    --config-path configs/ekf_tracker/customization.yaml
```

→ outputs `tests/visualization_pipeline/apple_in_the_tray/ekf_debug.mp4`.
See [`scripts/visualize_ekf_tracking.py`](scripts/visualize_ekf_tracking.md)
for every flag.

## Next

* [Public API](api.md) — the five-method facade.
* [Configuration](config.md) — every YAML knob.
