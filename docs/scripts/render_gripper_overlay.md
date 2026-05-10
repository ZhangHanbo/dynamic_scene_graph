# `scripts/render_gripper_overlay.py`

Projects the URDF-derived gripper geometry (inside-jaws AABB + finger
pads) onto a single RGB frame. Use this to verify the FK chain and
`T_bc` extrinsic before trusting any tracker run.

```bash
python scripts/render_gripper_overlay.py 488 \
    --trajectory apple_in_the_tray \
    --out gripper_overlay.png
```

| Positional / flag | Purpose |
|---|---|
| `<FRAME>` | Required positional — frame index. |
| `--trajectory` | Trajectory under `datasets/`. |
| `--out` | Output PNG path. |

If the projected boxes don't sit on the actual gripper in the image,
either the URDF FK or the `T_bc` extrinsic is wrong — fix that before
running any tracker.
