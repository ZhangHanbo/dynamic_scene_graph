# `scripts/visualize_arm_mask.py`

Renders per-link Fetch arm silhouettes onto RGB using
`utils.fetch_arm_mask.ArmMaskBuilder`. Validates the arm-mask pipeline
before its mask is fed to SLAM.

```bash
python scripts/visualize_arm_mask.py \
    --trajectory apple_in_the_tray \
    --start 0 --max-frame 50 --dilate-px 12
```

| CLI flag | Default | Purpose |
|---|---|---|
| `--trajectory` | required | Trajectory under `datasets/`. |
| `--start` | `0` | First frame. |
| `--max-frame` | end | Last frame. |
| `--dilate-px` | `12` | Dilation kernel size. Tune until every visible link is covered with a ~5-pixel margin. |

Output: each frame's RGB with the projected arm silhouette overlaid in
semi-transparent red.
