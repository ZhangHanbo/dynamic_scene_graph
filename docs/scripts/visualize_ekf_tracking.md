# `scripts/visualize_ekf_tracking.py`

Canonical EKF entry point. Replicates the production fast tier
(`InstrumentedTracker`), dumps per-frame state to JSON, and renders a
2 × 3 debug grid per frame.

```bash
python scripts/visualize_ekf_tracking.py \
    --trajectory apple_drop \
    --start 0 --max-frame 700 --step 1 \
    --config-path configs/ekf_tracker/customization.yaml
```

| CLI flag | Default | Purpose |
|---|---|---|
| `--trajectory` | `apple_in_the_tray` | Trajectory directory under `tests/visualization_pipeline/`. |
| `--max-frame` | `700` | Stop after this frame. |
| `--start` | `0` | First frame index. |
| `--step` | `1` | Stride. |
| `--pose-method` | `icp_chain` | Pose backend: `centroid` / `icp_chain` / `icp_anchor`. |
| `--out-subdir` | `ekf_debug` | PNG output subdirectory. |
| `--state-subdir` | `ekf_state` | JSON output subdirectory. |
| `--no-png` | off | Skip PNG rendering. |
| `--no-mp4` | off | Skip MP4 assembly. |
| `--fps` | `10.0` | MP4 frame rate. |
| `--config-path` | None | Override YAML config. |

**Outputs** under `tests/visualization_pipeline/<traj>/`:

| Path | Contents |
|---|---|
| `ekf_state/<frame>.json` | Per-frame state — pose, cov, `r`, label, sam2 tau. |
| `ekf_debug/frame_*.png` | Six-panel debug image. |
| `ekf_debug.mp4` | Frames assembled into a video. |
