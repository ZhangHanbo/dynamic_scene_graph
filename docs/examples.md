# Examples

Self-contained ~100-LOC recipes that exercise each tracker's public API
with a tiny synthetic dataset generated inline. No external data needed,
so they double as smoke tests in CI.

| File | Tracker | Purpose |
|---|---|---|
| `heuristic_offline.py` | `heuristic_tracker.ObjectTracker` | TSDF + ID-association on 5 synthetic frames |
| `ekf_offline.py` | `ekf_tracker.TwoTierOrchestratorGaussian` | Gaussian-backend EKF on the same 5 frames |
| `ekf_api_demo.py` | `ekf_tracker.EkfTracker` | All five public methods (`step`, `smooth`, `get_points`, `get_scene`, `detect`) on the same data |
| `demo_api.py` | `ekf_tracker.EkfTracker` | All five public methods on the cached `apple_drop` trajectory; emits `tests/visualization_pipeline/apple_drop/api_demo/summary.png` + per-object `.npy` point clouds |
| `visual_only_baseline.py` | `baselines.VisualOnlyTracker` | ICP-only reference baseline |
| `compare_trackers.py` | Both | Heuristic + EKF side-by-side; prints `T_wo` per frame |
| `track_apple_in_the_tray.py` | EKF | Pre-existing cached-trajectory end-to-end demo (needs cached data on disk) |

Run any example from the repo root:

```bash
python scripts/examples/heuristic_offline.py
python scripts/examples/ekf_offline.py
python scripts/examples/ekf_api_demo.py
python scripts/examples/visual_only_baseline.py
python scripts/examples/compare_trackers.py
```

The first four print one line per frame and exit with code 0 on success.
`track_apple_in_the_tray.py` requires the cached `apple_in_the_tray`
trajectory under `tests/visualization_pipeline/`.
