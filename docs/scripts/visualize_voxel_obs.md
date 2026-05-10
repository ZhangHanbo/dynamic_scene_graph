# `scripts/visualize_voxel_obs.py`

Open3D viewer for the `VoxelObservability` grid used by gravity-drop
prediction. Renders the workspace as a coloured point cloud.

```bash
python scripts/visualize_voxel_obs.py \
    --trajectory apple_drop \
    --animate-every 5 --show-unseen --with-tsdf \
    --column 0.0,0.6
```

| Color | Voxel state |
|---|---|
| Grey | Unobserved |
| Green | Empty (rays passed through) |
| Red | Occupied (rays terminated) |

| CLI flag | Default | Purpose |
|---|---|---|
| `--trajectory` | required | Trajectory under `tests/visualization_pipeline/`. |
| `--animate-every` | `1` | Re-render every N frames. |
| `--show-unseen` | off | Render grey unobserved voxels. |
| `--with-tsdf` | off | Overlay the TSDF mesh. |
| `--column X,Z` | None | Slice through a single workspace column. |
| `--mask-arm` | off | Mask out the Fetch arm via `utils.fetch_arm_mask`. |

Use this when a gravity prediction lands somewhere unexpected — if the
support voxel is grey, the predictor never saw the surface and fell
back to the floor plane.
