# Installation

```bash
git clone git@github.com:ZhangHanbo/dynamic_scene_graph.git
cd dynamic_scene_graph

conda create -n dynamic_scene_graph python=3.11
conda activate dynamic_scene_graph
pip install -r requirements.txt
pip install -e .
```

## Optional extras

| Need | Install |
|---|---|
| Live OWLv2 + SAM2 detection | `pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`, then drop OWLv2 + SAM ViT-B checkpoints under `scripts/rosbag2dataset/{owl,sam}/`. |
| Slow-tier pose-graph smoother | `pip install gtsam`. Without it, :py:meth:`ekf_tracker.api.EkfTracker.smooth` is unavailable; :py:meth:`step` works either way. |
| Build this site | `pip install -r requirements-docs.txt` |

## Verify

```bash
python demo/run_demo.py
```

Should extract `demo/apple_in_the_tray.zip` and print one tracker state line per frame. See [Quickstart](quickstart.md).
