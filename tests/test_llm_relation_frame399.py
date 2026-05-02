"""Smoke test for LLMRelationClient on apple_in_the_tray frame 399.

Loads the raw 640x480 RGB + the perception detection_h JSON (which has
per-object bboxes in image-pixel coordinates), normalises the bboxes,
calls the GPT-backed relation client, and prints the (N, N) p_parent
matrix plus the edges above a threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_update.relations.relation_client import LLMRelationClient


RGB_PATH = Path("/Volumes/External/Workspace/datasets/apple_in_the_tray/rgb/rgb_000399.png")
DET_PATH = Path(
    "/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP/"
    "tests/visualization_pipeline/apple_in_the_tray/perception/"
    "detection_h/detection_000399_final.json"
)


def main() -> int:
    rgb = Image.open(RGB_PATH).convert("RGB")
    W, H = rgb.size
    print(f"[rgb]   {RGB_PATH.name}  size={W}x{H}")

    with DET_PATH.open() as f:
        det_json = json.load(f)
    dets = det_json["detections"]
    print(f"[dets]  {len(dets)} objects from {DET_PATH.name}")

    bboxes_n = []
    labels = []
    for i, d in enumerate(dets):
        box = d.get("box")
        if box is None:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        bboxes_n.append([x0 / W, y0 / H, x1 / W, y1 / H])
        labels.append(d.get("label", f"obj{i}"))
        print(f"  {i}: {labels[-1]:8s} "
              f"box=({int(x0)}, {int(y0)}, {int(x1)}, {int(y1)})  "
              f"score={d.get('score', float('nan')):.2f}  "
              f"n_obs={d.get('n_obs', '-')}")
    bboxes_n = np.asarray(bboxes_n, dtype=np.float32)
    n = len(bboxes_n)
    if n < 2:
        print("[skip]  need at least 2 bboxes for relations")
        return 1

    client = LLMRelationClient(model_name="gpt-5.1")
    print(f"[client] backend={client.backend}  model={client._model_name}")
    print("[call]  LLMRelationClient.detect(rgb, bboxes_n) ...")

    p_parent = client.detect(rgb, bboxes_n)
    if p_parent is None:
        print("[fail]  detect() returned None (see warnings above)")
        print(f"[fail]  available={client.available}")
        return 2

    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print(f"\np_parent ({n}x{n}) — row i / col j = P(i is parent of j):")
    print("       " + "  ".join(f"{labels[j][:6]:>6s}" for j in range(n)))
    for i in range(n):
        row = "  ".join(f"{p_parent[i, j]:6.2f}" for j in range(n))
        print(f"{labels[i][:6]:>6s}  {row}")

    thr = 0.5
    print(f"\nEdges with score > {thr}:")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(p_parent[i, j])
            if s > thr:
                print(f"  {labels[i]} —({s:.2f})→ {labels[j]}   "
                      f"(i.e., {labels[j]} is on/in {labels[i]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
