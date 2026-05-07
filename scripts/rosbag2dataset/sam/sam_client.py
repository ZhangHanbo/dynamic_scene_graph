#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-in replacement for ``sam.py`` that calls the alpha_robot SAM server.

Server contract (``alpha_robot/service/sam/server.py``):

    POST <SAM_SERVER_URL>/sam_mask_by_bbox
    body = {
        "image":       base64 PNG,
        "bboxes":      [[[x1, y1, x2, y2]], [[...]]],   # 3-D list, pixel coords,
                                                         # one bbox per prediction
        "return_best": true,
    }
    response = {
        "result": [
            {"segmentation": base64(pickle(np.ndarray bool mask))},
            ...
        ]
    }

For each detection JSON under ``detection_boxes/`` this script:
  1. Reads the per-frame OWL output (boxes).
  2. Calls SAM once per frame with all boxes as a batch.
  3. Decodes the returned masks (pickle → numpy bool).
  4. Writes base64-encoded PNG masks back into the same JSON under the
     ``mask`` field — matching the format the old ``sam.py`` wrote, so
     ``track_object_ids.py`` and ``detection/mask_extractor.py`` work
     unchanged.
  5. Overlays masks on the existing ``detection_NNNNNN.png`` preview.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pickle
import sys
import time
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image

# Allow running as `python scripts/rosbag2dataset/sam/sam_client.py ...`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.rosbag2dataset.server_configs import (  # noqa: E402
    SAM_SERVER_URL, SAM_MASK_BY_BBOX_PATH,
)


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------

def _encode_png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_sam(rgb: np.ndarray,
             boxes_xyxy: List[List[int]],
             server_url: str = SAM_SERVER_URL,
             timeout: float = 120.0) -> List[np.ndarray]:
    """POST one frame + a batch of boxes; return one bool mask per box."""
    # The server expects a 3-D list (nb_predictions, nb_boxes_per_pred=1, 4).
    wrapped = [[list(map(int, b))] for b in boxes_xyxy]
    url = server_url.rstrip("/") + SAM_MASK_BY_BBOX_PATH
    payload = {
        "image":       _encode_png_b64(rgb),
        "bboxes":      wrapped,
        "return_best": True,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    masks: List[np.ndarray] = []
    for item in data.get("result", []):
        raw = base64.b64decode(item["segmentation"])
        m = pickle.loads(raw)
        # Server returns shape (1, H, W) torch/numpy tensor; squeeze.
        m = np.asarray(m).squeeze()
        if m.ndim == 3:
            m = m[0]
        masks.append(m.astype(bool))
    return masks


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def _mask_to_png_b64(mask: np.ndarray) -> str:
    """Encode a boolean mask as a base64 PNG (matches the legacy format)."""
    arr = (mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_OVERLAY_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]


def _overlay_masks(image_bgr: np.ndarray,
                   masks: List[np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    for i, m in enumerate(masks):
        color = np.array(_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)])
        layer = np.zeros_like(out)
        layer[m] = color
        cv2.addWeighted(out, 1.0, layer, 0.5, 0, out)
    return out


def process_frame(rgb_path: str,
                  det_json_path: str,
                  det_png_path: str,
                  server_url: str) -> int:
    """Return the number of masks added to this frame's JSON."""
    with open(det_json_path, "r") as f:
        data = json.load(f)
    dets = data.get("detections", [])
    if not dets:
        return 0

    # Build box list in pixel-xyxy (same as what OWL client wrote).
    boxes = []
    keep_dets = []
    for d in dets:
        bx = d.get("box")
        if not bx or len(bx) != 4:
            continue
        boxes.append([int(v) for v in bx])
        keep_dets.append(d)
    if not boxes:
        return 0

    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None:
        print(f"[warn] cannot read {rgb_path}")
        return 0
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masks = call_sam(img, boxes, server_url=server_url)
    assert len(masks) == len(keep_dets), \
        f"SAM returned {len(masks)} masks for {len(keep_dets)} boxes"

    # Stitch masks back into the JSON in base64-PNG form.
    for d, m in zip(keep_dets, masks):
        d["mask"] = _mask_to_png_b64(m)

    data["detections"] = dets    # unchanged ordering, just annotated
    with open(det_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Refresh the overlay preview
    preview_bgr = cv2.imread(det_png_path) if os.path.isfile(det_png_path) \
        else img_bgr.copy()
    preview_bgr = _overlay_masks(preview_bgr, masks)
    cv2.imwrite(det_png_path, preview_bgr)

    return len(masks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_name", help="dataset sub-dir under DATASET_PATH "
                                         "(or a full path)")
    ap.add_argument("--dataset-root", default=None,
                    help="dataset root; defaults to $DATASET_PATH or '.'")
    ap.add_argument("--det-dir", default="detection_boxes",
                    help="detection subdir written by owl_client (default: "
                         "detection_boxes)")
    ap.add_argument("--server", default=SAM_SERVER_URL,
                    help="SAM server URL (override with $SAM_SERVER_URL)")
    args = ap.parse_args()

    if os.path.isabs(args.dataset_name) and os.path.isdir(args.dataset_name):
        dataset_dir = args.dataset_name
    else:
        root = args.dataset_root or os.environ.get("DATASET_PATH", ".")
        dataset_dir = os.path.join(root, args.dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"no such dataset: {dataset_dir}")

    rgb_dir = os.path.join(dataset_dir, "rgb")
    det_dir = os.path.join(dataset_dir, args.det_dir)

    det_files = sorted(f for f in os.listdir(det_dir)
                       if f.startswith("detection_") and f.endswith(".json")
                       and not f.endswith("_final.json"))
    if not det_files:
        print(f"no OWL JSONs in {det_dir} — run owl_client first")
        return 1

    print(f"[sam] server={args.server}  "
          f"{len(det_files)} frames under {det_dir}")

    total = 0
    t0 = time.time()
    for i, f in enumerate(det_files):
        fid_str = f[len("detection_"):-len(".json")]
        try:
            fid = int(fid_str)
        except ValueError:
            continue
        rgb_path = os.path.join(rgb_dir, f"rgb_{fid:06d}.png")
        det_json = os.path.join(det_dir, f)
        det_png = os.path.join(det_dir, f.replace(".json", ".png"))

        if not os.path.isfile(rgb_path):
            print(f"[skip] missing RGB for frame {fid}: {rgb_path}")
            continue
        try:
            total += process_frame(rgb_path, det_json, det_png, args.server)
        except requests.RequestException as e:
            print(f"[http error] frame {fid}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(det_files)}] "
                  f"{(time.time() - t0) / (i + 1):.2f}s/frame  "
                  f"masks={total}")

    print(f"[done] {dataset_dir}: {total} masks in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
