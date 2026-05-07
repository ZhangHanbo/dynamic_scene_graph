#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live smoke-probe for the OWL + SAM detection servers.

Run this from a network that can reach the configured host (NUS VPN /
on-campus). Hits both endpoints with a tiny synthetic image, checks that
the response shapes match the server contract, and prints round-trip time.

    python rosbag2dataset/probe_servers.py
    SCENEREP_SERVER_HOST=crane5.ddns.comp.nus.edu.sg \
      python rosbag2dataset/probe_servers.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Allow running as `python scripts/rosbag2dataset/probe_servers.py` without PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.rosbag2dataset.server_configs import (
    OWL_SERVER_URL, SAM_SERVER_URL, SAM2_SERVER_URL,
    SAM2_START_PATH, SAM2_ADD_BOX_PATH, SAM2_PROPAGATE_PATH, SAM2_CLOSE_PATH,
)
import requests


def _synth_image(h: int = 96, w: int = 128) -> np.ndarray:
    """Small random image — big enough for the server's padding/resize
    to stay well-defined, small enough to keep the probe fast."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def probe_owl() -> bool:
    from scripts.rosbag2dataset.owl.owl_client import call_owl
    img = _synth_image()
    t0 = time.time()
    try:
        names, bboxes, scores = call_owl(
            img, text_queries=["apple", "cup"], bbox_conf_threshold=0.1,
            server_url=OWL_SERVER_URL, timeout=30.0,
        )
    except Exception as e:
        print(f"[OWL] FAIL  {OWL_SERVER_URL}  {type(e).__name__}: {e}")
        return False
    dt = time.time() - t0
    ok_shape = (
        isinstance(names, list)
        and bboxes.ndim == 2
        and (bboxes.size == 0 or bboxes.shape[1] == 4)
        and len(names) == len(scores) == bboxes.shape[0]
    )
    print(f"[OWL] OK    {OWL_SERVER_URL}  {dt:.2f}s  "
          f"{len(names)} detections  shapes={bboxes.shape},{scores.shape}")
    return ok_shape


def probe_sam() -> bool:
    from scripts.rosbag2dataset.sam.sam_client import call_sam
    img = _synth_image()
    # One box in the middle of the image.
    H, W = img.shape[:2]
    box = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]
    t0 = time.time()
    try:
        masks = call_sam(img, [box], server_url=SAM_SERVER_URL, timeout=60.0)
    except Exception as e:
        print(f"[SAM] FAIL  {SAM_SERVER_URL}  {type(e).__name__}: {e}")
        return False
    dt = time.time() - t0
    ok_shape = len(masks) == 1 and masks[0].shape == (H, W) \
        and masks[0].dtype == bool
    print(f"[SAM] OK    {SAM_SERVER_URL}  {dt:.2f}s  "
          f"mask shape={masks[0].shape if masks else None} "
          f"dtype={masks[0].dtype if masks else None} "
          f"nnz={int(masks[0].sum()) if masks else 0}")
    return ok_shape


def probe_sam_legacy() -> bool:
    """Hit the /sam_mask_by_bbox endpoint on the SAM URL.

    With the new server-side consolidation, ``SAM_SERVER_URL`` defaults
    to the SAM2 server, which exposes the legacy SAM v1 endpoints. This
    test verifies the old wire format (pickled-numpy mask in
    ``{result: [{segmentation: ...}]}``) is honoured.
    """
    from scripts.rosbag2dataset.sam.sam_client import call_sam
    img = _synth_image()
    H, W = img.shape[:2]
    box = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]
    t0 = time.time()
    try:
        masks = call_sam(img, [box], server_url=SAM_SERVER_URL, timeout=60.0)
    except Exception as e:
        print(f"[SAM(legacy)] FAIL  {SAM_SERVER_URL}  "
              f"{type(e).__name__}: {e}")
        return False
    dt = time.time() - t0
    ok = len(masks) == 1 and masks[0].shape == (H, W) \
        and masks[0].dtype == bool
    print(f"[SAM(legacy)] OK    {SAM_SERVER_URL}  {dt:.2f}s  "
          f"mask shape={masks[0].shape if masks else None} "
          f"dtype={masks[0].dtype if masks else None} "
          f"nnz={int(masks[0].sum()) if masks else 0}")
    return ok


def probe_sam2() -> bool:
    """Start a 3-frame session, box-prompt, propagate, close."""
    import base64, io
    from PIL import Image
    img = _synth_image()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    H, W = img.shape[:2]
    box = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]

    t0 = time.time()
    try:
        r = requests.post(SAM2_SERVER_URL.rstrip("/") + SAM2_START_PATH,
                          json={"frames": [b64, b64, b64]}, timeout=120)
        r.raise_for_status()
        session_id = r.json()["session_id"]

        r = requests.post(SAM2_SERVER_URL.rstrip("/") + SAM2_ADD_BOX_PATH,
                          json={"session_id": session_id,
                                "frame_idx": 0,
                                "object_id": 0,
                                "box": box},
                          timeout=60)
        r.raise_for_status()

        r = requests.post(SAM2_SERVER_URL.rstrip("/") + SAM2_PROPAGATE_PATH,
                          json={"session_id": session_id},
                          timeout=120)
        r.raise_for_status()
        results = r.json().get("results", [])

        requests.post(SAM2_SERVER_URL.rstrip("/") + SAM2_CLOSE_PATH,
                      json={"session_id": session_id}, timeout=30)
    except Exception as e:
        print(f"[SAM2] FAIL  {SAM2_SERVER_URL}  {type(e).__name__}: {e}")
        return False

    dt = time.time() - t0
    ok = (len(results) == 3
          and all("objects" in r for r in results)
          and any(r["objects"] for r in results))
    print(f"[SAM2] OK   {SAM2_SERVER_URL}  {dt:.2f}s  "
          f"{len(results)} frames propagated  "
          f"obj_present={[bool(r['objects']) for r in results]}")
    return ok


def main() -> int:
    print("probing:")
    print(f"  OWL_SERVER_URL  = {OWL_SERVER_URL}")
    print(f"  SAM_SERVER_URL  = {SAM_SERVER_URL}  "
          "(legacy /sam_* — should hit the SAM2 server by default)")
    print(f"  SAM2_SERVER_URL = {SAM2_SERVER_URL}")
    ok_owl       = probe_owl()
    ok_sam_leg   = probe_sam_legacy()
    ok_sam2      = probe_sam2()
    if ok_owl and ok_sam_leg and ok_sam2:
        print("\nall endpoints reachable and responses well-formed "
              "(OWL, legacy SAM, SAM2 video).")
        return 0
    print("\none or more endpoints failed — see errors above.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
