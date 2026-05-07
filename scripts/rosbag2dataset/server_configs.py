"""Alpha-robot detection-server endpoints.

Values copied from ``../projects/alpha_robot/arobot/configs.py`` — keep in
sync when the upstream addresses move. Override with environment variables
(``OWL_SERVER_URL`` / ``SAM_SERVER_URL``) without editing code.
"""

from __future__ import annotations

import os


# Ports match ``arobot.configs.IP_CONFIGS``:
#   OWLViT -> 4051
#   SAM2   -> 4057 (single server hosting both /sam2_* and legacy /sam_*;
#                   the old standalone SAM v1 service at 4057 is retired
#                   and SAM2 squats the same port).
# Host override: currently deployed on crane5 — flip SERVER_HOST or the
# per-service URLs to move without touching code.
SERVER_HOST = os.environ.get("SCENEREP_SERVER_HOST",
                             "crane5.ddns.comp.nus.edu.sg")

OWL_SERVER_URL = os.environ.get(
    "OWL_SERVER_URL", f"http://{SERVER_HOST}:4051")
# SAM2 server. Hosts both the new /sam2_* video tracking endpoints AND
# the legacy /sam_* per-image endpoints on the same port (4057), matching
# ``arobot.configs.IP_CONFIGS['SAM2']``.
SAM2_SERVER_URL = os.environ.get(
    "SAM2_SERVER_URL", f"http://{SERVER_HOST}:4057")
# SAM (legacy per-image API). Defaults to the SAM2 server which now
# implements the /sam_* endpoints with the exact legacy wire format.
SAM_SERVER_URL = os.environ.get("SAM_SERVER_URL", SAM2_SERVER_URL)

# Endpoint paths on each server (from service/owl_vit/server.py and
# service/sam/server.py).
OWL_DETECT_PATH         = "/owl_detect"
OWL_MATCH_BY_IMAGE_PATH = "/owl_match_by_image"
SAM_MASK_BY_BBOX_PATH   = "/sam_mask_by_bbox"
SAM_AUTO_MASK_PATH      = "/sam_auto_mask_generation"

# SAM2 session endpoints (service/sam2/server.py).
# Batch / offline (legacy; single /sam2_propagate over the whole video):
SAM2_START_PATH       = "/sam2_start_session"
SAM2_ADD_BOX_PATH     = "/sam2_add_box"
SAM2_ADD_POINTS_PATH  = "/sam2_add_points"
SAM2_PROPAGATE_PATH   = "/sam2_propagate"
SAM2_CLOSE_PATH       = "/sam2_close_session"
# Streaming / online (one frame at a time, prompts added at any frame).
SAM2_STREAM_INIT_PATH        = "/sam2_stream_init"
SAM2_STREAM_FRAME_PATH       = "/sam2_stream_frame"
SAM2_STREAM_ADD_BOX_PATH     = "/sam2_stream_add_box"
SAM2_STREAM_ADD_POINTS_PATH  = "/sam2_stream_add_points"
SAM2_STREAM_CLOSE_PATH       = "/sam2_stream_close"

# Default object vocabulary. Matches the old OBJECTS list in
# ``rosbag2dataset/owl/owl_object_scores.py``; extend per dataset.
DEFAULT_OBJECTS = [
    "apple", "cup", "bottle", "cola", "bowl", "tray", "cabinet",
]

# OWL request parameters. Applied uniformly: every frame, every class,
# no client-side post-processing. The server's own two-stage NMS is
# the only filter that runs.
OWL_BBOX_CONF: float = 0.05

# Cross-class NMS IoU: only suppress a box against a differently-labelled
# box when they overlap this much. Looser (0.7) -> allow overlapping
# classes (e.g. the same region labelled 'apple' and 'tomato') to coexist.
OWL_NMS_CROSS: float = 0.7

# Per-class NMS IoU: suppress same-class duplicates that overlap this
# much. Tighter (0.5) -> dedupe redundant boxes aggressively.
OWL_NMS_CAT:   float = 0.5
