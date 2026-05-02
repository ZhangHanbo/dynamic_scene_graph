"""Step 1 — single-frame LLM relation visualization.

Runs ``LLMRelationClient`` on one apple_in_the_tray RGB frame and overlays:
  * numbered red bounding boxes (the same boxes shown to the LLM), and
  * green parent→child arrows with the score, for pairs above ``THR``.

Saves ``tests/out/llm_relation_frame_<idx>.png`` and also prints the raw
``(N, N)`` p_parent matrix so the numeric output can be eyeballed next to
the picture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_update.relations.relation_client import LLMRelationClient, decode_mask_b64


# --------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------- #

FRAME_IDX = 399
THR = 0.5

ROOT = Path("/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP")
DATASET_RGB = Path(f"/Volumes/External/Workspace/datasets/apple_in_the_tray/rgb/rgb_{FRAME_IDX:06d}.png")
DET_JSON = (ROOT / "tests/visualization_pipeline/apple_in_the_tray/perception/"
            f"detection_h/detection_{FRAME_IDX:06d}_final.json")
OUT_PNG = ROOT / f"tests/out/llm_relation_frame_{FRAME_IDX:06d}.png"


# --------------------------------------------------------------------- #
#  Drawing helpers
# --------------------------------------------------------------------- #

def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_arrow(draw: ImageDraw.ImageDraw,
                p0: tuple, p1: tuple,
                color=(0, 200, 0), width: int = 3,
                head_len: int = 14, head_w: int = 10) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    # Shorten shaft so it doesn't stab through the arrowhead
    sx, sy = x1 - ux * head_len, y1 - uy * head_len
    draw.line([(x0, y0), (sx, sy)], fill=color, width=width)
    # Arrowhead triangle
    px, py = -uy, ux  # perpendicular
    tip = (x1, y1)
    left = (sx + px * head_w * 0.5, sy + py * head_w * 0.5)
    right = (sx - px * head_w * 0.5, sy - py * head_w * 0.5)
    draw.polygon([tip, left, right], fill=color)


def _bbox_center(b_pix: np.ndarray) -> tuple:
    x0, y0, x1, y1 = b_pix
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def render_relation_overlay(rgb: Image.Image,
                            bboxes_pix: np.ndarray,
                            labels: list,
                            p_parent: np.ndarray,
                            thr: float = THR,
                            caption_prefix: str = f"frame {FRAME_IDX}",
                            ) -> Image.Image:
    img = rgb.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(16)
    font_small = _load_font(12)

    # --- bboxes
    for i, b in enumerate(bboxes_pix):
        x0, y0, x1, y1 = [float(v) for v in b]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=2)
        tag = f"{i}:{labels[i]}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        tx, ty = x0 + 2, max(0.0, y0 - th - 2)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1],
                       fill=(255, 0, 0, 200))
        draw.text((tx, ty), tag, font=font, fill=(255, 255, 255, 255))

    # --- arrows (parent → child) for pairs above threshold
    n = p_parent.shape[0]
    arrow_entries = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(p_parent[i, j])
            if s >= thr:
                arrow_entries.append((s, i, j))
    arrow_entries.sort(reverse=True)
    for s, i, j in arrow_entries:
        pc = _bbox_center(bboxes_pix[i])
        cc = _bbox_center(bboxes_pix[j])
        _draw_arrow(draw, pc, cc, color=(0, 200, 0), width=3)
        midx = 0.5 * (pc[0] + cc[0])
        midy = 0.5 * (pc[1] + cc[1])
        tag = f"{s:.2f}"
        tw, th = draw.textbbox((0, 0), tag, font=font_small)[2:]
        draw.rectangle([midx - tw/2 - 2, midy - th/2 - 1,
                        midx + tw/2 + 2, midy + th/2 + 1],
                       fill=(0, 0, 0, 180))
        draw.text((midx - tw/2, midy - th/2), tag,
                  font=font_small, fill=(255, 255, 255, 255))

    # --- caption
    caption = (f"{caption_prefix} ({len(arrow_entries)} edges ≥ {thr:.2f})")
    tw, th = draw.textbbox((0, 0), caption, font=font)[2:]
    draw.rectangle([4, 4, 8 + tw, 8 + th], fill=(0, 0, 0, 200))
    draw.text((8, 6), caption, font=font, fill=(255, 255, 255, 255))
    return img


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #

def main() -> int:
    # (1) Load image + detections
    if not DATASET_RGB.exists():
        print(f"[fail]  missing RGB: {DATASET_RGB}")
        return 1
    if not DET_JSON.exists():
        print(f"[fail]  missing detections: {DET_JSON}")
        return 1

    rgb = Image.open(DATASET_RGB).convert("RGB")
    W, H = rgb.size
    print(f"[rgb]    {DATASET_RGB.name}  {W}x{H}")

    with DET_JSON.open() as f:
        det_json = json.load(f)
    dets = det_json["detections"]
    bboxes_pix, bboxes_n, labels, oids, masks = [], [], [], [], []
    for i, d in enumerate(dets):
        box = d.get("box")
        if box is None:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        bboxes_pix.append([x0, y0, x1, y1])
        bboxes_n.append([x0 / W, y0 / H, x1 / W, y1 / H])
        labels.append(d.get("label", f"obj{i}"))
        oids.append(d.get("object_id", i))
        m64 = d.get("mask")
        masks.append(decode_mask_b64(m64, size=(W, H)) if m64 else None)
        print(f"  [{i}]  oid={oids[-1]}  {labels[-1]:8s}  "
              f"box=({int(x0)}, {int(y0)}, {int(x1)}, {int(y1)})  "
              f"score={d.get('score', float('nan')):.2f}  "
              f"n_obs={d.get('n_obs', '-')}")
    bboxes_pix = np.asarray(bboxes_pix, dtype=np.float32)
    bboxes_n = np.asarray(bboxes_n, dtype=np.float32)
    n = len(bboxes_n)
    if n < 2:
        print("[skip]  need ≥2 bboxes")
        return 1

    # (2) Query LLM
    client = LLMRelationClient(model_name="gpt-5.1")
    print(f"\n[client] backend={client.backend}  model={client._model_name}")
    print("[call]   LLMRelationClient.detect(rgb, bboxes_n) ...")
    usable_masks = masks if all(m is not None for m in masks) else None
    p_parent = client.detect(rgb, bboxes_n, masks=usable_masks)
    if p_parent is None:
        print(f"[fail]   detect() returned None  (available={client.available})")
        return 2

    # (3) Print matrix
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print(f"\np_parent ({n}x{n}) — row i / col j = P(i parent of j):")
    print("       " + "  ".join(f"{labels[j][:6]:>6s}" for j in range(n)))
    for i in range(n):
        row = "  ".join(f"{p_parent[i, j]:6.2f}" for j in range(n))
        print(f"{labels[i][:6]:>6s}  {row}")
    edges = [(i, j, float(p_parent[i, j]))
             for i in range(n) for j in range(n)
             if i != j and p_parent[i, j] >= THR]
    print(f"\nEdges with score ≥ {THR}:")
    for i, j, s in sorted(edges, key=lambda t: -t[2]):
        print(f"  [{i}] {labels[i]} —({s:.2f})→ [{j}] {labels[j]}   "
              f"(oid {oids[i]} parent of oid {oids[j]})")

    # (4) Render overlay
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_relation_overlay(rgb, bboxes_pix, labels, p_parent,
                                      thr=THR,
                                      caption_prefix=f"frame {FRAME_IDX} — LLM relation overlay")
    overlay.save(OUT_PNG)
    print(f"\n[save]   {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
