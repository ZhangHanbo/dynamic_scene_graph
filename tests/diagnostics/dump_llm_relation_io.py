"""Dump the exact inputs/outputs of one ``LLMRelationClient.detect`` call.

Runs the real prompt-building path on apple_in_the_tray frame 399 and
captures:

  1. the annotated image sent to GPT (numbered bboxes drawn on the RGB),
     saved to ``tests/out/llm_io_frame000399_input.png``;
  2. the full text prompt appended to the system message, printed and
     saved to ``tests/out/llm_io_frame000399_prompt.txt``;
  3. the **raw** model response string (before any JSON parsing),
     saved to ``tests/out/llm_io_frame000399_response.txt``;
  4. what ``_extract_json`` parses out of that response.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ekf_tracker.relations.relation_client import (
    LLMRelationClient,
    _LLM_SYSTEM,
    _draw_mask_contours,
    _extract_json,
    decode_mask_b64,
)


FRAME_IDX = 399
ROOT = Path("/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP")
DATASET_RGB = Path(
    f"/Volumes/External/Workspace/datasets/apple_in_the_tray/rgb/rgb_{FRAME_IDX:06d}.png")
DET_JSON = (ROOT / "tests/visualization_pipeline/apple_in_the_tray/perception/"
            f"detection_h/detection_{FRAME_IDX:06d}_final.json")
OUT_DIR = ROOT / "tests/out"

OUT_IMG      = OUT_DIR / f"llm_io_frame{FRAME_IDX:06d}_input.png"
OUT_PROMPT   = OUT_DIR / f"llm_io_frame{FRAME_IDX:06d}_prompt.txt"
OUT_RESP     = OUT_DIR / f"llm_io_frame{FRAME_IDX:06d}_response.txt"
OUT_OVERLAY  = OUT_DIR / f"llm_io_frame{FRAME_IDX:06d}_response.png"


def main() -> int:
    rgb = Image.open(DATASET_RGB).convert("RGB")
    W, H = rgb.size
    with DET_JSON.open() as f:
        dets = json.load(f)["detections"]
    bboxes_n, labels, masks = [], [], []
    for d in dets:
        b = d.get("box")
        m64 = d.get("mask")
        if b is None or not m64:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)
        bboxes_n.append([x0 / W, y0 / H, x1 / W, y1 / H])
        labels.append(d.get("label", "?"))
        masks.append(decode_mask_b64(m64, size=(W, H)))  # reuses pre-computed SAM mask; no re-seg
    bboxes_n = np.asarray(bboxes_n, dtype=np.float32)
    n = len(bboxes_n)

    # --- 1) the annotated image the LLM actually sees (mask contours, no fill)
    img_annotated = _draw_mask_contours(rgb, masks)
    img_annotated.save(OUT_IMG)
    print(f"[img]    annotated input  →  {OUT_IMG}  ({W}x{H})")

    # --- 2) the text prompt (identical to what detect() builds)
    prompt = _LLM_SYSTEM + "\n\nObjects:\n" + "\n".join(
        f"  {i}: bbox [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"
        for i, b in enumerate(bboxes_n)
    )
    OUT_PROMPT.write_text(prompt + "\n")
    print(f"[prompt] saved           →  {OUT_PROMPT}  ({len(prompt)} chars)\n")

    banner = "═" * 70
    print(banner)
    print("PROMPT (text half of the user message; image goes in parallel):")
    print(banner)
    print(prompt)
    print(banner)

    for i, lab in enumerate(labels):
        print(f"  ({i} = {lab})")
    print()

    # --- 3) call the real LLM client and capture the raw text
    client = LLMRelationClient(model_name="gpt-5.1")
    if not client._lazy_init():
        print(f"[fail] client unavailable (available={client.available})")
        return 2

    # `client._llm.chat(...)` is the single line inside detect() that calls
    # the model. Replicate it exactly (note: detect() internally also uses
    # _draw_mask_contours when masks are supplied, which is what we built
    # img_annotated from above — so this round-trip is faithful).
    raw_text = client._llm.chat([prompt], image=img_annotated)
    if not isinstance(raw_text, str):
        raw_text = raw_text[0] if raw_text else ""
    OUT_RESP.write_text(raw_text + "\n")
    print(f"[save]   raw response    →  {OUT_RESP}  ({len(raw_text)} chars)\n")

    print(banner)
    print(f"RAW RESPONSE  (model = {client._model_name}):")
    print(banner)
    print(raw_text)
    print(banner)

    # --- 4) what detect() then parses out of that
    parsed = _extract_json(raw_text)
    print("\nParsed JSON:")
    print(json.dumps(parsed, indent=2) if parsed else "  (parse failed)")
    if parsed and "pairs" in parsed:
        print(f"\n{len(parsed['pairs'])} pair(s) would populate p_parent:")
        for p in parsed["pairs"]:
            i, j, s = p.get("i"), p.get("j"), p.get("score")
            name_i = labels[i] if isinstance(i, int) and 0 <= i < len(labels) else "?"
            name_j = labels[j] if isinstance(j, int) and 0 <= j < len(labels) else "?"
            print(f"  {name_i}[{i}] —({s})→ {name_j}[{j}]")

    # --- 5) overlay the response onto the image for visual check
    _render_response_overlay(img_annotated, masks, parsed, OUT_OVERLAY)
    print(f"\n[save]   response overlay →  {OUT_OVERLAY}")
    return 0


def _render_response_overlay(base_img: Image.Image,
                              masks,
                              parsed,
                              out_path: Path) -> None:
    """Draw parent→child arrows from the parsed LLM response on top of
    ``base_img`` (which is already the mask-contour annotated input)."""
    from PIL import ImageDraw, ImageFont
    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    def _font(sz):
        for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
            try:
                return ImageFont.truetype(name, size=sz)
            except Exception:
                pass
        return ImageFont.load_default()
    font = _font(14)

    # Mask COMs as arrow endpoints.
    centers = []
    for m in masks:
        m_ = np.asarray(m) > 0 if m is not None else None
        if m_ is None or not m_.any():
            centers.append(None)
            continue
        ys, xs = np.nonzero(m_)
        centers.append((float(xs.mean()), float(ys.mean())))

    def _arrow(p0, p1, color=(0, 230, 0, 255), width=3, head_len=14, head_w=10):
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / L, dy / L
        sx, sy = x1 - ux * head_len, y1 - uy * head_len
        draw.line([(x0, y0), (sx, sy)], fill=color, width=width)
        px, py = -uy, ux
        tip = (x1, y1)
        left = (sx + px * head_w * 0.5, sy + py * head_w * 0.5)
        right = (sx - px * head_w * 0.5, sy - py * head_w * 0.5)
        draw.polygon([tip, left, right], fill=color)

    n_drawn = 0
    if parsed and "pairs" in parsed:
        for p in parsed["pairs"]:
            try:
                i, j = int(p["i"]), int(p["j"])
                s = float(p["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= i < len(centers)) or not (0 <= j < len(centers)):
                continue
            if centers[i] is None or centers[j] is None:
                continue
            _arrow(centers[i], centers[j])
            mx = 0.5 * (centers[i][0] + centers[j][0])
            my = 0.5 * (centers[i][1] + centers[j][1])
            tag = f"{s:.2f}"
            tw, th = draw.textbbox((0, 0), tag, font=font,
                                   stroke_width=2)[2:]
            draw.text((mx - tw / 2, my - th / 2), tag, font=font,
                      fill=(0, 180, 0, 255),
                      stroke_width=2, stroke_fill=(255, 255, 255, 255))
            n_drawn += 1

    # Caption
    caption = f"frame {FRAME_IDX} — LLM response ({n_drawn} parent→child edges)"
    tw, th = draw.textbbox((0, 0), caption, font=font)[2:]
    draw.rectangle([4, 4, 8 + tw, 8 + th], fill=(0, 0, 0, 200))
    draw.text((8, 6), caption, font=font, fill=(255, 255, 255, 255))
    img.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
