"""Online relation-detector clients for the Bernoulli-EKF tracker.

Two backends share a single interface:

- :class:`RESTRelationClient` — wraps ``alpha_robot.client.SuppRelAfford``,
  calling its ``detect_support_graph`` endpoint (3-class:
  ``[parent, child, no_relation]``). Server URL defaults to the value in
  ``arobot.configs.IP_CONFIGS["SuppRelAfford"]``.
- :class:`LLMRelationClient` — prompts ``arobot.client.GPTChatBot`` with the
  RGB frame + numbered bounding boxes; parses JSON back to the same tensor.

The orchestrator consumes them through :func:`build_relation_client`, which
defers remote construction (server ping / API key check) until first use and
falls back to ``available=False`` on error so the tracker keeps running.

Both ``detect()`` calls return the same ``(N, N)`` ``p_parent`` matrix in
[0, 1], where ``p_parent[i, j]`` is the probability that object i is the
physical parent of j (i.e., j rests on / in i). Orchestrator per-edge EMA
(``RelationFilter``) handles temporal aggregation.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────
# Thread-local context for per-call metadata (used by the cache).
# ─────────────────────────────────────────────────────────────────────

_relation_ctx = threading.local()


def set_relation_context(frame_idx: Optional[int]) -> None:
    """Stash a per-call frame index so ``CachedRelationClient`` can use
    it as part of its cache key. The driver should call this immediately
    before invoking ``client.detect(...)`` so the metadata is in scope
    for the wrapped ``detect`` call.

    Pass ``None`` to clear (cache is then disabled for the next call).
    """
    if frame_idx is None:
        if hasattr(_relation_ctx, "frame"):
            del _relation_ctx.frame
    else:
        _relation_ctx.frame = int(frame_idx)


def _get_relation_frame() -> Optional[int]:
    return getattr(_relation_ctx, "frame", None)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Base interface
# ═════════════════════════════════════════════════════════════════════

class RelationClient:
    """Abstract base. Subclasses implement :meth:`detect`."""

    available: bool = False
    backend: str = "base"

    def detect(self,
               rgb: Image.Image,
               bboxes_norm: np.ndarray,
               masks: Optional[List[np.ndarray]] = None,
               ) -> Optional[np.ndarray]:
        """Return an ``(N, N)`` ``p_parent`` matrix, or ``None`` on failure.

        ``masks`` (optional) is a list of N binary ``(H, W)`` arrays aligned
        to ``bboxes_norm``. Backends that can use segmentation directly (LLM)
        will consume them; the REST server reads bboxes only.
        """
        raise NotImplementedError


class CachedRelationClient(RelationClient):
    """Drop-in wrapper around any ``RelationClient`` that persists each
    successful ``detect()`` result to disk and replays cached responses
    on identical inputs.

    Cache key fields:
      * frame index from the thread-local context (set via
        :func:`set_relation_context`);
      * number of detections ``n``;
      * per-detection bbox tuple, rounded to 3 decimals (suppresses
        floating-point noise on otherwise-identical inputs).

    Cache miss when the thread-local frame is ``None`` — the wrapped
    call still runs but the result is not persisted.
    """

    def __init__(self,
                 inner: RelationClient,
                 cache_dir: str,
                 verbose: bool = True):
        self._inner = inner
        self._cache_dir = cache_dir
        self._verbose = bool(verbose)
        self.available = bool(getattr(inner, "available", True))
        self.backend = f"cached:{getattr(inner, 'backend', 'base')}"
        os.makedirs(self._cache_dir, exist_ok=True)

    @staticmethod
    def _make_key(frame: int, bboxes_norm: np.ndarray) -> str:
        bb = np.asarray(bboxes_norm, dtype=np.float64).reshape(-1, 4)
        rounded = [tuple(round(float(v), 3) for v in row) for row in bb]
        payload = json.dumps(
            {"frame": int(frame), "n": int(bb.shape[0]),
             "bboxes": rounded},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, frame: int, key_hash: str) -> str:
        return os.path.join(
            self._cache_dir,
            f"relation_{int(frame):06d}_{key_hash}.json",
        )

    def detect(self,
               rgb: Image.Image,
               bboxes_norm: np.ndarray,
               masks: Optional[List[np.ndarray]] = None,
               ) -> Optional[np.ndarray]:
        frame = _get_relation_frame()
        if frame is None:
            # No frame context → no cache; fall through.
            return self._inner.detect(rgb, bboxes_norm, masks=masks)

        bb = np.asarray(bboxes_norm, dtype=np.float64).reshape(-1, 4)
        key_hash = self._make_key(frame, bb)
        path = self._cache_path(frame, key_hash)

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
                p = np.asarray(payload["p_parent"], dtype=np.float32)
                if self._verbose:
                    print(f"[relation] cache hit (frame={frame} "
                          f"key={key_hash} n={bb.shape[0]})")
                return p
            except Exception as e:
                logger.warning("relation cache load failed at %s: %s — "
                                "re-querying inner.", path, e)

        # Cache miss → call inner.
        result = self._inner.detect(rgb, bboxes_norm, masks=masks)
        if result is None:
            return None
        try:
            with open(path, "w") as f:
                json.dump({
                    "frame":     int(frame),
                    "key_hash":  key_hash,
                    "n":         int(bb.shape[0]),
                    "bboxes":    bb.tolist(),
                    "p_parent":  np.asarray(result, dtype=np.float32)
                                  .tolist(),
                }, f)
            if self._verbose:
                print(f"[relation] cache miss → wrote (frame={frame} "
                      f"key={key_hash} n={bb.shape[0]})")
        except Exception as e:
            logger.warning("relation cache write failed at %s: %s",
                           path, e)
        return result


def build_relation_client(
    backend: str = "rest",
    *,
    server_url: Optional[str] = None,
    llm_model: str = "gpt-5.1",
) -> RelationClient:
    """Factory. ``backend`` in ``{"rest", "llm"}``."""
    if backend == "rest":
        return RESTRelationClient(server_url=server_url)
    if backend == "llm":
        return LLMRelationClient(model_name=llm_model)
    raise ValueError(f"unknown relation backend: {backend!r}")


# ═════════════════════════════════════════════════════════════════════
#  REST backend — SuppRelAfford / detect_support_graph
# ═════════════════════════════════════════════════════════════════════

class RESTRelationClient(RelationClient):
    """Wraps ``SuppRelAfford.detect_support_graph``.

    Construction is lazy: the underlying ``SuppRelAfford`` client pings
    ``/available_apis`` on ``__init__``, which fails if the server is
    unreachable. We defer that to the first ``detect()`` call so the
    tracker can boot when the server is offline.
    """

    backend = "rest"

    def __init__(self, server_url: Optional[str] = None):
        self._server_url = server_url
        self._inner = None                 # SuppRelAfford instance (lazy)
        self._object_det_cls = None        # ObjectDetectionOutputs (lazy import)
        self._tried = False
        self.available = False

    def _lazy_init(self) -> bool:
        if self._tried:
            return self.available
        self._tried = True
        try:
            from arobot.client.relation_detectors.vmr_afford import SuppRelAfford
            from arobot.dtypes.objects import ObjectDetectionOutputs
            from arobot.configs import IP_CONFIGS
            url = self._server_url or IP_CONFIGS["SuppRelAfford"]
            self._inner = SuppRelAfford(server_url=url)
            self._object_det_cls = ObjectDetectionOutputs
            self.available = bool(getattr(self._inner, "is_supp_graph_available", False))
            if not self.available:
                logger.warning(
                    "SuppRelAfford server at %s has no /relation_det endpoint; "
                    "REST relation client disabled.", url,
                )
        except Exception as e:
            logger.warning("REST relation client unavailable: %s", e)
            self.available = False
        return self.available

    def detect(self,
               rgb: Image.Image,
               bboxes_norm: np.ndarray,
               masks: Optional[List[np.ndarray]] = None,
               ) -> Optional[np.ndarray]:
        if not self._lazy_init():
            return None
        if bboxes_norm is None or len(bboxes_norm) < 2:
            return None
        del masks  # REST server reads bboxes only
        try:
            objects = self._object_det_cls(image=rgb, bboxes=np.asarray(bboxes_norm))
            out = self._inner.detect_support_graph(image=rgb, objects=objects)
            # SceneGraphOutputs: relation_scores (N, N, 3) with
            # columns [parent, child, no_relation].
            scores = np.asarray(out.relation_scores, dtype=np.float32)
            if scores.ndim != 3 or scores.shape[-1] < 2:
                return None
            return scores[..., 0]  # p(i is parent of j)
        except Exception as e:
            logger.warning("REST relation detect() failed: %s", e)
            return None


# ═════════════════════════════════════════════════════════════════════
#  LLM backend — GPTChatBot
# ═════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = (
    "You are a scene-graph annotator. You receive an image in which each "
    "object is outlined by a thick colored segmentation contour (interior "
    "left transparent) and tagged with its integer index placed inside the "
    "mask. For every ordered pair (i, j), decide whether object i is the "
    "physical PARENT of j — that is, j rests on, or is contained inside, "
    "i. Output a single JSON object with no commentary:\n"
    "  { \"pairs\": [ { \"i\": int, \"j\": int, \"score\": float } ] }\n"
    "Include only pairs where the parent-of relation has score > 0.3. "
    "Score is your confidence in [0, 1]. Output JSON only."
)


class LLMRelationClient(RelationClient):
    """Prompts a VLM (default: ``GPTChatBot``) for the same support-graph output.

    Slower (~1-3 s per call) and non-deterministic but needs no model server.
    """

    backend = "llm"

    def __init__(self, model_name: str = "gpt-5.1"):
        self._model_name = model_name
        self._llm = None
        self._tried = False
        self.available = False

    def _lazy_init(self) -> bool:
        if self._tried:
            return self.available
        self._tried = True
        # Try alpha_robot's GPTChatBot first (both shadowed paths); if its
        # transitive imports fail (heavy arobot deps we don't need here),
        # fall back to a direct openai SDK call with the same API key
        # source (_personal_tokens.json["gpt"]).
        import importlib
        GPTChatBot = None
        for mod_path in ("client.chat_models.onlinechat",
                         "arobot.client.chat_models.onlinechat"):
            try:
                GPTChatBot = getattr(importlib.import_module(mod_path),
                                     "GPTChatBot")
                break
            except Exception as e:
                logger.debug("GPTChatBot import from %s failed: %s", mod_path, e)
        if GPTChatBot is not None:
            try:
                self._llm = GPTChatBot(model_name=self._model_name, temperature=0.0)
                self.available = True
                return True
            except Exception as e:
                logger.debug("GPTChatBot construction failed: %s", e)
        # Direct openai SDK fallback.
        try:
            from openai import OpenAI
            api_key = _load_openai_key()
            if not api_key:
                logger.warning("LLM relation client unavailable: "
                               "no OpenAI API key found")
                self.available = False
                return False
            self._llm = _DirectOpenAIChat(api_key=api_key,
                                          model_name=self._model_name)
            self.available = True
            return True
        except Exception as e:
            logger.warning("LLM relation client unavailable: %s", e)
            self.available = False
            return False

    def detect(self,
               rgb: Image.Image,
               bboxes_norm: np.ndarray,
               masks: Optional[List[np.ndarray]] = None,
               ) -> Optional[np.ndarray]:
        if not self._lazy_init():
            return None
        n = int(len(bboxes_norm))
        if n < 2:
            return None
        try:
            if masks is not None and len(masks) == n:
                img_annotated = _draw_mask_contours(rgb, masks)
            else:
                img_annotated = _draw_numbered_bboxes(rgb, bboxes_norm)
            prompt = _LLM_SYSTEM + "\n\nObjects:\n" + "\n".join(
                f"  {i}: bbox [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"
                for i, b in enumerate(bboxes_norm)
            )
            resp = self._llm.chat([prompt], image=img_annotated)
            text = resp if isinstance(resp, str) else (resp[0] if resp else "")
            obj = _extract_json(text)
            if obj is None or "pairs" not in obj:
                return None
            p = np.zeros((n, n), dtype=np.float32)
            for pair in obj["pairs"]:
                try:
                    i, j = int(pair["i"]), int(pair["j"])
                    s = float(pair["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= i < n and 0 <= j < n and i != j:
                    p[i, j] = max(0.0, min(1.0, s))
            return p
        except Exception as e:
            logger.warning("LLM relation detect() failed: %s", e)
            return None


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _draw_numbered_bboxes(rgb: Image.Image, bboxes_norm: np.ndarray) -> Image.Image:
    """Paint each bbox + its index onto a copy of ``rgb``."""
    from PIL import ImageDraw
    img = rgb.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(np.asarray(bboxes_norm)):
        x0, y0, x1, y1 = [float(v) for v in b]
        x0, x1 = x0 * W, x1 * W
        y0, y1 = y0 * H, y1 * H
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((x0 + 3, y0 + 3), str(i), fill=(255, 255, 0))
    return img


# tab10-style palette: bold, distinguishable, works on natural images.
_CONTOUR_PALETTE = [
    (31, 119, 180),    # blue
    (255, 127, 14),    # orange
    (44, 160, 44),     # green
    (214, 39, 40),     # red
    (148, 103, 189),   # purple
    (140, 86, 75),     # brown
    (227, 119, 194),   # pink
    (188, 189, 34),    # olive
    (23, 190, 207),    # cyan
    (127, 127, 127),   # gray
]


def _draw_mask_contours(rgb: Image.Image,
                        masks: List[np.ndarray],
                        thickness: int = 3) -> Image.Image:
    """Paint each binary mask's contour (no fill) in a distinct bold color,
    with its integer index drawn at the mask centroid.

    Parameters
    ----------
    rgb : PIL image (any mode; converted to RGB).
    masks : list of ``(H, W)`` arrays, convertible to boolean. Sizes must
        match the image (or each other; the image is resized to match if
        mismatched — but in the production pipeline the perception masks
        are already at full image resolution).
    thickness : pixel width of the drawn contour band.

    Interior is left untouched (fully transparent) so the LLM still sees
    the raw pixels inside each object.
    """
    from PIL import ImageDraw
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    img = rgb.convert("RGB").copy()
    W, H = img.size
    out = np.array(img, dtype=np.uint8)  # (H, W, 3)

    font_small = _load_font_compat(24)
    draw = ImageDraw.Draw(img)  # reused for text

    iters = max(1, int(thickness) // 2)
    for i, m in enumerate(masks):
        mask = np.asarray(m)
        if mask.dtype != bool:
            mask = mask > 0
        if mask.shape != (H, W):
            # Accept pix-values PIL masks too by last-chance resize.
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H))
            ) > 127
        if not mask.any():
            continue
        color = _CONTOUR_PALETTE[i % len(_CONTOUR_PALETTE)]
        dilated = binary_dilation(mask, iterations=iters)
        eroded = binary_erosion(mask, iterations=iters)
        contour = dilated & (~eroded)
        out[contour] = color

    # Write back the contour-painted pixels into the PIL image, then
    # overlay the index labels on top so text never gets covered by
    # another contour.
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for i, m in enumerate(masks):
        mask = np.asarray(m)
        if mask.dtype != bool:
            mask = mask > 0
        if not mask.any():
            continue
        # Label anchor: the point inside the mask that maximises
        #     score(p) = dist_to_boundary(p) − λ · ||p − COM||
        # which picks a thick, interior location while staying as close
        # to the mass centroid as possible. Handles non-convex, U/L-shaped,
        # and multi-component masks (the raw centroid would otherwise
        # fall outside or into a thin stripe).
        dist = distance_transform_edt(mask)
        ys_all, xs_all = np.nonzero(mask)
        cy_com = float(ys_all.mean())
        cx_com = float(xs_all.mean())
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        com_dist = np.sqrt((xx - cx_com) ** 2 + (yy - cy_com) ** 2)
        # λ chosen so a 1-px depth gain is worth ~0.5 px of proximity;
        # equivalent to "prefer deep, but don't wander past the body".
        score = np.where(mask, dist - 0.5 * com_dist, -np.inf)
        ly, lx = np.unravel_index(int(np.argmax(score)), score.shape)
        cx, cy = int(lx), int(ly)
        color = _CONTOUR_PALETTE[i % len(_CONTOUR_PALETTE)]
        tag = str(i)
        tw, th = draw.textbbox((0, 0), tag, font=font_small,
                               stroke_width=2)[2:]
        tx = max(0, min(W - tw - 2, cx - tw // 2))
        ty = max(0, min(H - th - 2, cy - th // 2))
        draw.text((tx, ty), tag, font=font_small, fill=color,
                  stroke_width=2, stroke_fill=(255, 255, 255))
    return img


def _load_font_compat(size: int):
    from PIL import ImageFont
    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def decode_mask_b64(b64_png: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Decode a base64-encoded PNG mask into an ``(H, W)`` bool array.

    Optional ``size = (W, H)`` resizes the decoded mask if its dimensions
    don't match the current frame. In the perception pipeline the masks
    are already at full image resolution so the resize branch is skipped.
    """
    raw = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(raw))
    if size is not None and img.size != size:
        img = img.resize(size)
    return np.asarray(img.convert("L"), dtype=np.uint8) > 127


def _load_openai_key() -> Optional[str]:
    """Locate an OpenAI API key.

    Search order:
      1) ``OPENAI_API_KEY`` env var.
      2) ``alpha_robot/arobot/_personal_tokens.json`` → ``"gpt"`` entry.
    """
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env
    try:
        from arobot.configs import PROJ_DIR
        tok_path = os.path.join(PROJ_DIR, "arobot", "_personal_tokens.json")
    except Exception:
        # Best-effort: guess the canonical sibling location.
        here = os.path.dirname(os.path.abspath(__file__))
        tok_path = os.path.join(
            here, "..", "..", "alpha_robot", "arobot", "_personal_tokens.json",
        )
    try:
        with open(tok_path) as f:
            tokens = json.load(f)
        key = tokens.get("gpt")
        if key and not key.startswith("REPLACE"):
            return key
    except Exception as e:
        logger.debug("token file %s unreadable: %s", tok_path, e)
    return None


class _DirectOpenAIChat:
    """Minimal OpenAI-SDK chat wrapper. Matches the one call site in
    :meth:`LLMRelationClient.detect`: ``chat(conversation, image=...)``."""

    def __init__(self, api_key: str, model_name: str = "gpt-5.1"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model_name

    def chat(self, conversation: List[str], image: Optional[Image.Image] = None,
              **_kwargs: Any) -> str:
        buf = io.BytesIO()
        (image or Image.new("RGB", (1, 1))).convert("RGB").save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        user_content: List[dict] = [
            {"type": "text", "text": conversation[0]},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
        messages = [{"role": "user", "content": user_content}]
        kwargs: dict = {"model": self.model, "messages": messages}
        # gpt-5 family uses max_completion_tokens instead of max_tokens.
        if self.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 2048
        else:
            kwargs["max_tokens"] = 2048
            kwargs["temperature"] = 0.0
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of an LLM response (may be code-fenced)."""
    if not text:
        return None
    # Strip common code fences.
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
