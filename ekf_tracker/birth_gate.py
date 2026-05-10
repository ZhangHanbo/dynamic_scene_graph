"""Tracker-side pending-birth buffer and admission policy (:func:`birth_admissible`)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _PendingBirth:
    """Holds the rolling state of a candidate birth: frames seen, score history, last fitness/RMSE."""
    source_id: Any              # perception's det["id"]; metadata only
    first_seen_frame: int
    last_seen_frame: int
    n_obs_tracker: int = 0      # frames seen unmatched in this tracker
    max_score: float = 0.0
    last_label: Optional[str] = None

    @classmethod
    def from_det(cls, det: Dict[str, Any], frame: int) -> "_PendingBirth":
        try:
            score = float(det.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return cls(
            source_id=det.get("id"),
            first_seen_frame=frame,
            last_seen_frame=frame,
            n_obs_tracker=0,
            max_score=score,
            last_label=det.get("label"),
        )

    def bump(self, det: Dict[str, Any], frame: int) -> None:
        self.last_seen_frame = frame
        self.n_obs_tracker += 1
        try:
            score = float(det.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > self.max_score:
            self.max_score = score
        if det.get("label"):
            self.last_label = det.get("label")


def birth_admissible(det: Dict[str, Any],
                      cfg,                       # BernoulliConfig (avoid import cycle)
                      image_shape: Optional[tuple],
                      *,
                      tracker_n_obs: Optional[int] = None,
                      tracker_max_score: Optional[float] = None,
                      require_pose: bool = True,
                      ) -> tuple:
    """True when a candidate has passed the temporal + score + fitness/RMSE birth gates."""
    if require_pose and det.get("T_co") is None:
        return False, "no_pose"

    margin = int(cfg.birth_border_margin_px)
    if margin > 0:
        box = det.get("box")
        if box is not None and image_shape is not None:
            try:
                x0, y0, x1, y1 = (float(b) for b in box)
                H_img, W_img = int(image_shape[0]), int(image_shape[1])
                if (x0 <= margin
                        or y0 <= margin
                        or x1 >= W_img - 1 - margin
                        or y1 >= H_img - 1 - margin):
                    return False, "border"
            except (TypeError, ValueError):
                pass

    k = int(cfg.birth_confirm_k)
    if k > 1:
        if tracker_n_obs is not None:
            if int(tracker_n_obs) < k:
                return False, "confirm"
        else:
            label = det.get("label")
            labels = det.get("labels") or {}
            n_obs = 0
            if (isinstance(labels, dict)
                    and isinstance(labels.get(label), dict)):
                try:
                    n_obs = int(labels[label].get("n_obs", 0))
                except (TypeError, ValueError):
                    n_obs = 0
            if n_obs == 0 and "n_obs" in det:
                try:
                    n_obs = int(det["n_obs"])
                except (TypeError, ValueError):
                    n_obs = 0
            if n_obs < k:
                return False, "confirm"

    if cfg.birth_score_min > 0.0:
        if tracker_max_score is not None:
            try:
                score = float(tracker_max_score)
            except (TypeError, ValueError):
                score = 0.0
        else:
            try:
                score = float(det.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
        if score < cfg.birth_score_min:
            return False, "score"

    if require_pose:
        if cfg.birth_fitness_min > 0.0:
            fit = det.get("fitness")
            if fit is not None:
                try:
                    if float(fit) < cfg.birth_fitness_min:
                        return False, "fitness"
                except (TypeError, ValueError):
                    pass
        if math.isfinite(cfg.birth_rmse_max):
            rmse = det.get("rmse")
            if rmse is not None:
                try:
                    if float(rmse) > cfg.birth_rmse_max:
                        return False, "rmse"
                except (TypeError, ValueError):
                    pass

    return True, "ok"
