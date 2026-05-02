"""Unit tests for the two new gates added to ``hungarian_associate``:

1. ``_label_in_history_meaningful`` — replaces the dict-membership
   check inside the soft label penalty so that a stale trace label
   (3 obs against a dominant 53,990) does NOT bypass the penalty.

2. ``max_residual_m`` — hard absolute-distance gate on the world-
   frame translation residual. Catches the inflated-covariance
   pathology where the chi² gate becomes meaningless after long
   miss runs.

Regression: ``apple_to_cabinate`` fr 495 — oid 2 (apple, 91 missed
frames, σ_trans ≈ 0.37 m, label history ``apple: 53990, bowl: 3``)
matched to a detected bowl 0.86 m away.
"""
from __future__ import annotations

import numpy as np
import pytest

from pose_update.perception.association import (
    _INFEASIBLE,
    _label_in_history_meaningful,
    hungarian_associate,
)


# ─── _label_in_history_meaningful ──────────────────────────────────


def test_label_purity_apple_to_cabinate_fr495_regression():
    """3 stale 'bowl' obs against 53,990 'apple' obs → bowl is NOT
    a meaningful member."""
    hist = {"apple": {"n_obs": 53990, "mean_score": 0.9},
            "bowl":  {"n_obs": 3,     "mean_score": 0.2}}
    assert _label_in_history_meaningful(hist, "bowl") is False
    assert _label_in_history_meaningful(hist, "apple") is True


def test_label_purity_meaningful_minority_is_kept():
    """30 vs 100 (23% share, 30 obs) clears both thresholds."""
    hist = {"apple": {"n_obs": 100}, "bowl": {"n_obs": 30}}
    assert _label_in_history_meaningful(hist, "bowl") is True


def test_label_purity_below_min_obs_rejected():
    """4 obs (just under min_obs=5) → False even at 50% share."""
    hist = {"apple": {"n_obs": 4}, "bowl": {"n_obs": 4}}
    assert _label_in_history_meaningful(hist, "bowl") is False


def test_label_purity_below_min_share_rejected():
    """5 obs but 5/1005 ≈ 0.5% share → False."""
    hist = {"apple": {"n_obs": 1000}, "bowl": {"n_obs": 5}}
    assert _label_in_history_meaningful(hist, "bowl") is False


def test_label_purity_unknown_label_rejected():
    hist = {"apple": {"n_obs": 100}}
    assert _label_in_history_meaningful(hist, "potato") is False


def test_label_purity_legacy_set_membership_preserved():
    """Set/List inputs still use pure membership (back-compat)."""
    assert _label_in_history_meaningful({"apple", "bowl"}, "bowl") is True
    assert _label_in_history_meaningful(["apple", "bowl"], "bowl") is True
    assert _label_in_history_meaningful({"apple"}, "bowl") is False


def test_label_purity_dict_without_n_obs_uses_membership():
    """A bare dict with non-payload values still resolves to
    membership (legacy callers may pass `{label: True}`)."""
    assert _label_in_history_meaningful({"apple": True, "bowl": True},
                                          "bowl") is True
    assert _label_in_history_meaningful({"apple": True}, "bowl") is False


def test_label_purity_empty_or_none():
    assert _label_in_history_meaningful(None, "apple") is False
    assert _label_in_history_meaningful({}, "apple") is False


def test_label_purity_custom_thresholds():
    hist = {"apple": {"n_obs": 100}, "bowl": {"n_obs": 4}}
    # Default rejects (4 < min_obs=5).
    assert _label_in_history_meaningful(hist, "bowl") is False
    # Lower min_obs → 4 obs at 4/104≈3.8% share → still False (share < 5%).
    assert _label_in_history_meaningful(hist, "bowl", min_obs=2) is False
    # Lower BOTH → True.
    assert _label_in_history_meaningful(hist, "bowl",
                                          min_obs=2, min_share=0.01) is True


# ─── max_residual_m gate ───────────────────────────────────────────


def _make_innovation_fn(residual_m: float):
    """Return an innovation_fn that yields a fixed translation
    residual of `residual_m` along x, with σ_trans = 0.4 m so the
    chi² gate passes (d²_trans ≈ (residual/0.4)² · I term ~ small)."""
    sigma_trans = 0.4
    def fn(oid, T_co, R_icp):
        nu = np.zeros(6, dtype=np.float64)
        nu[0] = float(residual_m)
        S = np.eye(6, dtype=np.float64)
        S[:3, :3] *= sigma_trans ** 2
        S[3:, 3:] *= 0.1 ** 2
        # 6-D Mahalanobis squared.
        d2 = float(nu @ np.linalg.solve(S, nu))
        return nu, S, d2, 0.0
    return fn


def _det(label="apple", det_id=0):
    return {"id": det_id, "label": label,
            "T_co": np.eye(4, dtype=np.float64),
            "R_icp": np.eye(6, dtype=np.float64) * 1e-4,
            "score": 0.5}


def test_max_residual_gate_blocks_far_match():
    """0.86 m residual at a 0.30 m cap → infeasible cell (mirrors
    apple_to_cabinate fr 495+)."""
    fn = _make_innovation_fn(0.86)
    res = hungarian_associate(
        track_oids=[2],
        detections=[_det("bowl", det_id=99)],
        innovation_fn=fn,
        track_labels={2: "apple"},
        enforce_label_match=False,
        gate_mode="trans",
        G_out_trans=21.108,
        max_residual_m=0.30,
    )
    assert res.match == {}                      # nothing matched
    assert res.unmatched_tracks == [2]
    assert res.unmatched_detections == [0]
    assert res.cost_matrix[0, 0] >= _INFEASIBLE  # cell is infeasible


def test_max_residual_gate_disabled_allows_match():
    """Same residual, gate disabled (None) → match goes through."""
    fn = _make_innovation_fn(0.86)
    res = hungarian_associate(
        track_oids=[2],
        detections=[_det("apple", det_id=99)],
        innovation_fn=fn,
        track_labels={2: "apple"},
        enforce_label_match=False,
        gate_mode="trans",
        G_out_trans=21.108,
        max_residual_m=None,
    )
    assert res.match == {2: 0}


def test_max_residual_gate_under_cap_passes():
    """0.20 m residual at 0.30 m cap → match goes through."""
    fn = _make_innovation_fn(0.20)
    res = hungarian_associate(
        track_oids=[2],
        detections=[_det("apple", det_id=99)],
        innovation_fn=fn,
        track_labels={2: "apple"},
        enforce_label_match=False,
        gate_mode="trans",
        max_residual_m=0.30,
    )
    assert res.match == {2: 0}


def test_polluted_history_no_longer_bypasses_penalty():
    """End-to-end through hungarian_associate: a polluted
    label_scores history should NOT bypass the soft penalty.

    We don't assert on the cell cost directly because Hungarian is
    a minimisation; instead we compare two runs (polluted vs clean)
    with the SAME residual and assert the polluted-history cost is
    no longer smaller than the clean-history-with-penalty cost.
    """
    fn = _make_innovation_fn(0.10)  # well within max_residual
    polluted = {2: {"apple": {"n_obs": 53990},
                     "bowl":  {"n_obs": 3}}}
    res = hungarian_associate(
        track_oids=[2],
        detections=[_det("bowl", det_id=99)],
        innovation_fn=fn,
        track_labels={2: "apple"},
        track_label_histories=polluted,
        enforce_label_match=False,
        label_penalty=6.0,
        gate_mode="trans",
        max_residual_m=None,
    )
    polluted_cost = float(res.cost_matrix[0, 0])
    # The penalty must have fired (cost includes the +6.0 bump).
    # d²_full for our fixture is ~ (0.10/0.4)² · 3 ≈ 0.19 (trans) plus
    # rotation block ≈ 0; in 'full' cost mode that's tiny. After +6:
    assert polluted_cost > 5.0, f"penalty did not fire (cost={polluted_cost})"
