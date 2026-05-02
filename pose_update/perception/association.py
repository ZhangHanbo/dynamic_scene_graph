"""
Data association for the Bernoulli-EKF tracker (bernoulli_ekf.tex §6).

Builds a cost matrix of Mahalanobis squared distances (with a chi^2 outer
gate, per-class feasibility, and optional SAM2 tracklet-ID continuity
bonus) and solves a linear assignment via scipy.optimize.linear_sum_assignment
(Jonker--Volgenant). The result is a dict track_oid -> det_index; tracks
not in the dict are unassigned, detection indices not hit are available
for birth.

A GT-oracle mode is provided for the degeneracy test: when invoked with
``oracle_mode=True``, the function bypasses the cost matrix and uses the
upstream ``det['id']`` field directly. Under oracle_mode, any track whose
oid does not appear in the detection list is marked unassigned, and
detections with an id not currently tracked become births. This replicates
the pre-Bernoulli behaviour of ``TwoTierOrchestrator._fast_tier`` for the
degeneracy check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# Large finite infeasibility cost (linear_sum_assignment does not accept
# true +inf with all-inf rows/columns without special-case handling).
_INFEASIBLE: float = 1e12


def _label_in_history_meaningful(
    t_history: Any,
    d_label: Optional[str],
    min_obs: int = 5,
    min_share: float = 0.05,
) -> bool:
    """Whether `d_label` is a *meaningful* member of a track's label
    history.

    A label "is in history" only if it has BOTH (a) ≥ ``min_obs``
    observations AND (b) ≥ ``min_share`` of the track's total
    observations. This guards the soft-label-penalty bypass against
    polluted histories — e.g., a track with 53,990 "apple" obs and 3
    stray "bowl" obs (purity 0.006%) should NOT be considered
    permissive of a bowl detection.

    Accepts the same shapes as the legacy membership check:
      * Mapping[label, {n_obs, ...}] — uses `n_obs` for purity.
      * Set / List / Mapping without n_obs — falls back to pure
        membership (legacy behaviour preserved).
      * None — always False.
    """
    if not t_history or d_label is None:
        return False
    if isinstance(t_history, Mapping):
        rec = t_history.get(d_label)
        if rec is None:
            return False
        if isinstance(rec, Mapping) and "n_obs" in rec:
            def _n(r: Any) -> int:
                return (int(r.get("n_obs", 0))
                        if isinstance(r, Mapping) else 1)
            n_d = _n(rec)
            if n_d < min_obs:
                return False
            total = sum(_n(r) for r in t_history.values()) or 1
            return (n_d / total) >= min_share
        # Mapping without n_obs payload — pure membership.
        return True
    return d_label in t_history


@dataclass
class AssociationResult:
    """Output of a single association pass.

    Attributes:
        match: dict mapping track oid -> detection index it was matched to.
               A track oid NOT in this dict is unassigned.
        unmatched_tracks: list of track oids with no detection.
        unmatched_detections: list of detection indices with no track.
        cost_matrix: (n_tracks, n_dets) total cost used (diagnostic).
        gated_pairs: number of (track, det) pairs inside the outer gate
                     that passed the label filter (diagnostic).
        d2_full_matrix: (n_tracks, n_dets) full 6-D Mahalanobis squared
                        distance per pair (NaN where infeasible). For
                        diagnostics: ν^T S^-1 ν.
        d2_trans_matrix: (n_tracks, n_dets) translation-only d^2 per
                         pair: ν_t^T S_tt^-1 ν_t (NaN where infeasible).
        d2_rot_matrix: (n_tracks, n_dets) rotation-only d^2 per pair
                       (NaN where infeasible).
    """
    match: Dict[int, int]
    unmatched_tracks: List[int]
    unmatched_detections: List[int]
    cost_matrix: np.ndarray
    gated_pairs: int
    d2_full_matrix: Optional[np.ndarray] = None
    d2_trans_matrix: Optional[np.ndarray] = None
    d2_rot_matrix: Optional[np.ndarray] = None


def hungarian_associate(
    track_oids: List[int],
    detections: List[Dict[str, Any]],
    innovation_fn: Callable[[int, np.ndarray, np.ndarray],
                            Optional[Tuple[np.ndarray, np.ndarray,
                                           float, float]]],
    track_labels: Mapping[int, str],
    track_tau: Optional[Mapping[int, int]] = None,
    alpha: float = 0.0,
    G_out: float = 25.0,
    enforce_label_match: bool = True,
    track_label_histories: Optional[Mapping[int, Any]] = None,
    label_penalty: float = 0.0,
    score_weight: float = 0.0,
    gate_mode: str = "full",
    G_out_trans: float = 21.108,
    G_out_rot: float = 21.108,
    cost_d2_mode: str = "full",
    max_residual_m: Optional[float] = None,
) -> AssociationResult:
    """Solve the Hungarian assignment of tracks to detections.

    Cost (paper-baseline eq:cost_sam2 plus the perception-pipeline soft
    label / score additions; the perception pipeline at
    `rosbag2dataset/sam2/sam2_client.py::_pair_cost` uses the same
    formula in IoU units):

        C[i, l] =   d2[i, l]
                  - alpha  * 1[tau_l == tau_i]
                  + label_penalty * 1[d_label NOT in track i's label history]
                  + score_weight  * (1 - score_l)
                  (feasible)
        C[i, l] = +INFEASIBLE      (infeasible)

    A pair is infeasible if EITHER:
      (a) `enforce_label_match=True` AND label strings disagree (paper
          hard-gate behaviour), OR
      (b) `d2 > G_out` (chi^2 outer gate, default chi^2_6(0.9997) = 25).

    Set `enforce_label_match=False` and `label_penalty>0` to switch to
    the perception-pipeline-style SOFT label gate: a class-label mismatch
    becomes a finite additive cost rather than an infeasibility, so a
    geometrically excellent (small d2) pair can override a label
    disagreement when the labels are noisy. This mirrors the perception
    pipeline's `(1 - IoU) + label_pen + score_pen` cost.

    Args:
        track_oids: ordered list of track object ids currently tracked.
        detections: per-frame detections in the orchestrator's canonical
            dict format (must contain 'T_co', 'R_icp', 'label'; 'sam2_id'
            and 'score' optional).
        innovation_fn: callable (oid, T_co, R_icp) -> (nu, S, d2, log_lik)
            or None when the innovation is undefined.
        track_labels: mapping track oid -> primary class label string.
        track_tau: optional mapping track oid -> last SAM2 tracklet id;
            missing / negative -> no alpha bonus.
        alpha: SAM2 continuity bonus in chi^2_6 units (paper default 4.4;
            0 disables).
        G_out: chi^2_6 outer-gate threshold. Pairs with d2 above this are
            infeasible regardless of label.
        enforce_label_match: True -> hard label gate (paper); False ->
            soft additive label penalty (perception-style).
        track_label_histories: optional mapping track oid -> container of
            labels ever associated with that track. Acceptable forms:
              * a Set[str] / List[str] / dict[str, ...]   (only key
                membership matters: `d_label in history`).
            When None, falls back to the singleton `{track_labels[oid]}`.
        label_penalty: chi^2_6-unit additive penalty applied to a pair
            when (1) `enforce_label_match=False` AND (2) the detection's
            label is NOT in the track's label history. Default 0 disables
            the soft gate. Reasonable values: 4-8 (less than G_out so a
            wrong-label same-position pair can still match; large enough
            that a same-label pair is preferred when both are feasible).
        score_weight: chi^2_6-unit additive weight on `(1 - det.score)`
            applied to every feasible pair. Mildly disfavours
            low-confidence detections without making them infeasible.
            Default 0 disables.
        gate_mode: which Mahalanobis component drives the outer gate.
            'full'  -> gate on the full 6-D d^2 vs `G_out`
                       (paper-baseline behaviour).
            'trans' -> gate on translation-only d^2 vs `G_out_trans`
                       (chi^2_3 quantile, default 21.108 = 99.97%).
                       Useful when ICP rotation is unreliable (chained
                       per-oid ICP drifts even at the same physical
                       position; the rot block of d^2 then saturates
                       the full gate spuriously).
            'trans_and_rot' -> gate on BOTH translation (vs `G_out_trans`)
                       and rotation (vs `G_out_rot`) blocks. A pair is
                       feasible iff both pass.
        G_out_trans: chi^2_3 outer-gate threshold for the translation
            block. Default 21.108 = chi^2_3(0.9997).
        G_out_rot: chi^2_3 outer-gate threshold for the rotation block,
            used only by `gate_mode='trans_and_rot'`. Default 21.108
            = chi^2_3(0.9997).
        cost_d2_mode: which d^2 enters the cost matrix.
            'full'  -> cost = d^2_full - alpha*..  (paper baseline)
            'trans' -> cost = d^2_trans - alpha*.. (drops rotation from
                       the cost; rotation only gates / not gates).
            'sum'   -> cost = d^2_trans + 0.1*d^2_rot - alpha*..
                       (heavy translation, light rotation tie-break).
        max_residual_m: optional hard cap on the world-frame
            translation residual ‖ν[:3]‖ (metres). Catches the
            inflated-covariance pathology: under long miss runs the
            chi² gate becomes meaningless because σ_trans grows
            without bound, and any "nearby-ish" detection passes.
            Default None disables the absolute gate.

    Returns:
        AssociationResult.
    """
    n_tracks = len(track_oids)
    n_dets = len(detections)

    cost = np.full((max(n_tracks, 1), max(n_dets, 1)), _INFEASIBLE,
                   dtype=np.float64)
    log_liks = np.full_like(cost, -np.inf, dtype=np.float64)
    d2_full_mat = np.full_like(cost, np.nan, dtype=np.float64)
    d2_trans_mat = np.full_like(cost, np.nan, dtype=np.float64)
    d2_rot_mat = np.full_like(cost, np.nan, dtype=np.float64)
    gated = 0

    if n_tracks == 0 or n_dets == 0:
        match: Dict[int, int] = {}
        unmatched_tracks = list(track_oids)
        unmatched_dets = list(range(n_dets))
        return AssociationResult(
            match=match,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
            cost_matrix=cost,
            gated_pairs=gated,
            d2_full_matrix=d2_full_mat,
            d2_trans_matrix=d2_trans_mat,
            d2_rot_matrix=d2_rot_mat,
        )

    # Fill the cost matrix.
    for i, oid in enumerate(track_oids):
        t_label = track_labels.get(oid, None)
        t_history = (track_label_histories.get(oid)
                      if track_label_histories is not None else None)
        t_tau = None
        if track_tau is not None:
            t_tau_val = track_tau.get(oid, -1)
            t_tau = int(t_tau_val) if (t_tau_val is not None
                                       and t_tau_val >= 0) else None
        for l, det in enumerate(detections):
            d_label = det.get("label", None)

            # (a) Hard label gate (paper default).
            if enforce_label_match:
                if t_label is not None and d_label is not None \
                        and d_label != t_label:
                    continue

            T_co = det.get("T_co")
            R_icp = det.get("R_icp")
            if T_co is None or R_icp is None:
                continue
            stats = innovation_fn(oid, np.asarray(T_co, dtype=np.float64),
                                  np.asarray(R_icp, dtype=np.float64))
            if stats is None:
                continue
            nu, S, d2, log_lik = stats

            # Block decomposition: translation (top-left 3x3 of S) and
            # rotation (bottom-right 3x3 of S) Mahalanobis distances.
            # Cross-coupling through the off-diagonal of S is dropped here;
            # for typical block-diagonal-dominant S (the case for
            # right-multiplicative SE(3) noise) the sum d2_t + d2_r is a
            # tight bound on d2_full.
            try:
                S_tt = S[:3, :3]
                S_rr = S[3:, 3:]
                Sinv_t_nu = np.linalg.solve(S_tt, nu[:3])
                Sinv_r_nu = np.linalg.solve(S_rr, nu[3:])
                d2_trans = float(nu[:3] @ Sinv_t_nu)
                d2_rot = float(nu[3:] @ Sinv_r_nu)
            except np.linalg.LinAlgError:
                d2_trans = float("inf")
                d2_rot = float("inf")
            d2_full_mat[i, l] = float(d2)
            d2_trans_mat[i, l] = d2_trans
            d2_rot_mat[i, l] = d2_rot

            # (b) Outer gate. The mode picks which Mahalanobis component
            # has to fit inside the chi^2 quantile.
            if gate_mode == "trans":
                if not math.isfinite(d2_trans) or d2_trans > G_out_trans:
                    continue
            elif gate_mode == "trans_and_rot":
                if (not math.isfinite(d2_trans)
                        or d2_trans > G_out_trans
                        or not math.isfinite(d2_rot)
                        or d2_rot > G_out_rot):
                    continue
            else:  # 'full' (paper baseline)
                if not math.isfinite(d2) or d2 > G_out:
                    continue

            # (b') Hard absolute-distance gate. The chi² gate above
            # is *probabilistic*: it goes lax when the track's σ has
            # ballooned during long miss runs. A 0.86 m residual
            # against σ ≈ 0.37 m gives d² ≈ 17 — under any chi² gate.
            # `max_residual_m` puts a sanity cap on the world-frame
            # translation residual itself.
            if max_residual_m is not None:
                if float(np.linalg.norm(nu[:3])) > max_residual_m:
                    continue

            # Pick which d^2 enters the cost.
            if cost_d2_mode == "trans":
                d2_for_cost = d2_trans
            elif cost_d2_mode == "sum":
                d2_for_cost = d2_trans + 0.1 * d2_rot
            else:
                d2_for_cost = d2

            # SAM2 continuity bonus.
            bonus = 0.0
            if alpha > 0.0 and t_tau is not None:
                d_tau_raw = det.get("sam2_id", det.get("id"))
                if d_tau_raw is not None:
                    d_tau = int(d_tau_raw)
                    if d_tau >= 0 and d_tau == t_tau:
                        bonus = alpha

            # Soft label penalty (perception-style). Only applies when
            # the hard label gate is OFF; otherwise the cell would have
            # been infeasible above anyway.
            label_pen = 0.0
            if (not enforce_label_match
                    and label_penalty > 0.0
                    and d_label is not None):
                # Purity-aware membership: a stale trace label (3 vs
                # 53,990) must NOT bypass the penalty. `t_history`
                # without n_obs payload still resolves to legacy
                # membership semantics for back-compat.
                if t_history is not None:
                    in_history = _label_in_history_meaningful(
                        t_history, d_label)
                else:
                    in_history = (t_label is not None
                                   and d_label == t_label)
                if not in_history:
                    label_pen = float(label_penalty)

            # Score penalty (perception-style). Per-detection bias --
            # mildly disfavours low-confidence detections.
            score_pen = 0.0
            if score_weight > 0.0:
                d_score = float(det.get("score", 1.0))
                score_pen = float(score_weight) * max(0.0, 1.0 - d_score)

            cost[i, l] = float(d2_for_cost) - bonus + label_pen + score_pen
            log_liks[i, l] = float(log_lik)
            gated += 1

    # Solve assignment. linear_sum_assignment minimises the total cost.
    # We pad the cost with _INFEASIBLE to handle rectangular inputs.
    row_ind, col_ind = linear_sum_assignment(cost[:n_tracks, :n_dets])

    match: Dict[int, int] = {}
    matched_track_rows = set()
    matched_det_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= _INFEASIBLE:
            continue
        oid = track_oids[r]
        match[oid] = int(c)
        matched_track_rows.add(r)
        matched_det_cols.add(c)

    unmatched_tracks = [track_oids[i] for i in range(n_tracks)
                        if i not in matched_track_rows]
    unmatched_dets = [l for l in range(n_dets)
                      if l not in matched_det_cols]

    return AssociationResult(
        match=match,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_dets,
        cost_matrix=cost[:n_tracks, :n_dets],
        gated_pairs=gated,
        d2_full_matrix=d2_full_mat[:n_tracks, :n_dets],
        d2_trans_matrix=d2_trans_mat[:n_tracks, :n_dets],
        d2_rot_matrix=d2_rot_mat[:n_tracks, :n_dets],
    )


def oracle_associate(
    track_oids: List[int],
    detections: List[Dict[str, Any]],
) -> AssociationResult:
    """Ground-truth association from the upstream ``det['id']`` field.

    Used by the degeneracy test: pre-Bernoulli ``_fast_tier`` looked up
    tracks directly by ``det['id']``. Calling this function and wiring
    ``AssociationResult.match`` into the same update path reproduces that
    behaviour exactly (substitution A.1 of the reduction in the paper).

    A track oid present in the track list but absent from any detection's
    id is unmatched (routed to the miss branch). A detection whose id is
    NOT in the track list is unmatched (routed to birth).
    """
    n_tracks = len(track_oids)
    n_dets = len(detections)
    track_set = set(track_oids)

    match: Dict[int, int] = {}
    matched_det_cols: set = set()
    for l, det in enumerate(detections):
        d_id = det.get("id")
        if d_id is None:
            continue
        d_id = int(d_id)
        if d_id not in track_set:
            continue
        # Only one detection per track under oracle mode (first wins).
        if d_id in match:
            continue
        match[d_id] = l
        matched_det_cols.add(l)

    unmatched_tracks = [oid for oid in track_oids if oid not in match]
    unmatched_dets = [l for l in range(n_dets) if l not in matched_det_cols]
    return AssociationResult(
        match=match,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_dets,
        cost_matrix=np.zeros((n_tracks, n_dets)),
        gated_pairs=len(match),
    )


def sam2_alpha_from_q_s(q_s: float) -> float:
    """Helper: alpha = 2 log(q_s / (1 - q_s)) (bernoulli_ekf.tex §6.1)."""
    q_s = float(q_s)
    if q_s <= 0.5:
        return 0.0
    if q_s >= 1.0:
        return float("inf")
    return 2.0 * math.log(q_s / (1.0 - q_s))
