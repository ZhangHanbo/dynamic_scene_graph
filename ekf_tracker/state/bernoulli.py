"""
Bernoulli existence-probability updates for the Bernoulli-EKF tracker
(bernoulli_ekf.tex §5 predict, §7 update, §8 birth).

Each tracked object carries a scalar existence probability r in [0, 1] --
the prior probability that the index refers to a physically-present
object. This module implements the four Bayes-derived updates on r:

  * r_predict(r, p_s)                    -- §5 (eq. eq:bern_pred_r)
  * r_assoc_update(r, L, p_d, lambda_c)  -- §7 (eq. eq:r_assoc)
  * r_miss_update(r, p_d_tilde)          -- §7 (eq. eq:r_miss)
  * r_birth(score, lambda_b, lambda_c)   -- §8 (eq. eq:birth_r)

All functions are scalar in, scalar out -- the caller loops over tracks.

A note on the SAM2-ID prior (bernoulli_ekf.tex §6.1): the continuity
bonus enters ONLY the association cost matrix, not the existence update.
Therefore r_assoc_update here consumes the pure pose likelihood L from
`rbpf_state.innovation_stats`; it is NOT multiplied by the SAM2-ID
likelihood ratio. This matches the `Scope of the modification' paragraph
in the paper and avoids double-counting (tau and z both derive from the
same mask).
"""

from __future__ import annotations

import math
from typing import Final

# Numerical floor to keep logits finite.
_EPS: Final[float] = 1e-12


def r_predict(r: float, p_s: float = 1.0) -> float:
    """Bernoulli prediction of the existence probability (eq. eq:bern_pred_r).

        r_{k|k-1} = p_s * r_{k-1|k-1}

    p_s is the observer-independent physical survival probability between
    consecutive frames. Default 1.0: rigid manipulation objects do not
    spontaneously disappear. Setting p_s < 1 models a known physical-removal
    hazard and decays r each frame even without observations.

    Clamps output to [0, 1] for numerical safety.
    """
    p_s = float(p_s)
    r = float(r)
    return max(0.0, min(1.0, p_s * r))


def r_assoc_update(r_pred: float,
                   L: float,
                   p_d: float = 0.9,
                   lambda_c: float = 1.0) -> float:
    """Bernoulli existence update on association (eq. eq:r_assoc).

        r^+ = p_d * L * r / (p_d * L * r + lambda_c * (1 - r))

    Args:
        r_pred:   predicted existence probability r_{k|k-1}.
        L:        pose-likelihood N(nu; 0, S), strictly positive; supply
                  exp(log_lik) where log_lik comes from
                  `rbpf_state.innovation_stats`. Log-space callers should
                  switch to `r_assoc_update_loglik` to avoid underflow.
        p_d:      detection probability of a present, visible object.
        lambda_c: clutter density (detections/frame).

    Returns:
        posterior r^+ in [0, 1].
    """
    r_pred = float(r_pred)
    L = float(L)
    p_d = float(p_d)
    lambda_c = float(lambda_c)

    num = p_d * L * r_pred
    den = num + lambda_c * (1.0 - r_pred)
    if den <= 0.0 or not math.isfinite(den):
        # If both numerator and denominator vanish, fall back to r_pred
        # (no update). If only the denominator vanishes numerically, the
        # association outcome is effectively impossible under both
        # hypotheses; keep r_pred.
        return max(0.0, min(1.0, r_pred))
    return max(0.0, min(1.0, num / den))


def r_assoc_update_loglik(r_pred: float,
                          log_L: float,
                          p_d: float = 0.9,
                          lambda_c: float = 1.0) -> float:
    """Log-space variant of `r_assoc_update`.

    Computes the same ratio using log-sum-exp to avoid underflow when the
    innovation is tens of sigma outside the gate (as can happen in initial
    frames before ICP converges).

        r^+ = exp(log(p_d r) + log_L) / (exp(log(p_d r) + log_L) +
                                          exp(log(lambda_c (1-r))))
    """
    r_pred = float(r_pred)
    p_d = float(p_d)
    lambda_c = float(lambda_c)
    log_L = float(log_L)

    # Edge cases
    if r_pred <= 0.0 or p_d <= 0.0:
        return 0.0
    if r_pred >= 1.0:
        return 1.0

    log_a = math.log(p_d * r_pred + _EPS) + log_L
    log_b = math.log(lambda_c * (1.0 - r_pred) + _EPS)
    m = max(log_a, log_b)
    if not math.isfinite(m):
        return max(0.0, min(1.0, r_pred))
    log_den = m + math.log(math.exp(log_a - m) + math.exp(log_b - m))
    log_post = log_a - log_den
    return max(0.0, min(1.0, math.exp(log_post)))


def r_miss_update(r_pred: float, p_d_tilde: float) -> float:
    """Bernoulli misdetection update (eq. eq:r_miss).

        r^+ = (1 - p_d_tilde) * r / ((1 - p_d_tilde) * r + (1 - r))

    Args:
        r_pred:    predicted existence probability r_{k|k-1}.
        p_d_tilde: state-dependent detection probability p_d * p_v^{(i)}
                   (pure p_d when the track is visible; 0 when it is
                   occluded/out-of-FOV, in which case r is preserved).

    Returns:
        posterior r^+ in [0, 1].
    """
    r_pred = float(r_pred)
    p_d_tilde = float(p_d_tilde)

    if r_pred <= 0.0:
        return 0.0
    if r_pred >= 1.0 and p_d_tilde >= 1.0:
        return 0.0
    if p_d_tilde <= 0.0:
        # No penalty for not seeing the unseen: r unchanged.
        return r_pred

    num = (1.0 - p_d_tilde) * r_pred
    den = num + (1.0 - r_pred)
    if den <= 0.0 or not math.isfinite(den):
        return max(0.0, min(1.0, r_pred))
    return max(0.0, min(1.0, num / den))


def r_birth(score: float,
            lambda_b: float = 1.0,
            lambda_c: float = 1.0) -> float:
    """Birth existence probability from a calibrated detection score
    (eq. eq:birth_r).

        r_new = lambda_b * s / (lambda_b * s + lambda_c)

    Reduces to s when lambda_b = lambda_c; interpolates between pure-score
    confidence (lambda_b >> lambda_c) and a clutter-dominated fallback
    (lambda_c >> lambda_b). The score itself is treated as a calibrated
    likelihood ratio between new-object and clutter hypotheses, so an
    uncalibrated detector should be temperature-scaled before calling.

    Returns:
        r_new in [0, 1]. If both rates are 0 or the score is NaN, returns
        0 to avoid spawning tracks on degenerate input.
    """
    score = float(score)
    lambda_b = float(lambda_b)
    lambda_c = float(lambda_c)
    score = max(0.0, min(1.0, score))
    num = lambda_b * score
    den = num + lambda_c
    if den <= 0.0 or not math.isfinite(den):
        return 0.0
    return max(0.0, min(1.0, num / den))
