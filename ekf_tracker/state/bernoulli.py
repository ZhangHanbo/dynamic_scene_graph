"""Pure-function Bernoulli existence updates: predict, association, miss, birth."""

from __future__ import annotations

import math
from typing import Final

# Numerical floor to keep logits finite.
_EPS: Final[float] = 1e-12


def r_predict(r: float, p_s: float = 1.0) -> float:
    r"""Bernoulli predict: :math:`r \leftarrow p_s\, r`."""
    p_s = float(p_s)
    r = float(r)
    return max(0.0, min(1.0, p_s * r))


def r_assoc_update(r_pred: float,
                   L: float,
                   p_d: float = 0.9,
                   lambda_c: float = 1.0) -> float:
    """Update :math:`r` after a successful association (likelihood ratio form)."""
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
    """Numerically-stable log-likelihood form of :func:`r_assoc_update`."""
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
    r"""Update :math:`r` for a missed detection: :math:`r \leftarrow \tfrac{(1-p_d) r}{1 - p_d r}`."""
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
    """Initial :math:`r` of a freshly admitted track from likelihood and clutter rates."""
    score = float(score)
    lambda_b = float(lambda_b)
    lambda_c = float(lambda_c)
    score = max(0.0, min(1.0, score))
    num = lambda_b * score
    den = num + lambda_c
    if den <= 0.0 or not math.isfinite(den):
        return 0.0
    return max(0.0, min(1.0, num / den))
