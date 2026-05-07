"""
Unit tests for pose_update/adaptive_kernel.py (Task 5).

Purely synthetic — no trajectory data required.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_adaptive_kernel.py -v
"""

import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from perception.adaptive_kernel import AdaptiveKernel, irls_weight, adapt_noise


class TestAlphaFitting:
    def test_alpha_near_2_for_gaussian_residuals(self):
        np.random.seed(0)
        residuals = np.random.normal(0, 1.0, size=500)
        kernel = AdaptiveKernel(c=1.0)
        alpha = kernel.fit_alpha(residuals)
        # For clean Gaussian data, α should be near its upper bound (L2)
        assert alpha >= 0.0, f"Expected α ≥ 0 for Gaussian residuals, got {alpha}"

    def test_alpha_negative_for_heavy_tails(self):
        np.random.seed(1)
        inliers = np.random.normal(0, 1.0, size=500)
        outliers = np.random.normal(0, 50.0, size=50)  # 10% contamination
        residuals = np.concatenate([inliers, outliers])
        kernel = AdaptiveKernel(c=1.0)
        alpha = kernel.fit_alpha(residuals)
        assert alpha < 1.5, \
            f"Expected α < 1.5 for heavy-tailed residuals, got {alpha}"

    def test_alpha_handles_empty_input(self):
        kernel = AdaptiveKernel(c=1.0)
        alpha = kernel.fit_alpha(np.array([]))
        # Default to L2 when no evidence
        assert alpha == kernel.alpha_max


class TestWeights:
    def test_weight_monotone_in_residual_for_negative_alpha(self):
        kernel = AdaptiveKernel(c=1.0)
        rs = np.linspace(0.1, 10.0, 20)
        w = kernel.weights(rs, alpha=-2.0)
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1] - 1e-10, \
                f"Weight not monotone decreasing at index {i}: {w[i]} → {w[i+1]}"

    def test_weight_near_zero_for_extreme_outlier(self):
        kernel = AdaptiveKernel(c=1.0)
        w = kernel.weights(np.array([100.0]), alpha=-5.0)[0]
        # Strong alpha + large residual → near-zero weight
        assert w < 1e-3

    def test_weight_for_L2_is_constant(self):
        kernel = AdaptiveKernel(c=1.0)
        rs = np.array([0.1, 1.0, 5.0, 20.0])
        w = kernel.weights(rs, alpha=2.0)
        # L2 has uniform weight 1/c² = 1
        np.testing.assert_array_almost_equal(w, np.ones(4))

    def test_irls_weight_standalone(self):
        rs = np.array([0.5, 1.0, 2.0])
        w = irls_weight(rs, alpha=0.0, c=1.0)  # Cauchy
        assert w.shape == rs.shape
        # Cauchy: w(r) = 2 / (r² + 2) when c=1
        expected = 2.0 / (rs * rs + 2.0)
        np.testing.assert_array_almost_equal(w, expected, decimal=5)


class TestInlierOutlierDiscrimination:
    def test_outlier_receives_much_lower_weight_than_inliers(self):
        np.random.seed(2)
        inliers = np.random.normal(0, 0.5, size=100)
        outliers = np.array([50.0])
        residuals = np.concatenate([inliers, outliers])

        kernel = AdaptiveKernel(c=1.0)
        alpha = kernel.fit_alpha(residuals)
        weights = kernel.weights(residuals, alpha)

        mean_inlier_w = np.mean(np.abs(weights[:-1]))
        outlier_w = np.abs(weights[-1])
        assert outlier_w < 0.1 * mean_inlier_w, \
            f"Outlier weight {outlier_w} not much smaller than " \
            f"mean inlier {mean_inlier_w}, α={alpha}"


class TestAdaptNoise:
    def test_adapt_noise_inflates_for_outliers(self):
        kernel = AdaptiveKernel(c=1.0)
        residuals = np.array([0.1, 0.1, 0.1, 50.0])
        base_noise = np.eye(3) * 0.01
        effective = adapt_noise(residuals, base_noise, kernel)

        # Shape: (N, 3, 3)
        assert effective.shape == (4, 3, 3)
        # Outlier's effective noise should be much larger
        assert np.trace(effective[-1]) > 100 * np.trace(effective[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
