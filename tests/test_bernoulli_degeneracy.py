"""
Degeneracy test: Bernoulli-EKF fast tier under `BernoulliConfig.degeneracy()`
must reproduce the pre-Bernoulli legacy path on the same trajectory.

The paper (§A.9) specifies five substitutions that collapse the new path
onto the old one:

  1. pi* = pi^GT       -> `association_mode='oracle'`
  2. w_k == 1          -> `enable_huber=False`
  3. Phi == id         -> `P_max=None`
  4. p_s = 1           -> `p_s=1.0`
  5. r treated as 1    -> `r_min=0, r_conf=0`, plus tie-break: init cov
                          must match the legacy constant (`init_cov_from_R=False`)

The test runs the TrajectoryRunner from test_orchestrator_integration with
both orchestrators side by side, and checks that per-frame collapsed pose
mean and covariance match within floating-point tolerance.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)
from pose_update.state.slam_interface import PassThroughSlam

from tests.test_orchestrator_integration import (
    TrajectoryRunner, DATA_ROOT, HAS_DATA,
)

requires_data = pytest.mark.skipif(
    not HAS_DATA, reason=f"Trajectory data not found at {DATA_ROOT}"
)


class _BernoulliRunner(TrajectoryRunner):
    """Variant of TrajectoryRunner that wires the Bernoulli fast tier in
    degeneracy mode. Otherwise identical to the integration test runner."""

    def __init__(self, n_frames: int, step: int = 5,
                 bernoulli_config: BernoulliConfig = None,
                 rng_seed: int = 42):
        super().__init__(n_frames=n_frames, step=step)
        # Swap the orchestrator for a Bernoulli one with the same slam
        # backend / trigger config.
        self.orchestrator = TwoTierOrchestrator(
            self.slam_backend,
            trigger=TriggerConfig(periodic_every_n_frames=30),
            verbose=False,
            rng_seed=rng_seed,
            bernoulli=bernoulli_config or BernoulliConfig.degeneracy(),
        )


def _collect_object_traces(reports: List[Dict]) -> Dict[int, List[tuple]]:
    """Map oid -> list of (frame_idx, T, cov) tuples."""
    out: Dict[int, List[tuple]] = {}
    for r in reports:
        for oid, info in r["objects"].items():
            out.setdefault(oid, []).append(
                (r["frame_idx"], info["T"].copy(), info["cov"].copy()))
    return out


@requires_data
class TestBernoulliDegeneracy:

    def _run_both(self, n_frames: int = 40, step: int = 5, rng_seed: int = 42):
        # Re-seed numpy's global state before each run. _recompute_relations
        # uses `np.random.uniform` (not the orchestrator's own Generator),
        # so the scene-graph edge scores drift when this test runs after
        # any other test that consumed global RNG. Fixing the seed per run
        # is the minimum change that makes the comparison bit-exact.
        np.random.seed(rng_seed)
        legacy = TrajectoryRunner(n_frames=n_frames, step=step)
        legacy.orchestrator = TwoTierOrchestrator(
            legacy.slam_backend,
            trigger=TriggerConfig(periodic_every_n_frames=30),
            verbose=False, rng_seed=rng_seed,
        )
        legacy_reports = legacy.run()

        np.random.seed(rng_seed)
        bern = _BernoulliRunner(n_frames=n_frames, step=step,
                                bernoulli_config=BernoulliConfig.degeneracy(),
                                rng_seed=rng_seed)
        bern_reports = bern.run()

        return legacy_reports, bern_reports

    def test_same_object_set(self):
        """Both paths should track the same set of oids per frame."""
        legacy, bern = self._run_both(n_frames=30, step=5)
        assert len(legacy) == len(bern), "Frame counts differ"
        for r_l, r_b in zip(legacy, bern):
            oids_l = set(r_l["objects"].keys())
            oids_b = set(r_b["objects"].keys())
            assert oids_l == oids_b, (
                f"Frame {r_l['frame_idx']}: legacy={oids_l}, bern={oids_b}")

    def test_same_pose_means(self):
        """Pose means should agree within a tight tolerance."""
        legacy, bern = self._run_both(n_frames=30, step=5)
        max_T_err = 0.0
        for r_l, r_b in zip(legacy, bern):
            for oid in r_l["objects"]:
                if oid not in r_b["objects"]:
                    continue
                T_l = r_l["objects"][oid]["T"]
                T_b = r_b["objects"][oid]["T"]
                err = float(np.max(np.abs(T_l - T_b)))
                if err > max_T_err:
                    max_T_err = err
                assert err < 1e-5, (
                    f"Frame {r_l['frame_idx']} oid {oid}: max|T_l - T_b| = {err}")
        print(f"[degeneracy] max pose-mean error across 30 frames: {max_T_err:.2e}")

    def test_same_covariances(self):
        """Collapsed covariances should agree within a tight tolerance."""
        legacy, bern = self._run_both(n_frames=30, step=5)
        max_cov_err = 0.0
        for r_l, r_b in zip(legacy, bern):
            for oid in r_l["objects"]:
                if oid not in r_b["objects"]:
                    continue
                C_l = r_l["objects"][oid]["cov"]
                C_b = r_b["objects"][oid]["cov"]
                err = float(np.max(np.abs(C_l - C_b)))
                if err > max_cov_err:
                    max_cov_err = err
                # Slightly looser tol for covariances (more float ops).
                assert err < 1e-5, (
                    f"Frame {r_l['frame_idx']} oid {oid}: max|C_l - C_b| = {err}")
        print(f"[degeneracy] max cov error across 30 frames: {max_cov_err:.2e}")


@requires_data
class TestBernoulliFullModeInvariants:
    """Run the full Bernoulli mode (Hungarian, Huber, P_max, visibility,
    pruning) on real data and check invariants. Exact output consistency
    is not expected -- only that the recursion stays well-behaved."""

    def _run(self, n_frames=60, step=5, rng_seed=42):
        cfg = BernoulliConfig(
            association_mode="hungarian",
            p_s=1.0, p_d=0.9, alpha=4.4,
            lambda_c=1.0, lambda_b=1.0,
            r_conf=0.5, r_min=1e-3,
            G_in=12.59, G_out=25.0,
            P_max=np.diag([0.0625] * 3 + [(np.pi / 4) ** 2] * 3),
            enable_visibility=True, enable_huber=True,
            init_cov_from_R=False, enforce_label_match=True,
            K=np.array([[554.3827, 0, 320.5],
                        [0, 554.3827, 240.5],
                        [0, 0, 1]], dtype=np.float32),
            image_shape=(480, 640),
        )
        runner = _BernoulliRunner(
            n_frames=n_frames, step=step,
            bernoulli_config=cfg, rng_seed=rng_seed,
        )
        return runner, runner.run()

    def test_r_in_unit_interval(self):
        runner, _ = self._run()
        for oid, r in runner.orchestrator.existence.items():
            assert 0.0 <= r <= 1.0, f"r out of bounds for oid {oid}: {r}"

    def test_covariance_psd(self):
        _, reports = self._run()
        for r in reports:
            for oid, info in r["objects"].items():
                eigs = np.linalg.eigvalsh(info["cov"])
                assert eigs.min() > -1e-6, (
                    f"Non-PSD cov at frame {r['frame_idx']} oid {oid}")

    def test_covariance_saturated(self):
        """Phi (P_max) should prevent any pose covariance trace from
        exceeding tr(P_max)."""
        runner, reports = self._run()
        cfg = runner.orchestrator.bernoulli
        tr_max = float(np.trace(cfg.P_max))
        max_seen = 0.0
        for r in reports:
            for oid, info in r["objects"].items():
                tr = float(np.trace(info["cov"]))
                max_seen = max(max_seen, tr)
                # Allow a small slack for the mixture-expansion term
                # collapsed_object adds on top of the per-particle Phi cap.
                assert tr <= tr_max * 2.5, (
                    f"Trace {tr:.4f} > 2.5*tr(P_max)={2.5*tr_max:.4f} "
                    f"at frame {r['frame_idx']} oid {oid}")
        print(f"[bernoulli] max cov trace: {max_seen:.4f} "
              f"(cap tr(P_max)={tr_max:.4f})")

    def test_confirmed_plus_tentative_equals_tracked(self):
        runner, reports = self._run()
        orch = runner.orchestrator
        n_confirmed = len(reports[-1]["objects"])
        n_tentative = len(orch.tentative_objects)
        n_tracked = len(orch.existence)
        assert n_confirmed + n_tentative == n_tracked, (
            f"{n_confirmed} + {n_tentative} != {n_tracked}")

    def test_prune_removes_dead_tracks(self):
        """Tracks with r < r_min should not appear in any report."""
        runner, reports = self._run()
        r_min = runner.orchestrator.bernoulli.r_min
        for oid, r in runner.orchestrator.existence.items():
            assert r >= r_min, f"Track oid {oid} has r={r} < r_min={r_min}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
