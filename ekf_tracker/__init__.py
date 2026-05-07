"""EKF scene-tracking package.

Public surface:
    EkfTracker / EkfObject / SceneView    — high-level facade in `api.py`
    GaussianEkfTracker                    — fast-tier base-frame Gaussian-EKF
    TwoTierOrchestratorGaussian           — fast tier + slow-tier pose-graph
    BernoulliConfig / TriggerConfig       — shared dataclasses
"""
from ekf_tracker.api import EkfObject, EkfTracker, SceneView  # noqa: F401
from ekf_tracker.config import BernoulliConfig, TriggerConfig  # noqa: F401
from ekf_tracker.gaussian_ekf_tracker import GaussianEkfTracker  # noqa: F401
from ekf_tracker.orchestrator_gaussian import TwoTierOrchestratorGaussian  # noqa: F401
