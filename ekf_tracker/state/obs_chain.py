"""Append-only camera-frame ICP observation chain so retroactive SLAM corrections can be re-projected to world without losing object information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from utils.ekf_se3 import se3_exp, se3_log, se3_adjoint


@dataclass
class ChainEntry:
    """One entry of an observation chain: timestamp, camera-frame ICP pose, fitness, RMSE."""
    frame: int
    T_co: np.ndarray            # (4, 4) ICP pose, camera frame
    R_co: np.ndarray            # (6, 6) ICP covariance, camera frame
    fitness: float = 1.0        # ICP fitness (diagnostic)
    rmse: float = 0.0           # ICP RMSE in metres (diagnostic)


@dataclass
class TrackObsChain:
    """Append-only chain of camera-frame ICP observations for one track."""
    entries: List[ChainEntry] = field(default_factory=list)

    def append(self,
               frame: int,
               T_co: np.ndarray,
               R_co: np.ndarray,
               fitness: float = 1.0,
               rmse: float = 0.0) -> None:
        self.entries.append(ChainEntry(
            frame=int(frame),
            T_co=np.asarray(T_co, dtype=np.float64).copy(),
            R_co=np.asarray(R_co, dtype=np.float64).copy(),
            fitness=float(fitness),
            rmse=float(rmse),
        ))

    def __len__(self) -> int:
        return len(self.entries)

    def cap(self, max_len: int) -> None:
        """Keep only the most recent `max_len` entries (drop oldest)."""
        if max_len <= 0 or len(self.entries) <= max_len:
            return
        self.entries = self.entries[-max_len:]

    # --------------------------------------------------------------- #
    #  World-frame composition
    # --------------------------------------------------------------- #

    def world_frame_estimate(
            self,
            T_wb_history: Mapping[int, np.ndarray],
            T_bc: np.ndarray,
            Sigma_wb_history: Optional[Mapping[int, np.ndarray]] = None,
            max_iter: int = 5,
            tol: float = 1e-6,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Compose the per-entry camera-frame poses with the cached :math:`T_{wb}` to produce world-frame poses."""
        if not self.entries:
            return None
        T_bc = np.asarray(T_bc, dtype=np.float64)
        Ad_bc = se3_adjoint(T_bc)

        T_wos: List[np.ndarray] = []
        R_wos: List[np.ndarray] = []
        for e in self.entries:
            T_wb = T_wb_history.get(e.frame)
            if T_wb is None:
                continue
            T_wb = np.asarray(T_wb, dtype=np.float64)
            T_wo = T_wb @ T_bc @ e.T_co
            R_wo = Ad_bc @ e.R_co @ Ad_bc.T
            if Sigma_wb_history is not None:
                Sigma_wb = Sigma_wb_history.get(e.frame)
                if Sigma_wb is not None:
                    inv_T_wo = np.linalg.inv(T_wo)
                    Ad_inv = se3_adjoint(inv_T_wo)
                    R_wo = R_wo + Ad_inv @ np.asarray(Sigma_wb,
                                                       dtype=np.float64) @ Ad_inv.T
            R_wo = 0.5 * (R_wo + R_wo.T)
            T_wos.append(T_wo)
            R_wos.append(R_wo)

        if not T_wos:
            return None

        # Initial weights from per-entry inverse trace; kept as a
        # rough info-weight (full Σ inversion happens inside the GN loop).
        weights = np.array([1.0 / max(np.trace(R), 1e-12) for R in R_wos])
        weights = weights / weights.sum()

        # Initial reference: weighted-translation-mean as a quick seed.
        T_ref = T_wos[int(np.argmax(weights))].copy()

        # Gauss-Newton on SE(3): re-linearise innovations at T_ref,
        # take the information-weighted mean correction, repeat.
        I_tot = np.zeros((6, 6))
        for _ in range(max_iter):
            xis = []
            I_tot = np.zeros((6, 6))
            b = np.zeros(6)
            for T_n, R_n in zip(T_wos, R_wos):
                xi_n = se3_log(np.linalg.inv(T_ref) @ T_n)
                xis.append(xi_n)
                try:
                    R_inv = np.linalg.inv(R_n)
                except np.linalg.LinAlgError:
                    R_inv = np.eye(6) * 1e-3
                I_tot = I_tot + R_inv
                b = b + R_inv @ xi_n
            try:
                delta = np.linalg.solve(I_tot, b)
            except np.linalg.LinAlgError:
                # Fall back to weighted-mean if the info matrix is singular.
                delta = np.sum(weights[:, None] *
                                np.stack(xis, axis=0), axis=0)
            T_ref = T_ref @ se3_exp(delta)
            if float(np.linalg.norm(delta)) < tol:
                break

        # Posterior cov in the tangent at T_ref.
        try:
            Sigma_post = np.linalg.inv(I_tot)
            Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
        except np.linalg.LinAlgError:
            Sigma_post = np.eye(6)

        return T_ref, Sigma_post, len(T_wos)

    # --------------------------------------------------------------- #
    #  Diagnostic
    # --------------------------------------------------------------- #

    def to_jsonable(self,
                     max_dump: int = 50) -> Dict[str, object]:
        """Serialize the chain to a JSON-friendly dict."""
        out: Dict[str, object] = {"len": len(self.entries)}
        if not self.entries:
            return out
        first = self.entries[0]
        last = self.entries[-1]
        out["first_frame"] = int(first.frame)
        out["last_frame"] = int(last.frame)
        if max_dump > 0:
            recent = self.entries[-max_dump:]
            out["entries"] = [
                {"frame": int(e.frame),
                 "T_co": e.T_co.tolist(),
                 "R_co_diag": np.diag(e.R_co).tolist(),
                 "fitness": float(e.fitness),
                 "rmse": float(e.rmse)}
                for e in recent
            ]
        return out


class ChainStore:
    """Per-track append-only ICP chain registry."""

    def __init__(self) -> None:
        self.chains: Dict[int, TrackObsChain] = {}
        self.T_wb_history: Dict[int, np.ndarray] = {}
        self.Sigma_wb_history: Dict[int, np.ndarray] = {}

    # --- chain management ---
    def append(self,
                oid: int,
                frame: int,
                T_co: np.ndarray,
                R_co: np.ndarray,
                fitness: float = 1.0,
                rmse: float = 0.0,
                max_len: int = 200) -> None:
        ch = self.chains.setdefault(oid, TrackObsChain())
        ch.append(frame, T_co, R_co, fitness, rmse)
        if max_len > 0:
            ch.cap(max_len)

    def delete(self, oid: int) -> None:
        self.chains.pop(oid, None)

    def get(self, oid: int) -> Optional[TrackObsChain]:
        return self.chains.get(oid)

    # --- SLAM history cache ---
    def record_pose(self, frame: int, T_wb: np.ndarray,
                     Sigma_wb: Optional[np.ndarray] = None) -> None:
        self.T_wb_history[int(frame)] = np.asarray(T_wb, dtype=np.float64).copy()
        if Sigma_wb is not None:
            self.Sigma_wb_history[int(frame)] = np.asarray(
                Sigma_wb, dtype=np.float64).copy()

    def revise_pose(self, frame: int, T_wb: np.ndarray,
                     Sigma_wb: Optional[np.ndarray] = None) -> None:
        """Re-project a chain to world after a retroactive SLAM correction."""
        self.record_pose(frame, T_wb, Sigma_wb)

    # --- world-frame queries ---
    def world_frame_estimate(self,
                              oid: int,
                              T_bc: np.ndarray,
                              ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        ch = self.chains.get(oid)
        if ch is None:
            return None
        return ch.world_frame_estimate(
            T_wb_history=self.T_wb_history,
            T_bc=T_bc,
            Sigma_wb_history=(self.Sigma_wb_history
                              if self.Sigma_wb_history else None),
        )
