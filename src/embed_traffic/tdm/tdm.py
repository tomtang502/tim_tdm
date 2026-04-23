"""TDM (Traffic Decision Model) — simple trajectory-based collision reasoning.

Inputs per frame:
  1. `TIMFrameOutput` — from TIM (requires a calibrated TIM to get world-space
     pedestrian positions and velocities in meters).
  2. `CarState`       — ego-vehicle pose (position, velocity, acceleration)
                         in the same ground frame as the TIM output.

Output:
  `TDMOutput` carrying one of {NONE, CAUTION, SLOW_DOWN, BRAKE} plus the
  closest-approach prediction that drove the decision.

Model assumptions (intentionally simple, per "simple trajectory modeling"):
  - Pedestrian motion is constant velocity within the prediction horizon.
    (TIM's LSTM/Kalman already feeds us the current velocity.)
  - Car motion is constant acceleration within the horizon.
  - Ground is flat; all motion happens in the (X, Z) plane.
  - "Collision" is approximated by Euclidean distance between point
    particles. A scalar safety radius encodes car+ped combined extent.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from embed_traffic.inference.schema import PedestrianInfo, TIMFrameOutput
from embed_traffic.tdm.schema import (
    AlertLevel,
    CarState,
    CollisionPrediction,
    TDMOutput,
)


class TDM:
    """Rule-based trajectory TDM.

    Args:
        horizon_s:      How far into the future to roll out trajectories.
        dt_s:           Discretization step. 0.1 s gives 80 samples at 8 s horizon.
        r_brake_m:      Min-distance threshold for BRAKE alerts.
        r_slow_m:       Min-distance threshold for SLOW_DOWN.
        r_caution_m:    Min-distance threshold for CAUTION (beyond this → NONE).
        ttc_brake_s:    TTC cap for BRAKE (collision imminent).
        ttc_slow_s:     TTC cap for SLOW_DOWN (collision likely soon).
        ttc_caution_s:  TTC cap for CAUTION (monitor).
        min_pedestrian_history: ignore pedestrians tracked fewer than this many
                                frames (noisy velocity).
        require_world_space: if True, raise when a TIM output has no world-space
                             fields; if False, pedestrians without them are
                             skipped silently.
    """

    def __init__(
        self,
        horizon_s: float = 8.0,
        dt_s: float = 0.1,
        r_brake_m: float = 1.5,
        r_slow_m: float = 2.5,
        r_caution_m: float = 5.0,
        ttc_brake_s: float = 2.0,
        ttc_slow_s: float = 4.0,
        ttc_caution_s: float = 6.0,
        min_pedestrian_history: int = 3,
        require_world_space: bool = True,
    ) -> None:
        if not (r_brake_m <= r_slow_m <= r_caution_m):
            raise ValueError("radii must satisfy r_brake ≤ r_slow ≤ r_caution")
        if not (ttc_brake_s <= ttc_slow_s <= ttc_caution_s):
            raise ValueError("TTC thresholds must satisfy brake ≤ slow ≤ caution")

        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.r_brake_m = float(r_brake_m)
        self.r_slow_m = float(r_slow_m)
        self.r_caution_m = float(r_caution_m)
        self.ttc_brake_s = float(ttc_brake_s)
        self.ttc_slow_s = float(ttc_slow_s)
        self.ttc_caution_s = float(ttc_caution_s)
        self.min_pedestrian_history = int(min_pedestrian_history)
        self.require_world_space = bool(require_world_space)

        # Precompute the time grid used by every prediction
        n_steps = int(round(self.horizon_s / self.dt_s)) + 1
        self._t_grid = np.linspace(0.0, self.horizon_s, n_steps)

    # ─────────────────────────────────────────────────────────────────
    # Core kinematics
    # ─────────────────────────────────────────────────────────────────

    def _predict_closest_approach(
        self,
        ped_pos: np.ndarray,           # (2,)  (X, Z)
        ped_vel: np.ndarray,           # (2,)  (vx, vz)
        car: CarState,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Return (t*, d_min, car_pos(t*), ped_pos(t*)).

        Pedestrian uses constant velocity; car uses constant acceleration.
        """
        t = self._t_grid                          # (T,)
        # Pedestrian path (constant velocity): shape (T, 2)
        ped_path = ped_pos[None, :] + ped_vel[None, :] * t[:, None]

        # Car path (constant acceleration): (T, 2)
        p0 = np.asarray(car.position_m, dtype=np.float64)
        v0 = np.asarray(car.velocity_m_s, dtype=np.float64)
        a0 = np.asarray(car.acceleration_m_s2, dtype=np.float64)
        car_path = (
            p0[None, :]
            + v0[None, :] * t[:, None]
            + 0.5 * a0[None, :] * (t[:, None] ** 2)
        )

        # Distance at each timestep
        diff = car_path - ped_path                # (T, 2)
        dists = np.linalg.norm(diff, axis=1)       # (T,)
        k = int(np.argmin(dists))
        return float(t[k]), float(dists[k]), car_path[k], ped_path[k]

    # ─────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────

    def _classify(self, time_s: float, dist_m: float) -> AlertLevel:
        """Map (TTC, min distance) to an alert level."""
        if time_s > self.ttc_caution_s or dist_m > self.r_caution_m:
            return AlertLevel.NONE

        # Beyond this point: within caution range on both axes.
        if dist_m <= self.r_brake_m and time_s <= self.ttc_brake_s:
            return AlertLevel.BRAKE
        if dist_m <= self.r_slow_m and time_s <= self.ttc_slow_s:
            return AlertLevel.SLOW_DOWN
        return AlertLevel.CAUTION

    # ─────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────

    def decide(self, tim_out: TIMFrameOutput, car: CarState) -> TDMOutput:
        preds: List[CollisionPrediction] = []
        per_ped_alerts: List[tuple[AlertLevel, CollisionPrediction]] = []

        for ped in tim_out.pedestrians:
            pp, pv = self._ped_state(ped)
            if pp is None:
                continue

            t_star, d_min, car_pt, ped_pt = self._predict_closest_approach(
                pp, pv, car
            )

            # Weight crossing pedestrians more aggressively: shrink effective
            # distance when we're confident they're crossing. This nudges
            # (d, t) into a higher severity bucket.
            effective_d = d_min
            if ped.crossing_intent == "crossing":
                boost = 1.5 * float(ped.crossing_prob or 0.5)
                effective_d = max(0.0, d_min - boost)

            level = self._classify(t_star, effective_d)

            pred = CollisionPrediction(
                ped_id=int(ped.ped_id),
                time_to_min_dist_s=t_star,
                min_distance_m=d_min,
                car_pos_m_at_min=(float(car_pt[0]), float(car_pt[1])),
                ped_pos_m_at_min=(float(ped_pt[0]), float(ped_pt[1])),
                ped_pos_m_t0=(float(pp[0]), float(pp[1])),
                ped_vel_m_s=(float(pv[0]), float(pv[1])),
            )
            preds.append(pred)
            per_ped_alerts.append((level, pred))

        # Pick the worst alert; break ties by smallest distance, then smallest TTC
        if per_ped_alerts:
            per_ped_alerts.sort(
                key=lambda x: (x[0].severity, -x[1].min_distance_m, -x[1].time_to_min_dist_s),
                reverse=True,
            )
            alert, worst = per_ped_alerts[0]
            reason = self._reason(alert, worst)
            return TDMOutput(
                frame_id=tim_out.frame_id,
                frame_time_s=tim_out.frame_time_s,
                alert=alert,
                reason=reason,
                at_risk_ped_id=worst.ped_id,
                time_to_min_dist_s=worst.time_to_min_dist_s,
                min_distance_m=worst.min_distance_m,
                per_ped_predictions=preds,
                car_state=car,
            )

        return TDMOutput(
            frame_id=tim_out.frame_id,
            frame_time_s=tim_out.frame_time_s,
            alert=AlertLevel.NONE,
            reason="No pedestrians with usable trajectories",
            per_ped_predictions=preds,
            car_state=car,
        )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _ped_state(
        self, ped: PedestrianInfo
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (position_m, velocity_m_s) for a pedestrian, or (None, None)
        if we can't trust their state yet."""
        if ped.track_length < self.min_pedestrian_history:
            return None, None
        if ped.position_m_ground is None or ped.velocity_m_s is None:
            if self.require_world_space:
                raise ValueError(
                    f"Pedestrian {ped.ped_id} has no world-space fields. "
                    "Construct TIM with a CameraCalibration, or set "
                    "require_world_space=False on TDM."
                )
            return None, None
        pp = np.asarray(ped.position_m_ground, dtype=np.float64)
        pv = np.asarray(ped.velocity_m_s, dtype=np.float64)
        return pp, pv

    @staticmethod
    def _reason(alert: AlertLevel, pred: CollisionPrediction) -> str:
        if alert == AlertLevel.NONE:
            return "Closest approach is clear of the safety envelope"
        return (
            f"{alert.value.upper()} ← ped {pred.ped_id}: "
            f"min-dist {pred.min_distance_m:.2f} m at t={pred.time_to_min_dist_s:.2f} s"
        )
