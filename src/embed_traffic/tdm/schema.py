"""TDM types: CarState, AlertLevel, CollisionPrediction, TDMOutput.

All kinematic quantities are expressed in the same ground frame as the TIM
output (meters, with X = lateral-right, Y = up, Z = forward-from-camera).
Camera sits at the ground-frame origin at height `camera_height_m`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Alert levels
# ─────────────────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    """Four-level alert, ordered from least to most severe."""
    NONE = "none"           # no pedestrian risk
    CAUTION = "caution"     # pedestrian visible, monitor
    SLOW_DOWN = "slow_down"  # potential collision, reduce speed
    BRAKE = "brake"          # imminent collision, brake immediately

    @property
    def severity(self) -> int:
        return _SEVERITY_RANK[self]

    @classmethod
    def max_of(cls, levels: List["AlertLevel"]) -> "AlertLevel":
        if not levels:
            return cls.NONE
        return max(levels, key=lambda a: a.severity)


_SEVERITY_RANK = {
    AlertLevel.NONE: 0,
    AlertLevel.CAUTION: 1,
    AlertLevel.SLOW_DOWN: 2,
    AlertLevel.BRAKE: 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Car state (TDM input #2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CarState:
    """Driver-side ego-vehicle state, expressed in the camera's ground frame.

    Coordinate convention: X = lateral (+right), Z = longitudinal (+forward
    from camera into the scene). An approaching car has Z_position > 0 and
    velocity_z < 0 (closing distance).

    All fields are in SI units (meters, m/s, m/s²).
    """

    position_m: Tuple[float, float]         # (X, Z)
    velocity_m_s: Tuple[float, float]       # (vx, vz)
    acceleration_m_s2: Tuple[float, float]  # (ax, az)

    @property
    def speed_m_s(self) -> float:
        vx, vz = self.velocity_m_s
        return float((vx * vx + vz * vz) ** 0.5)

    @property
    def heading_deg(self) -> float:
        """Direction of travel in degrees. 0° = +Z (away from camera along road
        ahead), 180° = -Z (approaching camera), +90° = +X (lateral right)."""
        import math
        vx, vz = self.velocity_m_s
        return float(math.degrees(math.atan2(vx, vz)))


# ─────────────────────────────────────────────────────────────────────────────
# Collision prediction (per pedestrian)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CollisionPrediction:
    """Closest-approach prediction between the ego car and a single pedestrian."""

    ped_id: int
    time_to_min_dist_s: float                # when closest approach happens (≥ 0)
    min_distance_m: float                    # |car − ped| at that time
    car_pos_m_at_min: Tuple[float, float]
    ped_pos_m_at_min: Tuple[float, float]

    # Raw inputs echoed so consumers can rebuild the trajectory if needed
    ped_pos_m_t0: Tuple[float, float]
    ped_vel_m_s: Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level TDM output (one per frame)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TDMOutput:
    """Result of one TDM decision step."""

    frame_id: int
    frame_time_s: float

    alert: AlertLevel
    reason: str                                        # human-readable explanation

    at_risk_ped_id: Optional[int] = None
    time_to_min_dist_s: Optional[float] = None         # for the selected risk ped
    min_distance_m: Optional[float] = None

    per_ped_predictions: List[CollisionPrediction] = field(default_factory=list)

    # Echoed inputs (useful for the demo overlay / debugging)
    car_state: Optional[CarState] = None

    # ── Serialization ──
    def to_json(self) -> str:
        # enum → str
        payload = asdict(self)
        payload["alert"] = self.alert.value
        return json.dumps(payload)
