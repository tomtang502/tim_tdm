"""Synthetic car-state generators for TDM testing.

Produces a `CarState` for each frame so TDM can be exercised end-to-end on
TIM outputs. These are NOT training data — they're just reasonable patterns
for driving the demo and visualizing decisions.

Coordinate convention: camera's ground frame.
    X: lateral (+right from camera POV)
    Z: longitudinal (+forward, into the scene)

A typical "approaching car" scenario:
    - Starts at Z = 30 m (ahead of camera along the road)
    - Moves toward camera: v_z < 0
    - Slight lateral offset (x ≈ 1 m for a right-side lane)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from embed_traffic.tdm.schema import CarState


# Callable(frame_time_s) -> CarState
CarStateSource = Callable[[float], CarState]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario factories
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApproachingCar:
    """Car driving toward the camera at a constant (or decelerating) speed.

    Starts at `z_start_m` and moves in the -Z direction.

    Args:
        z_start_m: initial longitudinal distance from camera (m)
        x_lane_m:  lateral lane offset (m, positive = right of camera centerline)
        speed_m_s: initial speed (m/s). Will be applied as velocity = (0, -speed)
        decel_m_s2: deceleration magnitude (m/s²). Applied as a = (0, +decel),
                    i.e. opposite to motion. 0 = constant speed.
    """

    z_start_m: float = 30.0
    x_lane_m: float = 1.0
    speed_m_s: float = 10.0
    decel_m_s2: float = 0.0

    def __call__(self, t_s: float) -> CarState:
        # Current velocity (decelerating: |v| shrinks, direction stays −Z while v<0)
        v_z = -self.speed_m_s + self.decel_m_s2 * t_s
        # Don't let the car reverse just because deceleration keeps integrating
        v_z = min(v_z, 0.0)
        # Position from kinematics (constant deceleration until v_z hits 0)
        if v_z < 0.0:
            z = self.z_start_m - self.speed_m_s * t_s + 0.5 * self.decel_m_s2 * t_s * t_s
        else:
            # Stopped — compute stop time and position
            t_stop = self.speed_m_s / max(self.decel_m_s2, 1e-6)
            z = self.z_start_m - 0.5 * self.speed_m_s * t_stop
        return CarState(
            position_m=(self.x_lane_m, float(z)),
            velocity_m_s=(0.0, float(v_z)),
            acceleration_m_s2=(0.0, float(self.decel_m_s2) if v_z < 0.0 else 0.0),
        )


@dataclass
class StationaryCar:
    """Stationary car at (x, z)."""

    x_m: float = 0.0
    z_m: float = 20.0

    def __call__(self, _t_s: float) -> CarState:
        return CarState(
            position_m=(self.x_m, self.z_m),
            velocity_m_s=(0.0, 0.0),
            acceleration_m_s2=(0.0, 0.0),
        )


@dataclass
class ConstantVelocityCar:
    """Car with prescribed constant velocity (approach or depart)."""

    start_pos_m: tuple[float, float] = (1.0, 30.0)
    velocity_m_s: tuple[float, float] = (0.0, -10.0)

    def __call__(self, t_s: float) -> CarState:
        x = self.start_pos_m[0] + self.velocity_m_s[0] * t_s
        z = self.start_pos_m[1] + self.velocity_m_s[1] * t_s
        return CarState(
            position_m=(float(x), float(z)),
            velocity_m_s=self.velocity_m_s,
            acceleration_m_s2=(0.0, 0.0),
        )


@dataclass
class ApproachAndStop:
    """Car approaches and then starts braking at a given time (a driver reaction).

    Before `brake_at_t_s`: constant speed, -Z direction.
    After: constant deceleration, comes to rest.
    """

    z_start_m: float = 30.0
    x_lane_m: float = 1.0
    speed_m_s: float = 12.0
    brake_at_t_s: float = 3.0
    decel_m_s2: float = 4.0

    def __call__(self, t_s: float) -> CarState:
        if t_s <= self.brake_at_t_s:
            return CarState(
                position_m=(self.x_lane_m, self.z_start_m - self.speed_m_s * t_s),
                velocity_m_s=(0.0, -self.speed_m_s),
                acceleration_m_s2=(0.0, 0.0),
            )
        dt = t_s - self.brake_at_t_s
        z_at_brake = self.z_start_m - self.speed_m_s * self.brake_at_t_s
        v_z = -self.speed_m_s + self.decel_m_s2 * dt
        if v_z < 0.0:
            z = z_at_brake - self.speed_m_s * dt + 0.5 * self.decel_m_s2 * dt * dt
            a_z = self.decel_m_s2
        else:
            # Stopped
            t_to_stop = self.speed_m_s / self.decel_m_s2
            z = z_at_brake - 0.5 * self.speed_m_s * t_to_stop
            v_z = 0.0
            a_z = 0.0
        return CarState(
            position_m=(self.x_lane_m, float(z)),
            velocity_m_s=(0.0, float(v_z)),
            acceleration_m_s2=(0.0, float(a_z)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, Callable[[], CarStateSource]] = {
    # Car approaching the intersection straight along the camera's road (from
    # down the street, driving toward the camera).
    "approaching":       lambda: ApproachingCar(),
    "approaching_fast":  lambda: ApproachingCar(speed_m_s=15.0, z_start_m=40.0),
    "approach_and_stop": lambda: ApproachAndStop(),

    # Parked car just inside the intersection.
    "stationary":        lambda: StationaryCar(),

    # Car entering the intersection from a SIDE street (crossing the camera's
    # axis roughly perpendicular). Starts 14 m right of camera, 18 m ahead
    # (|start| ≈ 22.8 m), moving left and forward — the kind of turn-in you'd
    # see at a real junction. Lateral speed dominates.
    "side_approach":     lambda: ConstantVelocityCar(
        start_pos_m=(14.0, 18.0), velocity_m_s=(-6.0, -2.0),
    ),

    # Car approaching diagonally (entering intersection from a feeder road).
    # Starts ~29 m from the intersection.
    "diagonal_approach": lambda: ConstantVelocityCar(
        start_pos_m=(6.0, 28.0), velocity_m_s=(-1.5, -9.0),
    ),

    # Generic constant-velocity override (use keyword args to customize).
    "constant_velocity": lambda: ConstantVelocityCar(),
}


def make_scenario(name: str) -> CarStateSource:
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Choose from: {sorted(SCENARIOS)}"
        )
    return SCENARIOS[name]()
