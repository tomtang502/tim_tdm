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

# ─────────────────────────────────────────────────────────────────────────────
# Periodic car-spawn infrastructure
#   - A `Spawn` is a single car instantiated at `spawn_t_s` with a specific
#     scenario pattern and a unique `car_id`.
#   - `spawn_schedule()` produces a list of Spawns over the video timeline,
#     one every `period_s` seconds up to `duration_s - end_buffer_s`, cycling
#     through the supplied scenario list.
#   - `Spawn.state_at(global_t)` returns a CarState using LOCAL time
#     (global_t - spawn_t_s) or None if the car hasn't spawned or has
#     travelled off-scene.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Spawn:
    scenario_name: str
    spawn_t_s: float
    car_id: str
    # Filled lazily on first call so we can pickle / re-use the schedule cheaply.
    _source: CarStateSource | None = None

    # Off-scene thresholds (ground-frame meters). A car whose pose leaves
    # this envelope is treated as "expired" for the demo — no TDM decision,
    # no rendering.
    x_abs_max_m: float = 25.0
    z_min_m: float = -8.0
    z_max_m: float = 55.0

    def _is_expired(self, state: CarState) -> bool:
        x, z = state.position_m
        return (abs(x) > self.x_abs_max_m) or (z < self.z_min_m) or (z > self.z_max_m)

    def state_at(self, global_t_s: float) -> CarState | None:
        """CarState at video-global time, or None if the spawn is inactive."""
        if global_t_s < self.spawn_t_s:
            return None
        if self._source is None:
            self._source = make_scenario(self.scenario_name)
        state = self._source(global_t_s - self.spawn_t_s)
        if self._is_expired(state):
            return None
        # Stamp the unique per-spawn car_id onto the returned state
        return CarState(
            position_m=state.position_m,
            velocity_m_s=state.velocity_m_s,
            acceleration_m_s2=state.acceleration_m_s2,
            car_id=self.car_id,
        )


def spawn_schedule(
    scenarios: list[str],
    duration_s: float,
    period_s: float = 5.0,
    end_buffer_s: float = 5.0,
    start_t_s: float = 0.0,
) -> list[Spawn]:
    """Build a rotating-spawn schedule over a video timeline.

    Spawns land at `start_t_s, start_t_s + period_s, ...` up to and including
    the last time ≤ `duration_s - end_buffer_s`. Each spawn picks the next
    scenario from the rotation and gets `car_id = f"{scenario}_{idx}"`.

    Example (20 s video, 5 s period, 5 s tail, 2 scenarios):
      t=0  → scenarios[0] "approaching_0"
      t=5  → scenarios[1] "side_approach_1"
      t=10 → scenarios[0] "approaching_2"
      t=15 → scenarios[1] "side_approach_3"
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    last_t = duration_s - end_buffer_s
    if last_t < start_t_s:
        # Video too short — at least spawn one at t=0
        return [Spawn(scenarios[0], start_t_s, f"{scenarios[0]}_0")]
    spawns: list[Spawn] = []
    idx = 0
    t = start_t_s
    while t <= last_t + 1e-9:
        name = scenarios[idx % len(scenarios)]
        spawns.append(Spawn(scenario_name=name, spawn_t_s=float(t),
                             car_id=f"{name}_{idx}"))
        idx += 1
        t += period_s
    return spawns


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
