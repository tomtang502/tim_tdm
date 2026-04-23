# TDM — Traffic Decision Model

TDM runs on the **driver / cyclist app**. Each frame it combines a
`TIMFrameOutput` (from the camera node) with the driver's own `CarState`
(position, velocity, acceleration) and decides whether to alert the driver.

> **How the scene is set up.** The video comes from a **stationary street
> dashcam** mounted near an intersection (`plan.txt §IX`). TIM observes
> pedestrians at that intersection. The **car that TDM is alerting is NOT
> visible in the video** — it is a separate vehicle (driven by the user of
> the driver app) that is **approaching the same intersection from some
> direction**. In this prototype we don't have real car telemetry, so the
> demo **simulates** the car in the camera's ground frame: it starts far
> away (e.g. 30 m from the intersection) and drives in toward it. The car
> only appears in the top-down panel (as a rectangle oriented along its
> velocity) and in the alert banner. Every alert is issued for the driver
> of this simulated car.

| Alert | Severity | When |
|---|---|---|
| `NONE` | 0 | No pedestrian on a collision course |
| `CAUTION` | 1 | Pedestrian within 5 m & 6 s of closest approach — stay alert |
| `SLOW_DOWN` | 2 | Closest approach ≤ 2.5 m & 4 s — reduce speed |
| `BRAKE` | 3 | Closest approach ≤ 1.5 m & 2 s — brake now |

The model is deliberately simple: **constant-velocity pedestrian** and
**constant-acceleration car**, evaluated on a discrete time grid up to an 8 s
horizon. See `plan.txt §XI` for the design rationale.

---

## 1. Inputs

### 1.1 `TIMFrameOutput`

Must come from a **calibrated** TIM (i.e. `TIM(camera_calibration=...)`).
TDM reads each pedestrian's:

| Field | Used for |
|---|---|
| `ped_id` | identity across frames |
| `position_m_ground` | ground-plane position in meters `[X, Z]` |
| `velocity_m_s` | ground velocity in m/s `[vx, vz]` |
| `track_length` | ignore very short tracks (noisy velocity) |
| `crossing_intent`, `crossing_prob` | bumps severity for crossing peds |

Pedestrians without `position_m_ground`/`velocity_m_s` are skipped. If you
pass an uncalibrated TIM output, construct TDM with
`TDM(require_world_space=False)` to silence the error.

### 1.2 `CarState`

```python
@dataclass
class CarState:
    position_m:         (X, Z)   # meters, in the camera's ground frame
    velocity_m_s:       (vx, vz) # m/s
    acceleration_m_s2:  (ax, az) # m/s²
```

Coordinate convention (same as TIM):

- `X`: lateral (+ right from camera POV)
- `Z`: longitudinal (+ forward into the scene)
- A car approaching the camera has `Z > 0` and `vz < 0`.

**We don't have real car telemetry in this prototype.** For testing we
simulate the car with `embed_traffic.tdm.simulator`:

| Scenario | Direction into intersection | Description |
|---|---|---|
| `approaching` | straight (down camera's road) | Constant 10 m/s approach from Z = 30 m |
| `approaching_fast` | straight | 15 m/s from Z = 40 m |
| `approach_and_stop` | straight | 12 m/s until t = 3 s, then 4 m/s² braking |
| `side_approach` | from camera's right-hand feeder road | Starts at (+8 m, 15 m), velocity (−6 m/s, −2 m/s) |
| `diagonal_approach` | diagonal from upper-right | Starts at (+5 m, 28 m), velocity (−1.5 m/s, −9 m/s) |
| `stationary` | — | Parked at (0, 20) inside the intersection |
| `constant_velocity` | user-specified | Arbitrary constant-velocity override |

All directions are the car's approach toward the intersection (Z = 0 near the
camera). In the top-down panel, the car rectangle is **oriented along its
velocity vector**, so a `side_approach` car visibly enters from the right
pointing toward the camera.

---

## 2. Output

```python
@dataclass
class TDMOutput:
    frame_id:            int
    frame_time_s:        float
    alert:               AlertLevel              # NONE | CAUTION | SLOW_DOWN | BRAKE
    reason:              str                     # human-readable explanation
    at_risk_ped_id:      Optional[int]           # None if alert == NONE
    time_to_min_dist_s:  Optional[float]         # TTC to closest approach (s)
    min_distance_m:      Optional[float]         # min |car − ped| (m)
    per_ped_predictions: list[CollisionPrediction]
    car_state:           Optional[CarState]      # echoed back
```

Each `CollisionPrediction` carries the per-pedestrian rollout — `t*`, `d*`,
car and pedestrian positions at `t*`, and the original `t=0` ped state.

Serialize with `TDMOutput.to_json()`.

---

## 3. Public API

```python
from embed_traffic.tdm import TDM, CarState, AlertLevel
from embed_traffic.tdm.simulator import make_scenario

tdm = TDM(
    horizon_s=8.0,      # how far into the future to roll out
    dt_s=0.1,           # sample step (81 samples at 8 s horizon)
    r_brake_m=1.5,      # closest-approach thresholds (meters)
    r_slow_m=2.5,
    r_caution_m=5.0,
    ttc_brake_s=2.0,    # time-to-closest-approach thresholds (seconds)
    ttc_slow_s=4.0,
    ttc_caution_s=6.0,
    min_pedestrian_history=3,   # ignore ped tracks shorter than this
    require_world_space=True,   # raise if TIM wasn't calibrated
)

# Per-frame call (TIM runs separately, yields tim_out)
car_at_t = make_scenario("approaching")(tim_out.frame_time_s)
tdm_out = tdm.decide(tim_out, car_at_t)

print(tdm_out.alert.value, tdm_out.reason)
```

Every pedestrian in the TIM output is rolled forward with constant velocity;
the car is rolled forward with constant acceleration. The time grid that
minimizes |car(t) − ped(t)| for each pedestrian gives `t*` and `d*`. The
severity rules in §1 above are then applied; if any pedestrian lands in
BRAKE, that wins; else the worst among {SLOW_DOWN, CAUTION, NONE}.

Crossing pedestrians (`crossing_intent == "crossing"`) get an effective
distance shrink of `1.5 × crossing_prob` meters — this nudges them into a
higher severity tier when they're confidently crossing.

---

## 4. Math (short version)

Given ped state at t=0: `p_p(0)`, `v_p`; car state: `p_c(0)`, `v_c`, `a_c`.

```
p_p(t) = p_p(0) + v_p t
p_c(t) = p_c(0) + v_c t + (1/2) a_c t²
D(t)   = | p_c(t) − p_p(t) |
```

TDM samples `t ∈ {0, dt, 2dt, …, horizon}` (81 points by default) and picks
`t* = argmin_t D(t)`, `d* = D(t*)`. Alert level is a function of `(t*, d*)`:

```
if t* > ttc_caution or d* > r_caution            →  NONE
elif d* ≤ r_brake    and t* ≤ ttc_brake          →  BRAKE
elif d* ≤ r_slow     and t* ≤ ttc_slow           →  SLOW_DOWN
else                                              →  CAUTION
```

Discretization error ≤ `dt_s / 2` in TTC. At `dt_s = 0.1 s` this is 50 ms —
well below the decision thresholds (2 s, 4 s, 6 s).

---

## 5. End-to-end demo

```bash
python demo/demo_tdm.py                               # both default clips
python demo/demo_tdm.py --scenario approach_and_stop  # change car behavior
python demo/demo_tdm.py --video a.mp4 --video b.mp4   # custom videos
```

Per video, writes to `outputs/demo/tdm/`:

| Artifact | Contents |
|---|---|
| `<stem>.mp4` | Side-by-side. Left: overlay with TDM alert banner at the top (color-coded). Right: top-down with the ego car drawn as a rectangle, trajectory trail, velocity arrow, closest-approach segment highlighted in alert color. |
| `<stem>.jsonl` | Per-frame `{"tim": {...}, "tdm": {...}}`. |
| `<stem>_calibration/` | Same 8 depth panels used for calibration. |

Sample run on JAAD video_0135 with `scenario=approach_and_stop`:

```
TDM alerts:
  none      :  348  ( 68.2%)
  caution   :  114  ( 22.4%)
  slow_down :   20  (  3.9%)
  brake     :   28  (  5.5%)
```

This is a busy intersection and the simulated car drives straight through
without actually stopping for pedestrians — the BRAKE frames reflect the
model correctly flagging that behavior.

---

## 6. Tuning & known limits

- **Thresholds are deliberately conservative.** Real deployment should tune
  them against labeled driving data (false-alarm rate vs missed-collision rate).
- **Point-particle collisions.** Car and pedestrian are both treated as
  points; a single scalar `r_brake_m` absorbs their combined footprint.
- **Constant-velocity pedestrian.** If the Kalman-derived velocity is noisy
  at a given frame, `t*` / `d*` will be noisy too. `min_pedestrian_history`
  filters the worst cases; consider averaging over the last few frames for
  production.
- **No lateral reasoning for the car.** The demo assumes the ego car moves
  mostly along Z. `CarState` does carry a full `(vx, vz)` so arbitrary
  headings work, but the simulator scenarios all drive straight forward.
- **Single-ego.** TDM is per-driver; multi-vehicle cooperation is out of scope.
- **Pedestrians without world coords are silently skipped** (unless
  `require_world_space=True`, which raises). Upgrade TIM to use a camera
  calibration — see `guide/TIM.md §8`.
