"""TDM visualization overlays.

Draws alert banners on top of TIM overlay frames and augments the top-down
view with the ego car + its predicted trajectory + the worst-case pedestrian
highlight.
"""

from __future__ import annotations

import cv2
import numpy as np

from embed_traffic.inference.schema import TIMFrameOutput
from embed_traffic.inference.topdown import (
    DEFAULT_XRANGE,
    DEFAULT_ZRANGE,
    _build_world_to_canvas,
    render_topdown_frame,
)
from embed_traffic.tdm.schema import AlertLevel, TDMOutput

__all__ = [
    "ALERT_COLORS",
    "CAR_PALETTE",
    "CarColorAllocator",
    "draw_alert_banner",
    "draw_alert_banner_multi",
    "render_topdown_frame_with_car",
    "render_topdown_frame_with_cars",
]


# Fixed car palette (BGR, fairly saturated, high-contrast). Cycled by index.
CAR_PALETTE: list[tuple[int, int, int]] = [
    (240, 200, 80),    # light blue/teal
    (80, 200, 240),    # amber/yellow
    (240, 80, 200),    # magenta
    (80, 240, 80),     # lime
    (120, 80, 240),    # purple
]


def _color_for_car(index: int) -> tuple[int, int, int]:
    return CAR_PALETTE[index % len(CAR_PALETTE)]


class CarColorAllocator:
    """Assigns a stable color to each car_id in first-seen order.

    With multiple spawns over time, palette-by-index would re-color cars
    every frame as the active list changes. This class fixes the mapping
    once per car_id so a given car keeps its color for its entire lifetime.
    """

    def __init__(self, palette: list[tuple[int, int, int]] = CAR_PALETTE) -> None:
        self.palette = palette
        self._map: dict[str, tuple[int, int, int]] = {}

    def color_for(self, car_id: str | None, fallback_index: int = 0) -> tuple[int, int, int]:
        if not car_id:
            return self.palette[fallback_index % len(self.palette)]
        if car_id not in self._map:
            self._map[car_id] = self.palette[len(self._map) % len(self.palette)]
        return self._map[car_id]


# BGR colors — bright enough to cut through busy scenes
ALERT_COLORS: dict[AlertLevel, tuple[int, int, int]] = {
    AlertLevel.NONE:      (100, 200, 100),   # green
    AlertLevel.CAUTION:   (40, 200, 240),    # amber / yellow-orange
    AlertLevel.SLOW_DOWN: (40, 140, 240),    # orange
    AlertLevel.BRAKE:     (40, 40, 230),     # red
}


# ─────────────────────────────────────────────────────────────────────────────
# Alert banner (drawn on top of overlay frame)
# ─────────────────────────────────────────────────────────────────────────────

def draw_alert_banner(frame: np.ndarray, tdm_out: TDMOutput) -> None:
    """Draw a colored TDM alert banner across the top of `frame` in-place."""
    H, W = frame.shape[:2]
    color = ALERT_COLORS[tdm_out.alert]

    # Semi-transparent band over the image top
    band_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, band_h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Label: big alert name
    label = tdm_out.alert.value.replace("_", " ").upper()
    cv2.putText(
        frame, label, (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA,
    )

    # Reason: smaller sub-text
    cv2.putText(
        frame, tdm_out.reason, (14, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Car info on the right of the banner (if provided)
    if tdm_out.car_state is not None:
        cs = tdm_out.car_state
        info = (
            f"Car  (x={cs.position_m[0]:+.1f}, z={cs.position_m[1]:+.1f}) m   "
            f"|v|={cs.speed_m_s:.1f} m/s"
        )
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            frame, info, (W - tw - 14, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Top-down view with car added
# ─────────────────────────────────────────────────────────────────────────────

def _draw_one_car(
    canvas: np.ndarray,
    tdm_out: TDMOutput,
    car_history: list[tuple[float, float]] | None,
    w2c,
    car_color: tuple[int, int, int],
    label_suffix: str = "",
) -> None:
    """Draw one simulated car + its closest-approach segment onto `canvas`.

    `car_color` tints the body, wheels stay dark, alert-colored ring surrounds
    the body when the alert is more severe than NONE.
    """
    alert_color = ALERT_COLORS[tdm_out.alert]

    # ── Car history trail (grayscale) ──
    if car_history is not None and tdm_out.car_state is not None:
        car_history.append(tdm_out.car_state.position_m)
        if len(car_history) > 60:
            del car_history[:-60]
        if len(car_history) >= 2:
            pts = np.array([w2c(*p) for p in car_history], dtype=np.int32)
            for i in range(1, len(pts)):
                a = i / len(pts)
                faded = tuple(int(c * a) for c in car_color)
                cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), faded, 1)

    # ── Simulated car marker (car-shaped, oriented by velocity) ──
    if tdm_out.car_state is not None:
        cs = tdm_out.car_state
        cx, cz = cs.position_m
        vx, vz = cs.velocity_m_s

        # Car dimensions in world meters (compact sedan-ish: 1.8 m × 4.5 m)
        car_half_w_m = 0.9
        car_half_l_m = 2.25
        wheel_l_m = 0.55      # longitudinal wheel length (along body axis)
        wheel_w_m = 0.25      # lateral wheel width
        axle_offset_l = 1.55  # how far front / back axles sit from car center
        axle_offset_w = 0.85  # how far wheels sit from body centerline

        speed = float(np.hypot(vx, vz))
        if speed > 1e-3:
            ux, uz = vx / speed, vz / speed            # forward unit vec (world)
        else:
            ux, uz = 0.0, -1.0                          # default: pointing toward camera
        px, pz = uz, -ux                                # lateral (right-of-forward)

        def world_to_px(x_m: float, z_m: float) -> tuple[int, int]:
            return w2c(x_m, z_m)

        def body_polygon(center_x: float, center_z: float,
                          half_l: float, half_w: float) -> np.ndarray:
            pts = [
                (center_x + half_l * ux + half_w * px,
                 center_z + half_l * uz + half_w * pz),   # front-right
                (center_x + half_l * ux - half_w * px,
                 center_z + half_l * uz - half_w * pz),   # front-left
                (center_x - half_l * ux - half_w * px,
                 center_z - half_l * uz - half_w * pz),   # back-left
                (center_x - half_l * ux + half_w * px,
                 center_z - half_l * uz + half_w * pz),   # back-right
            ]
            return np.array([world_to_px(*p) for p in pts], dtype=np.int32)

        # 1) Main body — color REFLECTS ALERT STATE.
        # When NONE, use the car's identity color so multiple cars are easy
        # to tell apart. When alerting, the body turns the alert color and
        # the outline switches to the identity color so the car is still
        # identifiable ("that red-outlined yellow-body car is car B").
        body = body_polygon(cx, cz, car_half_l_m, car_half_w_m)
        if tdm_out.alert == AlertLevel.NONE:
            body_color = car_color
            outline_color = (255, 255, 255)
            outline_thickness = 1
        else:
            body_color = alert_color
            outline_color = car_color
            outline_thickness = 3
        cv2.fillPoly(canvas, [body], body_color)
        cv2.polylines(canvas, [body], True, outline_color, outline_thickness)

        # 2) Windshield — a trapezoid on the front half
        ws_front = car_half_l_m * 0.85
        ws_back = car_half_l_m * 0.35
        ws = np.array([
            world_to_px(cx + ws_front * ux + 0.60 * car_half_w_m * px,
                        cz + ws_front * uz + 0.60 * car_half_w_m * pz),
            world_to_px(cx + ws_front * ux - 0.60 * car_half_w_m * px,
                        cz + ws_front * uz - 0.60 * car_half_w_m * pz),
            world_to_px(cx + ws_back * ux - 0.70 * car_half_w_m * px,
                        cz + ws_back * uz - 0.70 * car_half_w_m * pz),
            world_to_px(cx + ws_back * ux + 0.70 * car_half_w_m * px,
                        cz + ws_back * uz + 0.70 * car_half_w_m * pz),
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [ws], (235, 235, 235))

        # 3) Four wheels — rotated rectangles at the axle positions
        for side_sign in (+1, -1):
            for axle_sign in (+1, -1):
                wheel_cx = cx + axle_sign * axle_offset_l * ux + side_sign * axle_offset_w * px
                wheel_cz = cz + axle_sign * axle_offset_l * uz + side_sign * axle_offset_w * pz
                wheel = body_polygon(wheel_cx, wheel_cz, wheel_l_m / 2, wheel_w_m / 2)
                cv2.fillPoly(canvas, [wheel], (18, 18, 18))
                cv2.polylines(canvas, [wheel], True, (255, 255, 255), 1)

        # 4) Heading tick on the FRONT bumper (two short lines as "headlights")
        for side_sign in (+1, -1):
            bumper_x = cx + car_half_l_m * ux + side_sign * 0.55 * car_half_w_m * px
            bumper_z = cz + car_half_l_m * uz + side_sign * 0.55 * car_half_w_m * pz
            tip_x = bumper_x + 0.5 * ux
            tip_z = bumper_z + 0.5 * uz
            cv2.line(canvas, world_to_px(bumper_x, bumper_z),
                     world_to_px(tip_x, tip_z), (255, 245, 180), 2)

        # 5) Velocity arrow (~2 s lookahead)
        cx_pt, cz_pt = world_to_px(cx, cz)
        arrow_end = world_to_px(cx + vx * 2.0, cz + vz * 2.0)
        cv2.arrowedLine(canvas, (cx_pt, cz_pt), arrow_end,
                         (255, 255, 255), 2, tipLength=0.25)

        # 6) Labels
        label = "SIM CAR"
        if label_suffix:
            label = f"{label} {label_suffix}"
        cv2.putText(canvas, label,
                     (cx_pt + 12, cz_pt - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, f"{cs.speed_m_s:.1f} m/s  hdg={cs.heading_deg:+.0f}°",
                     (cx_pt + 12, cz_pt + 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Closest-approach segment to at-risk ped (alert-colored) ──
    if tdm_out.at_risk_ped_id is not None:
        for pred in tdm_out.per_ped_predictions:
            if pred.ped_id != tdm_out.at_risk_ped_id:
                continue
            a = w2c(*pred.car_pos_m_at_min)
            b = w2c(*pred.ped_pos_m_at_min)
            cv2.line(canvas, a, b, alert_color, 2)
            cv2.circle(canvas, a, 5, alert_color, 2)
            cv2.circle(canvas, b, 5, alert_color, 2)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(
                canvas,
                f"t*={pred.time_to_min_dist_s:.1f}s  d*={pred.min_distance_m:.2f}m",
                (mid[0] + 6, mid[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, alert_color, 1,
            )
            break


# ─────────────────────────────────────────────────────────────────────────────
# Single-car convenience (unchanged public API)
# ─────────────────────────────────────────────────────────────────────────────

def render_topdown_frame_with_car(
    tim_out: TIMFrameOutput,
    tdm_out: TDMOutput,
    track_histories: dict[int, list[tuple[float, float]]],
    canvas_size: tuple[int, int] = (800, 900),
    xrange: tuple[float, float] = DEFAULT_XRANGE,
    zrange: tuple[float, float] = DEFAULT_ZRANGE,
    hfov_deg: float = 60.0,
    car_history: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Render the TIM top-down view and overlay ONE simulated car."""
    canvas = render_topdown_frame(
        tim_out, track_histories,
        canvas_size=canvas_size, xrange=xrange, zrange=zrange, hfov_deg=hfov_deg,
    )
    w2c, _, _ = _build_world_to_canvas(canvas_size, xrange, zrange)
    _draw_one_car(canvas, tdm_out, car_history, w2c, CAR_PALETTE[0])

    # Alert banner at the bottom of the top-down panel
    H, W = canvas.shape[:2]
    alert_color = ALERT_COLORS[tdm_out.alert]
    cv2.rectangle(canvas, (0, H - 34), (W, H), alert_color, -1)
    cv2.putText(canvas, f"TDM: {tdm_out.alert.value.upper()}",
                 (10, H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Multi-car entry points
# ─────────────────────────────────────────────────────────────────────────────

def render_topdown_frame_with_cars(
    tim_out: TIMFrameOutput,
    tdm_outs: list[TDMOutput],
    track_histories: dict[int, list[tuple[float, float]]],
    car_histories: list[list[tuple[float, float]]] | None = None,
    canvas_size: tuple[int, int] = (800, 900),
    xrange: tuple[float, float] = DEFAULT_XRANGE,
    zrange: tuple[float, float] = DEFAULT_ZRANGE,
    hfov_deg: float = 60.0,
    color_allocator: "CarColorAllocator | None" = None,
) -> np.ndarray:
    """Render the TIM top-down view with N simulated cars.

    Each car's body color changes to reflect its alert state (NONE = identity
    color from `color_allocator`, else = alert color). The outline uses the
    identity color when alerting so the car remains identifiable.

    Args:
        tim_out:         the single frame's TIM output (shared across all cars).
        tdm_outs:        one TDMOutput per car, same order as passed to TDM.
        track_histories: pedestrian histories (managed by caller across frames).
        car_histories:   per-car history lists (managed by caller across frames).
                          If None, no car trail is drawn. Length must match
                          tdm_outs when provided.
        color_allocator: optional `CarColorAllocator` that assigns a stable
                          color per car_id across frames. If None, falls back
                          to palette-by-index (not stable when cars come and go).
    """
    canvas = render_topdown_frame(
        tim_out, track_histories,
        canvas_size=canvas_size, xrange=xrange, zrange=zrange, hfov_deg=hfov_deg,
    )
    w2c, _, _ = _build_world_to_canvas(canvas_size, xrange, zrange)

    if car_histories is not None and len(car_histories) != len(tdm_outs):
        raise ValueError(
            f"car_histories length ({len(car_histories)}) must match "
            f"tdm_outs length ({len(tdm_outs)})"
        )

    for i, tdm_out in enumerate(tdm_outs):
        history = car_histories[i] if car_histories is not None else None
        suffix = tdm_out.car_id if tdm_out.car_id else f"#{i + 1}"
        if color_allocator is not None:
            car_color = color_allocator.color_for(tdm_out.car_id, fallback_index=i)
        else:
            car_color = _color_for_car(i)
        _draw_one_car(canvas, tdm_out, history, w2c, car_color,
                       label_suffix=f"({suffix})")

    # ── Combined footer showing the worst alert across all cars ──
    H, W = canvas.shape[:2]
    worst = max(tdm_outs, key=lambda o: o.alert.severity) if tdm_outs else None
    if worst is not None:
        alert_color = ALERT_COLORS[worst.alert]
        cv2.rectangle(canvas, (0, H - 34), (W, H), alert_color, -1)
        label = f"TDM worst: {worst.alert.value.upper()}"
        if worst.car_id:
            label = f"{label}  ({worst.car_id})"
        cv2.putText(canvas, label, (10, H - 11),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def draw_alert_banner_multi(
    frame: np.ndarray,
    tdm_outs: list[TDMOutput],
    color_allocator: "CarColorAllocator | None" = None,
) -> None:
    """Draw a stacked alert banner — one row per car — at the top of `frame`.

    Color is the worst alert across all cars. Each row shows the car's id
    (if any) + its own alert level + a short kinematic summary. The small
    identity dot uses `color_allocator` when provided so it matches the
    top-down color.
    """
    if not tdm_outs:
        return
    H, W = frame.shape[:2]
    worst = max(tdm_outs, key=lambda o: o.alert.severity)
    color = ALERT_COLORS[worst.alert]

    row_h = 44
    band_h = min(H // 3, 28 + row_h * len(tdm_outs))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, band_h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Top-of-band summary
    summary = f"TDM — worst: {worst.alert.value.upper()}"
    if worst.car_id:
        summary = f"{summary}  ({worst.car_id})"
    cv2.putText(frame, summary, (14, 26),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Per-car rows
    for i, out in enumerate(tdm_outs):
        y = 28 + i * row_h
        if y + 30 > band_h:
            break
        if color_allocator is not None:
            car_color = color_allocator.color_for(out.car_id, fallback_index=i)
        else:
            car_color = _color_for_car(i)
        alert_color = ALERT_COLORS[out.alert]
        # small colored dot for the car's identity
        cv2.circle(frame, (24, y + 18), 7, car_color, -1)
        cv2.circle(frame, (24, y + 18), 7, (255, 255, 255), 1)
        # car id + alert
        cid = out.car_id if out.car_id else f"car_{i+1}"
        line1 = f"{cid}: {out.alert.value.upper()}"
        cv2.putText(frame, line1, (40, y + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2, cv2.LINE_AA)
        # reason
        cv2.putText(frame, out.reason, (40, y + 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        # car kinematics on the right
        if out.car_state is not None:
            cs = out.car_state
            info = (
                f"({cs.position_m[0]:+.1f}, {cs.position_m[1]:+.1f}) m  "
                f"|v|={cs.speed_m_s:.1f} m/s"
            )
            (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, info, (W - tw - 14, y + 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
