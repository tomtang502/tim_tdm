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
    "draw_alert_banner",
    "render_topdown_frame_with_car",
]


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
    """Render the TIM top-down view and overlay the car + predicted trajectory.

    Extra elements drawn on top of `render_topdown_frame`:
      - car marker (rectangle-ish) at car position
      - projected car trajectory for the next ~3 s (dashed)
      - highlighted "at-risk" pedestrian with the predicted closest-approach
        segment connecting car(t*) ←→ ped(t*)
    """
    canvas = render_topdown_frame(
        tim_out,
        track_histories,
        canvas_size=canvas_size,
        xrange=xrange,
        zrange=zrange,
        hfov_deg=hfov_deg,
    )
    w2c, _, _ = _build_world_to_canvas(canvas_size, xrange, zrange)

    alert_color = ALERT_COLORS[tdm_out.alert]

    # ── Car history trail ──
    if car_history is not None and tdm_out.car_state is not None:
        car_history.append(tdm_out.car_state.position_m)
        if len(car_history) > 60:
            del car_history[:-60]
        if len(car_history) >= 2:
            pts = np.array([w2c(*p) for p in car_history], dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                faded = tuple(int(c * alpha) for c in (220, 220, 220))
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

        # 1) Main body
        body = body_polygon(cx, cz, car_half_l_m, car_half_w_m)
        cv2.fillPoly(canvas, [body], alert_color)
        cv2.polylines(canvas, [body], True, (255, 255, 255), 1)

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
        cv2.putText(canvas, "SIM CAR",
                     (cx_pt + 12, cz_pt - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, f"{cs.speed_m_s:.1f} m/s  hdg={cs.heading_deg:+.0f}°",
                     (cx_pt + 12, cz_pt + 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Closest-approach segment to at-risk ped ──
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

    # ── Alert banner at the bottom of the top-down panel ──
    H = canvas.shape[0]
    W = canvas.shape[1]
    cv2.rectangle(canvas, (0, H - 34), (W, H), alert_color, -1)
    cv2.putText(
        canvas, f"TDM: {tdm_out.alert.value.upper()}",
        (10, H - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )

    return canvas
