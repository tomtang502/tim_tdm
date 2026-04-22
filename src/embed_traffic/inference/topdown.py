"""Top-down simulation visualization.

Takes a list of `TIMFrameOutput`s with calibrated world-space fields and
renders an animated 2D bird's-eye view showing:
  • the stationary camera (at origin)
  • the camera's field of view
  • pedestrians as colored dots at (X, Z) meters
  • a trailing history per pedestrian
  • velocity arrows (if available)
  • speed label in m/s
  • frame time HUD

Axes convention (matches `CameraCalibration.pixel_to_ground`):
  X: lateral, meters (positive = right of camera)
  Z: forward, meters (positive = away from camera)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from embed_traffic.inference.demo import color_for_id
from embed_traffic.inference.schema import TIMFrameOutput

__all__ = ["render_topdown_frame", "render_topdown_video"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_XRANGE = (-15.0, 15.0)   # meters, X (lateral)
DEFAULT_ZRANGE = (0.0, 35.0)     # meters, Z (forward from camera)
DEFAULT_CANVAS_SIZE = (800, 900)  # (W, H) pixels
TRAIL_LEN = 40                    # per-pedestrian history length

_BG = (30, 30, 30)        # dark background
_GRID = (55, 55, 55)      # subtle grid
_GRID_MAJOR = (80, 80, 80)
_FOV = (70, 70, 120)      # bluish FoV wedge
_CAMERA = (255, 255, 255)
_AXIS_TXT = (200, 200, 200)
_INTENT_COLORS = {
    "crossing": (70, 70, 255),
    "not-crossing": (70, 200, 70),
}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate mapping
# ─────────────────────────────────────────────────────────────────────────────

def _build_world_to_canvas(
    canvas_size: tuple[int, int],
    xrange: tuple[float, float],
    zrange: tuple[float, float],
):
    W, H = canvas_size
    x_span = xrange[1] - xrange[0]
    z_span = zrange[1] - zrange[0]
    sx = W / x_span
    sz = H / z_span

    def w2c(x_m: float, z_m: float) -> tuple[int, int]:
        px = int(round((x_m - xrange[0]) * sx))
        # Z = 0 at the bottom of the canvas (camera), larger Z upwards.
        py = int(round(H - (z_m - zrange[0]) * sz))
        return px, py

    return w2c, sx, sz


# ─────────────────────────────────────────────────────────────────────────────
# Single frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_topdown_frame(
    out: TIMFrameOutput,
    track_histories: dict[int, list[tuple[float, float]]],
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    xrange: tuple[float, float] = DEFAULT_XRANGE,
    zrange: tuple[float, float] = DEFAULT_ZRANGE,
    hfov_deg: float = 60.0,
) -> np.ndarray:
    """Render a single top-down canvas for `out`. Mutates `track_histories` in place."""
    W, H = canvas_size
    canvas = np.full((H, W, 3), _BG, dtype=np.uint8)
    w2c, sx, sz = _build_world_to_canvas(canvas_size, xrange, zrange)

    # ── Grid ──
    for x_m in range(int(np.ceil(xrange[0])), int(np.floor(xrange[1])) + 1):
        color = _GRID_MAJOR if x_m % 5 == 0 else _GRID
        cv2.line(canvas, w2c(x_m, zrange[0]), w2c(x_m, zrange[1]), color, 1)
    for z_m in range(int(np.ceil(zrange[0])), int(np.floor(zrange[1])) + 1):
        color = _GRID_MAJOR if z_m % 5 == 0 else _GRID
        cv2.line(canvas, w2c(xrange[0], z_m), w2c(xrange[1], z_m), color, 1)

    # ── Camera field of view (triangle) ──
    fov_half = np.deg2rad(hfov_deg / 2.0)
    fov_depth = zrange[1]
    fov_x_extent = fov_depth * np.tan(fov_half)
    pts = np.array(
        [w2c(0, 0), w2c(-fov_x_extent, fov_depth), w2c(fov_x_extent, fov_depth)],
        dtype=np.int32,
    )
    fov_layer = canvas.copy()
    cv2.fillPoly(fov_layer, [pts], _FOV)
    cv2.addWeighted(fov_layer, 0.25, canvas, 0.75, 0, dst=canvas)

    # ── Camera marker (triangle pointing up = +Z forward) ──
    cam_xy = w2c(0, 0)
    cam_size = 10
    cam_tri = np.array(
        [
            (cam_xy[0], cam_xy[1] - cam_size),
            (cam_xy[0] - cam_size, cam_xy[1] + int(cam_size * 0.6)),
            (cam_xy[0] + cam_size, cam_xy[1] + int(cam_size * 0.6)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [cam_tri], _CAMERA)
    cv2.putText(canvas, "camera", (cam_xy[0] - 28, cam_xy[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _CAMERA, 1)

    # ── Pedestrians ──
    live_ids = set()
    for ped in out.pedestrians:
        if ped.position_m_ground is None:
            continue
        x, z = ped.position_m_ground
        # Clip to viewport so we don't spam off-canvas points in history
        if not (xrange[0] <= x <= xrange[1] and zrange[0] <= z <= zrange[1]):
            continue

        tid = ped.ped_id
        live_ids.add(tid)
        hist = track_histories.setdefault(tid, [])
        hist.append((x, z))
        if len(hist) > TRAIL_LEN:
            del hist[:-TRAIL_LEN]

        color = color_for_id(tid)

        # Trail
        if len(hist) >= 2:
            pts_px = np.array([w2c(xx, zz) for (xx, zz) in hist], dtype=np.int32)
            for i in range(1, len(pts_px)):
                alpha = i / len(pts_px)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(canvas, tuple(pts_px[i - 1]), tuple(pts_px[i]), faded, 2)

        # Current position
        pt = w2c(x, z)
        ring = _INTENT_COLORS.get(ped.crossing_intent or "", color)
        cv2.circle(canvas, pt, 7, ring, 2)
        cv2.circle(canvas, pt, 5, color, -1)

        # Velocity arrow
        if ped.velocity_m_s is not None:
            vx, vz = ped.velocity_m_s
            arrow_end = w2c(x + vx, z + vz)
            cv2.arrowedLine(canvas, pt, arrow_end, color, 1, tipLength=0.3)

        # Label
        parts = [f"ID {tid}"]
        if ped.speed_m_s is not None:
            parts.append(f"{ped.speed_m_s:.1f}m/s")
        cv2.putText(
            canvas,
            " ".join(parts),
            (pt[0] + 9, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    # ── Fade dead trails so we don't keep them forever ──
    for tid in list(track_histories.keys()):
        if tid not in live_ids:
            del track_histories[tid]

    # ── Axis labels ──
    cv2.putText(canvas, "X (m, lateral)", (W - 170, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _AXIS_TXT, 1)
    cv2.putText(canvas, "Z (m, forward)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _AXIS_TXT, 1)
    # Scale ticks on the sides
    for x_m in (xrange[0], 0, xrange[1]):
        p = w2c(x_m, zrange[0])
        cv2.putText(canvas, f"{int(x_m)}", (p[0] - 6, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _AXIS_TXT, 1)
    for z_m in (0, 10, 20, 30):
        if zrange[0] <= z_m <= zrange[1]:
            p = w2c(xrange[0], z_m)
            cv2.putText(canvas, f"{z_m}", (6, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _AXIS_TXT, 1)

    # ── HUD ──
    hud = (
        f"Top-down view  |  frame {out.frame_id}  t={out.frame_time_s:.2f}s  |  "
        f"peds: {out.num_pedestrians}"
    )
    cv2.putText(canvas, hud, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (230, 230, 230), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Video writer
# ─────────────────────────────────────────────────────────────────────────────

def render_topdown_video(
    outputs: Iterable[TIMFrameOutput],
    output_path: str | Path,
    fps: float,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    xrange: tuple[float, float] = DEFAULT_XRANGE,
    zrange: tuple[float, float] = DEFAULT_ZRANGE,
    hfov_deg: float = 60.0,
) -> int:
    """Write a top-down mp4 summarizing a whole TIM run. Returns frames written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, canvas_size)
    histories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    n_written = 0
    try:
        for out in outputs:
            frame = render_topdown_frame(
                out, histories,
                canvas_size=canvas_size,
                xrange=xrange,
                zrange=zrange,
                hfov_deg=hfov_deg,
            )
            writer.write(frame)
            n_written += 1
    finally:
        writer.release()
    return n_written
