"""Overlay drawing + demo-video generation for TIM inference."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from embed_traffic.inference.schema import PedestrianInfo, TIMFrameOutput
from embed_traffic.inference.tim import TIM

# Stable palette for track-id coloring (BGR)
_RNG = np.random.default_rng(42)
_PALETTE: List[tuple[int, int, int]] = [
    (int(r), int(g), int(b)) for r, g, b in _RNG.integers(50, 255, (256, 3))
]

INTENT_COLORS = {
    "crossing": (0, 0, 255),        # red
    "not-crossing": (0, 200, 0),    # green
}


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_pedestrian(frame: np.ndarray, ped: PedestrianInfo) -> None:
    """Draw a single pedestrian overlay onto `frame` in-place.

    When the pedestrian has a `speed_m_s` (from a calibrated TIM), the label
    shows real-world m/s + ground position; otherwise it falls back to px/s.
    """
    color = color_for_id(ped.ped_id)
    border = INTENT_COLORS.get(ped.crossing_intent or "", color)
    x1, y1, x2, y2 = (int(v) for v in ped.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)

    # Primary label (ID + speed)
    if ped.speed_m_s is not None:
        speed_str = f"{ped.speed_m_s:.1f}m/s"
    else:
        speed_str = f"{ped.speed_px_s:.0f}px/s"
    parts = [f"ID:{ped.ped_id}", speed_str]
    if ped.crossing_intent:
        tag = "CROSS" if ped.crossing_intent == "crossing" else "NO-X"
        parts.append(tag)
        if ped.crossing_prob is not None:
            parts.append(f"{ped.crossing_prob:.0%}")
    cv2.putText(
        frame,
        " ".join(parts),
        (x1, max(y1 - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        border,
        1,
    )

    # Secondary label: ground-plane position if calibrated
    if ped.position_m_ground is not None:
        gx, gz = ped.position_m_ground
        cv2.putText(
            frame,
            f"({gx:+.1f}, {gz:+.1f})m",
            (x1, min(y2 + 18, frame.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            border,
            1,
        )

    # Predicted future path — faded tail
    if ped.predicted_path:
        pts = np.asarray(ped.predicted_path, dtype=np.int32)
        for j in range(1, len(pts)):
            alpha = 1.0 - j / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tuple(pts[j - 1]), tuple(pts[j]), faded, 1)


def draw_hud(frame: np.ndarray, out: TIMFrameOutput) -> None:
    """Draw top-left HUD with frame id / time / ped count / latency."""
    msg = (
        f"Frame {out.frame_id} | t={out.frame_time_s:.2f}s | "
        f"Peds: {out.num_pedestrians} | {out.processing_time_ms:.0f}ms"
    )
    cv2.putText(
        frame,
        msg,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def overlay_frame(frame: np.ndarray, out: TIMFrameOutput, tim: TIM) -> np.ndarray:
    """Return a copy of `frame` with TIM overlays (bboxes, trails, HUD)."""
    canvas = frame.copy()

    # Draw trajectory trails from estimator state (past N centers)
    for ped in out.pedestrians:
        traj = tim.estimator.get_trajectory(ped.ped_id)
        if traj and traj.length > 1:
            pts = np.asarray(traj.centers[-30:], dtype=np.int32)
            color = color_for_id(ped.ped_id)
            for j in range(1, len(pts)):
                cv2.line(canvas, tuple(pts[j - 1]), tuple(pts[j]), color, 2)

    for ped in out.pedestrians:
        draw_pedestrian(canvas, ped)

    draw_hud(canvas, out)
    return canvas


def render_demo_video(
    tim: TIM,
    video_path: str,
    output_path: str,
    max_frames: int | None = None,
) -> None:
    """Run TIM on `video_path` and write an overlay mp4 to `output_path`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    def cb(frame: np.ndarray, out: TIMFrameOutput) -> None:
        writer.write(overlay_frame(frame, out, tim))

    try:
        tim.process_video(video_path, max_frames=max_frames, callback=cb)
    finally:
        writer.release()
