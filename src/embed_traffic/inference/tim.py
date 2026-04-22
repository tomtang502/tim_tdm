"""Traffic Info Model (TIM) — end-to-end inference pipeline.

Chains: YOLO detection → ByteTrack tracking → Kalman trajectory →
        LSTM crossing-intent classifier.

This is the dashcam-only deployment wrapper (see plan.txt [VII]). It produces
`TIMFrameOutput` records that TDM consumes directly.

Usage (Python):
    from embed_traffic.inference import TIM
    tim = TIM()  # uses checkpoints/ped_dashcam by default
    frame_out = tim.process_frame(bgr_frame, frame_id=0)

Usage (video):
    outputs = tim.process_video("path/to/clip.mp4")

Usage (CLI):
    python -m embed_traffic.inference clip.mp4 --output out.jsonl
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np
import torch

from embed_traffic.calibration.schema import CameraCalibration
from embed_traffic.inference.schema import PedestrianInfo, TIMFrameOutput
from embed_traffic.models.intent_model import CrossingIntentLSTM, FEATURE_DIM, SEQ_LEN
from embed_traffic.models.trajectory import PedestrianTrajectory, TrajectoryEstimator
from embed_traffic.paths import detector_weights, intent_weights

# Default checkpoint names — dashcam-only deployment
DEFAULT_DETECTOR_RUN = "ped_dashcam"
DEFAULT_INTENT_RUN = "intent_default"

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TIM:
    """Traffic Info Model inference wrapper.

    Detect → Track → Trajectory → Intent, for the dashcam deployment.

    Args:
        detector_run_name: checkpoints/<name>/weights/best.pt is loaded as the
            YOLO detector. Defaults to 'ped_dashcam'.
        intent_run_name:   checkpoints/<name>/intent_lstm.pt is loaded as the
            crossing-intent classifier. Defaults to 'intent_default'.
            If the file is missing, TIM runs without intent (None/None fields).
        detector_weights_path: bypass run_name resolution and pass an absolute path.
        intent_weights_path:   same, for the intent classifier.
        tracker: ultralytics tracker config ('bytetrack.yaml' or 'botsort.yaml').
        fps: video frame rate for speed/time calculations.
        predict_steps: number of future frames to predict per pedestrian.
        imgsz: inference resolution.
        device: torch device ('cuda:0', 'cpu', or torch.device).
    """

    def __init__(
        self,
        detector_run_name: str = DEFAULT_DETECTOR_RUN,
        intent_run_name: str = DEFAULT_INTENT_RUN,
        detector_weights_path: Optional[str] = None,
        intent_weights_path: Optional[str] = None,
        tracker: str = "bytetrack.yaml",
        fps: float = 30.0,
        predict_steps: int = 30,
        imgsz: int = 1280,
        device: Optional[torch.device] = None,
        camera_calibration: Optional[str | Path | CameraCalibration] = None,
    ) -> None:
        self.fps = fps
        self.predict_steps = predict_steps
        self.imgsz = imgsz
        self.device = torch.device(device) if device is not None else _DEVICE

        # ── Camera calibration (optional) ──
        # If provided, TIM additionally emits position_m_ground / speed_m_s /
        # velocity_m_s per pedestrian. Accepts either a loaded CameraCalibration
        # instance or a path to a JSON written by `embed_traffic.calibration`.
        self.calibration: Optional[CameraCalibration]
        if camera_calibration is None:
            self.calibration = None
        elif isinstance(camera_calibration, CameraCalibration):
            self.calibration = camera_calibration
        else:
            self.calibration = CameraCalibration.load(camera_calibration)

        # Per-track history of ground positions (for metric speed via finite diff)
        self._ground_history: dict[int, list[tuple[int, np.ndarray]]] = {}

        # ── Detection + Tracking ──
        if detector_weights_path is None:
            detector_weights_path = str(detector_weights(detector_run_name))
        if not Path(detector_weights_path).exists():
            raise FileNotFoundError(
                f"Detector weights not found: {detector_weights_path}\n"
                f"Train with `scripts/run_train_dashcam.sh` or pass "
                f"`detector_run_name`/`detector_weights_path`."
            )
        self.estimator = TrajectoryEstimator(
            model_path=detector_weights_path, tracker=tracker, fps=fps
        )
        self._detector_path = detector_weights_path

        # ── Crossing-intent classifier ──
        if intent_weights_path is None:
            intent_weights_path = str(intent_weights(intent_run_name))
        self._intent_path = intent_weights_path

        if Path(intent_weights_path).exists():
            self.intent_model: Optional[CrossingIntentLSTM] = CrossingIntentLSTM().to(
                self.device
            )
            self.intent_model.load_state_dict(
                torch.load(intent_weights_path, map_location=self.device, weights_only=True)
            )
            self.intent_model.eval()
        else:
            self.intent_model = None

        # Sliding-window feature buffers, one per tracked pedestrian
        self._feature_buffer: dict[int, list[list[float]]] = {}

    # ─────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────

    @property
    def has_intent_model(self) -> bool:
        return self.intent_model is not None

    @property
    def detector_path(self) -> str:
        return self._detector_path

    @property
    def intent_path(self) -> str:
        return self._intent_path

    # ─────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset tracking state for a new video."""
        self.estimator.reset()
        self._feature_buffer.clear()
        self._ground_history.clear()

    # ─────────────────────────────────────────────────────────────────
    # World-space projection
    # ─────────────────────────────────────────────────────────────────

    def _update_ground_state(
        self,
        track_id: int,
        bbox: list[float],
        frame_id: int,
    ) -> tuple[Optional[list[float]], Optional[float], Optional[list[float]]]:
        """Project bbox foot-point to ground; return (position, speed, velocity).

        Foot point = (bbox center x, bbox bottom y). Uses the camera calibration's
        homography. Returns (None, None, None) if no calibration is loaded.
        """
        if self.calibration is None:
            return None, None, None

        x1, y1, x2, y2 = bbox
        foot_u = 0.5 * (x1 + x2)
        foot_v = y2  # bottom of the bbox
        ground = self.calibration.pixel_to_ground(np.asarray([[foot_u, foot_v]]))[0]
        # ground = (X, Z) in meters, or NaN if foot point is above the horizon
        if not np.all(np.isfinite(ground)):
            return None, None, None

        history = self._ground_history.setdefault(track_id, [])
        history.append((frame_id, ground.copy()))
        # Keep a bounded window (≤ 30 entries)
        if len(history) > 30:
            del history[:-30]

        # Instantaneous velocity from last two samples
        velocity_m_s: Optional[list[float]] = None
        speed_m_s: Optional[float] = None
        if len(history) >= 2:
            (f_prev, p_prev), (f_curr, p_curr) = history[-2], history[-1]
            dt = max((f_curr - f_prev) / self.fps, 1e-6)
            vx = (p_curr[0] - p_prev[0]) / dt
            vz = (p_curr[1] - p_prev[1]) / dt
            velocity_m_s = [float(vx), float(vz)]
            speed_m_s = float(np.hypot(vx, vz))

        return [float(ground[0]), float(ground[1])], speed_m_s, velocity_m_s

    # ─────────────────────────────────────────────────────────────────
    # Intent classification
    # ─────────────────────────────────────────────────────────────────

    def _update_intent_features(
        self,
        track_id: int,
        traj: PedestrianTrajectory,
        img_w: int,
        img_h: int,
    ) -> list[list[float]]:
        """Append one normalized feature row and return the trailing window."""
        buf = self._feature_buffer.setdefault(track_id, [])

        bbox = traj.bboxes[-1]
        cx_norm = ((bbox[0] + bbox[2]) / 2) / img_w
        cy_norm = ((bbox[1] + bbox[3]) / 2) / img_h
        w_norm = abs(bbox[2] - bbox[0]) / img_w
        h_norm = abs(bbox[3] - bbox[1]) / img_h
        area_norm = w_norm * h_norm

        dx, dy, speed_norm = 0.0, 0.0, 0.0
        if len(traj.centers) >= 2:
            prev = traj.centers[-2]
            curr = traj.centers[-1]
            dx = (curr[0] - prev[0]) / img_w
            dy = (curr[1] - prev[1]) / img_h
            speed_norm = float(np.sqrt(dx * dx + dy * dy))

        buf.append([cx_norm, cy_norm, w_norm, h_norm, dx, dy, speed_norm, area_norm])
        if len(buf) > SEQ_LEN:
            del buf[:-SEQ_LEN]
        return buf

    def _classify_intent(
        self,
        track_id: int,
        traj: PedestrianTrajectory,
        img_w: int,
        img_h: int,
    ) -> tuple[Optional[str], Optional[float]]:
        """Classify crossing intent if the track is long enough."""
        if self.intent_model is None:
            return None, None

        features = self._update_intent_features(track_id, traj, img_w, img_h)
        if len(features) < SEQ_LEN:
            return None, None

        seq = np.asarray(features[-SEQ_LEN:], dtype=np.float32)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.intent_model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(logits.argmax(dim=1).item())
            prob_cross = float(probs[0, 1].item())
        return ("crossing" if pred_idx == 1 else "not-crossing"), prob_cross

    # ─────────────────────────────────────────────────────────────────
    # Core inference
    # ─────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_id: int) -> TIMFrameOutput:
        """Run TIM on a single BGR frame. Returns TIMFrameOutput."""
        t0 = time.perf_counter()
        img_h, img_w = frame.shape[:2]

        active = self.estimator.process_frame(frame, frame_id)

        peds: List[PedestrianInfo] = []
        for track in active:
            tid = track["track_id"]
            traj = self.estimator.get_trajectory(tid)
            if traj is None:
                continue

            intent, prob = self._classify_intent(tid, traj, img_w, img_h)

            predicted_path: Optional[List[List[float]]] = None
            if traj.length >= 5:
                predicted_path = traj.predict_future(self.predict_steps)

            pos_ground, speed_ms, vel_ms = self._update_ground_state(
                tid, track["bbox"], frame_id
            )

            peds.append(
                PedestrianInfo(
                    ped_id=int(tid),
                    bbox=[float(v) for v in track["bbox"]],
                    center=[float(v) for v in track["center"]],
                    confidence=float(track["conf"]),
                    speed_px_s=float(track["speed_px_s"]),
                    avg_speed_px_s=float(track["avg_speed_px_s"]),
                    direction=[float(v) for v in track["direction"]],
                    track_length=int(track["track_length"]),
                    crossing_intent=intent,
                    crossing_prob=prob,
                    predicted_path=predicted_path,
                    position_m_ground=pos_ground,
                    speed_m_s=speed_ms,
                    velocity_m_s=vel_ms,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        frame_time_s = float(frame_id) / self.fps if self.fps > 0 else 0.0
        return TIMFrameOutput(
            frame_id=frame_id,
            pedestrians=peds,
            frame_width=img_w,
            frame_height=img_h,
            frame_time_s=frame_time_s,
            processing_time_ms=elapsed_ms,
        )

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray, TIMFrameOutput], None]] = None,
    ) -> List[TIMFrameOutput]:
        """Run TIM over an entire video. Returns one TIMFrameOutput per frame.

        Args:
            video_path: path to a video file (mp4, avi, ...).
            max_frames: optional cap; if None, process the entire video.
            callback: optional `callback(frame, output)` invoked per frame —
                      useful for visualization or streaming to TDM.
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = video_fps
        self.estimator.fps = video_fps

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total = min(total, max_frames)

        outputs: List[TIMFrameOutput] = []
        try:
            for fid in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                out = self.process_frame(frame, fid)
                outputs.append(out)
                if callback is not None:
                    callback(frame, out)
        finally:
            cap.release()
        return outputs

    def stream(
        self,
        frames: Iterable[np.ndarray],
        start_frame_id: int = 0,
    ) -> Iterable[TIMFrameOutput]:
        """Iterate frames → TIMFrameOutput lazily (for live streams)."""
        self.reset()
        for i, frame in enumerate(frames):
            yield self.process_frame(frame, start_frame_id + i)
