"""TIM (Traffic Info Model) Integration.

Combines all components into a single pipeline:
  Input:  video frame
  Output: per-pedestrian info {ped_id, bbox, speed, trajectory, crossing_intent,
          traffic_light_state}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from embed_traffic.models.intent_model import (
    CrossingIntentLSTM,
    FEATURE_DIM,
    SEQ_LEN,
)
from embed_traffic.models.trajectory import (
    PedestrianTrajectory,
    TrajectoryEstimator,
)
from embed_traffic.paths import detector_weights, intent_weights

MODEL_PATH_DASHCAM = str(detector_weights("ped_dashcam"))
MODEL_PATH_TRAFFIC_LIGHT = str(detector_weights("ped_traffic_light"))
INTENT_MODEL_PATH = str(intent_weights("intent_default"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class PedestrianInfo:
    """Output per pedestrian per frame from TIM."""
    ped_id: int
    bbox: list                    # [x1, y1, x2, y2]
    center: list                  # [cx, cy]
    confidence: float
    speed_px_s: float             # instantaneous speed in pixels/sec
    avg_speed_px_s: float         # average speed over track
    direction: list               # [dx, dy] unit vector
    track_length: int             # number of frames tracked
    crossing_intent: Optional[str] = None   # "crossing", "not-crossing", or None
    crossing_prob: Optional[float] = None   # probability of crossing
    predicted_path: Optional[list] = None   # list of [cx, cy] future positions


@dataclass
class TIMOutput:
    """Full TIM output for a single frame."""
    frame_id: int
    pedestrians: list             # list of PedestrianInfo
    traffic_light_state: Optional[str] = None  # "red", "green", "yellow", or None
    num_pedestrians: int = 0
    processing_time_ms: float = 0.0


class TIM:
    """
    Traffic Info Model — end-to-end pedestrian detection, tracking,
    trajectory estimation, and crossing intent classification.
    """

    def __init__(self, model_path=None, view="traffic_light",
                 intent_model_path=INTENT_MODEL_PATH,
                 tracker="bytetrack.yaml", fps=30.0, predict_steps=30):
        if model_path is None:
            model_path = MODEL_PATH_TRAFFIC_LIGHT if view == "traffic_light" else MODEL_PATH_DASHCAM
        self.fps = fps
        self.predict_steps = predict_steps

        # Detection + Tracking
        self.estimator = TrajectoryEstimator(
            model_path=model_path, tracker=tracker, fps=fps
        )

        # Crossing intent classifier
        self.intent_model = None
        if Path(intent_model_path).exists():
            self.intent_model = CrossingIntentLSTM().to(DEVICE)
            self.intent_model.load_state_dict(
                torch.load(intent_model_path, map_location=DEVICE, weights_only=True)
            )
            self.intent_model.eval()
            print(f"  Intent model loaded from {intent_model_path}")
        else:
            print(f"  WARNING: Intent model not found at {intent_model_path}, skipping intent classification")

        # Feature buffer for intent classification
        self._feature_buffer = {}  # track_id -> list of feature vectors

    def reset(self):
        """Reset all state for a new video."""
        self.estimator.reset()
        self._feature_buffer = {}

    def _extract_intent_features(self, track_id, traj, img_w=1920, img_h=1080):
        """Build feature vector for intent classification from trajectory."""
        if track_id not in self._feature_buffer:
            self._feature_buffer[track_id] = []

        bbox = traj.bboxes[-1]
        cx = ((bbox[0] + bbox[2]) / 2) / img_w
        cy = ((bbox[1] + bbox[3]) / 2) / img_h
        w = abs(bbox[2] - bbox[0]) / img_w
        h = abs(bbox[3] - bbox[1]) / img_h
        area = w * h

        dx, dy, speed = 0.0, 0.0, 0.0
        if len(traj.centers) >= 2:
            prev = traj.centers[-2]
            curr = traj.centers[-1]
            dx = (curr[0] - prev[0]) / img_w
            dy = (curr[1] - prev[1]) / img_h
            speed = np.sqrt(dx**2 + dy**2)

        feat = [cx, cy, w, h, dx, dy, speed, area]
        self._feature_buffer[track_id].append(feat)

        # Keep only last SEQ_LEN features
        if len(self._feature_buffer[track_id]) > SEQ_LEN:
            self._feature_buffer[track_id] = self._feature_buffer[track_id][-SEQ_LEN:]

        return self._feature_buffer[track_id]

    def _classify_intent(self, track_id, traj):
        """Classify crossing intent for a pedestrian."""
        if self.intent_model is None:
            return None, None

        features = self._extract_intent_features(track_id, traj)

        if len(features) < SEQ_LEN:
            return None, None

        # Prepare input tensor
        seq = np.array(features[-SEQ_LEN:])
        x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.intent_model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            prob = probs[0, 1].item()  # probability of crossing

        intent = "crossing" if pred == 1 else "not-crossing"
        return intent, prob

    def process_frame(self, frame, frame_id) -> TIMOutput:
        """
        Process a single video frame through the full TIM pipeline.

        Args:
            frame: BGR image (numpy array)
            frame_id: frame index

        Returns:
            TIMOutput with all pedestrian information
        """
        import time
        t0 = time.time()

        # Step 1: Detect + Track
        active_tracks = self.estimator.process_frame(frame, frame_id)

        pedestrians = []
        for track in active_tracks:
            tid = track["track_id"]
            traj = self.estimator.get_trajectory(tid)

            # Step 2: Crossing intent
            intent, prob = self._classify_intent(tid, traj)

            # Step 3: Future trajectory prediction
            predicted_path = None
            if traj and traj.length >= 5:
                predicted_path = traj.predict_future(self.predict_steps)

            ped = PedestrianInfo(
                ped_id=tid,
                bbox=track["bbox"],
                center=track["center"],
                confidence=track["conf"],
                speed_px_s=track["speed_px_s"],
                avg_speed_px_s=track["avg_speed_px_s"],
                direction=track["direction"],
                track_length=track["track_length"],
                crossing_intent=intent,
                crossing_prob=prob,
                predicted_path=predicted_path,
            )
            pedestrians.append(ped)

        elapsed_ms = (time.time() - t0) * 1000

        return TIMOutput(
            frame_id=frame_id,
            pedestrians=pedestrians,
            num_pedestrians=len(pedestrians),
            processing_time_ms=elapsed_ms,
        )

    def process_video(self, video_path, max_frames=None, callback=None):
        """
        Process an entire video through TIM.

        Args:
            video_path: path to video file
            max_frames: optional frame limit
            callback: optional function called with (frame, tim_output) per frame

        Returns:
            list of TIMOutput
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        outputs = []
        for fid in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame, fid)
            outputs.append(result)

            if callback:
                callback(frame, result)

        cap.release()
        return outputs


def generate_tim_demo(video_path, output_path, max_frames=200):
    """Generate a demo video showing full TIM output overlay."""
    tim = TIM()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    np.random.seed(42)
    colors = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(50, 255, (200, 3))]

    def draw_callback(frame, result):
        for ped in result.pedestrians:
            color = colors[ped.ped_id % len(colors)]
            x1, y1, x2, y2 = [int(v) for v in ped.bbox]

            # Intent-based color override
            if ped.crossing_intent == "crossing":
                border_color = (0, 0, 255)  # red
            elif ped.crossing_intent == "not-crossing":
                border_color = (0, 255, 0)  # green
            else:
                border_color = color

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

            # Label
            label_parts = [f"ID:{ped.ped_id}", f"{ped.speed_px_s:.0f}px/s"]
            if ped.crossing_intent:
                label_parts.append(f"{'CROSS' if ped.crossing_intent == 'crossing' else 'NO-X'}")
                if ped.crossing_prob is not None:
                    label_parts.append(f"{ped.crossing_prob:.0%}")
            label = " ".join(label_parts)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

            # Draw trajectory trail
            traj = tim.estimator.get_trajectory(ped.ped_id)
            if traj and traj.length > 1:
                pts = np.array(traj.centers[-30:], dtype=np.int32)
                for j in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), color, 2)

            # Draw predicted path
            if ped.predicted_path:
                pts = np.array(ped.predicted_path, dtype=np.int32)
                for j in range(1, len(pts)):
                    alpha = 1.0 - j / len(pts)
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), faded, 1)

        # HUD
        cv2.putText(frame, f"Frame {result.frame_id} | "
                    f"Peds: {result.num_pedestrians} | "
                    f"{result.processing_time_ms:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    tim.process_video(video_path, max_frames=max_frames, callback=draw_callback)
    writer.release()
    print(f"  Wrote {max_frames} frames to {output_path}")


def main():
    from embed_traffic.data.loader import UnifiedDataLoader
    from embed_traffic.paths import OUTPUTS_DIR

    loader = UnifiedDataLoader()
    out_dir = OUTPUTS_DIR / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 6: TIM Integration Demo ===\n")

    # Demo on JAAD video
    print("--- JAAD demo ---")
    jaad_video = str(loader.jaad_clips_dir / "video_0297.mp4")
    generate_tim_demo(jaad_video, str(out_dir / "demo_tim_jaad.mp4"), max_frames=200)

    # Demo on PIE video (with close pedestrians)
    print("\n--- PIE demo ---")
    pie_video = str(loader.pie_clips_dir / "set03" / "video_0015.mp4")
    generate_tim_demo(pie_video, str(out_dir / "demo_tim_pie.mp4"), max_frames=200)

    # Quick latency benchmark
    print("\n--- Latency benchmark ---")
    tim = TIM()
    cap = cv2.VideoCapture(jaad_video)
    times = []
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        result = tim.process_frame(frame, i)
        times.append(result.processing_time_ms)
    cap.release()

    print(f"  Per-frame latency: mean={np.mean(times):.1f}ms, "
          f"p50={np.percentile(times, 50):.1f}ms, "
          f"p95={np.percentile(times, 95):.1f}ms")
    print(f"  Throughput: {1000 / np.mean(times):.1f} fps")

    print("\nTIM pipeline complete!")


if __name__ == "__main__":
    main()
