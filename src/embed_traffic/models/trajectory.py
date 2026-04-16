"""Speed & Trajectory Estimation.

4.1 Compute per-pedestrian trajectory from tracked bounding boxes
4.2 Estimate walking speed from bbox displacement
4.3 Predict future trajectory via Kalman filter
4.4 Validate against JAAD ground truth
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pykalman import KalmanFilter
from ultralytics import YOLO

from embed_traffic.paths import detector_weights

# Default detector checkpoint for trajectory estimation. Override by passing
# model_path to TrajectoryEstimator or by exporting TIM_DETECTOR_RUN.
DEFAULT_RUN_NAME = "ped_dashcam"
MODEL_PATH = str(detector_weights(DEFAULT_RUN_NAME))


class PedestrianTrajectory:
    """Stores and processes trajectory for a single tracked pedestrian."""

    def __init__(self, track_id, fps=30.0):
        self.track_id = track_id
        self.fps = fps
        self.frames = []       # frame indices
        self.bboxes = []       # [x1, y1, x2, y2]
        self.centers = []      # [cx, cy]
        self.speeds = []       # px/s
        self._kf = None
        self._kf_states = []

    def add(self, frame_id, bbox):
        self.frames.append(frame_id)
        self.bboxes.append(bbox)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.centers.append([cx, cy])

        # Compute instantaneous speed
        if len(self.centers) >= 2:
            prev = self.centers[-2]
            dt = (self.frames[-1] - self.frames[-2]) / self.fps
            if dt > 0:
                dist = np.sqrt((cx - prev[0])**2 + (cy - prev[1])**2)
                self.speeds.append(dist / dt)
            else:
                self.speeds.append(0.0)
        else:
            self.speeds.append(0.0)

    @property
    def length(self):
        return len(self.frames)

    @property
    def avg_speed(self):
        return np.mean(self.speeds) if self.speeds else 0.0

    @property
    def last_center(self):
        return self.centers[-1] if self.centers else None

    @property
    def last_speed(self):
        return self.speeds[-1] if self.speeds else 0.0

    @property
    def bbox_height(self):
        """Average bbox height — proxy for distance to camera."""
        if not self.bboxes:
            return 0
        return np.mean([b[3] - b[1] for b in self.bboxes])

    def direction_vector(self, window=5):
        """Compute average movement direction over last N frames."""
        if len(self.centers) < 2:
            return np.array([0.0, 0.0])
        pts = np.array(self.centers[-window:])
        if len(pts) < 2:
            return np.array([0.0, 0.0])
        diff = pts[-1] - pts[0]
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 0 else np.array([0.0, 0.0])

    def fit_kalman(self):
        """Fit Kalman filter to observed trajectory for smoothing + prediction."""
        if len(self.centers) < 3:
            return

        observations = np.array(self.centers)
        dt = 1.0 / self.fps

        # State: [x, y, vx, vy]
        # Observation: [x, y]
        self._kf = KalmanFilter(
            transition_matrices=np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]),
            observation_matrices=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]),
            initial_state_mean=np.array([
                observations[0, 0], observations[0, 1], 0, 0
            ]),
            observation_covariance=np.eye(2) * 5.0,
            transition_covariance=np.eye(4) * 0.5,
            em_vars=["transition_covariance", "observation_covariance"],
        )

        # Fit with EM
        self._kf = self._kf.em(observations, n_iter=5)
        self._kf_states, _ = self._kf.smooth(observations)

    def predict_future(self, n_steps=30):
        """
        Predict future positions for n_steps frames.
        Returns list of [cx, cy] predicted positions.
        """
        if self._kf is None:
            self.fit_kalman()
        if self._kf is None:
            # Fallback: linear extrapolation
            return self._linear_predict(n_steps)

        # Start from last smoothed state
        state = self._kf_states[-1]
        predictions = []
        for _ in range(n_steps):
            state = self._kf.transition_matrices @ state
            predictions.append([state[0], state[1]])

        return predictions

    def _linear_predict(self, n_steps):
        """Simple linear extrapolation fallback."""
        if len(self.centers) < 2:
            return [self.centers[-1]] * n_steps if self.centers else [[0, 0]] * n_steps
        direction = self.direction_vector()
        speed_px_per_frame = self.last_speed / self.fps if self.fps > 0 else 0
        last = np.array(self.centers[-1])
        return [(last + direction * speed_px_per_frame * (i + 1)).tolist() for i in range(n_steps)]


class TrajectoryEstimator:
    """Runs YOLOv8 + ByteTrack and builds pedestrian trajectories."""

    def __init__(self, model_path=MODEL_PATH, tracker="bytetrack.yaml", fps=30.0):
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.fps = fps
        self.trajectories = {}  # track_id -> PedestrianTrajectory

    def reset(self):
        """Reset tracker state and trajectories."""
        self.model = YOLO(MODEL_PATH)
        self.trajectories = {}

    def process_frame(self, frame, frame_id):
        """
        Process a single frame: detect + track + update trajectories.
        Returns list of active track dicts.
        """
        results = self.model.track(
            frame, persist=True, tracker=self.tracker,
            classes=[0], verbose=False, imgsz=1280,
        )

        active_tracks = []
        for r in results:
            if r.boxes.id is not None:
                for box, track_id, conf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.id.cpu().numpy().astype(int),
                    r.boxes.conf.cpu().numpy(),
                ):
                    tid = int(track_id)
                    bbox = box.tolist()

                    if tid not in self.trajectories:
                        self.trajectories[tid] = PedestrianTrajectory(tid, self.fps)
                    self.trajectories[tid].add(frame_id, bbox)

                    traj = self.trajectories[tid]
                    active_tracks.append({
                        "track_id": tid,
                        "bbox": bbox,
                        "conf": float(conf),
                        "center": traj.last_center,
                        "speed_px_s": traj.last_speed,
                        "avg_speed_px_s": traj.avg_speed,
                        "direction": traj.direction_vector().tolist(),
                        "track_length": traj.length,
                    })

        return active_tracks

    def get_trajectory(self, track_id):
        return self.trajectories.get(track_id)

    def predict_all(self, n_steps=30, min_track_length=5):
        """Predict future positions for all active trajectories."""
        predictions = {}
        for tid, traj in self.trajectories.items():
            if traj.length >= min_track_length:
                traj.fit_kalman()
                predictions[tid] = traj.predict_future(n_steps)
        return predictions


def validate_on_jaad(max_videos=3, max_frames=200):
    """
    Validate trajectory estimation against JAAD ground truth.
    Compare predicted trajectories vs actual GT positions.
    """
    from embed_traffic.data.loader import UnifiedDataLoader

    loader = UnifiedDataLoader()
    jaad_test = loader.get_jaad_samples(split="test")

    # Group by video
    videos = defaultdict(list)
    for s in jaad_test:
        videos[s.video_id].append(s)

    # Pick videos with most annotations
    sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)[:max_videos]

    all_speed_errors = []
    all_pos_errors = []

    for video_id, vid_samples in sorted_videos:
        video_path = str(loader.jaad_clips_dir / f"{video_id}.mp4")
        if not Path(video_path).exists():
            continue

        print(f"\n  Validating on {video_id}...")

        # Build GT trajectories: {ped_id: {frame_id: [cx, cy]}}
        gt_trajs = defaultdict(dict)
        for s in vid_samples:
            if s.occlusion < 2:
                cx = (s.bbox[0] + s.bbox[2]) / 2
                cy = (s.bbox[1] + s.bbox[3]) / 2
                gt_trajs[s.ped_id][s.frame_id] = [cx, cy]

        # Run tracker
        estimator = TrajectoryEstimator()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for fid in range(min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            estimator.process_frame(frame, fid)

        cap.release()

        # Compare predicted speeds with GT speeds
        for tid, traj in estimator.trajectories.items():
            if traj.length < 10:
                continue

            # Find best matching GT pedestrian by IoU at first frame
            best_gt_id = None
            best_overlap = 0
            first_bbox = traj.bboxes[0]
            first_frame = traj.frames[0]

            for gt_id, gt_frames in gt_trajs.items():
                if first_frame not in gt_frames:
                    continue
                # Find GT bbox for this frame
                gt_sample = [s for s in vid_samples if s.ped_id == gt_id and s.frame_id == first_frame]
                if not gt_sample:
                    continue
                gb = gt_sample[0].bbox
                # IoU
                x1 = max(first_bbox[0], gb[0])
                y1 = max(first_bbox[1], gb[1])
                x2 = min(first_bbox[2], gb[2])
                y2 = min(first_bbox[3], gb[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (first_bbox[2] - first_bbox[0]) * (first_bbox[3] - first_bbox[1])
                a2 = (gb[2] - gb[0]) * (gb[3] - gb[1])
                iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
                if iou > best_overlap:
                    best_overlap = iou
                    best_gt_id = gt_id

            if best_gt_id is None or best_overlap < 0.3:
                continue

            gt_positions = gt_trajs[best_gt_id]

            # Compare center positions at shared frames
            for i, fid in enumerate(traj.frames):
                if fid in gt_positions:
                    pred_pos = traj.centers[i]
                    gt_pos = gt_positions[fid]
                    err = np.sqrt((pred_pos[0] - gt_pos[0])**2 + (pred_pos[1] - gt_pos[1])**2)
                    all_pos_errors.append(err)

            # Compare speeds
            gt_speeds = []
            gt_frame_list = sorted(gt_positions.keys())
            for j in range(1, len(gt_frame_list)):
                f_prev, f_curr = gt_frame_list[j-1], gt_frame_list[j]
                p_prev, p_curr = gt_positions[f_prev], gt_positions[f_curr]
                dt = (f_curr - f_prev) / fps
                if dt > 0:
                    d = np.sqrt((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2)
                    gt_speeds.append(d / dt)

            if gt_speeds and traj.speeds:
                pred_avg = traj.avg_speed
                gt_avg = np.mean(gt_speeds)
                all_speed_errors.append(abs(pred_avg - gt_avg))

    # Report
    print(f"\n=== Trajectory Validation Results ===")
    if all_pos_errors:
        print(f"  Position error (px): mean={np.mean(all_pos_errors):.1f}, "
              f"median={np.median(all_pos_errors):.1f}, p95={np.percentile(all_pos_errors, 95):.1f}")
    if all_speed_errors:
        print(f"  Speed error (px/s):  mean={np.mean(all_speed_errors):.1f}, "
              f"median={np.median(all_speed_errors):.1f}, p95={np.percentile(all_speed_errors, 95):.1f}")

    return all_pos_errors, all_speed_errors


def generate_trajectory_video(video_path, output_path, max_frames=200):
    """Generate visualization with trajectories and predictions."""
    estimator = TrajectoryEstimator()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    np.random.seed(42)
    colors = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(50, 255, (200, 3))]

    for fid in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        active = estimator.process_frame(frame, fid)

        for track in active:
            tid = track["track_id"]
            traj = estimator.get_trajectory(tid)
            color = colors[tid % len(colors)]
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with speed
            speed = track["speed_px_s"]
            label = f"ID:{tid} {speed:.0f}px/s"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Draw trajectory trail (last 30 centers)
            if traj and traj.length > 1:
                pts = np.array(traj.centers[-30:], dtype=np.int32)
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    thick = max(1, int(alpha * 3))
                    cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), color, thick)

            # Draw predicted future (if enough history)
            if traj and traj.length >= 10:
                future = traj.predict_future(n_steps=20)
                pts_future = np.array(future, dtype=np.int32)
                for j in range(1, len(pts_future)):
                    alpha = 1.0 - j / len(pts_future)
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, tuple(pts_future[j-1]), tuple(pts_future[j]), faded, 1)
                # Mark predicted endpoint
                cv2.circle(frame, tuple(pts_future[-1]), 4, color, -1)

        cv2.putText(frame, f"Frame {fid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"  Wrote {min(max_frames, fid+1)} frames to {output_path}")


def main():
    print("=== Step 4: Trajectory Estimation ===\n")

    # 4.4 Validate on JAAD
    print("--- Validating trajectory estimation on JAAD ---")
    pos_errors, speed_errors = validate_on_jaad(max_videos=3, max_frames=200)

    # Generate visualization
    print("\n--- Generating trajectory visualization ---")
    from embed_traffic.data.loader import UnifiedDataLoader
    loader = UnifiedDataLoader()

    jaad_video = str(loader.jaad_clips_dir / "video_0297.mp4")
    from embed_traffic.paths import OUTPUTS_DIR
    out_dir = OUTPUTS_DIR / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_trajectory_video(jaad_video, str(out_dir / "sample_trajectory_jaad.mp4"), max_frames=200)

    pie_video = str(loader.pie_clips_dir / "set03" / "video_0015.mp4")
    generate_trajectory_video(pie_video, str(out_dir / "sample_trajectory_pie.mp4"), max_frames=200)

    print("\nDone!")


if __name__ == "__main__":
    main()
