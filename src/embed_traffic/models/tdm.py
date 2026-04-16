"""TDM (Traffic Decision Model).

Rule-based baseline using Time-To-Collision (TTC).

Decision logic:
  - TTC < 2s  → STOP
  - 2s <= TTC < 5s → SLOW DOWN
  - TTC >= 5s or no pedestrian in path → NO-OP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from embed_traffic.models.tim import PedestrianInfo, TIMOutput


@dataclass
class TDMOutput:
    """Output from the Traffic Decision Model."""
    decision: str                    # "STOP", "SLOW_DOWN", or "NO_OP"
    ttc: Optional[float] = None      # time-to-collision in seconds (None if no risk)
    risk_ped_id: Optional[int] = None  # pedestrian ID causing the decision
    risk_level: float = 0.0          # 0.0 (safe) to 1.0 (imminent collision)
    reason: str = ""                 # human-readable explanation


class TDM:
    """
    Traffic Decision Model — rule-based baseline.
    Takes TIM output + vehicle info and decides: STOP, SLOW_DOWN, or NO_OP.
    """

    def __init__(self,
                 ttc_stop=2.0,
                 ttc_slow=5.0,
                 vehicle_width_px=200,
                 crossing_weight=2.0,
                 min_ped_height=30):
        """
        Args:
            ttc_stop: TTC threshold for STOP (seconds)
            ttc_slow: TTC threshold for SLOW_DOWN (seconds)
            vehicle_width_px: assumed vehicle path width in pixels
            crossing_weight: multiply risk for pedestrians with crossing intent
            min_ped_height: ignore pedestrians smaller than this (too far away)
        """
        self.ttc_stop = ttc_stop
        self.ttc_slow = ttc_slow
        self.vehicle_width_px = vehicle_width_px
        self.crossing_weight = crossing_weight
        self.min_ped_height = min_ped_height

    def compute_ttc(self, ped: PedestrianInfo, vehicle_speed_px_s: float,
                    vehicle_pos: tuple = None, frame_width: int = 1920,
                    frame_height: int = 1080) -> Optional[float]:
        """
        Compute Time-To-Collision between vehicle and pedestrian.

        For dashcam view: vehicle moves forward (toward top of frame),
            TTC = vertical distance / closing speed

        For traffic light view: uses predicted trajectory intersection
            with vehicle path.

        Returns TTC in seconds, or None if no collision risk.
        """
        if not ped.center or not ped.predicted_path:
            return None

        ped_cx, ped_cy = ped.center
        ped_h = abs(ped.bbox[3] - ped.bbox[1])

        # Skip tiny/distant pedestrians
        if ped_h < self.min_ped_height:
            return None

        # Method 1: Simple approach — use pedestrian predicted trajectory
        # Check if any predicted future position enters the vehicle's path
        if vehicle_pos is None:
            # Assume vehicle is at bottom-center of frame (dashcam)
            vehicle_pos = (frame_width / 2, frame_height)

        vx, vy = vehicle_pos
        half_width = self.vehicle_width_px / 2

        # Check each predicted future position
        for i, (px, py) in enumerate(ped.predicted_path):
            # Is the pedestrian in the vehicle's lateral path?
            in_path = abs(px - vx) < half_width

            if in_path:
                # Time to reach this position (based on prediction steps)
                fps = 30.0  # default
                ttc = (i + 1) / fps

                # Adjust by pedestrian speed (faster ped = sooner collision)
                if ped.speed_px_s > 0:
                    dist = np.sqrt((px - ped_cx)**2 + (py - ped_cy)**2)
                    ttc_speed = dist / ped.speed_px_s
                    ttc = min(ttc, ttc_speed)

                return ttc

        # Method 2: Distance-based fallback
        if vehicle_speed_px_s > 0:
            # Vertical distance between pedestrian and vehicle
            dist_y = abs(vy - ped_cy)
            if dist_y > 0:
                # Simple TTC from closing speed
                ttc = dist_y / vehicle_speed_px_s
                # Only relevant if pedestrian is in lateral path
                if abs(ped_cx - vx) < half_width * 2:
                    return ttc

        return None

    def decide(self, tim_output: TIMOutput,
               vehicle_speed_px_s: float = 0.0,
               vehicle_pos: tuple = None,
               frame_width: int = 1920,
               frame_height: int = 1080) -> TDMOutput:
        """
        Make a driving decision based on TIM output.

        Args:
            tim_output: output from TIM pipeline
            vehicle_speed_px_s: vehicle speed in pixels/sec
            vehicle_pos: (x, y) vehicle position in frame coords
            frame_width, frame_height: frame dimensions

        Returns:
            TDMOutput with decision
        """
        if not tim_output.pedestrians:
            return TDMOutput(
                decision="NO_OP",
                risk_level=0.0,
                reason="No pedestrians detected",
            )

        min_ttc = float("inf")
        risk_ped = None
        max_risk = 0.0

        for ped in tim_output.pedestrians:
            ttc = self.compute_ttc(
                ped, vehicle_speed_px_s, vehicle_pos,
                frame_width, frame_height
            )

            if ttc is None:
                continue

            # Adjust risk based on crossing intent
            risk = 1.0 / max(ttc, 0.1)
            if ped.crossing_intent == "crossing":
                risk *= self.crossing_weight
            if ped.crossing_prob is not None:
                risk *= (0.5 + ped.crossing_prob)  # scale by probability

            if ttc < min_ttc:
                min_ttc = ttc
                risk_ped = ped
                max_risk = min(risk, 1.0)

        # Decision based on minimum TTC
        if min_ttc <= self.ttc_stop:
            decision = "STOP"
            reason = f"Pedestrian {risk_ped.ped_id} TTC={min_ttc:.1f}s"
            if risk_ped.crossing_intent == "crossing":
                reason += " (crossing)"
        elif min_ttc <= self.ttc_slow:
            decision = "SLOW_DOWN"
            reason = f"Pedestrian {risk_ped.ped_id} TTC={min_ttc:.1f}s"
            if risk_ped.crossing_intent == "crossing":
                reason += " (crossing)"
        else:
            decision = "NO_OP"
            reason = f"Min TTC={min_ttc:.1f}s (safe)"
            max_risk = max(0.0, 1.0 - min_ttc / 10.0)

        return TDMOutput(
            decision=decision,
            ttc=min_ttc if min_ttc < float("inf") else None,
            risk_ped_id=risk_ped.ped_id if risk_ped else None,
            risk_level=max_risk,
            reason=reason,
        )


def evaluate_on_pie():
    """
    Evaluate TDM on PIE dataset where we have vehicle OBD speed.
    Compare decisions against actual driver behavior.
    """
    from embed_traffic.data.loader import UnifiedDataLoader
    from embed_traffic.models.tim import TIM
    import cv2
    import time

    print("--- Evaluating TDM on PIE (with OBD speed) ---")
    loader = UnifiedDataLoader()
    pie_test = loader.get_pie_samples(split="test")

    # Use dashcam model for PIE
    tim = TIM(view="dashcam")
    tdm = TDM()

    # Pick a video with crossing events
    from collections import defaultdict
    video_crossing = defaultdict(int)
    video_speed = defaultdict(list)
    for s in pie_test:
        if s.crossing_intent == 1:
            video_crossing[s.video_id] += 1
        if s.vehicle_speed is not None:
            video_speed[s.video_id].append(s.vehicle_speed)

    # Pick top video with crossing events that has speed data
    best_vid = None
    for vid, count in sorted(video_crossing.items(), key=lambda x: x[1], reverse=True):
        if vid in video_speed and len(video_speed[vid]) > 100:
            best_vid = vid
            break

    if best_vid is None:
        print("  No suitable PIE video found")
        return

    print(f"  Using {best_vid} ({video_crossing[best_vid]} crossing annotations)")

    # Get avg speed for the video (convert km/h to rough px/s estimate)
    avg_speed_kmh = np.mean(video_speed[best_vid])
    # Rough conversion: 1 km/h ≈ 5 px/s at typical dashcam distance
    avg_speed_px_s = avg_speed_kmh * 5.0
    print(f"  Avg vehicle speed: {avg_speed_kmh:.1f} km/h (~{avg_speed_px_s:.0f} px/s)")

    # Build GT per-frame data
    gt_frames = defaultdict(list)
    for s in pie_test:
        if s.video_id == best_vid:
            gt_frames[s.frame_id].append(s)

    # Find start frame with annotations
    annotated_frames = sorted(gt_frames.keys())
    start_frame = annotated_frames[0]

    # Process video
    parts = best_vid.split("/")
    video_path = str(loader.pie_clips_dir / parts[0] / f"{parts[1]}.mp4")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    decisions = {"STOP": 0, "SLOW_DOWN": 0, "NO_OP": 0}
    gt_crossing_frames = 0
    correct_alerts = 0
    total_frames = 0
    max_frames = 300

    for fid in range(start_frame, start_frame + max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        tim_out = tim.process_frame(frame, fid)

        # Get vehicle speed for this frame
        gt = gt_frames.get(fid, [])
        frame_speed = avg_speed_px_s
        for s in gt:
            if s.vehicle_speed is not None:
                frame_speed = s.vehicle_speed * 5.0
                break

        tdm_out = tdm.decide(tim_out, vehicle_speed_px_s=frame_speed)
        decisions[tdm_out.decision] += 1

        # Check if GT has crossing pedestrians
        has_crossing = any(s.crossing_intent == 1 for s in gt)
        if has_crossing:
            gt_crossing_frames += 1
            if tdm_out.decision in ["STOP", "SLOW_DOWN"]:
                correct_alerts += 1

        total_frames += 1

    cap.release()

    print(f"\n  Results over {total_frames} frames:")
    print(f"  Decisions: STOP={decisions['STOP']}, SLOW_DOWN={decisions['SLOW_DOWN']}, NO_OP={decisions['NO_OP']}")
    print(f"  GT crossing frames: {gt_crossing_frames}")
    if gt_crossing_frames > 0:
        print(f"  Alert rate on crossing: {correct_alerts}/{gt_crossing_frames} "
              f"({100*correct_alerts/gt_crossing_frames:.1f}%)")

    return decisions


def generate_tdm_demo(video_path, output_path, view="dashcam",
                      vehicle_speed_px_s=100.0, max_frames=200):
    """Generate a demo video with TDM decision overlay."""
    from embed_traffic.models.tim import TIM
    import cv2

    tim = TIM(view=view)
    tdm = TDM()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    np.random.seed(42)
    colors = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(50, 255, (200, 3))]

    DECISION_COLORS = {
        "STOP": (0, 0, 255),
        "SLOW_DOWN": (0, 165, 255),
        "NO_OP": (0, 255, 0),
    }

    for fid in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        tim_out = tim.process_frame(frame, fid)
        tdm_out = tdm.decide(tim_out, vehicle_speed_px_s=vehicle_speed_px_s,
                             frame_width=w, frame_height=h)

        # Draw pedestrians
        for ped in tim_out.pedestrians:
            color = colors[ped.ped_id % len(colors)]
            x1, y1, x2, y2 = [int(v) for v in ped.bbox]

            # Highlight risk pedestrian
            if ped.ped_id == tdm_out.risk_ped_id:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3),
                              DECISION_COLORS[tdm_out.decision], 4)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{ped.ped_id}"
            if ped.crossing_intent:
                label += f" {'X' if ped.crossing_intent == 'crossing' else 'OK'}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw predicted path
            if ped.predicted_path:
                pts = np.array(ped.predicted_path[:15], dtype=np.int32)
                for j in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), color, 1)

        # Decision banner
        dec_color = DECISION_COLORS[tdm_out.decision]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"TDM: {tdm_out.decision}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, dec_color, 3)

        if tdm_out.ttc is not None:
            cv2.putText(frame, f"TTC: {tdm_out.ttc:.1f}s", (350, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, tdm_out.reason, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Risk bar
        bar_w = int(tdm_out.risk_level * 300)
        cv2.rectangle(frame, (w - 320, 10), (w - 320 + bar_w, 30), dec_color, -1)
        cv2.rectangle(frame, (w - 320, 10), (w - 20, 30), (200, 200, 200), 1)
        cv2.putText(frame, "Risk", (w - 320, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"  Wrote {max_frames} frames to {output_path}")


def main():
    from embed_traffic.data.loader import UnifiedDataLoader
    from embed_traffic.paths import OUTPUTS_DIR

    print("=== TDM (Traffic Decision Model) ===\n")

    # Step 1: Evaluate rule-based TDM on PIE
    print("--- Step 1: Rule-based TDM evaluation ---")
    evaluate_on_pie()

    # Generate demo videos
    loader = UnifiedDataLoader()
    out_dir = OUTPUTS_DIR / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating TDM demo (dashcam) ---")
    jaad_video = str(loader.jaad_clips_dir / "video_0297.mp4")
    generate_tdm_demo(jaad_video, str(out_dir / "demo_tdm_dashcam.mp4"),
                      view="dashcam", vehicle_speed_px_s=100.0)

    print("\n--- Generating TDM demo (traffic light) ---")
    # Use an Intersection-Flow image sequence isn't video, so use PIE with traffic light model
    pie_video = str(loader.pie_clips_dir / "set03" / "video_0015.mp4")
    generate_tdm_demo(pie_video, str(out_dir / "demo_tdm_traffic_light.mp4"),
                      view="traffic_light", vehicle_speed_px_s=80.0)

    print("\nTDM complete!")


if __name__ == "__main__":
    main()
