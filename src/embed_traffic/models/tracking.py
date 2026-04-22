"""Pedestrian Tracking.

Integrates BoT-SORT/ByteTrack with fine-tuned YOLOv8 detections.
Evaluates tracking consistency on PIE and JAAD video sequences.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from embed_traffic.data.loader import UnifiedDataLoader
from embed_traffic.paths import detector_weights

DEFAULT_RUN_NAME = "ped_dashcam"


def default_model_path() -> str:
    return str(detector_weights(DEFAULT_RUN_NAME))


def track_video(model, video_path, start_frame=0, max_frames=None, tracker="botsort.yaml"):
    """
    Run tracking on a video file starting from start_frame.
    Returns dict of {frame_id: [{track_id, bbox, conf}, ...]}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total, start_frame + max_frames) if max_frames else total

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_tracks = {}
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            classes=[0],
            verbose=False,
            imgsz=1280,
        )

        frame_tracks = []
        for r in results:
            if r.boxes.id is not None:
                for box, track_id, conf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.id.cpu().numpy().astype(int),
                    r.boxes.conf.cpu().numpy(),
                ):
                    frame_tracks.append({
                        "track_id": int(track_id),
                        "bbox": box.tolist(),
                        "conf": float(conf),
                    })

        all_tracks[frame_idx] = frame_tracks
        frame_idx += 1

    cap.release()
    return all_tracks, fps


def compute_tracking_metrics(pred_tracks, gt_data, iou_threshold=0.5):
    """
    Compute tracking metrics: MOTA, IDF1, ID switches, track fragmentation.

    pred_tracks: dict {frame_id: [{track_id, bbox, conf}, ...]}
    gt_data: dict {frame_id: [{ped_id, bbox}, ...]}
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0

    # Map from gt_id -> last matched pred_id
    gt_to_pred = {}

    # Evaluate on all frames that have GT or predictions
    all_frames = sorted(set(list(pred_tracks.keys()) + list(gt_data.keys())))

    for frame_idx in all_frames:
        pred_frame = pred_tracks.get(frame_idx, [])
        gt_frame = gt_data.get(frame_idx, [])
        total_gt += len(gt_frame)

        if not gt_frame:
            total_fp += len(pred_frame)
            continue

        if not pred_frame:
            total_fn += len(gt_frame)
            continue

        # Compute IoU matrix
        gt_boxes = np.array([g["bbox"] for g in gt_frame])
        pred_boxes = np.array([p["bbox"] for p in pred_frame])

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gb in enumerate(gt_boxes):
            for j, pb in enumerate(pred_boxes):
                x1 = max(gb[0], pb[0])
                y1 = max(gb[1], pb[1])
                x2 = min(gb[2], pb[2])
                y2 = min(gb[3], pb[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
                area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                union = area_g + area_p - inter
                iou_matrix[i, j] = inter / union if union > 0 else 0

        # Greedy matching
        matched_gt = set()
        matched_pred = set()

        # Sort by IoU descending
        indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
        for gi, pi in zip(indices[0], indices[1]):
            if gi in matched_gt or pi in matched_pred:
                continue
            if iou_matrix[gi, pi] < iou_threshold:
                break

            matched_gt.add(gi)
            matched_pred.add(pi)
            total_tp += 1

            # Check ID switch
            gt_id = gt_frame[gi]["ped_id"]
            pred_id = pred_frame[pi]["track_id"]
            if gt_id in gt_to_pred:
                if gt_to_pred[gt_id] != pred_id:
                    total_id_switches += 1
            gt_to_pred[gt_id] = pred_id

        total_fp += len(pred_frame) - len(matched_pred)
        total_fn += len(gt_frame) - len(matched_gt)

    # MOTA = 1 - (FN + FP + ID_switches) / total_gt
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0.0

    # IDF1 = 2 * TP / (2 * TP + FP + FN)
    idf1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "ID_switches": total_id_switches,
        "total_gt": total_gt,
    }


def build_gt_by_frame(samples, video_id):
    """Build ground truth dict: {frame_id: [{ped_id, bbox}, ...]}"""
    gt = defaultdict(list)
    for s in samples:
        if s.video_id == video_id and s.occlusion < 2:
            gt[s.frame_id].append({"ped_id": s.ped_id, "bbox": s.bbox})
    return dict(gt)


def evaluate_tracking_on_dataset(model, loader, dataset, samples, tracker, num_videos=5, max_frames=300):
    """Evaluate tracking on a few videos from a dataset."""
    # Group by video
    videos = defaultdict(list)
    for s in samples:
        videos[s.video_id].append(s)

    # Pick videos with most annotations
    sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)[:num_videos]

    all_metrics = []
    for video_id, vid_samples in sorted_videos:
        if dataset == "pie":
            parts = video_id.split("/")
            video_path = str(loader.pie_dir / "PIE_clips" / parts[0] / f"{parts[1]}.mp4")
        else:
            video_path = str(loader.jaad_dir / "JAAD_clips" / f"{video_id}.mp4")

        if not Path(video_path).exists():
            print(f"  Skipping {video_id} (video not found)")
            continue

        gt_data = build_gt_by_frame(vid_samples, video_id)
        if not gt_data:
            print(f"  Skipping {video_id} (no GT in frame range)")
            continue

        # Start tracking from where annotations begin
        gt_frames = sorted(gt_data.keys())
        start_frame = gt_frames[0]

        print(f"  Tracking {video_id} (frames {start_frame}-{start_frame + max_frames})...")
        t0 = time.time()

        # Reset tracker state for new video
        model_fresh = YOLO(default_model_path())
        pred_tracks, fps = track_video(
            model_fresh, video_path, start_frame=start_frame, max_frames=max_frames, tracker=tracker
        )

        metrics = compute_tracking_metrics(pred_tracks, gt_data)
        elapsed = time.time() - t0
        track_fps = len(pred_tracks) / elapsed

        print(f"    MOTA={metrics['MOTA']:.3f} IDF1={metrics['IDF1']:.3f} "
              f"IDsw={metrics['ID_switches']} "
              f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} "
              f"({track_fps:.1f} fps)")

        all_metrics.append(metrics)

    # Average metrics
    if all_metrics:
        avg = {
            "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
            "IDF1": np.mean([m["IDF1"] for m in all_metrics]),
            "ID_switches": sum(m["ID_switches"] for m in all_metrics),
            "total_gt": sum(m["total_gt"] for m in all_metrics),
        }
        return avg
    return None


def generate_tracking_video(model, video_path, output_path, max_frames=150, tracker="botsort.yaml"):
    """Generate a visualization video with tracking overlays."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Color palette for track IDs
    np.random.seed(42)
    colors = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(50, 255, (200, 3))]

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, tracker=tracker, classes=[0], verbose=False, imgsz=1280
        )

        for r in results:
            if r.boxes.id is not None:
                for box, track_id, conf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.id.cpu().numpy().astype(int),
                    r.boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = colors[track_id % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Wrote {frame_idx} frames to {output_path}")


def main():
    model = YOLO(default_model_path())
    loader = UnifiedDataLoader()

    # --- Evaluate BoT-SORT on JAAD ---
    print("=== JAAD Tracking (BoT-SORT) ===")
    jaad_test = loader.get_jaad_samples(split="test")
    jaad_botsort = evaluate_tracking_on_dataset(
        model, loader, "jaad", jaad_test, tracker="botsort.yaml", num_videos=5, max_frames=300
    )
    if jaad_botsort:
        print(f"\n  Avg MOTA={jaad_botsort['MOTA']:.3f} IDF1={jaad_botsort['IDF1']:.3f} "
              f"IDsw={jaad_botsort['ID_switches']}")

    # --- Evaluate BoT-SORT on PIE ---
    print("\n=== PIE Tracking (BoT-SORT) ===")
    pie_test = loader.get_pie_samples(split="test")
    pie_botsort = evaluate_tracking_on_dataset(
        model, loader, "pie", pie_test, tracker="botsort.yaml", num_videos=5, max_frames=300
    )
    if pie_botsort:
        print(f"\n  Avg MOTA={pie_botsort['MOTA']:.3f} IDF1={pie_botsort['IDF1']:.3f} "
              f"IDsw={pie_botsort['ID_switches']}")

    # --- Evaluate ByteTrack on JAAD for comparison ---
    print("\n=== JAAD Tracking (ByteTrack) ===")
    jaad_bytetrack = evaluate_tracking_on_dataset(
        model, loader, "jaad", jaad_test, tracker="bytetrack.yaml", num_videos=5, max_frames=300
    )
    if jaad_bytetrack:
        print(f"\n  Avg MOTA={jaad_bytetrack['MOTA']:.3f} IDF1={jaad_bytetrack['IDF1']:.3f} "
              f"IDsw={jaad_bytetrack['ID_switches']}")

    # --- Generate sample tracking videos ---
    print("\n=== Generating tracking visualization ===")
    from embed_traffic.paths import OUTPUTS_DIR
    out_dir = OUTPUTS_DIR / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JAAD sample
    jaad_video = str(loader.jaad_clips_dir / "video_0297.mp4")
    model_viz = YOLO(default_model_path())
    generate_tracking_video(model_viz, jaad_video, str(out_dir / "sample_tracking_jaad.mp4"), max_frames=150)

    # PIE sample
    pie_video = str(loader.pie_clips_dir / "set01" / "video_0002.mp4")
    model_viz2 = YOLO(default_model_path())
    generate_tracking_video(model_viz2, pie_video, str(out_dir / "sample_tracking_pie.mp4"), max_frames=150)

    loader.release_all()
    print("\nDone!")


if __name__ == "__main__":
    main()
