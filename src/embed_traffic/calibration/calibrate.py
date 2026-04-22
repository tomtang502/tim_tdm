"""End-to-end camera calibration.

Pipeline:
  1. Grab N frames from a video (or a list of image paths).
  2. Run a monocular metric-depth model on each.
  3. Mask out dynamic regions using the TIM detector (pedestrians, vehicles).
  4. Backproject static pixels into 3D camera-frame points.
  5. RANSAC-fit a ground plane across the combined point cloud.
  6. Derive camera intrinsics (from field-of-view assumption or override) and
     ground-to-camera extrinsics.
  7. Serialize a `CameraCalibration` JSON to `configs/cameras/<camera_id>.json`.

The depth model is used only here — runtime TIM does not depend on it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from embed_traffic.calibration.depth import DEFAULT_DEPTH_MODEL, DepthEstimator
from embed_traffic.calibration.ground_plane import (
    depth_to_point_cloud,
    extrinsics_from_plane,
    fit_plane_ransac,
    heuristic_ground_mask,
)
from embed_traffic.calibration.schema import (
    CameraCalibration,
    GroundExtrinsics,
    Intrinsics,
)
from embed_traffic.paths import REPO_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames_from_video(video_path: str, n_frames: int) -> List[np.ndarray]:
    """Sample `n_frames` evenly-spaced BGR frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    idx = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    frames: List[np.ndarray] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return frames


def load_image_frames(paths: Iterable[str]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append(img)
    if not frames:
        raise RuntimeError("No input images given.")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic-region masking (via YOLO)
# ─────────────────────────────────────────────────────────────────────────────

_DYNAMIC_CLASSES_COCO = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck


def _build_dynamic_mask(
    frame: np.ndarray,
    yolo,
    dynamic_classes: list[int] = _DYNAMIC_CLASSES_COCO,
    pad_px: int = 10,
) -> np.ndarray:
    """Return a boolean mask where dynamic objects have been EXCLUDED (True = keep)."""
    H, W = frame.shape[:2]
    keep = np.ones((H, W), dtype=bool)

    results = yolo.predict(frame, classes=dynamic_classes, verbose=False)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1) - pad_px)
            y1 = max(0, int(y1) - pad_px)
            x2 = min(W, int(x2) + pad_px)
            y2 = min(H, int(y2) + pad_px)
            keep[y1:y2, x1:x2] = False
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration driver
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    frames: List[np.ndarray],
    camera_id: str,
    depth_model_id: str = DEFAULT_DEPTH_MODEL,
    hfov_deg: float = 60.0,
    intrinsics: Optional[Intrinsics] = None,
    ransac_iterations: int = 1000,
    inlier_threshold_m: float = 0.08,
    min_inliers: int = 500,
    mask_dynamic: bool = True,
    verbose: bool = True,
    return_depth_maps: bool = False,
) -> CameraCalibration | tuple[CameraCalibration, List[np.ndarray]]:
    """Run depth → RANSAC → extrinsics and return a CameraCalibration.

    Args:
        frames: list of BGR frames (uint8, HxWx3).
        camera_id: label for this camera (used in output filename).
        depth_model_id: HF model id; default is DA-v2 metric outdoor.
        hfov_deg: horizontal FoV assumption (used only if intrinsics is None).
        intrinsics: override intrinsics entirely.
        ransac_iterations, inlier_threshold_m, min_inliers: RANSAC params.
        mask_dynamic: if True, mask out pedestrians/vehicles using the TIM detector.
        verbose: print progress.
    """
    if not frames:
        raise ValueError("At least one frame is required.")

    H, W = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (H, W):
            raise ValueError(
                f"All frames must share resolution; got {f.shape[:2]} vs {(H, W)}."
            )

    # Intrinsics
    K = intrinsics if intrinsics is not None else Intrinsics.from_fov(W, H, hfov_deg)
    if verbose:
        print(f"[calibrate] frames:    {len(frames)}  resolution: {W}x{H}")
        print(f"[calibrate] intrinsics: fx={K.fx:.1f} fy={K.fy:.1f} "
              f"cx={K.cx:.1f} cy={K.cy:.1f}")
        print(f"[calibrate] depth model: {depth_model_id}")

    # ── Depth ──
    depther = DepthEstimator(depth_model_id)
    depth_maps: List[np.ndarray] = []
    for i, f in enumerate(frames):
        d = depther.predict(f)
        depth_maps.append(d)
        if verbose:
            print(
                f"[calibrate]  frame {i+1}/{len(frames)}: "
                f"depth min={d.min():.2f} max={d.max():.2f} mean={d.mean():.2f} m"
            )

    # ── Mask dynamic objects ──
    keep_masks: List[np.ndarray] = []
    if mask_dynamic:
        from ultralytics import YOLO
        from embed_traffic.paths import detector_weights

        weights = detector_weights("ped_dashcam")
        yolo = YOLO(str(weights)) if Path(weights).exists() else YOLO("yolov8n.pt")
        for f in frames:
            keep_masks.append(_build_dynamic_mask(f, yolo))
    else:
        keep_masks = [np.ones((H, W), dtype=bool) for _ in frames]

    # ── Combined point cloud (static, bottom-band pixels only) ──
    all_points: List[np.ndarray] = []
    for d, keep in zip(depth_maps, keep_masks):
        ground_mask = heuristic_ground_mask(d)
        mask = keep & ground_mask
        if mask.sum() < 100:
            continue
        # Subsample to keep memory bounded (~50k pts per frame max)
        pts = depth_to_point_cloud(d, K, valid_mask=mask)
        if len(pts) > 50_000:
            idx = np.random.default_rng(0).choice(len(pts), size=50_000, replace=False)
            pts = pts[idx]
        all_points.append(pts)
    if not all_points:
        raise RuntimeError("No valid ground points after masking — check FoV / inputs.")
    pts = np.concatenate(all_points, axis=0)
    if verbose:
        print(f"[calibrate] point cloud: {len(pts)} static points fed to RANSAC")

    # ── RANSAC plane fit ──
    n_hat, d_offset, inliers = fit_plane_ransac(
        pts,
        n_iterations=ransac_iterations,
        inlier_threshold_m=inlier_threshold_m,
        min_inliers=min_inliers,
    )
    if verbose:
        print(f"[calibrate] plane: n={tuple(round(float(x), 3) for x in n_hat)} "
              f"d={d_offset:.3f} inliers={len(inliers)}")

    # ── Derive extrinsics ──
    ext = extrinsics_from_plane(n_hat, d_offset)
    if verbose:
        print(f"[calibrate] camera_height ≈ {ext.camera_height_m:.2f} m,  "
              f"pitch ≈ {ext.pitch_deg:.1f}°,  roll ≈ {ext.roll_deg:.1f}°")

    # Sanity warning
    if not (1.0 < ext.camera_height_m < 8.0):
        print(
            f"[calibrate] WARNING: camera_height = {ext.camera_height_m:.2f} m is "
            "outside a reasonable pole-mount range (1–8 m). Check FoV / depth scale.",
            file=sys.stderr,
        )

    calib = CameraCalibration(
        camera_id=camera_id,
        image_size=(W, H),
        intrinsics=K,
        extrinsics=ext,
        depth_model=depth_model_id,
        source_frames=len(frames),
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=(
            f"Auto-calibrated with {len(frames)} frames via {depth_model_id}. "
            f"Dynamic objects masked: {mask_dynamic}. "
            f"HFOV assumption: {hfov_deg}° (unless intrinsics overridden)."
        ),
    )
    if return_depth_maps:
        return calib, depth_maps
    return calib


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _default_output_path(camera_id: str) -> Path:
    return REPO_ROOT / "configs" / "cameras" / f"{camera_id}.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m embed_traffic.calibration",
        description="Auto-calibrate a stationary camera via monocular depth.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to a video file.")
    src.add_argument("--images", nargs="+", help="List of image paths.")

    p.add_argument("--camera-id", type=str, required=True,
                   help="Camera identifier, used for the output filename.")
    p.add_argument("--n-frames", type=int, default=8,
                   help="How many frames to sample from --video. Default 8.")
    p.add_argument("--depth-model", type=str, default=DEFAULT_DEPTH_MODEL,
                   help="HF model id for depth estimation.")
    p.add_argument("--hfov-deg", type=float, default=60.0,
                   help="Horizontal field-of-view in degrees (used for intrinsics).")
    p.add_argument("--no-mask-dynamic", action="store_true",
                   help="Disable masking of people/vehicles via YOLO.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path. Default: configs/cameras/<camera_id>.json")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args(argv)

    if args.video:
        frames = sample_frames_from_video(args.video, args.n_frames)
    else:
        frames = load_image_frames(args.images)

    calib = calibrate(
        frames=frames,
        camera_id=args.camera_id,
        depth_model_id=args.depth_model,
        hfov_deg=args.hfov_deg,
        mask_dynamic=not args.no_mask_dynamic,
        verbose=not args.quiet,
    )

    out_path = Path(args.output) if args.output else _default_output_path(args.camera_id)
    calib.save(out_path)
    if not args.quiet:
        print(f"[calibrate] wrote → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
