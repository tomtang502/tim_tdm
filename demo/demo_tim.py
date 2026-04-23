"""End-to-end demo of the TIM flow on a single video.

Steps:
  1. Read the first 8 frames of the video and auto-calibrate the camera via
     a monocular depth model (Depth-Anything-V2-Metric-Outdoor-Large).
     Save the 8 predicted depth maps (colorized PNGs) for inspection.
  2. Process the entire video with TIM + the fresh calibration.
  3. Write three artifacts to outputs/demo/tim/:
       a) <stem>.mp4             — SIDE-BY-SIDE video: left panel is the
                                    original video with bbox/ID/m-s overlay;
                                    right panel is the animated top-down
                                    (bird's-eye) view with camera + pedestrians
                                    in world coordinates. Both panels are in
                                    lockstep (same frame time).
       b) <stem>.jsonl           — one TIMFrameOutput per line (JSON).
       c) <stem>_calibration/
             depth_00.png ...    — 8 colormapped depth predictions used for
                                    calibration.

Run:
    source scripts/_common.sh
    python demo/demo_tim.py                              # default JAAD / PIE clip
    python demo/demo_tim.py --video path/to/clip.mp4
    python demo/demo_tim.py --video clip.mp4 --n-cal-frames 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from collections import defaultdict

from embed_traffic.calibration import calibrate
from embed_traffic.calibration.depth import DEFAULT_DEPTH_MODEL
from embed_traffic.inference import TIM, TIMFrameOutput
from embed_traffic.inference.demo import overlay_frame
from embed_traffic.inference.topdown import (
    DEFAULT_XRANGE,
    DEFAULT_ZRANGE,
    render_topdown_frame,
)
from embed_traffic.paths import DATA_DIR, OUTPUTS_DIR, REPO_ROOT

DEMO_OUT_DIR = OUTPUTS_DIR / "demo" / "tim"
CONFIGS_DIR = REPO_ROOT / "configs" / "cameras"

# All existing candidates are processed (one side-by-side mp4 per video).
# video_0001 is 20s and essentially stationary (mean frame-to-frame pixel diff
# ≈ 2.5), matching the pole-mounted street-camera deployment.
DEFAULT_VIDEO_CANDIDATES = [
    DATA_DIR / "JAAD_clips" / "video_0001.mp4",   # ~20 s, static dashcam
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pick_default_videos() -> list[Path]:
    """Return all default candidates that exist on disk."""
    found = [v for v in DEFAULT_VIDEO_CANDIDATES if v.exists()]
    if not found:
        raise FileNotFoundError(
            "No default demo video found. Run `bash scripts/set_up_data.sh` "
            "or pass --video explicitly."
        )
    return found


def read_first_n_frames(video_path: Path, n: int) -> list[np.ndarray]:
    """Read the first `n` BGR frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return frames


def summarize(outputs: list[TIMFrameOutput]) -> None:
    """Print a concise summary of an inference run."""
    if not outputs:
        print("  (no frames processed)")
        return
    latencies = np.asarray([o.processing_time_ms for o in outputs])
    total_dets = sum(o.num_pedestrians for o in outputs)
    unique_ids = {p.ped_id for o in outputs for p in o.pedestrians}
    world_speed = sum(
        1 for o in outputs for p in o.pedestrians if p.speed_m_s is not None
    )
    crossing = sum(
        1 for o in outputs for p in o.pedestrians if p.crossing_intent == "crossing"
    )
    print(
        f"  frames:        {len(outputs)} "
        f"(0.0–{outputs[-1].frame_time_s:.2f}s video time)"
    )
    print(
        f"  latency:       mean={latencies.mean():.1f}ms "
        f"p95={np.percentile(latencies, 95):.1f}ms "
        f"→ {1000.0 / latencies.mean():.1f} fps"
    )
    print(
        f"  detections:    {total_dets}  "
        f"unique_ids={len(unique_ids)}  "
        f"w/ world speed={world_speed}  "
        f"crossing_flag={crossing}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

def _save_depth_pngs(frames: list[np.ndarray], depth_maps: list[np.ndarray],
                      out_dir: Path) -> None:
    """Save side-by-side (RGB | depth) colormapped PNGs, plus a summary of ranges."""
    out_dir.mkdir(parents=True, exist_ok=True)
    global_min = min(float(d.min()) for d in depth_maps)
    global_max = max(float(d.max()) for d in depth_maps)
    span = max(global_max - global_min, 1e-6)

    for i, (frame, d) in enumerate(zip(frames, depth_maps)):
        # Normalize using the global depth range so the colormap is comparable
        norm = np.clip((d - global_min) / span, 0.0, 1.0)
        depth_u8 = (norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
        # Match heights so we can hstack the two
        h = min(frame.shape[0], depth_color.shape[0])
        combo = np.concatenate(
            [frame[:h, :frame.shape[1]], depth_color[:h, :depth_color.shape[1]]],
            axis=1,
        )
        # Annotate depth range on the depth panel
        x0 = frame.shape[1] + 10
        cv2.putText(combo, f"depth_{i:02d}  range {global_min:.1f}-{global_max:.1f} m",
                    (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"depth_{i:02d}.png"), combo)
    print(f"  ↳ saved {len(depth_maps)} depth panels → {out_dir}")
    print(f"    (global range: {global_min:.2f}–{global_max:.2f} m)")


def do_calibrate(video_path: Path, n_frames: int, camera_id: str, depth_model: str,
                 hfov_deg: float, depth_out_dir: Path):
    """Read the first N frames of the video and calibrate from them.

    Also saves colormapped depth predictions to `depth_out_dir`.
    """
    print(f"\n[1/3] Calibrating camera from first {n_frames} frames of {video_path.name}")
    t0 = time.perf_counter()
    frames = read_first_n_frames(video_path, n_frames)
    print(f"  loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")
    calib, depth_maps = calibrate(
        frames=frames,
        camera_id=camera_id,
        depth_model_id=depth_model,
        hfov_deg=hfov_deg,
        mask_dynamic=True,
        verbose=True,
        return_depth_maps=True,
    )
    out_json = CONFIGS_DIR / f"{camera_id}.json"
    calib.save(out_json)

    _save_depth_pngs(frames, depth_maps, depth_out_dir)

    dt = time.perf_counter() - t0
    print(f"  ↳ camera_height={calib.extrinsics.camera_height_m:.2f}m  "
          f"pitch={calib.extrinsics.pitch_deg:+.1f}°  "
          f"saved → {out_json}  ({dt:.1f}s total)")
    return calib


def do_inference(
    video_path: Path,
    calib_or_path,
    jsonl_out: Path,
    mp4_out: Path,
    hfov_deg: float,
):
    """Process the entire video with TIM+calibration, writing JSONL + a
    side-by-side mp4 (overlay | top-down) both synchronized per frame."""
    print(f"\n[2/3] Running TIM on {video_path.name}")

    # Probe video for writer properties
    cap_probe = cv2.VideoCapture(str(video_path))
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()
    print(f"  video: {w}x{h} @ {fps:.1f}fps  ({total} frames)")

    tim = TIM(camera_calibration=calib_or_path)
    print(f"  detector:    {tim.detector_path}")
    print(f"  intent:      {tim.intent_path} (loaded={tim.has_intent_model})")
    print(f"  calibration: {tim.calibration.camera_id}  "
          f"height={tim.calibration.extrinsics.camera_height_m:.2f}m")

    # Side-by-side layout. Top-down panel height matches overlay height so we
    # can hstack without resizing. Top-down width chosen for ~8:9 aspect ratio
    # (matches the default (800, 900) canvas), rounded to even pixels.
    topdown_h = h
    topdown_w = int(round(h * 8 / 9))
    topdown_w -= topdown_w % 2
    combined_w = w + topdown_w
    combined_h = h
    td_canvas_size = (topdown_w, topdown_h)
    print(f"  panels: overlay {w}x{h}  |  top-down {topdown_w}x{topdown_h}  "
          f"→ combined {combined_w}x{combined_h}")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    mp4_out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_out), fourcc, fps, (combined_w, combined_h))
    outputs: list[TIMFrameOutput] = []
    td_histories: dict[int, list[tuple[float, float]]] = defaultdict(list)

    jsonl_fh = jsonl_out.open("w")

    def cb(frame: np.ndarray, out: TIMFrameOutput) -> None:
        outputs.append(out)
        jsonl_fh.write(json.dumps(asdict(out)) + "\n")

        overlay = overlay_frame(frame, out, tim)   # shape (h, w, 3)
        topdown = render_topdown_frame(
            out,
            td_histories,
            canvas_size=td_canvas_size,
            xrange=DEFAULT_XRANGE,
            zrange=DEFAULT_ZRANGE,
            hfov_deg=hfov_deg,
        )                                           # shape (topdown_h, topdown_w, 3)
        combined = np.hstack([overlay, topdown])   # shape (h, combined_w, 3)
        writer.write(combined)

    try:
        tim.process_video(str(video_path), max_frames=None, callback=cb)
    finally:
        jsonl_fh.close()
        writer.release()

    print(f"  ↳ JSONL:            {jsonl_out}")
    print(f"  ↳ side-by-side mp4: {mp4_out}")
    return outputs, fps


def do_summarize(outputs: list[TIMFrameOutput]) -> None:
    print("\n[3/3] Summary")
    summarize(outputs)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    video_path: Path,
    args: argparse.Namespace,
    index: int | None = None,
    total: int | None = None,
) -> None:
    """Run calibrate → inference → summary on a single video."""
    video_stem = video_path.stem
    camera_id = args.camera_id or f"demo_{video_stem}"
    depth_out_dir = DEMO_OUT_DIR / f"{video_stem}_calibration"

    header = "=" * 72
    tag = f" [{index}/{total}]" if index and total else ""
    print(f"\n{header}\n{tag} {video_path.name}\n{header}")
    print(f"Video:      {video_path}")
    print(f"Camera id:  {camera_id}")
    print(f"Output dir: {DEMO_OUT_DIR}")

    # (1) Calibrate (saves 8 colormapped depth panels alongside)
    calib = do_calibrate(
        video_path=video_path,
        n_frames=args.n_cal_frames,
        camera_id=camera_id,
        depth_model=args.depth_model,
        hfov_deg=args.hfov_deg,
        depth_out_dir=depth_out_dir,
    )

    # (2) Run TIM on the full video → JSONL + side-by-side mp4 (overlay | top-down)
    jsonl_out = DEMO_OUT_DIR / f"{video_stem}.jsonl"
    mp4_out = DEMO_OUT_DIR / f"{video_stem}.mp4"
    outputs, _ = do_inference(
        video_path, calib, jsonl_out, mp4_out, hfov_deg=args.hfov_deg
    )

    # (3) Report
    do_summarize(outputs)

    print("\nArtifacts:")
    print(f"  side-by-side mp4: {mp4_out}")
    print(f"  jsonl output:     {jsonl_out}")
    print(f"  depth panels:     {depth_out_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="TIM flow demo with auto-calibration.")
    parser.add_argument("--video", type=Path, default=None, action="append",
                        help="Input video (repeatable). Defaults to all bundled samples.")
    parser.add_argument("--n-cal-frames", type=int, default=8,
                        help="Frames used for one-shot calibration (default: 8).")
    parser.add_argument("--camera-id", type=str, default=None,
                        help="Camera id for configs/cameras/<id>.json "
                             "(default: demo_<video_stem>). Only honored when a "
                             "single --video is passed.")
    parser.add_argument("--depth-model", type=str, default=DEFAULT_DEPTH_MODEL,
                        help="HF depth model for calibration.")
    parser.add_argument("--hfov-deg", type=float, default=60.0,
                        help="Horizontal FoV assumption (degrees).")
    args = parser.parse_args()

    videos = args.video if args.video else pick_default_videos()
    if args.camera_id and len(videos) > 1:
        print(
            "WARNING: --camera-id was provided but multiple videos were given; "
            "it will be reused for each and the saved calibration will be "
            "overwritten. Omit --camera-id to get one per video.",
            file=sys.stderr,
        )

    for i, v in enumerate(videos, start=1):
        run_one(v, args, index=i, total=len(videos))

    print(f"\n✓ Done. {len(videos)} video(s) processed → {DEMO_OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
