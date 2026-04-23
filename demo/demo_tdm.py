"""End-to-end demo of the TDM flow on a single video.

Pipeline:
  1. Calibrate the camera from the first N frames (depth model → ground plane).
  2. Run TIM on the full video to get per-frame pedestrian world-space info.
  3. Simulate a reasonable ego-vehicle trajectory (see simulator scenarios).
  4. Feed (TIM output, CarState) into TDM to get per-frame alert decisions.
  5. Write a side-by-side mp4 (overlay with alert banner | top-down with car)
     plus a JSONL of TIM + TDM records.

Default outputs (per video): outputs/demo/tdm/<stem>.{mp4,jsonl}  plus
8 calibration depth panels under <stem>_calibration/.

Run:
    source scripts/_common.sh
    python demo/demo_tdm.py                                  # both default clips
    python demo/demo_tdm.py --video a.mp4 --scenario approach_and_stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from embed_traffic.calibration import calibrate
from embed_traffic.calibration.depth import DEFAULT_DEPTH_MODEL
from embed_traffic.inference import TIM, TIMFrameOutput
from embed_traffic.inference.demo import overlay_frame
from embed_traffic.paths import DATA_DIR, OUTPUTS_DIR, REPO_ROOT
from embed_traffic.tdm import TDM, AlertLevel, CarState, TDMOutput
from embed_traffic.tdm.demo import draw_alert_banner, render_topdown_frame_with_car
from embed_traffic.tdm.simulator import SCENARIOS, make_scenario

DEMO_OUT_DIR = OUTPUTS_DIR / "demo" / "tdm"
CONFIGS_DIR = REPO_ROOT / "configs" / "cameras"

# JAAD video_0001 is 20s long and the dashcam is essentially stationary
# (mean frame-to-frame pixel diff ≈ 2.5), making it a great stand-in for a
# pole-mounted street camera watching an intersection.
DEFAULT_VIDEO_CANDIDATES = [
    DATA_DIR / "JAAD_clips" / "video_0001.mp4",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared structure with demo_tim.py)
# ─────────────────────────────────────────────────────────────────────────────

def pick_default_videos() -> list[Path]:
    found = [v for v in DEFAULT_VIDEO_CANDIDATES if v.exists()]
    if not found:
        raise FileNotFoundError(
            "No default demo video found. Run `bash scripts/set_up_data.sh` "
            "or pass --video explicitly."
        )
    return found


def read_first_n_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        for _ in range(n):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    return frames


def save_depth_pngs(frames: list[np.ndarray], depth_maps: list[np.ndarray],
                     out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dmin = min(float(d.min()) for d in depth_maps)
    dmax = max(float(d.max()) for d in depth_maps)
    span = max(dmax - dmin, 1e-6)
    for i, (f, d) in enumerate(zip(frames, depth_maps)):
        norm = np.clip((d - dmin) / span, 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
        h = min(f.shape[0], depth_color.shape[0])
        combo = np.concatenate([f[:h], depth_color[:h]], axis=1)
        cv2.putText(
            combo, f"depth_{i:02d} range {dmin:.1f}-{dmax:.1f}m",
            (f.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2,
        )
        cv2.imwrite(str(out_dir / f"depth_{i:02d}.png"), combo)


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

def do_calibrate(video: Path, n_frames: int, camera_id: str,
                 depth_model: str, hfov_deg: float, depth_out: Path):
    print(f"\n[1/3] Calibrating from first {n_frames} frames of {video.name}")
    t0 = time.perf_counter()
    frames = read_first_n_frames(video, n_frames)
    calib, depth_maps = calibrate(
        frames=frames, camera_id=camera_id,
        depth_model_id=depth_model, hfov_deg=hfov_deg,
        mask_dynamic=True, verbose=True, return_depth_maps=True,
    )
    out_json = CONFIGS_DIR / f"{camera_id}.json"
    calib.save(out_json)
    save_depth_pngs(frames, depth_maps, depth_out)
    print(
        f"  ↳ camera_height={calib.extrinsics.camera_height_m:.2f}m  "
        f"pitch={calib.extrinsics.pitch_deg:+.1f}°  ({time.perf_counter()-t0:.1f}s)"
    )
    return calib


def do_inference(
    video: Path,
    calib,
    scenario_name: str,
    jsonl_out: Path,
    mp4_out: Path,
    hfov_deg: float,
) -> list[tuple[TIMFrameOutput, TDMOutput]]:
    print(f"\n[2/3] Running TIM + TDM on {video.name}  (scenario='{scenario_name}')")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  video: {w}x{h} @ {fps:.1f}fps  ({total} frames)")

    tim = TIM(camera_calibration=calib)
    tdm = TDM(require_world_space=False)   # skip peds without world coords
    car_source = make_scenario(scenario_name)

    # Side-by-side layout — same dims as demo_tim.py for visual parity
    topdown_h = h
    topdown_w = (int(h * 8 / 9) // 2) * 2
    combined_w = w + topdown_w
    combined_h = h
    td_canvas = (topdown_w, topdown_h)
    print(f"  panels: overlay {w}x{h}  |  top-down {topdown_w}x{topdown_h}  "
          f"→ combined {combined_w}x{combined_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_out), fourcc, fps, (combined_w, combined_h))

    td_histories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    car_history: list[tuple[float, float]] = []

    joint: list[tuple[TIMFrameOutput, TDMOutput]] = []
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_fh = jsonl_out.open("w")

    def cb(frame: np.ndarray, tim_out: TIMFrameOutput) -> None:
        car = car_source(tim_out.frame_time_s)
        tdm_out = tdm.decide(tim_out, car)
        joint.append((tim_out, tdm_out))
        jsonl_fh.write(json.dumps({
            "tim": asdict(tim_out),
            "tdm": {**asdict(tdm_out), "alert": tdm_out.alert.value},
        }) + "\n")

        overlay = overlay_frame(frame, tim_out, tim)
        draw_alert_banner(overlay, tdm_out)
        topdown = render_topdown_frame_with_car(
            tim_out, tdm_out, td_histories,
            canvas_size=td_canvas, hfov_deg=hfov_deg,
            car_history=car_history,
        )
        writer.write(np.hstack([overlay, topdown]))

    try:
        tim.process_video(str(video), max_frames=None, callback=cb)
    finally:
        jsonl_fh.close()
        writer.release()

    print(f"  ↳ JSONL:            {jsonl_out}")
    print(f"  ↳ side-by-side mp4: {mp4_out}")
    return joint


def summarize(joint: list[tuple[TIMFrameOutput, TDMOutput]]) -> None:
    print("\n[3/3] Summary")
    if not joint:
        print("  (no frames)")
        return
    tim_lats = np.array([t.processing_time_ms for t, _ in joint])
    counts: dict[str, int] = defaultdict(int)
    for _, d in joint:
        counts[d.alert.value] += 1
    total = len(joint)
    dur = joint[-1][0].frame_time_s
    print(f"  frames:   {total}  (0.0–{dur:.2f}s video time)")
    print(f"  TIM lat:  mean={tim_lats.mean():.1f}ms  p95={np.percentile(tim_lats, 95):.1f}ms")
    print("  TDM alerts:")
    for lvl in ("none", "caution", "slow_down", "brake"):
        n = counts.get(lvl, 0)
        pct = 100.0 * n / total
        print(f"    {lvl:10s}: {n:4d}  ({pct:5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_one(video: Path, args: argparse.Namespace, idx: int, total: int) -> None:
    stem = video.stem
    camera_id = args.camera_id or f"demo_{stem}"
    depth_out = DEMO_OUT_DIR / f"{stem}_calibration"

    print(f"\n{'=' * 72}\n [{idx}/{total}] {video.name}\n{'=' * 72}")
    print(f"Video:      {video}")
    print(f"Camera id:  {camera_id}")
    print(f"Output dir: {DEMO_OUT_DIR}")
    print(f"Scenario:   {args.scenario}")

    calib = do_calibrate(
        video, args.n_cal_frames, camera_id,
        args.depth_model, args.hfov_deg, depth_out,
    )

    jsonl_out = DEMO_OUT_DIR / f"{stem}.jsonl"
    mp4_out = DEMO_OUT_DIR / f"{stem}.mp4"
    joint = do_inference(video, calib, args.scenario, jsonl_out, mp4_out, args.hfov_deg)

    summarize(joint)

    print("\nArtifacts:")
    print(f"  side-by-side mp4: {mp4_out}")
    print(f"  jsonl output:     {jsonl_out}")
    print(f"  depth panels:     {depth_out}/")


def main() -> int:
    p = argparse.ArgumentParser(description="TDM flow demo.")
    p.add_argument("--video", type=Path, default=None, action="append",
                   help="Input video (repeatable). Defaults to bundled samples.")
    p.add_argument("--scenario", type=str, default="approaching",
                   choices=sorted(SCENARIOS.keys()),
                   help="Synthetic car-state scenario to simulate.")
    p.add_argument("--n-cal-frames", type=int, default=8)
    p.add_argument("--camera-id", type=str, default=None,
                   help="Only honored with a single --video.")
    p.add_argument("--depth-model", type=str, default=DEFAULT_DEPTH_MODEL)
    p.add_argument("--hfov-deg", type=float, default=60.0)
    args = p.parse_args()

    videos = args.video if args.video else pick_default_videos()
    if args.camera_id and len(videos) > 1:
        print("WARNING: --camera-id with multiple videos will overwrite each other.",
              file=sys.stderr)

    for i, v in enumerate(videos, start=1):
        run_one(v, args, i, len(videos))

    print(f"\n✓ Done. {len(videos)} video(s) processed → {DEMO_OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
