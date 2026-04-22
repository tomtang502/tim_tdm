"""CLI for TIM inference.

Examples:
  # Write JSONL (one TIMFrameOutput per line) for a whole video
  python -m embed_traffic.inference clip.mp4 --output out.jsonl

  # Also render an overlay video
  python -m embed_traffic.inference clip.mp4 --demo out.mp4 --max-frames 200

  # Override which detector checkpoint to use
  python -m embed_traffic.inference clip.mp4 \
      --detector-run-name ped_dashcam --intent-run-name intent_default
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embed_traffic.inference.demo import render_demo_video
from embed_traffic.inference.tim import (
    DEFAULT_DETECTOR_RUN,
    DEFAULT_INTENT_RUN,
    TIM,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m embed_traffic.inference",
        description="Run TIM on a video and emit per-frame TIMFrameOutput records.",
    )
    p.add_argument("video", type=str, help="Path to input video (mp4, avi, ...).")
    p.add_argument(
        "--output", type=str, default=None,
        help="JSONL output path. One TIMFrameOutput per line. If omitted, nothing is written.",
    )
    p.add_argument(
        "--demo", type=str, default=None,
        help="Optional overlay mp4 path. If provided, generates a demo video.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the number of frames processed. Default: process entire video.",
    )
    p.add_argument(
        "--detector-run-name", type=str, default=DEFAULT_DETECTOR_RUN,
        help=f"Checkpoint dir under checkpoints/. Default: {DEFAULT_DETECTOR_RUN}.",
    )
    p.add_argument(
        "--intent-run-name", type=str, default=DEFAULT_INTENT_RUN,
        help=f"Intent-LSTM checkpoint dir. Default: {DEFAULT_INTENT_RUN}.",
    )
    p.add_argument(
        "--detector-weights", type=str, default=None,
        help="Absolute path to detector weights (overrides --detector-run-name).",
    )
    p.add_argument(
        "--intent-weights", type=str, default=None,
        help="Absolute path to intent weights (overrides --intent-run-name).",
    )
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--predict-steps", type=int, default=30)
    p.add_argument("--device", type=str, default=None,
                   help="torch device string, e.g. 'cuda:0' or 'cpu'.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress prints.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not Path(args.video).exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 2

    tim = TIM(
        detector_run_name=args.detector_run_name,
        intent_run_name=args.intent_run_name,
        detector_weights_path=args.detector_weights,
        intent_weights_path=args.intent_weights,
        tracker=args.tracker,
        imgsz=args.imgsz,
        predict_steps=args.predict_steps,
        device=args.device,
    )

    if not args.quiet:
        print(f"Detector: {tim.detector_path}")
        print(f"Intent:   {tim.intent_path} (loaded={tim.has_intent_model})")
        print(f"Video:    {args.video}")

    jsonl_fh = open(args.output, "w") if args.output else None

    if args.demo:
        # Render demo video — this also writes JSONL if requested.
        def cb(frame, out):
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps(asdict(out)) + "\n")

        render_demo_video(
            tim, args.video, args.demo, max_frames=args.max_frames or 200
        )
        # render_demo_video already ran process_video; if we also want JSONL,
        # do a second pass below.
        if jsonl_fh is not None:
            jsonl_fh.close()
            jsonl_fh = open(args.output, "w")
            outputs = tim.process_video(args.video, max_frames=args.max_frames)
            for out in outputs:
                jsonl_fh.write(json.dumps(asdict(out)) + "\n")
    else:
        outputs = tim.process_video(args.video, max_frames=args.max_frames)
        if jsonl_fh is not None:
            for out in outputs:
                jsonl_fh.write(json.dumps(asdict(out)) + "\n")

        if not args.quiet and outputs:
            latencies = np.asarray([o.processing_time_ms for o in outputs])
            print(
                f"Processed {len(outputs)} frames | "
                f"latency mean={latencies.mean():.1f}ms "
                f"p50={np.percentile(latencies, 50):.1f}ms "
                f"p95={np.percentile(latencies, 95):.1f}ms | "
                f"throughput={1000.0 / latencies.mean():.1f} fps"
            )

    if jsonl_fh is not None:
        jsonl_fh.close()
        if not args.quiet:
            print(f"Wrote JSONL → {args.output}")

    if args.demo and not args.quiet:
        print(f"Wrote demo video → {args.demo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
