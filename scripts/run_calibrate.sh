#!/bin/bash
# One-time camera calibration: run a monocular depth model over a few frames
# and extract the ground-plane geometry. Output: configs/cameras/<CAMERA_ID>.json
#
# Usage:
#   RUN_NAME=calib_junction_01 \
#   CAMERA_ID=junction_01 \
#   VIDEO=data/JAAD_clips/video_0297.mp4 \
#   N_FRAMES=8 \
#       bash scripts/run_calibrate.sh
#
# Optional:
#   DEPTH_MODEL=depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
#   HFOV_DEG=60.0

set -eo pipefail

# ── User-editable ──
RUN_NAME="${RUN_NAME:-camera_calibrate}"
CAMERA_ID="${CAMERA_ID:?CAMERA_ID is required (label for configs/cameras/<id>.json)}"
VIDEO="${VIDEO:-}"
IMAGES="${IMAGES:-}"   # space-separated list of image paths (alternative to VIDEO)
N_FRAMES="${N_FRAMES:-8}"
DEPTH_MODEL="${DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf}"
HFOV_DEG="${HFOV_DEG:-60.0}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Camera Calibration ===" | tee "$LOG"
echo "RUN_NAME:    $RUN_NAME"      | tee -a "$LOG"
echo "CAMERA_ID:   $CAMERA_ID"      | tee -a "$LOG"
echo "VIDEO:       $VIDEO"          | tee -a "$LOG"
echo "IMAGES:      $IMAGES"         | tee -a "$LOG"
echo "N_FRAMES:    $N_FRAMES"       | tee -a "$LOG"
echo "DEPTH_MODEL: $DEPTH_MODEL"    | tee -a "$LOG"
echo "HFOV_DEG:    $HFOV_DEG"       | tee -a "$LOG"
echo "Started:     $(date)"         | tee -a "$LOG"

ARGS=(
  --camera-id "$CAMERA_ID"
  --n-frames "$N_FRAMES"
  --depth-model "$DEPTH_MODEL"
  --hfov-deg "$HFOV_DEG"
)

if [ -n "$VIDEO" ]; then
    ARGS+=(--video "$VIDEO")
elif [ -n "$IMAGES" ]; then
    # shellcheck disable=SC2086
    ARGS+=(--images $IMAGES)
else
    echo "ERROR: set VIDEO=... or IMAGES=\"path1 path2 ...\"" | tee -a "$LOG"
    exit 2
fi

python -m embed_traffic.calibration "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
