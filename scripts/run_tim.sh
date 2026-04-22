#!/bin/bash
# Run TIM inference on video(s) — end-to-end: detect → track → trajectory → intent.
#
# By default:
#   * Uses detector: checkpoints/ped_dashcam/weights/best.pt
#   * Uses intent:   checkpoints/intent_default/intent_lstm.pt
#   * Runs on two demo videos (JAAD + PIE) and writes both JSONL + demo mp4s
#
# Override:
#   RUN_NAME=my_tim_run \
#   DETECTOR_RUN=ped_dashcam \
#   INTENT_RUN=intent_default \
#   VIDEO="data/JAAD_clips/video_0297.mp4" \
#   MAX_FRAMES=200 \
#       bash scripts/run_tim.sh

set -eo pipefail

# ── User-editable ──
RUN_NAME="${RUN_NAME:-tim_infer}"
DETECTOR_RUN="${DETECTOR_RUN:-ped_dashcam}"
INTENT_RUN="${INTENT_RUN:-intent_default}"
MAX_FRAMES="${MAX_FRAMES:-200}"
# If VIDEO is set, run on just that video. Otherwise loop over demo defaults.
VIDEO="${VIDEO:-}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"
OUT_DIR="outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

echo "=== TIM Inference ===" | tee "$LOG"
echo "RUN_NAME:     $RUN_NAME"      | tee -a "$LOG"
echo "DETECTOR_RUN: $DETECTOR_RUN"  | tee -a "$LOG"
echo "INTENT_RUN:   $INTENT_RUN"    | tee -a "$LOG"
echo "MAX_FRAMES:   $MAX_FRAMES"    | tee -a "$LOG"
echo "OUT_DIR:      $OUT_DIR"       | tee -a "$LOG"
echo "Started:      $(date)"         | tee -a "$LOG"

run_one () {
    local video="$1"
    local tag
    tag="$(basename "${video%.*}")"
    local jsonl="$OUT_DIR/${tag}.jsonl"
    local demo="$OUT_DIR/${tag}.mp4"
    echo ""                                                 | tee -a "$LOG"
    echo "--- $video ---"                                   | tee -a "$LOG"
    python -m embed_traffic.inference "$video" \
        --detector-run-name "$DETECTOR_RUN" \
        --intent-run-name "$INTENT_RUN" \
        --max-frames "$MAX_FRAMES" \
        --output "$jsonl" \
        --demo "$demo" 2>&1 | tee -a "$LOG"
}

if [ -n "$VIDEO" ]; then
    run_one "$VIDEO"
else
    for v in data/JAAD_clips/video_0297.mp4 data/PIE_clips/set03/video_0015.mp4; do
        if [ -f "$v" ]; then run_one "$v"; fi
    done
fi

echo ""                         | tee -a "$LOG"
echo "Finished: $(date)"         | tee -a "$LOG"
