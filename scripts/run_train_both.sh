#!/bin/bash
# Train both dashcam and traffic-light YOLO detectors sequentially.
# Each sub-training gets its own log file and checkpoint directory.
# Usage:
#   RUN_NAME_DASHCAM=ped_dashcam RUN_NAME_TL=ped_traffic_light \
#       bash scripts/run_train_both.sh
#
# Optional: override pretrained weights (defaults to yolo26x.pt from each module).
#   BASE_MODEL=yolo11x.pt bash scripts/run_train_both.sh
# Or set per-model:
#   BASE_MODEL_DASHCAM=yolo11x.pt BASE_MODEL_TL=yolo26x.pt bash scripts/run_train_both.sh

set -eo pipefail

# ── User-editable ──
RUN_NAME="init version"
RUN_NAME_DASHCAM="${RUN_NAME_DASHCAM:-ped_dashcam}"
RUN_NAME_TL="${RUN_NAME_TL:-ped_traffic_light}"
BASE_MODEL="yolo26x"
BASE_MODEL_DASHCAM="${BASE_MODEL_DASHCAM:-${BASE_MODEL:-}}"
BASE_MODEL_TL="${BASE_MODEL_TL:-${BASE_MODEL:-}}"
# Global batch size (total across all GPUs). Leave empty to use TRAIN_DEFAULTS.
BATCH=32
BATCH_DASHCAM="${BATCH_DASHCAM:-$BATCH}"
BATCH_TL="${BATCH_TL:-$BATCH}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_both_${TIMESTAMP}.log"

echo "=== Training Both Models ===" | tee "$LOG"
echo "Dashcam run:       $RUN_NAME_DASHCAM" | tee -a "$LOG"
echo "Traffic-light run: $RUN_NAME_TL"      | tee -a "$LOG"
echo "Base model dashcam: ${BASE_MODEL_DASHCAM:-<module default>}" | tee -a "$LOG"
echo "Base model TL:      ${BASE_MODEL_TL:-<module default>}"      | tee -a "$LOG"
echo "Batch dashcam:      ${BATCH_DASHCAM:-<module default>}"      | tee -a "$LOG"
echo "Batch TL:           ${BATCH_TL:-<module default>}"           | tee -a "$LOG"
echo "Started: $(date)"                      | tee -a "$LOG"

DASHCAM_ARGS=(--run-name "$RUN_NAME_DASHCAM")
[ -n "$BASE_MODEL_DASHCAM" ] && DASHCAM_ARGS+=(--pretrained "$BASE_MODEL_DASHCAM")
[ -n "$BATCH_DASHCAM" ]      && DASHCAM_ARGS+=(--batch "$BATCH_DASHCAM")

TL_ARGS=(--run-name "$RUN_NAME_TL")
[ -n "$BASE_MODEL_TL" ] && TL_ARGS+=(--pretrained "$BASE_MODEL_TL")
[ -n "$BATCH_TL" ]      && TL_ARGS+=(--batch "$BATCH_TL")

echo ""                                                       | tee -a "$LOG"
echo "===== 1. Dashcam Model (PIE + JAAD) =====" | tee -a "$LOG"
python -m embed_traffic.train.dashcam "${DASHCAM_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo ""                                                       | tee -a "$LOG"
echo "===== 2. Traffic Light Model (IFlow + MIO-TCD) =====" | tee -a "$LOG"
python -m embed_traffic.train.traffic_light "${TL_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
