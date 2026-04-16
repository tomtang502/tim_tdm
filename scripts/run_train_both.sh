#!/bin/bash
# Train both dashcam and traffic-light YOLO detectors sequentially.
# Each sub-training gets its own log file and checkpoint directory.
# Usage:
#   RUN_NAME_DASHCAM=ped_dashcam RUN_NAME_TL=ped_traffic_light \
#       bash scripts/run_train_both.sh

set -e

# ── User-editable ──
RUN_NAME_DASHCAM="${RUN_NAME_DASHCAM:-ped_dashcam}"
RUN_NAME_TL="${RUN_NAME_TL:-ped_traffic_light}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_both_${TIMESTAMP}.log"

echo "=== Training Both Models ===" | tee "$LOG"
echo "Dashcam run:       $RUN_NAME_DASHCAM" | tee -a "$LOG"
echo "Traffic-light run: $RUN_NAME_TL"      | tee -a "$LOG"
echo "Started: $(date)"                      | tee -a "$LOG"

echo ""                                                       | tee -a "$LOG"
echo "===== 1. Dashcam Model (PIE + JAAD) =====" | tee -a "$LOG"
python -m embed_traffic.train.dashcam --run-name "$RUN_NAME_DASHCAM" 2>&1 | tee -a "$LOG"

echo ""                                                       | tee -a "$LOG"
echo "===== 2. Traffic Light Model (IFlow + MIO-TCD) =====" | tee -a "$LOG"
python -m embed_traffic.train.traffic_light --run-name "$RUN_NAME_TL" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
