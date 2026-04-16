#!/bin/bash
# Fine-tune YOLO detector for dashcam view (PIE + JAAD).
# Usage: RUN_NAME=ped_dashcam bash scripts/run_train_dashcam.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-ped_dashcam}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Fine-tune YOLO: Dashcam (PIE + JAAD) ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                           | tee -a "$LOG"
echo "Started: $(date)"                              | tee -a "$LOG"
echo "Log:     $LOG"                                 | tee -a "$LOG"

python -m embed_traffic.train.dashcam --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
