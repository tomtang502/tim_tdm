#!/bin/bash
# Fine-tune YOLO detector for dashcam view (PIE + JAAD).
# Usage: RUN_NAME=ped_dashcam bash scripts/run_train_dashcam.sh

set -eo pipefail

# ── User-editable ──
RUN_NAME="${RUN_NAME:-ped_dashcam}"
BASE_MODEL="yolo26x"
# Global batch size. Leave empty to use TRAIN_DEFAULTS.
BATCH=32

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Fine-tune YOLO: Dashcam (PIE + JAAD) ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                           | tee -a "$LOG"
echo "Base model: ${BASE_MODEL:-<module default>}"  | tee -a "$LOG"
echo "Batch:      ${BATCH:-<module default>}"       | tee -a "$LOG"
echo "Started: $(date)"                              | tee -a "$LOG"
echo "Log:     $LOG"                                 | tee -a "$LOG"

ARGS=(--run-name "$RUN_NAME")
[ -n "$BASE_MODEL" ] && ARGS+=(--pretrained "$BASE_MODEL")
[ -n "$BATCH" ]      && ARGS+=(--batch "$BATCH")

python -m embed_traffic.train.dashcam "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
