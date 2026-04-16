#!/bin/bash
# Fine-tune YOLO detector for traffic-light view (Intersection-Flow + MIO-TCD).
# Usage: RUN_NAME=ped_traffic_light bash scripts/run_train_traffic_light.sh

set -eo pipefail

# ── User-editable ──
RUN_NAME="${RUN_NAME:-ped_traffic_light}"
BASE_MODEL="yolo26x"
# Global batch size. Leave empty to use TRAIN_DEFAULTS.
BATCH=32

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Fine-tune YOLO: Traffic Light (IFlow + MIO-TCD) ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                       | tee -a "$LOG"
echo "Base model: ${BASE_MODEL:-<module default>}"              | tee -a "$LOG"
echo "Batch:      ${BATCH:-<module default>}"                   | tee -a "$LOG"
echo "Started: $(date)"                                          | tee -a "$LOG"
echo "Log:     $LOG"                                             | tee -a "$LOG"

ARGS=(--run-name "$RUN_NAME")
[ -n "$BASE_MODEL" ] && ARGS+=(--pretrained "$BASE_MODEL")
[ -n "$BATCH" ]      && ARGS+=(--batch "$BATCH")

python -m embed_traffic.train.traffic_light "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
