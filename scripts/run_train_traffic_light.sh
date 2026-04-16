#!/bin/bash
# Fine-tune YOLO detector for traffic-light view (Intersection-Flow + MIO-TCD).
# Usage: RUN_NAME=ped_traffic_light bash scripts/run_train_traffic_light.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-ped_traffic_light}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Fine-tune YOLO: Traffic Light (IFlow + MIO-TCD) ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                       | tee -a "$LOG"
echo "Started: $(date)"                                          | tee -a "$LOG"
echo "Log:     $LOG"                                             | tee -a "$LOG"

python -m embed_traffic.train.traffic_light --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
