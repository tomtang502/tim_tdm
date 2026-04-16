#!/bin/bash
# Run TIM integration demo (detection + tracking + intent + trajectory).
# Usage: RUN_NAME=tim_demo bash scripts/run_tim.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-tim_demo}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== TIM Integration Demo ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"            | tee -a "$LOG"
echo "Started: $(date)"               | tee -a "$LOG"

python -m embed_traffic.models.tim 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
