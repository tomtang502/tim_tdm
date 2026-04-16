#!/bin/bash
# Run tracking evaluation (BoT-SORT + ByteTrack on JAAD/PIE).
# Usage: RUN_NAME=tracking_demo bash scripts/run_tracking.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-tracking_demo}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Tracking Evaluation ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"            | tee -a "$LOG"
echo "Started: $(date)"               | tee -a "$LOG"

python -m embed_traffic.models.tracking 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
