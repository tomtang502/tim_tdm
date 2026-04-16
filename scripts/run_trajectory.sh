#!/bin/bash
# Run trajectory estimation + validation.
# Usage: RUN_NAME=trajectory_demo bash scripts/run_trajectory.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-trajectory_demo}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Trajectory Estimation ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"              | tee -a "$LOG"
echo "Started: $(date)"                 | tee -a "$LOG"

python -m embed_traffic.models.trajectory 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
