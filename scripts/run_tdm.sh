#!/bin/bash
# Run TDM (Traffic Decision Model) evaluation + demo videos.
# Usage: RUN_NAME=tdm_demo bash scripts/run_tdm.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-tdm_demo}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== TDM Evaluation ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"      | tee -a "$LOG"
echo "Started: $(date)"         | tee -a "$LOG"

python -m embed_traffic.models.tdm 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
