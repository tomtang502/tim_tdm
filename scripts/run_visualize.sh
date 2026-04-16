#!/bin/bash
# Generate sample annotation videos for each dataset.
# Usage: RUN_NAME=visualize_samples bash scripts/run_visualize.sh

set -e

# ── User-editable ──
RUN_NAME="${RUN_NAME:-visualize_samples}"

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"

echo "=== Dataset Sample Visualization ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                    | tee -a "$LOG"
echo "Started: $(date)"                       | tee -a "$LOG"

python -m embed_traffic.data.visualize 2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
