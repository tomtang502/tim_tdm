#!/bin/bash
#SBATCH --job-name=train_both
#SBATCH --account=cis240018p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_train_both_%j.out
#SBATCH --error=logs/slurm_train_both_%j.err
#
# Submit from the repo root:
#   sbatch scripts/slurm/run_train_both.sh
#
# Env-var overrides (same as run_train_both.sh) are supported:
#   sbatch --export=ALL,BASE_MODEL=yolo11x,BATCH=24 scripts/slurm/run_train_both.sh

set -eo pipefail

# ── Training knobs passed through to scripts/run_train_both.sh ──
export WORKERS=12            # dataloader workers; should be <= --cpus-per-task

# Resolve repo root. SLURM copies the batch script to a spool dir, so
# ${BASH_SOURCE[0]} is unreliable here — use SLURM_SUBMIT_DIR.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Load conda (batch jobs do not source ~/.bashrc, so we do it explicitly)
module load anaconda3/2024.10-1
conda activate 17422

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Node: $SLURMD_NODENAME"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  Mem: ${SLURM_MEM_PER_NODE:-?}M  GPUs: ${SLURM_GPUS_ON_NODE:-?}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_train_both.sh
