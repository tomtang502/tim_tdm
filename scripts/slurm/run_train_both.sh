#!/bin/bash
#SBATCH --job-name=train_both
#SBATCH --account=cis240018p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=42:00:00
#SBATCH --output=logs/slurm_train_both_%j.out
#SBATCH --error=logs/slurm_train_both_%j.err
#
# Submit with:
#   sbatch scripts/slurm/run_train_both.sh
#
# Env-var overrides (same as run_train_both.sh) are supported:
#   sbatch --export=ALL,BASE_MODEL=yolo11x,BATCH=24 scripts/slurm/run_train_both.sh

set -eo pipefail

# Resolve repo root (two levels up from this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Node: $SLURMD_NODENAME  GPUs: $SLURM_GPUS_ON_NODE  CPUs: $SLURM_CPUS_ON_NODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_train_both.sh
