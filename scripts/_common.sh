#!/bin/bash
# Common helpers sourced by every run_*.sh script.
#
# Responsibilities:
#   1. cd to repo root
#   2. Activate conda env (defaults to $CONDA_ENV or "17422")
#   3. Prepend NVIDIA site-packages libs to LD_LIBRARY_PATH
#   4. Ensure logs/ and checkpoints/ exist

set -e

# Resolve repo root (parent of this script's directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# Activate conda env
CONDA_ENV="${CONDA_ENV:-17422}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Locate NVIDIA CUDA libs in the Python env's site-packages and add to LD_LIBRARY_PATH
PY_BIN="$(command -v python)"
if [ -n "$PY_BIN" ]; then
    PY_VER=$("$PY_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
    for prefix in "$CONDA_PREFIX" "$HOME/.local"; do
        [ -z "$prefix" ] && continue
        NV_BASE="$prefix/lib/$PY_VER/site-packages/nvidia"
        if [ -d "$NV_BASE" ]; then
            NV_LIBS=$(find "$NV_BASE" -name lib -type d 2>/dev/null | tr '\n' ':')
            if [ -n "$NV_LIBS" ]; then
                export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
            fi
        fi
    done
fi

# Ensure standard output directories exist
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints" "$REPO_ROOT/outputs/demo" "$REPO_ROOT/outputs/samples"

cd "$REPO_ROOT"
