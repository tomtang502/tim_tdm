"""Centralized path constants for the embed_traffic repo.

Resolves the repo root as the parent of `src/`, so all paths are stable regardless
of where Python is invoked from.
"""

from __future__ import annotations

from pathlib import Path

# Repo root is two levels up from this file: src/embed_traffic/paths.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Top-level directories
SRC_DIR: Path = REPO_ROOT / "src"
DATASETS_DIR: Path = REPO_ROOT / "datasets"     # vendored dataset code (PIE, JAAD)
DATA_DIR: Path = REPO_ROOT / "data"             # downloaded + generated data (gitignored)
CHECKPOINTS_DIR: Path = REPO_ROOT / "checkpoints"
LOGS_DIR: Path = REPO_ROOT / "logs"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"

# Dataset vendored code (contains Python loaders + annotation XMLs/zips)
PIE_CODE_DIR: Path = DATASETS_DIR / "PIE"
JAAD_CODE_DIR: Path = DATASETS_DIR / "JAAD"

# Cache directories for generated .pkl files (kept outside submodules so
# `git status` inside datasets/ stays clean).
CACHE_DIR: Path = DATA_DIR / "cache"
PIE_CACHE_DIR: Path = CACHE_DIR / "pie"
JAAD_CACHE_DIR: Path = CACHE_DIR / "jaad"

# Downloaded video/image data
PIE_CLIPS_DIR: Path = DATA_DIR / "PIE_clips"
JAAD_CLIPS_DIR: Path = DATA_DIR / "JAAD_clips"
INTERSECTION_FLOW_DIR: Path = DATA_DIR / "Intersection-Flow-5K"
MIO_TCD_DIR: Path = DATA_DIR / "MIO-TCD"

# Derived YOLO-format datasets
YOLO_DATASET_DIR: Path = DATA_DIR / "yolo_dataset"
YOLO_DATASET_DASHCAM_DIR: Path = DATA_DIR / "yolo_dataset_dashcam"
YOLO_DATASET_TRAFFIC_LIGHT_DIR: Path = DATA_DIR / "yolo_dataset_traffic_light"


def checkpoint_dir(run_name: str) -> Path:
    """Directory for checkpoints of a specific training run."""
    return CHECKPOINTS_DIR / run_name


def detector_weights(run_name: str, which: str = "best") -> Path:
    """Path to a YOLO detector checkpoint for a training run.

    Ultralytics writes to <project>/<name>/weights/{best,last}.pt when training
    with `project=checkpoints, name=<run_name>`.
    """
    return checkpoint_dir(run_name) / "weights" / f"{which}.pt"


def intent_weights(run_name: str = "intent_default") -> Path:
    """Path to intent classifier checkpoint."""
    return checkpoint_dir(run_name) / "intent_lstm.pt"


def ensure_dirs() -> None:
    """Create standard output directories if they don't exist."""
    for d in (DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR,
              OUTPUTS_DIR / "demos", OUTPUTS_DIR / "samples"):
        d.mkdir(parents=True, exist_ok=True)
