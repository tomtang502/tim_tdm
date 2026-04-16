#!/bin/bash
# Download and prepare all 4 datasets: PIE, JAAD, Intersection-Flow-5K, MIO-TCD.
#
# Directory layout after completion:
#   datasets/PIE/             — vendored PIE code + annotations (unzipped)
#   datasets/JAAD/            — vendored JAAD code + annotations
#   data/PIE_clips/           — 53 MP4 videos (~74 GB)
#   data/JAAD_clips/          — 346 MP4 videos (~3 GB)
#   data/Intersection-Flow-5K/— 6,928 labeled frames (~5.75 GB)
#   data/MIO-TCD/             — 137K traffic surveillance frames (~3.5 GB)
#   data/yolo_dataset/        — merged YOLO-format training data (derived)
#
# Each step is idempotent: re-running skips already-downloaded datasets.
#
# Usage: bash scripts/set_up_data.sh

set -e

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/set_up_data_${TIMESTAMP}.log"

{
    echo "=== Setting up datasets ==="
    echo "Started: $(date)"
    echo ""

    mkdir -p data datasets

    # ------------------------------------------------------------------
    # 1. PIE — annotations live in datasets/PIE, videos in data/PIE_clips
    # ------------------------------------------------------------------
    echo "--- PIE ---"
    if [ ! -d datasets/PIE ]; then
        echo "Cloning PIE repo into datasets/PIE ..."
        git clone https://github.com/aras62/PIE.git datasets/PIE
    else
        echo "datasets/PIE already present; skipping clone"
    fi

    # Patch pie_data.py so it tolerates non-directory entries in annotations/
    if ! grep -q 'isdir(join(self._annotation_path, f))' datasets/PIE/utilities/pie_data.py 2>/dev/null; then
        echo "Patching datasets/PIE/utilities/pie_data.py (listdir -> filter isdir)..."
        python - <<'PY'
from pathlib import Path
p = Path("datasets/PIE/utilities/pie_data.py")
src = p.read_text()
old = "set_ids = [f for f in sorted(listdir(self._annotation_path))]"
new = "set_ids = [f for f in sorted(listdir(self._annotation_path)) if isdir(join(self._annotation_path, f))]"
if old in src:
    src = src.replace(old, new)
    # Ensure isdir is imported alongside isfile
    if "from os.path import join, abspath, isfile" in src and "isdir" not in src.split("from os.path import")[1].splitlines()[0]:
        src = src.replace("from os.path import join, abspath, isfile", "from os.path import join, abspath, isfile, isdir")
    p.write_text(src)
    print("  patched.")
PY
    fi

    # Unzip PIE annotation zips in place
    cd datasets/PIE/annotations
    for z in annotations.zip annotations_attributes.zip annotations_vehicle.zip; do
        [ -f "$z" ] || continue
        # Check if already unzipped by looking for one of the set directories
        if [ ! -d "set01" ] || [ "$(ls set01 2>/dev/null | wc -l)" = "0" ]; then
            echo "Unzipping $z ..."
            unzip -qo "$z"
        fi
    done
    cd "$REPO_ROOT"

    # Videos
    if [ ! -d data/PIE_clips ] || [ "$(find data/PIE_clips -name '*.mp4' 2>/dev/null | wc -l)" = "0" ]; then
        echo "Downloading PIE videos (~74 GB) ..."
        mkdir -p data/PIE_clips
        (cd data && wget -N --recursive --no-parent -nH --cut-dirs=1 \
            -R "index.html*" \
            https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/ .)
    else
        echo "data/PIE_clips already populated; skipping"
    fi

    # ------------------------------------------------------------------
    # 2. JAAD — annotations in datasets/JAAD (already in repo), videos in data/JAAD_clips
    # ------------------------------------------------------------------
    echo ""
    echo "--- JAAD ---"
    if [ ! -d datasets/JAAD ]; then
        echo "Cloning JAAD repo into datasets/JAAD ..."
        git clone https://github.com/ykotseruba/JAAD.git datasets/JAAD
    else
        echo "datasets/JAAD already present; skipping clone"
    fi

    if [ ! -d data/JAAD_clips ] || [ "$(find data/JAAD_clips -name '*.mp4' 2>/dev/null | wc -l)" = "0" ]; then
        echo "Downloading JAAD videos (~3 GB) ..."
        (cd data && wget -q http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip)
        (cd data && unzip -q JAAD_clips.zip && rm JAAD_clips.zip)
    else
        echo "data/JAAD_clips already populated; skipping"
    fi

    # ------------------------------------------------------------------
    # 3. Intersection-Flow-5K via Kaggle
    # ------------------------------------------------------------------
    echo ""
    echo "--- Intersection-Flow-5K ---"
    if [ ! -d data/Intersection-Flow-5K/Intersection-Flow-5K ]; then
        if ! command -v kaggle >/dev/null 2>&1; then
            echo "ERROR: kaggle CLI not installed (pip install kaggle) or not on PATH" >&2
            exit 1
        fi
        if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
            echo "ERROR: ~/.kaggle/kaggle.json not found. Set up Kaggle API credentials first." >&2
            exit 1
        fi
        mkdir -p data/Intersection-Flow-5K
        (cd data/Intersection-Flow-5K && \
            kaggle datasets download -d starsw/intersection-flow-5k && \
            unzip -q intersection-flow-5k.zip && \
            rm intersection-flow-5k.zip)
    else
        echo "data/Intersection-Flow-5K already populated; skipping"
    fi

    # ------------------------------------------------------------------
    # 4. MIO-TCD Localization
    # ------------------------------------------------------------------
    echo ""
    echo "--- MIO-TCD Localization ---"
    if [ ! -d data/MIO-TCD/MIO-TCD-Localization ]; then
        mkdir -p data/MIO-TCD
        (cd data/MIO-TCD && \
            wget -q https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar && \
            tar xf MIO-TCD-Localization.tar && \
            rm MIO-TCD-Localization.tar)
    else
        echo "data/MIO-TCD already populated; skipping"
    fi

    # ------------------------------------------------------------------
    # 5. Build unified YOLO dataset
    # ------------------------------------------------------------------
    echo ""
    echo "--- Building unified YOLO dataset ---"
    python -m embed_traffic.data.prepare_yolo
    python -m embed_traffic.data.integrate

    echo ""
    echo "Finished: $(date)"
} 2>&1 | tee "$LOG"
