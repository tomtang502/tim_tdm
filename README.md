# TIM/TDM: Cooperative Edge Perception for Intersection Safety

Embedded traffic perception system with two cooperating nodes:

- **TIM (Traffic Info Model)** — runs on the traffic-light camera node. Detects pedestrians, tracks them across frames, estimates walking speed and future trajectory, classifies crossing intent, and reads the traffic-light state.
- **TDM (Traffic Decision Model)** — runs on the driver/cyclist app. Consumes TIM's per-frame output and emits a decision: **STOP**, **SLOW_DOWN**, or **NO_OP**.

The TIM detector is trained as two view-specific models: a **dashcam** model (from PIE + JAAD) and a **traffic-light** model (from Intersection-Flow-5K + MIO-TCD). The traffic-light model is what ships with the camera node; the dashcam model is useful for offline validation against PIE's OBD-synced driving data.

---

## Quick start (on a fresh machine)

Prerequisites:

- Linux, Python ≥ 3.10, CUDA 12.8 GPU drivers
- Conda (or any Python env manager)
- ~**500 GB** of disk (videos + derived YOLO dataset)
- ~32 GB GPU VRAM (training was developed on 2× RTX 5090)

```bash
# 1. Clone and enter the repo
git clone <your fork>
cd embed_traffic

# 2. Create an env and install (CUDA 12.8 wheels from PyTorch index)
conda create -n embed_traffic python=3.12 -y
conda activate embed_traffic
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128

# 3. Download and prepare all four datasets (~85 GB total)
#    Requires ~/.kaggle/kaggle.json for Intersection-Flow-5K.
bash scripts/set_up_data.sh

# 4. Train. Each script exposes a RUN_NAME variable at the top; checkpoints go
#    into checkpoints/<RUN_NAME>/ and logs into logs/<RUN_NAME>_<ts>.log.
RUN_NAME_DASHCAM=ped_dashcam RUN_NAME_TL=ped_traffic_light \
    bash scripts/run_train_both.sh

# 5. Train the crossing-intent classifier
RUN_NAME=intent_default bash scripts/run_train_intent.sh

# 6. Run demos
bash scripts/run_tim.sh        # outputs/demos/demo_tim_*.mp4
bash scripts/run_tdm.sh        # outputs/demos/demo_tdm_*.mp4
```

If your conda env is named something other than `embed_traffic`, set `CONDA_ENV=<name>` before the scripts.

---

## Repository layout

```
embed_traffic/
├── pyproject.toml                 # pip install -e . metadata
├── README.md
├── .gitignore
│
├── src/embed_traffic/             # core Python package
│   ├── paths.py                   # centralized path constants
│   ├── data/
│   │   ├── loader.py              # UnifiedDataLoader (PIE + JAAD)
│   │   ├── prepare_yolo.py        # build yolo_dataset/ from PIE+JAAD
│   │   ├── integrate.py           # merge IFlow + MIO-TCD + inD + WTS
│   │   └── visualize.py           # generate sample mp4s
│   ├── train/
│   │   ├── config.py              # shared hyperparameters
│   │   ├── dashcam.py             # fine-tune YOLO on PIE+JAAD
│   │   ├── traffic_light.py       # fine-tune YOLO on IFlow+MIO-TCD
│   │   └── intent.py              # train crossing-intent LSTM
│   ├── models/
│   │   ├── intent_model.py        # CrossingIntentLSTM (bi-LSTM)
│   │   ├── trajectory.py          # Kalman filter trajectories
│   │   ├── tracking.py            # ByteTrack / BoT-SORT wrappers
│   │   ├── tim.py                 # full TIM pipeline
│   │   └── tdm.py                 # rule-based TTC decisions
│   ├── eval/
│   │   └── yolo_zeroshot.py       # COCO-pretrained YOLO baseline
│   └── utils/
│       └── env.py                 # NVIDIA LD_LIBRARY_PATH helper
│
├── scripts/                       # bash entrypoints; every script has a
│   ├── _common.sh                 # RUN_NAME at the top and writes
│   ├── set_up_data.sh             # logs/<RUN_NAME>_<ts>.log
│   ├── run_train_dashcam.sh
│   ├── run_train_traffic_light.sh
│   ├── run_train_both.sh
│   ├── run_train_intent.sh
│   ├── run_tim.sh
│   ├── run_tdm.sh
│   ├── run_trajectory.sh
│   ├── run_tracking.sh
│   └── run_visualize.sh
│
├── datasets/                      # vendored dataset code (annotations + loaders)
│   ├── PIE/                       # from github.com/aras62/PIE
│   └── JAAD/                      # from github.com/ykotseruba/JAAD
│
├── data/                          # downloaded + generated data (GITIGNORED)
│   ├── PIE_clips/                 # 53 MP4 videos, ~74 GB
│   ├── JAAD_clips/                # 346 MP4 videos, ~3 GB
│   ├── Intersection-Flow-5K/      # 6.9K labeled frames, ~5.75 GB
│   ├── MIO-TCD/                   # 137K surveillance frames, ~3.5 GB
│   ├── yolo_dataset/              # derived merged YOLO dataset
│   ├── yolo_dataset_dashcam/      # symlinked PIE+JAAD subset
│   └── yolo_dataset_traffic_light/  # symlinked IFlow+MIO-TCD subset
│
├── checkpoints/                   # model checkpoints (GITIGNORED)
│   └── <RUN_NAME>/
│       ├── weights/best.pt        # YOLO detectors
│       └── intent_lstm.pt         # intent classifier
│
├── logs/                          # training logs (GITIGNORED)
│   └── <RUN_NAME>_<timestamp>.log
│
└── outputs/                       # demo videos (GITIGNORED)
    ├── demos/
    └── samples/
```

---

## Datasets

| Dataset | Camera view | Role | Size | Access |
|---|---|---|---|---|
| PIE   | Dashcam    | Pedestrian crossing intent + OBD vehicle speed | 74 GB video | Direct download |
| JAAD  | Dashcam    | Pedestrian behavioural labels                   | 3 GB video  | Direct download |
| Intersection-Flow-5K | Fixed infrastructure | Fixed-camera pedestrian annotations | 5.75 GB | Kaggle |
| MIO-TCD Localization | Fixed surveillance  | Real traffic-camera frames             | 3.5 GB  | Direct download |

`scripts/set_up_data.sh` handles downloads idempotently. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) before running — required for Intersection-Flow-5K.

---

## Training overview

Both detector training runs use the same hyperparameters (defined in `src/embed_traffic/train/config.py`):

- Base: `yolo26x.pt` (downloaded automatically by ultralytics)
- `epochs=80, imgsz=1280, batch=20, nbs=60`
- Strong augmentation: `mosaic=1.0, mixup=0.2, scale=0.5, shear=2.0`, random erasing, copy-paste
- Optimizer: AdamW, `lr0=0.005, lrf=0.01`, cosine LR with 5-epoch warmup
- `cache="disk"` (dashcam) or `cache=True` (traffic-light) — RAM cache is safe for the smaller traffic-light dataset
- DDP across two GPUs (`device=[0, 1]`). `amp=True` by default; avoid `half=True` during training (EMA can go NaN).

The intent classifier (`src/embed_traffic/train/intent.py`) is a small bi-LSTM trained on PIE+JAAD crossing-intent labels.

### Checkpoint layout

Ultralytics writes to `<project>/<name>/weights/`; we pass `project=checkpoints, name=<RUN_NAME>` so checkpoints end up at:

```
checkpoints/<RUN_NAME>/
├── weights/best.pt   # best by val mAP@0.5:0.95
└── weights/last.pt   # last epoch
```

The intent classifier saves `checkpoints/<RUN_NAME>/intent_lstm.pt` and a `training_log.json`.

---

## Inference

The `TIM` class in `src/embed_traffic/models/tim.py` is the main entrypoint:

```python
from embed_traffic.models.tim import TIM

tim = TIM(view="traffic_light")     # or "dashcam"; picks the right checkpoint
for frame_id in range(num_frames):
    frame = cv2.imread(...)
    out = tim.process_frame(frame, frame_id)
    for ped in out.pedestrians:
        print(ped.ped_id, ped.bbox, ped.crossing_intent, ped.speed_px_s,
              ped.predicted_path)
```

The `TDM` class in `src/embed_traffic/models/tdm.py` consumes `TIMOutput` and returns a decision:

```python
from embed_traffic.models.tdm import TDM

tdm = TDM()
decision = tdm.decide(tim_out, vehicle_speed_px_s=...)
# decision.decision -> "STOP" | "SLOW_DOWN" | "NO_OP"
```

---

## Script conventions

Every script in `scripts/` follows the same contract:

```bash
#!/bin/bash
set -e

# ── User-editable ──
RUN_NAME="..."

# ── Boilerplate ──
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/${RUN_NAME}_${TIMESTAMP}.log"
...
```

`scripts/_common.sh` handles:
1. `cd` into the repo root
2. `conda activate $CONDA_ENV` (default `17422`)
3. Prepending NVIDIA `site-packages` lib dirs to `LD_LIBRARY_PATH` (fixes `libcudnn.so.9` import errors)
4. Creating `logs/`, `checkpoints/`, `outputs/` if missing

---

## Known issues

- **EMA NaN with `compile=True` + DDP**: ultralytics has an edge case where `torch.compile` combined with multi-GPU DDP can corrupt EMA weights. Our config uses `compile=True` but the detectors will skip corrupted checkpoint saves rather than crash. If you see `Skipping checkpoint save at epoch N: EMA contains NaN/Inf`, the best fix is to drop to single-GPU or disable `compile=True`.
- **`half=True` during training slows training**: despite the name, enabling `half=True` in `model.train()` on top of AMP doubles FP16 work. Keep `half=True` only on `model.val()` inference calls.
- **`libcudnn.so.9` not found**: the conda env doesn't prepend NVIDIA's site-packages lib dirs automatically. `scripts/_common.sh` handles this; if running Python directly, call `from embed_traffic.utils.env import setup_nvidia_libs; setup_nvidia_libs()` first.

---

## Results (YOLO26x on test split)

| Model              | Datasets              | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|--------------------|-----------------------|:-------:|:------------:|:---------:|:------:|
| ped_dashcam        | PIE + JAAD            | 0.483   | 0.297        | 0.469     | 0.513  |
| ped_traffic_light  | IFlow + MIO-TCD       | **0.801** | **0.423**  | **0.804** | **0.731** |

The fixed-camera model is significantly stronger — its training data is more consistent in perspective and pedestrian scale. For the deployment TIM node, use the traffic-light model.

Intent classifier (PIE + JAAD test set): **83.7% accuracy** (macro F1 ~0.80).

TIM pipeline end-to-end latency (RTX 5090, 1280px input, ByteTrack): **~10.6 ms/frame (~94 fps)**.

---

## License

See individual dataset licenses for dataset-specific terms (PIE and JAAD are under MIT; Intersection-Flow-5K under CC BY-NC-SA 4.0; MIO-TCD under CC BY-NC-SA 4.0).
