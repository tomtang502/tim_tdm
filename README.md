# TIM/TDM: Cooperative Edge Perception for Intersection Safety

Embedded traffic perception system with two cooperating nodes:

- **TIM (Traffic Info Model)** — runs on the traffic-light camera node. Detects pedestrians, tracks them across frames, estimates walking speed and future trajectory, classifies crossing intent, and (with one-time camera calibration) emits real-world positions and speeds in meters.
- **TDM (Traffic Decision Model)** — runs on the driver/cyclist app. Consumes TIM's per-frame output and emits a decision: **STOP**, **SLOW_DOWN**, or **NO_OP**.

**Deployment note:** the shipped camera is a **stationary pole-mounted unit** at a traffic junction pointed down the road. That viewpoint (low-to-mid height, road-parallel) matches what PIE/JAAD dashcams see, so the dashcam-trained detector transfers directly. There is no ego-motion at inference time. See `plan.txt §IX`.

For the complete TIM API (input/output types, shapes, calibration, usage patterns, troubleshooting), read **[guide/TIM.md](guide/TIM.md)**.

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

# 4. Train the detector. Checkpoints go to checkpoints/<RUN_NAME>/
#    and logs to logs/<RUN_NAME>_<ts>.log.
RUN_NAME_DASHCAM=ped_dashcam bash scripts/run_train_dashcam.sh

# 5. Train the crossing-intent classifier
RUN_NAME=intent_default bash scripts/run_train_intent.sh

# 6. End-to-end TIM demo (auto-calibrates via Depth-Anything-V2,
#    writes a side-by-side mp4 + per-frame JSONL + 8 depth panels)
python demo/demo_tim.py
```

If your conda env is named something other than `embed_traffic`, set `CONDA_ENV=<name>` before the scripts.

---

## What TIM produces

For every frame, TIM emits a `TIMFrameOutput` with:

- `frame_id`, `frame_time_s` (video-time in seconds), `processing_time_ms`
- a list of `PedestrianInfo` records, each containing:
  - `ped_id` (persistent tracker ID)
  - `bbox` (pixels), `center`, `confidence`
  - `speed_px_s`, `avg_speed_px_s`, `direction` (pixel-space kinematics)
  - `track_length`
  - `crossing_intent`, `crossing_prob` (from the LSTM, after ≥ 15 frames of history)
  - `predicted_path` (Kalman prediction for next N frames)
  - **With camera calibration:** `position_m_ground` (X, Z in meters), `speed_m_s`, `velocity_m_s`

The full output schema, including the JSON wire format for TDM, is in **[guide/TIM.md §4](guide/TIM.md)**.

---

## Demo

```bash
# Defaults to processing two JAAD clips bundled in the repo
python demo/demo_tim.py

# Or pass your own video(s); flag is repeatable
python demo/demo_tim.py --video clip.mp4 --video other.mp4
```

Per video, writes to `outputs/demo/tim/`:

| Artifact | What it is |
|---|---|
| `<stem>.mp4` | **Side-by-side** video. Left panel: overlay with bbox, ID, m/s speed, ground (X, Z). Right panel: top-down bird's-eye view with camera, FoV wedge, and pedestrian dots moving in world coordinates. Both panels are synchronized in time. |
| `<stem>.jsonl` | One `TIMFrameOutput` per line (JSON). |
| `<stem>_calibration/depth_00..07.png` | The 8 depth-model predictions that were used for one-shot camera calibration. |

The calibration itself is persisted to `configs/cameras/<camera_id>.json` and reused on subsequent runs. See `guide/TIM.md §8` for how calibration works.

### TDM demo

```bash
python demo/demo_tdm.py --scenario approach_and_stop
```

Runs the full TIM → TDM pipeline, simulating a car approaching the camera. Writes to `outputs/demo/tdm/`:

- `<stem>.mp4` — side-by-side (overlay + alert banner | top-down with ego car drawn in)
- `<stem>.jsonl` — combined `{"tim": {...}, "tdm": {...}}` per frame

Scenarios: `approaching`, `approaching_fast`, `approach_and_stop`, `stationary`, `constant_velocity`. See `guide/TDM.md`.

---

## Repository layout

```
embed_traffic/
├── pyproject.toml                 # pip install -e . metadata
├── README.md
├── plan.txt                       # project history / design decisions
├── .gitignore
│
├── guide/
│   ├── TIM.md                     # full TIM API reference
│   └── TDM.md                     # (reserved)
│
├── demo/
│   └── demo_tim.py                # end-to-end demo (calibrate + run + visualize)
│
├── src/embed_traffic/             # core Python package
│   ├── paths.py                   # centralized path constants
│   ├── data/
│   │   ├── loader.py              # UnifiedDataLoader (PIE + JAAD)
│   │   ├── prepare_yolo.py        # build yolo_dataset/ from PIE+JAAD
│   │   ├── integrate.py           # merge IFlow + MIO-TCD
│   │   └── visualize.py           # generate sample mp4s
│   ├── train/
│   │   ├── config.py              # shared hyperparameters
│   │   ├── dashcam.py             # fine-tune YOLO on PIE+JAAD
│   │   ├── traffic_light.py       # [archived] fine-tune on IFlow+MIO-TCD
│   │   └── intent.py              # train crossing-intent LSTM
│   ├── models/
│   │   ├── intent_model.py        # CrossingIntentLSTM (bi-LSTM)
│   │   ├── trajectory.py          # Kalman filter trajectories
│   │   ├── tracking.py            # ByteTrack / BoT-SORT wrappers
│   │   ├── tim.py                 # legacy shim → embed_traffic.inference.TIM
│   │   └── tdm.py                 # rule-based TTC decisions
│   ├── inference/                 # ← canonical TIM inference API
│   │   ├── tim.py                 # TIM class (Detector + Tracker + Kalman + Intent)
│   │   ├── schema.py              # PedestrianInfo, TIMFrameOutput, JSON I/O
│   │   ├── demo.py                # bbox/ID/m-s overlay drawing
│   │   └── topdown.py             # top-down bird's-eye animation
│   ├── calibration/               # ← one-shot camera calibration
│   │   ├── schema.py              # CameraCalibration, Intrinsics, pixel_to_ground
│   │   ├── depth.py               # Depth-Anything-V2 wrapper
│   │   ├── ground_plane.py        # RANSAC plane fit, extrinsics derivation
│   │   └── calibrate.py           # end-to-end driver + CLI
│   ├── tdm/                       # ← canonical TDM decision API
│   │   ├── tdm.py                 # TDM class (trajectory-based classifier)
│   │   ├── schema.py              # CarState, AlertLevel, TDMOutput
│   │   ├── simulator.py           # synthetic car trajectories (no real data yet)
│   │   └── demo.py                # alert banner + top-down with car marker
│   ├── eval/
│   │   └── yolo_zeroshot.py       # COCO-pretrained YOLO baseline
│   └── utils/
│       └── env.py                 # NVIDIA LD_LIBRARY_PATH helper
│
├── scripts/                       # bash entrypoints; every script has a
│   ├── _common.sh                 # RUN_NAME at the top and writes
│   ├── set_up_data.sh             # logs/<RUN_NAME>_<ts>.log
│   ├── run_train_dashcam.sh
│   ├── run_train_traffic_light.sh     # [archived]
│   ├── run_train_both.sh              # [archived]
│   ├── run_train_intent.sh
│   ├── run_calibrate.sh            # one-shot camera calibration
│   ├── run_tim.sh                  # batch TIM over one or more videos
│   ├── run_tdm.sh
│   ├── run_trajectory.sh
│   ├── run_tracking.sh
│   └── run_visualize.sh
│
├── configs/cameras/                # per-camera calibration JSONs
│   └── <camera_id>.json
│
├── datasets/                       # vendored dataset code (annotations + loaders)
│   ├── PIE/                        # from github.com/aras62/PIE
│   └── JAAD/                       # from github.com/ykotseruba/JAAD
│
├── data/                           # downloaded + generated data (GITIGNORED)
│   ├── PIE_clips/                  # 53 MP4 videos, ~74 GB
│   ├── JAAD_clips/                 # 346 MP4 videos, ~3 GB
│   ├── Intersection-Flow-5K/       # 6.9K labeled frames, ~5.75 GB
│   ├── MIO-TCD/                    # 137K surveillance frames, ~3.5 GB
│   ├── yolo_dataset/               # derived merged YOLO dataset
│   ├── yolo_dataset_dashcam/       # symlinked PIE+JAAD subset
│   └── yolo_dataset_traffic_light/ # [archived] IFlow+MIO-TCD subset
│
├── checkpoints/                    # model checkpoints (GITIGNORED)
│   └── <RUN_NAME>/
│       ├── weights/best.pt         # YOLO detectors
│       └── intent_lstm.pt          # intent classifier
│
├── logs/                           # training logs (GITIGNORED)
│   └── <RUN_NAME>_<timestamp>.log
│
└── outputs/                        # demo + inference artifacts (GITIGNORED)
    └── demo/
        ├── tim/                    # demo_tim.py artifacts
        │   ├── <stem>.mp4          # side-by-side (overlay | top-down)
        │   ├── <stem>.jsonl        # per-frame TIM records
        │   └── <stem>_calibration/ # 8 depth panels
        └── tdm/                    # demo_tdm.py artifacts
            ├── <stem>.mp4          # side-by-side (overlay+banner | top-down+car)
            ├── <stem>.jsonl        # per-frame {"tim": ..., "tdm": ...}
            └── <stem>_calibration/
```

---

## Datasets

| Dataset | Camera view | Role | Size | Access |
|---|---|---|---|---|
| PIE   | Dashcam    | Pedestrian crossing intent + OBD vehicle speed | 74 GB video | Direct download |
| JAAD  | Dashcam    | Pedestrian behavioural labels                   | 3 GB video  | Direct download |
| Intersection-Flow-5K | Fixed infrastructure | [archived] Fixed-camera pedestrians | 5.75 GB | Kaggle |
| MIO-TCD Localization | Fixed surveillance  | [archived] Traffic-camera frames   | 3.5 GB  | Direct download |

Only the **dashcam** track (PIE + JAAD) is active for production — see `plan.txt §VII`. The fixed-camera datasets are kept for reproducibility but no longer drive training.

`scripts/set_up_data.sh` handles downloads idempotently. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) before running for Intersection-Flow-5K.

---

## Training overview

Detector training (`src/embed_traffic/train/dashcam.py`, base `yolo26x.pt`):

- `epochs=80, imgsz=1280, batch=20, nbs=60`
- Strong augmentation: `mosaic=1.0, mixup=0.2, scale=0.5, shear=2.0`, random erasing, copy-paste
- Optimizer: AdamW, `lr0=0.005, lrf=0.01`, cosine LR with 5-epoch warmup
- `cache="disk"` (dashcam-full training)
- DDP across two GPUs (`device=[0, 1]`). `amp=True` by default; **do not** set `half=True` during training (EMA can go NaN).

The intent classifier (`src/embed_traffic/train/intent.py`) is a small bi-LSTM trained on PIE+JAAD crossing-intent labels.

### Checkpoint layout

Ultralytics writes to `<project>/<name>/weights/`; we pass `project=checkpoints, name=<RUN_NAME>` so checkpoints land at:

```
checkpoints/<RUN_NAME>/
├── weights/best.pt   # best by val mAP@0.5:0.95
└── weights/last.pt   # last epoch
```

The intent classifier saves `checkpoints/<RUN_NAME>/intent_lstm.pt` and a `training_log.json`.

---

## Inference

Canonical API is in `embed_traffic.inference`:

```python
from embed_traffic.inference import TIM

tim = TIM()                                      # default: ped_dashcam + intent_default
# or with real-world coordinates:
tim = TIM(camera_calibration="configs/cameras/junction_01.json")

for frame_id in range(num_frames):
    frame = cv2.imread(...)                       # BGR (H,W,3) uint8
    out = tim.process_frame(frame, frame_id)
    for ped in out.pedestrians:
        print(ped.ped_id, ped.bbox, ped.crossing_intent,
              ped.speed_px_s, ped.speed_m_s, ped.position_m_ground)
```

TDM (simple trajectory collision model) consumes `TIMFrameOutput` plus an ego `CarState`:

```python
from embed_traffic.tdm import TDM, CarState
from embed_traffic.tdm.simulator import make_scenario

tdm = TDM()                                  # default thresholds
car = make_scenario("approaching")(0.0)       # synthetic car at t=0
out = tdm.decide(tim_out, car)
# out.alert -> NONE | CAUTION | SLOW_DOWN | BRAKE
```

See **[guide/TIM.md](guide/TIM.md)** and **[guide/TDM.md](guide/TDM.md)** for full API, CLI, and schema documentation.

---

## One-shot camera calibration

A stationary pole-mounted camera only needs to be calibrated once. Our pipeline runs a monocular metric-depth model (Depth-Anything-V2-Metric-Outdoor-Large by default) on ~8 frames, RANSAC-fits the ground plane, and writes a `CameraCalibration` JSON.

```bash
# Shell wrapper
CAMERA_ID=junction_01 VIDEO=data/JAAD_clips/video_0297.mp4 \
    N_FRAMES=8 bash scripts/run_calibrate.sh

# Or python directly
python -m embed_traffic.calibration --video clip.mp4 --camera-id junction_01
```

Output → `configs/cameras/junction_01.json`. Once saved, runtime TIM uses it with zero additional model calls — calibration is pure geometry at inference.

**Ground-frame convention:** `X=right, Y=up, Z=forward`. This is a *left-handed* frame (det `R_cam_to_ground` = −1) chosen because it keeps "right" consistent between image and world — the visually intuitive convention for top-down views in driving / AR. See `plan.txt §X` and `guide/TIM.md §8`.

---

## Script conventions

Every script in `scripts/` follows the same contract:

```bash
#!/bin/bash
set -eo pipefail

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

- **EMA NaN with `compile=True` + DDP**: ultralytics has an edge case where `torch.compile` combined with multi-GPU DDP can corrupt EMA weights. The current training config disables `compile=True` as a result.
- **`half=True` during training slows training**: despite the name, `half=True` in `model.train()` on top of AMP doubles FP16 work. Keep `half=True` only on `model.val()` inference calls.
- **`libcudnn.so.9` not found**: the conda env doesn't prepend NVIDIA's site-packages lib dirs automatically. `scripts/_common.sh` handles this; for direct Python execution, source the same script first.

---

## Results (YOLO26x on test split)

| Model              | Datasets              | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|--------------------|-----------------------|:-------:|:------------:|:---------:|:------:|
| **ped_dashcam (shipping)** | PIE + JAAD    | **0.797** | **0.555**  | **0.823** | **0.740** |
| ped_traffic_light (archived) | IFlow + MIO-TCD | 0.673 | 0.310      | 0.706     | 0.603  |

Intent classifier (PIE + JAAD test set): **83.7% accuracy**, macro-F1 ≈ 0.80.

TIM pipeline end-to-end latency (RTX 5090, 1280 px input, ByteTrack, intent LSTM): **~18 ms/frame (~55 fps)**. Adding camera calibration is ~0.1 ms overhead per pedestrian.

See `plan.txt §VIII` for the full training history.

---

## License

See individual dataset licenses for dataset-specific terms (PIE and JAAD are under MIT; Intersection-Flow-5K under CC BY-NC-SA 4.0; MIO-TCD under CC BY-NC-SA 4.0).
