# TIM — Traffic Info Model

TIM turns a video frame (or an entire video) into per-pedestrian information
for the driver app (TDM) to consume. It is the dashcam-only deployment (see
`plan.txt` §VII); there is no traffic-light-state output.

> **Deployment note (plan.txt §IX):** "dashcam view" refers to the camera
> **viewpoint** — low-to-mid height, road-parallel line of sight — **not** to
> the camera being mounted on a vehicle. The shipped camera is a stationary
> pole-mounted unit at a traffic junction that happens to share the PIE/JAAD
> training viewpoint, which is why the dashcam-trained model transfers
> directly. At inference time there is **no ego-motion** in the frame.

---

## 1. Pipeline

```
BGR frame  →  YOLO26x detection  →  ByteTrack tracking  →  Kalman trajectory  →  LSTM intent
  (H,W,3)      boxes+scores           persistent IDs        past + 30-step future   P(crossing)
```

Stage-by-stage:

| Stage | Component | Weights | Output added |
|---|---|---|---|
| 1. Detect | YOLO26x fine-tuned on PIE+JAAD | `checkpoints/ped_dashcam/weights/best.pt` | bbox, confidence |
| 2. Track | ByteTrack (via ultralytics) | — (stateful) | persistent `ped_id` |
| 3. Trajectory | Kalman filter on bbox centers | — | center, speed, direction, predicted future path |
| 4. Intent | Bi-directional LSTM | `checkpoints/intent_default/intent_lstm.pt` | crossing/not-crossing, P(crossing) |

The intent model requires **≥ 15 frames of history** per pedestrian. Until
then, `crossing_intent` and `crossing_prob` are `None`.

The trajectory predictor requires **≥ 5 frames of history**; otherwise
`predicted_path` is `None`.

---

## 2. Public API

```python
from embed_traffic.inference import TIM, PedestrianInfo, TIMFrameOutput
```

### 2.1 `TIM` constructor

```python
tim = TIM(
    detector_run_name="ped_dashcam",     # checkpoints/<name>/weights/best.pt
    intent_run_name="intent_default",    # checkpoints/<name>/intent_lstm.pt
    detector_weights_path=None,           # absolute path override
    intent_weights_path=None,             # absolute path override
    tracker="bytetrack.yaml",             # or "botsort.yaml"
    fps=30.0,                             # used for px/s and dt computations
    predict_steps=30,                     # future frames to predict
    imgsz=1280,                           # detector inference resolution
    device=None,                          # auto-detect cuda, or "cuda:0" / "cpu"
    camera_calibration=None,              # str | Path | CameraCalibration | None
                                          # if provided, enables world-space fields
)
```

### 2.2 Processing methods

**Single frame:**

```python
out: TIMFrameOutput = tim.process_frame(frame, frame_id=0)
```

**Entire video:**

```python
outputs: list[TIMFrameOutput] = tim.process_video(
    video_path,
    max_frames=None,          # or an int cap
    callback=None,             # optional fn(frame, TIMFrameOutput)
)
```

**Live stream:**

```python
for out in tim.stream(frame_iterator, start_frame_id=0):
    ...  # produces one TIMFrameOutput per frame, lazily
```

**Reset** (when switching video):

```python
tim.reset()  # clears tracker state + intent feature buffers
```

---

## 3. Input spec

### 3.1 `process_frame(frame, frame_id)`

| Arg | Type | Shape | dtype | Constraints |
|---|---|---|---|---|
| `frame` | `np.ndarray` | `(H, W, 3)` | `uint8` | **BGR** (OpenCV convention), not RGB |
| `frame_id` | `int` | scalar | — | Monotonic ID for tracking continuity |

`H` and `W` can vary per frame; the detector resizes internally to `imgsz`.

### 3.2 `process_video(video_path, max_frames=None, callback=None)`

| Arg | Type | Notes |
|---|---|---|
| `video_path` | `str \| Path` | Any format OpenCV can open: mp4, avi, mov, … |
| `max_frames` | `int \| None` | Stop after N frames; `None` processes the whole video |
| `callback` | `Callable[[np.ndarray, TIMFrameOutput], None] \| None` | Invoked per frame — useful for rendering or streaming to TDM |

`fps` is auto-read from the video file.

---

## 4. Output spec

### 4.1 `TIMFrameOutput` (per frame)

```python
@dataclass
class TIMFrameOutput:
    frame_id:            int
    pedestrians:         list[PedestrianInfo]     # length = N detected peds
    frame_width:         int                       # pixels
    frame_height:        int                       # pixels
    frame_time_s:        float                     # video time: frame_id / fps
    processing_time_ms:  float                     # wall-clock for this frame
    # Convenience:
    num_pedestrians:     int                       # == len(pedestrians)
```

### 4.2 `PedestrianInfo` (per pedestrian per frame)

| Field | Type | Shape | Units / range | Notes |
|---|---|---|---|---|
| `ped_id` | `int` | scalar | — | Persistent across frames from ByteTrack |
| `bbox` | `list[float]` | `[4]` | pixels `[x1, y1, x2, y2]` | Top-left / bottom-right corners |
| `center` | `list[float]` | `[2]` | pixels `[cx, cy]` | bbox center |
| `confidence` | `float` | scalar | `[0, 1]` | Detector confidence |
| `speed_px_s` | `float` | scalar | px/s | Instantaneous (last-two-centers) |
| `avg_speed_px_s` | `float` | scalar | px/s | Averaged over tracked history |
| `direction` | `list[float]` | `[2]` | unit vector `[dx, dy]` | Avg over last 5 frames; `[0, 0]` if unknown |
| `track_length` | `int` | scalar | frames | How long this ID has been tracked |
| `crossing_intent` | `str \| None` | — | `"crossing"`, `"not-crossing"`, or `None` | `None` until ≥ 15 frame history |
| `crossing_prob` | `float \| None` | scalar | `[0, 1]` | P(crossing); pairs with `crossing_intent` |
| `predicted_path` | `list[list[float]] \| None` | `[predict_steps, 2]` | pixels | Future `[cx, cy]` for `predict_steps` frames; `None` until ≥ 5 frame history |
| `position_m_ground` | `list[float] \| None` | `[2]` | meters `[X, Z]` | Ground-plane position; **requires calibration** (see §8). `None` otherwise |
| `speed_m_s` | `float \| None` | scalar | m/s | Ground speed; **requires calibration** and ≥ 2 frames of track history |
| `velocity_m_s` | `list[float] \| None` | `[2]` | m/s `[vx, vz]` | Ground velocity vector; same requirements as `speed_m_s` |

### 4.3 JSON wire format

```python
from embed_traffic.inference import serialize_frame_output, deserialize_frame_output
wire = serialize_frame_output(out)       # compact JSON string
out2 = deserialize_frame_output(wire)     # TIMFrameOutput (round-trip safe)
```

This is what the driver node receives over the network (gRPC/MQTT/HTTP).

### 4.4 Example JSON

```json
{
  "frame_id": 42,
  "frame_width": 1920,
  "frame_height": 1080,
  "frame_time_s": 1.400,
  "processing_time_ms": 27.4,
  "pedestrians": [
    {
      "ped_id": 3,
      "bbox": [1231.5, 412.8, 1302.1, 653.9],
      "center": [1266.8, 533.4],
      "confidence": 0.91,
      "speed_px_s": 78.2,
      "avg_speed_px_s": 65.0,
      "direction": [0.12, -0.99],
      "track_length": 34,
      "crossing_intent": "crossing",
      "crossing_prob": 0.87,
      "predicted_path": [[1265.8, 534.1], [1264.7, 534.8], ...],
      "position_m_ground": [2.45, 9.62],
      "speed_m_s": 1.38,
      "velocity_m_s": [0.21, -1.36]
    }
  ]
}
```

(`position_m_ground`, `speed_m_s`, `velocity_m_s` are `null` when no camera calibration is loaded.)

---

## 5. Usage patterns

> The fastest way to see everything working is to run
> `python demo/demo_tim.py`. It auto-calibrates from the first 8 frames of the
> default clip(s), then writes a side-by-side (overlay | top-down) mp4 + JSONL
> + 8 depth panels to `outputs/demo/tim/`. Source for that demo is the
> reference implementation of the patterns below.

### 5.1 Python — single frame

```python
import cv2
from embed_traffic.inference import TIM

tim = TIM()  # loads checkpoints/ped_dashcam + checkpoints/intent_default
frame = cv2.imread("frame_0042.jpg")      # BGR, uint8, (H, W, 3)
out = tim.process_frame(frame, frame_id=42)

for p in out.pedestrians:
    print(p.ped_id, p.bbox, p.crossing_intent, p.crossing_prob)
```

### 5.2 Python — whole video, stream results

```python
from embed_traffic.inference import TIM, serialize_frame_output

tim = TIM()
with open("out.jsonl", "w") as f:
    def cb(frame, out):
        f.write(serialize_frame_output(out) + "\n")
    tim.process_video("clip.mp4", max_frames=500, callback=cb)
```

### 5.3 Python — with overlay demo video

```python
from embed_traffic.inference import TIM
from embed_traffic.inference.demo import render_demo_video

tim = TIM()
render_demo_video(tim, "clip.mp4", "overlay.mp4", max_frames=300)
```

### 5.4 CLI

```bash
# JSONL only
python -m embed_traffic.inference data/JAAD_clips/video_0297.mp4 \
    --output outputs/demo.jsonl

# JSONL + demo video
python -m embed_traffic.inference data/JAAD_clips/video_0297.mp4 \
    --output outputs/demo.jsonl --demo outputs/demo.mp4 --max-frames 200

# Use a different checkpoint
python -m embed_traffic.inference data/JAAD_clips/video_0297.mp4 \
    --detector-run-name my_custom_run --intent-run-name intent_v2
```

### 5.5 Shell wrapper (batch of videos)

```bash
RUN_NAME=prod_eval DETECTOR_RUN=ped_dashcam MAX_FRAMES=500 \
    bash scripts/run_tim.sh
```

Writes per-video JSONL + demo mp4 to `outputs/$RUN_NAME/`.

### 5.6 End-to-end demo (calibration + inference + visualization)

```bash
python demo/demo_tim.py                                   # all default clips
python demo/demo_tim.py --video a.mp4 --video b.mp4       # custom videos
python demo/demo_tim.py --n-cal-frames 12 --hfov-deg 50   # tweaks
```

Per video, produces under `outputs/demo/tim/`:

| Artifact | Contents |
|---|---|
| `<stem>.mp4` | Side-by-side video. Left: overlay with bbox/ID/m-s labels. Right: top-down bird's-eye showing camera + pedestrians in world meters. Both panels synchronized by frame. |
| `<stem>.jsonl` | One `TIMFrameOutput` per line (JSON). |
| `<stem>_calibration/` | 8 colormapped depth panels used for one-shot calibration. |

Calibration is cached in `configs/cameras/demo_<stem>.json` for reuse.

---

## 6. Performance

Measured on a single RTX 5090 with YOLO26x @ imgsz=1280, ByteTrack, 15-frame
LSTM intent.

| Metric | Value |
|---|---|
| Mean frame latency | ~25–35 ms |
| Throughput | ~30–40 fps |
| VRAM footprint | ~4 GB |

Real-time at 30 fps is achievable on one RTX 5090.

---

## 7. What the downstream TDM expects

TDM only consumes `PedestrianInfo` fields — it does **not** depend on the
frame image itself. A minimal TDM input per frame is:

```python
{
    "frame_id": int,
    "frame_time_s": float,
    "frame_width": int,
    "frame_height": int,
    "pedestrians": [
        {
            "ped_id": int,
            "bbox":           [x1, y1, x2, y2],
            "center":         [cx, cy],
            "speed_px_s":     float,
            "direction":      [dx, dy],
            "crossing_intent": "crossing" | "not-crossing" | None,
            "crossing_prob":  float | None,
            "predicted_path": [[cx, cy], ...] | None,
            # Present only when TIM was constructed with a CameraCalibration:
            "position_m_ground": [X_m, Z_m] | None,
            "speed_m_s":         float | None,
            "velocity_m_s":      [vx_m_s, vz_m_s] | None,
        },
        ...
    ],
}
```

For real-world TTC decisions (meters and m/s rather than pixels), construct
TIM with a `CameraCalibration` — see §8. Without one, TIM still emits the
pixel-space fields and TDM falls back to pixel-based reasoning.

---

## 8. Camera calibration — real-world speed / position (optional)

TIM's pixel-space outputs work without any calibration, but TDM often needs
real-world units (meters / m/s). Because the camera is stationary at a
junction, calibration is a one-time install task.

### 8.1 What calibration does

Given a **CameraCalibration** at init, TIM populates the world-space fields on
every `PedestrianInfo`:

| Field | Units |
|---|---|
| `position_m_ground` | `[X, Z]` meters on the ground plane |
| `speed_m_s` | scalar m/s |
| `velocity_m_s` | `[vx, vz]` m/s |

Method: take the bbox foot point (bottom-center), project through the camera
intrinsics, and intersect with the pre-computed ground plane. Pure linear
algebra at runtime, no ML model.

### 8.2 Auto-calibration via monocular depth (recommended)

The `embed_traffic.calibration` package runs a depth model (Depth-Anything-V2
metric-outdoor by default) on a handful of frames, RANSAC-fits the ground
plane, and emits a `CameraCalibration` JSON. This only happens at install time.

**CLI:**

```bash
# Video mode — sample N frames from a video
RUN_NAME=calib_junction_01 CAMERA_ID=junction_01 \
    VIDEO=data/JAAD_clips/video_0297.mp4 N_FRAMES=8 \
    bash scripts/run_calibrate.sh

# Or: python directly
python -m embed_traffic.calibration \
    --video data/JAAD_clips/video_0297.mp4 \
    --camera-id junction_01 --n-frames 8
```

Output → `configs/cameras/junction_01.json`.

**Using the calibration in TIM:**

```python
from embed_traffic.inference import TIM

tim = TIM(camera_calibration="configs/cameras/junction_01.json")
out = tim.process_frame(frame, frame_id=0)
for p in out.pedestrians:
    print(p.ped_id, p.position_m_ground, p.speed_m_s)
```

### 8.3 What's in the JSON

```json
{
  "camera_id": "junction_01",
  "image_size": [1920, 1080],
  "intrinsics": {"fx": 1663.0, "fy": 1663.0, "cx": 960.0, "cy": 540.0},
  "extrinsics": {
    "plane_normal_cam": [0.02, -0.98, -0.18],
    "plane_offset_cam": 3.15,
    "R_cam_to_ground": [[...]],
    "t_cam_to_ground": [0.0, 3.15, 0.0],
    "pitch_deg": 10.2,
    "roll_deg": 0.4,
    "camera_height_m": 3.15
  },
  "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
  "source_frames": 8,
  "created_at": "2026-04-22T12:34:56+00:00",
  "notes": "Auto-calibrated with 8 frames..."
}
```

### 8.4 When to re-calibrate

- Camera is physically moved or repositioned.
- Camera lens / zoom changes (intrinsics invalidated).
- Ground surface changes significantly (e.g., road re-graded).

Otherwise the calibration is stable indefinitely — it's just geometry.

### 8.5 Known limits

- RANSAC needs a visible ground region. If the camera points too steeply
  downward, or the road is fully occluded by foreground objects, calibration
  will fail. Re-run with clearer frames.
- Depth model is *approximate* metric — expect ~5–15% error on
  `camera_height_m`. Good enough for TTC decisions; not good enough for
  centimeter-level mapping.
- HFOV assumption (default 60°) may be off for wide-angle pole cams. Set
  `--hfov-deg` to your lens spec or provide full intrinsics programmatically.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Detector weights not found` | `ped_dashcam` run hasn't been trained | `bash scripts/run_train_dashcam.sh` or pass `detector_weights_path=` |
| `crossing_intent=None` forever | Track is <15 frames long, or intent weights missing | Lengthen the clip, or check `tim.has_intent_model` |
| `predicted_path=None` | Track is <5 frames long | Wait a few frames after first detection |
| Zero pedestrians for many frames | Scene has no people, or detector confidence threshold too high | Check with a demo render; adjust `imgsz` / retrain with wider data |
| `libcudnn.so.9: cannot open shared object file` | NVIDIA libs not on `LD_LIBRARY_PATH` | `source scripts/_common.sh` or run via `bash scripts/run_tim.sh` |
