"""Unified data loader for PIE and JAAD datasets.

Loads frames on-the-fly from video files via OpenCV VideoCapture (no frame
extraction). Outputs a unified `PedestrianSample` format.

The PIE and JAAD loaders live in `datasets/PIE/utilities/` and `datasets/JAAD/`
respectively; this module inserts those paths into sys.path on import so that
`from pie_data import PIE` and `from jaad_data import JAAD` resolve.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from embed_traffic.paths import (
    DATA_DIR,
    JAAD_CACHE_DIR,
    JAAD_CLIPS_DIR,
    JAAD_CODE_DIR,
    PIE_CACHE_DIR,
    PIE_CLIPS_DIR,
    PIE_CODE_DIR,
)

# Wire vendored dataset loaders onto sys.path
_pie_util_dir = PIE_CODE_DIR / "utilities"
if _pie_util_dir.exists() and str(_pie_util_dir) not in sys.path:
    sys.path.insert(0, str(_pie_util_dir))
if JAAD_CODE_DIR.exists() and str(JAAD_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(JAAD_CODE_DIR))

from pie_data import PIE as _PIEBase  # noqa: E402
from jaad_data import JAAD as _JAADBase  # noqa: E402


class PIE(_PIEBase):
    """Vendored PIE loader with cache redirected outside the submodule."""

    @property
    def cache_path(self):  # type: ignore[override]
        PIE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return str(PIE_CACHE_DIR)


class JAAD(_JAADBase):
    """Vendored JAAD loader with cache redirected outside the submodule."""

    @property
    def cache_path(self):  # type: ignore[override]
        JAAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return str(JAAD_CACHE_DIR)


@dataclass
class PedestrianSample:
    """Unified pedestrian annotation for a single frame."""

    dataset: str          # 'pie' or 'jaad'
    video_id: str         # e.g. 'set01/video_0001' (PIE) or 'video_0001' (JAAD)
    frame_id: int         # 0-indexed frame number
    ped_id: str           # unique pedestrian id
    bbox: list            # [x1, y1, x2, y2]
    crossing_intent: int  # 1=crossing, 0=not-crossing, -1=irrelevant
    action: int           # 0=standing, 1=walking
    look: int             # 0=not-looking, 1=looking
    occlusion: int        # 0=none, 1=partial, 2=full
    traffic_light_state: Optional[int] = None
    vehicle_speed: Optional[float] = None   # OBD speed in km/h (PIE only)


class VideoReader:
    """Lazy OpenCV VideoCapture that seeks to specific frames on demand."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")

    def read_frame(self, frame_id: int) -> np.ndarray:
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_id} from {self.video_path}")
        return frame

    @property
    def frame_count(self) -> int:
        self._open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        self._open()
        return self._cap.get(cv2.CAP_PROP_FPS)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()


class UnifiedDataLoader:
    """Unified data loader for PIE + JAAD datasets.

    Uses the vendored PIE/JAAD annotation parsers (from the upstream repos) and
    reads video frames on-the-fly. PIE and JAAD *code* lives in `datasets/`;
    *video clips* live in `data/PIE_clips/` and `data/JAAD_clips/`.
    """

    def __init__(
        self,
        pie_code_dir: Optional[Path] = None,
        jaad_code_dir: Optional[Path] = None,
        pie_clips_dir: Optional[Path] = None,
        jaad_clips_dir: Optional[Path] = None,
    ):
        self.pie_code_dir = Path(pie_code_dir) if pie_code_dir else PIE_CODE_DIR
        self.jaad_code_dir = Path(jaad_code_dir) if jaad_code_dir else JAAD_CODE_DIR
        self.pie_clips_dir = Path(pie_clips_dir) if pie_clips_dir else PIE_CLIPS_DIR
        self.jaad_clips_dir = Path(jaad_clips_dir) if jaad_clips_dir else JAAD_CLIPS_DIR

        # Backwards-compat aliases used elsewhere in the codebase
        self.pie_dir = self.pie_code_dir
        self.jaad_dir = self.jaad_code_dir

        self._pie: Optional[PIE] = None
        self._jaad: Optional[JAAD] = None
        self._pie_db = None
        self._jaad_db = None
        self._video_readers: dict[str, VideoReader] = {}

    # --- Database generation ---

    def _load_pie_db(self):
        if self._pie_db is not None:
            return self._pie_db
        self._pie = PIE(data_path=str(self.pie_code_dir))
        self._pie_db = self._pie.generate_database()
        return self._pie_db

    def _load_jaad_db(self):
        if self._jaad_db is not None:
            return self._jaad_db
        self._jaad = JAAD(data_path=str(self.jaad_code_dir))
        self._jaad_db = self._jaad.generate_database()
        return self._jaad_db

    # --- Video reading ---

    def _get_video_reader(self, video_path: str) -> VideoReader:
        if video_path not in self._video_readers:
            self._video_readers[video_path] = VideoReader(video_path)
        return self._video_readers[video_path]

    def get_frame(self, dataset: str, video_id: str, frame_id: int) -> np.ndarray:
        """Read a single frame on-the-fly from the video file."""
        if dataset == "pie":
            parts = video_id.split("/")
            if len(parts) != 2:
                raise ValueError(
                    f"PIE video_id must be 'set_id/video_id', got: {video_id}"
                )
            set_id, vid = parts
            video_path = str(self.pie_clips_dir / set_id / f"{vid}.mp4")
        elif dataset == "jaad":
            video_path = str(self.jaad_clips_dir / f"{video_id}.mp4")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        reader = self._get_video_reader(video_path)
        return reader.read_frame(frame_id)

    # --- Sample generation ---

    def get_pie_samples(self, split: str = "all") -> list[PedestrianSample]:
        """Pedestrian samples from PIE. split: 'train', 'val', 'test', 'all'."""
        db = self._load_pie_db()
        set_map = {
            "train": ["set01", "set02", "set04"],
            "val": ["set05", "set06"],
            "test": ["set03"],
            "all": ["set01", "set02", "set03", "set04", "set05", "set06"],
        }
        target_sets = set_map[split]
        samples: list[PedestrianSample] = []

        for set_id in target_sets:
            if set_id not in db:
                continue
            for vid, vid_data in db[set_id].items():
                veh_data = vid_data.get("vehicle_annotations", {})

                tl_states: dict[int, int] = {}
                for _, obj_data in vid_data.get("traffic_annotations", {}).items():
                    if obj_data.get("obj_class") == "traffic_light":
                        for i, frame in enumerate(obj_data["frames"]):
                            tl_states[frame] = obj_data["state"][i]

                for ped_id, ped_data in vid_data.get("ped_annotations", {}).items():
                    behavior = ped_data.get("behavior", {})
                    cross_list = behavior.get("cross", [])
                    action_list = behavior.get("action", [])
                    look_list = behavior.get("look", [])

                    for i, frame in enumerate(ped_data["frames"]):
                        speed = None
                        if frame in veh_data and isinstance(veh_data[frame], dict):
                            speed = veh_data[frame].get("OBD_speed")

                        samples.append(
                            PedestrianSample(
                                dataset="pie",
                                video_id=f"{set_id}/{vid}",
                                frame_id=frame,
                                ped_id=ped_id,
                                bbox=ped_data["bbox"][i],
                                crossing_intent=(
                                    cross_list[i] if i < len(cross_list) else -1
                                ),
                                action=action_list[i] if i < len(action_list) else 0,
                                look=look_list[i] if i < len(look_list) else 0,
                                occlusion=ped_data["occlusion"][i],
                                traffic_light_state=tl_states.get(frame),
                                vehicle_speed=speed,
                            )
                        )

        return samples

    def get_jaad_samples(
        self, split: str = "all", subset: str = "default"
    ) -> list[PedestrianSample]:
        """Pedestrian samples from JAAD. split: 'train', 'val', 'test', 'all'."""
        db = self._load_jaad_db()

        if split == "all":
            target_vids = set(db.keys())
        else:
            split_file = self.jaad_code_dir / "split_ids" / subset / f"{split}.txt"
            with open(split_file, "r") as f:
                target_vids = {x.strip() for x in f.readlines()}

        samples: list[PedestrianSample] = []
        for vid, vid_data in db.items():
            if vid not in target_vids:
                continue

            traffic_data = vid_data.get("traffic_annotations", {})

            for ped_id, ped_data in vid_data.get("ped_annotations", {}).items():
                behavior = ped_data.get("behavior", {})
                if not behavior:
                    continue

                cross_list = behavior.get("cross", [])
                action_list = behavior.get("action", [])
                look_list = behavior.get("look", [])

                for i, frame in enumerate(ped_data["frames"]):
                    tl_state = None
                    if isinstance(traffic_data, dict) and frame in traffic_data:
                        tl_state = traffic_data[frame].get("traffic_light")

                    samples.append(
                        PedestrianSample(
                            dataset="jaad",
                            video_id=vid,
                            frame_id=frame,
                            ped_id=ped_id,
                            bbox=ped_data["bbox"][i],
                            crossing_intent=(
                                cross_list[i] if i < len(cross_list) else -1
                            ),
                            action=action_list[i] if i < len(action_list) else 0,
                            look=look_list[i] if i < len(look_list) else 0,
                            occlusion=ped_data["occlusion"][i],
                            traffic_light_state=tl_state,
                            vehicle_speed=None,
                        )
                    )

        return samples

    def get_all_samples(self, split: str = "all") -> list[PedestrianSample]:
        """Combined samples from both PIE and JAAD."""
        samples: list[PedestrianSample] = []
        if self.pie_code_dir.exists():
            samples.extend(self.get_pie_samples(split))
        if self.jaad_code_dir.exists():
            samples.extend(self.get_jaad_samples(split))
        return samples

    # --- Trajectory helpers ---

    def get_pedestrian_trajectory(
        self, samples: list[PedestrianSample], ped_id: str
    ) -> list[PedestrianSample]:
        """All frames for a specific pedestrian, sorted by frame_id."""
        traj = [s for s in samples if s.ped_id == ped_id]
        traj.sort(key=lambda s: s.frame_id)
        return traj

    def compute_speed(
        self, trajectory: list[PedestrianSample], fps: float = 30.0
    ) -> list[float]:
        """Pixel-space speed (px/s) per frame from bbox center displacement."""
        speeds = [0.0]
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            cx_prev = (prev.bbox[0] + prev.bbox[2]) / 2
            cy_prev = (prev.bbox[1] + prev.bbox[3]) / 2
            cx_curr = (curr.bbox[0] + curr.bbox[2]) / 2
            cy_curr = (curr.bbox[1] + curr.bbox[3]) / 2
            dt = (curr.frame_id - prev.frame_id) / fps
            if dt > 0:
                dist = np.sqrt((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2)
                speeds.append(dist / dt)
            else:
                speeds.append(0.0)
        return speeds

    def release_all(self) -> None:
        """Release all open video captures."""
        for reader in self._video_readers.values():
            reader.release()
        self._video_readers.clear()


def export_yolo_labels(
    samples: list[PedestrianSample],
    img_width: int = 1920,
    img_height: int = 1080,
) -> dict[tuple[str, str, int], list[tuple[int, float, float, float, float]]]:
    """Convert samples to YOLO-format labels keyed by (dataset, video_id, frame_id)."""
    labels: dict[tuple[str, str, int], list[tuple[int, float, float, float, float]]] = {}
    for s in samples:
        key = (s.dataset, s.video_id, s.frame_id)
        x1, y1, x2, y2 = s.bbox
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        labels.setdefault(key, []).append((0, x_center, y_center, w, h))
    return labels


def main() -> None:
    """Smoke test: load a few samples from each dataset."""
    loader = UnifiedDataLoader()

    print("Loading JAAD annotations...")
    jaad_samples = loader.get_jaad_samples(split="train")
    print(f"  JAAD train samples: {len(jaad_samples)}")
    if jaad_samples:
        s = jaad_samples[0]
        print(
            f"  First sample: video={s.video_id}, frame={s.frame_id}, "
            f"ped={s.ped_id}, bbox={s.bbox}, cross={s.crossing_intent}"
        )

    print("\nLoading PIE annotations...")
    pie_samples = loader.get_pie_samples(split="train")
    print(f"  PIE train samples: {len(pie_samples)}")
    if pie_samples:
        s = pie_samples[0]
        print(
            f"  First sample: video={s.video_id}, frame={s.frame_id}, "
            f"ped={s.ped_id}, bbox={s.bbox}, cross={s.crossing_intent}, "
            f"speed={s.vehicle_speed}"
        )

    print(f"\nTotal combined samples: {len(jaad_samples) + len(pie_samples)}")
    loader.release_all()


if __name__ == "__main__":
    main()
