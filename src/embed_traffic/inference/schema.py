"""TIM output schema — the contract TDM consumes.

Per the dashcam-only decision (plan.txt [VII]), TIM does NOT emit traffic-light
state. Downstream TDM should use TTC + crossing_intent instead.

Wire format: JSON. Dataclasses are JSON-serializable via `serialize_frame_output`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Per-pedestrian record
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class PedestrianInfo:
    """Everything TIM knows about a single pedestrian in a single frame.

    All coordinates are in pixel space of the input frame unless noted.
    """

    # Identity
    ped_id: int                              # persistent tracker ID across frames

    # Localization
    bbox: List[float]                        # [x1, y1, x2, y2]
    center: List[float]                      # [cx, cy] — bbox center
    confidence: float                        # detector confidence [0, 1]

    # Kinematics (pixel space, per second)
    speed_px_s: float                        # instantaneous speed
    avg_speed_px_s: float                    # average speed over tracked history
    direction: List[float]                   # [dx, dy] — unit direction vector
    track_length: int                        # frames this ped_id has been tracked

    # Intent (may be None if track is too short)
    crossing_intent: Optional[str] = None    # "crossing" | "not-crossing" | None
    crossing_prob: Optional[float] = None    # P(crossing) ∈ [0, 1]

    # Motion prediction (list of [cx, cy] future positions, may be None)
    predicted_path: Optional[List[List[float]]] = None

    # World-space fields — populated only when TIM has a CameraCalibration loaded.
    # (See plan.txt §X and src/embed_traffic/calibration/.)
    position_m_ground: Optional[List[float]] = None   # [X, Z] in meters on ground plane
    speed_m_s: Optional[float] = None                  # scalar ground speed in m/s
    velocity_m_s: Optional[List[float]] = None         # [vx, vz] ground velocity in m/s


# ────────────────────────────────────────────────────────────────────────────
# Per-frame output
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TIMFrameOutput:
    """TIM output for a single frame."""

    frame_id: int
    pedestrians: List[PedestrianInfo] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    frame_time_s: float = 0.0          # time within video: frame_id / fps
    processing_time_ms: float = 0.0    # wall-clock latency for this frame

    @property
    def num_pedestrians(self) -> int:
        return len(self.pedestrians)


# ────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ────────────────────────────────────────────────────────────────────────────

def serialize_frame_output(out: TIMFrameOutput) -> str:
    """Serialize to JSON string (compact). Used over the wire to TDM."""
    return json.dumps(asdict(out))


def deserialize_frame_output(payload: str) -> TIMFrameOutput:
    """Parse a JSON payload back into a TIMFrameOutput."""
    data = json.loads(payload)
    peds = [PedestrianInfo(**p) for p in data.pop("pedestrians", [])]
    return TIMFrameOutput(pedestrians=peds, **data)
