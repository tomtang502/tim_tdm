"""Camera calibration package.

One-time, install-time calibration from a handful of frames. Uses a monocular
depth model (Depth-Anything-V2-Metric by default) to extract the ground plane,
then saves a `CameraCalibration` JSON that TIM loads at init to emit real-world
positions and speeds.

Entry points:
    calibrate(frames, camera_id, ...)          → CameraCalibration
    python -m embed_traffic.calibration ...    → CLI writing configs/cameras/<id>.json
"""

from embed_traffic.calibration.schema import (
    CameraCalibration,
    GroundExtrinsics,
    Intrinsics,
)
from embed_traffic.calibration.calibrate import calibrate

__all__ = [
    "CameraCalibration",
    "GroundExtrinsics",
    "Intrinsics",
    "calibrate",
]
