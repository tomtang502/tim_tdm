"""Camera calibration schema.

A `CameraCalibration` captures the geometry needed to project pixel coordinates
onto the ground plane. Persisted as JSON under `configs/cameras/<camera_id>.json`
and passed to `TIM` at init time.

Conventions (OpenCV camera frame):
    X: right, Y: down, Z: forward (into scene)

Ground frame (Y up, image-right = world-right):
    X: right (matches camera X at zero roll),  Y: up (ground normal),  Z: forward
    Camera sits at (0, camera_height_m, 0) looking along +Z with pitch rotation.

    Note: this is a left-handed frame (Y × Z = −X). We prefer it because it
    keeps "right" the same in both image space and world space, which is the
    convention used by most driving / AR top-down views. If you need a
    right-handed world frame, negate the X column of R_cam_to_ground.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_fov(cls, image_w: int, image_h: int, hfov_deg: float = 60.0) -> "Intrinsics":
        """Approximate intrinsics from horizontal field of view."""
        fx = 0.5 * image_w / np.tan(np.deg2rad(hfov_deg) / 2.0)
        fy = fx  # assume square pixels
        return cls(fx=fx, fy=fy, cx=image_w / 2.0, cy=image_h / 2.0)


@dataclass
class GroundExtrinsics:
    """Rigid transform between camera frame and ground frame.

    plane_normal_cam:   unit normal to ground in camera frame (points up ≈ -Y_cam)
    plane_offset_cam:   distance from camera origin to ground plane along normal
                         (equals camera_height_m)
    R_cam_to_ground:    3x3 rotation
    t_cam_to_ground:    translation (camera position in ground frame)
    pitch_deg:          downward pitch of camera (0 = horizontal, +ve = looking down)
    roll_deg:           in-plane rotation estimate
    camera_height_m:    camera height above ground (metric)
    """

    plane_normal_cam: Tuple[float, float, float]
    plane_offset_cam: float
    R_cam_to_ground: list[list[float]]  # 3x3
    t_cam_to_ground: Tuple[float, float, float]
    pitch_deg: float
    roll_deg: float
    camera_height_m: float


@dataclass
class CameraCalibration:
    camera_id: str
    image_size: Tuple[int, int]  # (W, H)
    intrinsics: Intrinsics
    extrinsics: GroundExtrinsics
    depth_model: str
    source_frames: int
    created_at: str
    notes: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        data = json.loads(Path(path).read_text())
        data["intrinsics"] = Intrinsics(**data["intrinsics"])
        data["extrinsics"] = GroundExtrinsics(**data["extrinsics"])
        data["image_size"] = tuple(data["image_size"])
        return cls(**data)

    # ─────────────────────────────────────────────────────────────────
    # Geometry — the hot path used by TIM at inference time
    # ─────────────────────────────────────────────────────────────────

    def pixel_to_ground(
        self,
        uv: np.ndarray,
        mask_above_horizon: bool = True,
    ) -> np.ndarray:
        """Project image pixels to ground-plane (X, Z) in meters.

        Args:
            uv: shape (N, 2) array of pixel coordinates.
            mask_above_horizon: if True, pixels whose ray doesn't intersect the
                ground in front of the camera (e.g., sky pixels above the
                horizon) are returned as NaN so they can be filtered. If False,
                those pixels get a large negative-Z extrapolation.

        Returns:
            (N, 2) array of (X, Z) in meters. Z is distance forward along road,
            X is lateral. Y is 0 (on ground).
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        K_inv = np.linalg.inv(self.intrinsics.matrix())
        R = np.asarray(self.extrinsics.R_cam_to_ground, dtype=np.float64)
        t = np.asarray(self.extrinsics.t_cam_to_ground, dtype=np.float64)

        # Homogeneous pixel rays in camera frame
        ones = np.ones((uv.shape[0], 1), dtype=np.float64)
        pixels_h = np.concatenate([uv, ones], axis=1)       # (N, 3)
        rays_cam = (K_inv @ pixels_h.T).T                    # (N, 3)
        rays_ground = (R @ rays_cam.T).T                      # (N, 3)

        # Ray p(s) = t + s * rays_ground in the ground frame. The ground plane
        # is Y = 0. Solve: t[1] + s * rays_ground[:, 1] = 0 for s.
        #
        # With camera above ground (t[1] = +h > 0), a valid intersection
        # requires rays_ground[:, 1] < 0 (ray pointing downward) AND s > 0
        # (in front of the camera). Otherwise the pixel is above the horizon.
        denom = rays_ground[:, 1]
        safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        s = -t[1] / safe
        points = t + rays_ground * s[:, None]

        xz = points[:, [0, 2]]
        if mask_above_horizon:
            invalid = (s <= 0) | ~np.isfinite(s)
            xz[invalid] = np.nan
        return xz

    @classmethod
    def default_identity(
        cls, image_w: int, image_h: int, camera_id: str = "identity",
        camera_height_m: float = 2.5,
    ) -> "CameraCalibration":
        """Fallback calibration: camera at `camera_height_m` above ground, looking
        horizontally forward with zero roll. Useful for tests / smoke checks only.
        """
        K = Intrinsics.from_fov(image_w, image_h, 60.0)
        # Camera looks horizontally: plane normal in cam frame = (0, -1, 0) (up = -Y_cam).
        # Ground-X = camera-X (image-right ↔ world-right, left-handed ground frame),
        # Ground-Y = -camera-Y (up),  Ground-Z = camera-Z (forward).
        R_cam_to_ground = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0]]
        )
        ext = GroundExtrinsics(
            plane_normal_cam=(0.0, -1.0, 0.0),
            plane_offset_cam=float(camera_height_m),
            R_cam_to_ground=R_cam_to_ground.tolist(),
            t_cam_to_ground=(0.0, float(camera_height_m), 0.0),
            pitch_deg=0.0,
            roll_deg=0.0,
            camera_height_m=float(camera_height_m),
        )
        return cls(
            camera_id=camera_id,
            image_size=(image_w, image_h),
            intrinsics=K,
            extrinsics=ext,
            depth_model="identity",
            source_frames=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes="Synthetic default calibration; for tests only, not auto-calibrated.",
        )
