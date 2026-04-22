"""Ground-plane extraction from a metric depth map.

Given one (or more) metric depth maps and intrinsics, this module fits a plane
to the "ground" region via RANSAC and derives camera height + pitch.
"""

from __future__ import annotations

import numpy as np

from embed_traffic.calibration.schema import (
    GroundExtrinsics,
    Intrinsics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backprojection
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_point_cloud(
    depth: np.ndarray,
    K: Intrinsics,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject a depth map into 3D camera-frame points.

    Args:
        depth: (H, W) metric depth in meters.
        K: intrinsics.
        valid_mask: (H, W) bool array; if provided, only those pixels are kept.
    Returns:
        (N, 3) array of (X, Y, Z) camera-frame points.
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    if valid_mask is None:
        valid_mask = np.ones_like(depth, dtype=bool)
    valid_mask = valid_mask & (depth > 0) & np.isfinite(depth)

    uu = uu[valid_mask].astype(np.float64)
    vv = vv[valid_mask].astype(np.float64)
    d = depth[valid_mask].astype(np.float64)

    X = (uu - K.cx) * d / K.fx
    Y = (vv - K.cy) * d / K.fy
    Z = d
    return np.stack([X, Y, Z], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC plane fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_plane_ransac(
    points: np.ndarray,
    n_iterations: int = 500,
    inlier_threshold_m: float = 0.05,
    min_inliers: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane n·p + d = 0 to a point cloud via RANSAC.

    Args:
        points: (N, 3) point cloud.
        n_iterations: RANSAC iterations.
        inlier_threshold_m: max orthogonal distance for inlier classification.
        min_inliers: minimum inlier count to consider the fit valid.
    Returns:
        n_hat: unit normal (3,)
        d:     offset scalar (so n_hat · p + d = 0 for on-plane points)
        inliers: (M, 3) inlier points.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = points.shape[0]
    if N < 3:
        raise ValueError(f"Need at least 3 points, got {N}")

    best_n = None
    best_d = None
    best_inlier_count = -1
    best_inlier_mask = None

    for _ in range(n_iterations):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -np.dot(normal, p1)

        dists = np.abs(points @ normal + offset)
        inliers = dists < inlier_threshold_m
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_n = normal
            best_d = offset
            best_inlier_mask = inliers

    if best_n is None or best_inlier_count < min_inliers:
        raise RuntimeError(
            f"RANSAC failed to find a plane with ≥{min_inliers} inliers "
            f"(best={best_inlier_count})."
        )

    # Refine with least squares on the inliers
    inlier_pts = points[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    refined_normal = Vt[-1]
    refined_normal = refined_normal / np.linalg.norm(refined_normal)
    refined_offset = -float(refined_normal @ centroid)

    # Orient normal so it points "up" in camera frame (negative Y component)
    if refined_normal[1] > 0:
        refined_normal = -refined_normal
        refined_offset = -refined_offset

    return refined_normal, refined_offset, inlier_pts


# ─────────────────────────────────────────────────────────────────────────────
# Ground mask heuristic
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_ground_mask(
    depth: np.ndarray,
    top_frac: float = 0.45,
    bottom_frac: float = 1.0,
    min_depth: float = 1.0,
    max_depth: float = 60.0,
) -> np.ndarray:
    """Heuristic mask of 'likely ground' pixels.

    Keeps the bottom band of the image (where the road usually sits) and
    filters out extreme depths. Cheap and works well for road-facing cameras.
    """
    H, W = depth.shape
    mask = np.zeros_like(depth, dtype=bool)
    y_start = int(H * top_frac)
    y_end = int(H * bottom_frac)
    mask[y_start:y_end] = True
    mask &= (depth > min_depth) & (depth < max_depth)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Extrinsics from plane
# ─────────────────────────────────────────────────────────────────────────────

def extrinsics_from_plane(
    plane_normal_cam: np.ndarray,
    plane_offset_cam: float,
) -> GroundExtrinsics:
    """Build a ground-frame transform from the camera-frame ground plane.

    We define the ground frame such that:
      - Ground is Y = 0.
      - Camera Z roughly aligns with ground Z (forward).
      - Ground Y axis = -plane_normal_cam (world up).
    """
    n = plane_normal_cam / np.linalg.norm(plane_normal_cam)
    camera_height_m = float(abs(plane_offset_cam))

    # Ground Y axis in camera frame = -n (since n points from ground up into camera frame
    # so -n points from camera down to ground; world up = -n reversed = -n).
    # Actually: n is the plane normal pointing toward the camera (upward), so "up" in
    # world = n. We'll use world-up = n (unit vector from ground plane to camera).
    y_world_in_cam = n / np.linalg.norm(n)

    # Forward direction (Z_world_in_cam): project camera Z axis onto the plane
    z_cam_axis = np.array([0.0, 0.0, 1.0])
    z_world_in_cam = z_cam_axis - np.dot(z_cam_axis, y_world_in_cam) * y_world_in_cam
    z_world_in_cam = z_world_in_cam / np.linalg.norm(z_world_in_cam)

    # Right direction (X_world_in_cam): we want ground-X to match camera-X
    # (image-right ↔ world-right). With Y_world = up and Z_world = forward, the
    # consistent "right" direction in the camera frame is Z × Y (this produces
    # a left-handed world frame, which is the visually intuitive convention for
    # top-down views — like most driving/AR systems).
    x_world_in_cam = np.cross(z_world_in_cam, y_world_in_cam)
    x_world_in_cam = x_world_in_cam / np.linalg.norm(x_world_in_cam)

    # R_world_in_cam has columns [X_w_in_c, Y_w_in_c, Z_w_in_c]
    R_world_in_cam = np.stack([x_world_in_cam, y_world_in_cam, z_world_in_cam], axis=1)
    # R_cam_to_ground rotates camera-frame vectors to ground-frame. This is R_world_in_cam^T.
    R_cam_to_ground = R_world_in_cam.T

    # Translation: camera position in ground frame. Camera is at height camera_height_m above
    # ground (Y_world = camera_height_m), at ground-frame origin in X/Z.
    t_cam_to_ground = np.array([0.0, camera_height_m, 0.0])

    # Derive pitch: angle between camera Z axis and ground XZ plane
    # pitch_deg > 0 when camera looks downward.
    cam_z_in_ground = R_cam_to_ground @ z_cam_axis
    horiz_norm = np.linalg.norm([cam_z_in_ground[0], cam_z_in_ground[2]])
    pitch_rad = np.arctan2(-cam_z_in_ground[1], horiz_norm)
    pitch_deg = float(np.rad2deg(pitch_rad))

    # Roll: rotation of camera-X about camera-Z. Measured as angle between
    # (R_cam_to_ground @ e_x) projected onto ground XZ and the ground X axis.
    cam_x_in_ground = R_cam_to_ground @ np.array([1.0, 0.0, 0.0])
    roll_rad = np.arctan2(cam_x_in_ground[1], cam_x_in_ground[0])
    roll_deg = float(np.rad2deg(roll_rad))

    return GroundExtrinsics(
        plane_normal_cam=tuple(float(x) for x in n),
        plane_offset_cam=float(plane_offset_cam),
        R_cam_to_ground=R_cam_to_ground.tolist(),
        t_cam_to_ground=tuple(float(x) for x in t_cam_to_ground),
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        camera_height_m=camera_height_m,
    )
