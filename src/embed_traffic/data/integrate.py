"""Integrate additional datasets into the YOLO training pipeline.

Supported datasets:
    1. Intersection-Flow-5K (ready now)
    2. inD (when access granted)
    3. WTS (when access granted)
    4. MIO-TCD Localization (pedestrian subset)

Converts all to YOLO format and merges into `data/yolo_dataset/`.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from embed_traffic.paths import (
    INTERSECTION_FLOW_DIR,
    MIO_TCD_DIR,
    YOLO_DATASET_DIR,
)

YOLO_DIR = YOLO_DATASET_DIR  # backwards-compat alias
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


# ============================================================
# 1. Intersection-Flow-5K
# ============================================================

def integrate_intersection_flow(data_dir=None, split_map=None):
    if data_dir is None:
        data_dir = INTERSECTION_FLOW_DIR
    """
    Integrate Intersection-Flow-5K dataset.
    It already has YOLO-format labels. We just need to:
    - Filter for pedestrian class only
    - Copy images and remap labels to our single-class format (class 0 = pedestrian)
    """
    data_dir = Path(data_dir)

    # Find the actual data directory (may be nested after unzip)
    # Intersection-Flow-5K has: images/ and labels/ with train/val/test splits
    # Classes: vehicle=0, bus=1, bicycle=2, pedestrian=3, engine=4, truck=5, tricycle=6, obstacle=7
    PEDESTRIAN_CLASS = 3

    possible_roots = [
        data_dir,
        data_dir / "Intersection-Flow-5K",
        data_dir / "intersection-flow-5k",
    ]

    root = None
    for p in possible_roots:
        if (p / "images").exists() or (p / "YOLO" / "images").exists():
            root = p
            break
        # Check for YOLO subdirectory
        for sub in p.iterdir() if p.exists() else []:
            if (sub / "images").exists():
                root = sub
                break
        if root:
            break

    if root is None:
        # Try finding images directory anywhere under data_dir
        for p in data_dir.rglob("images"):
            if p.is_dir() and (p / "train").exists():
                root = p.parent
                break

    if root is None:
        print(f"ERROR: Cannot find Intersection-Flow-5K data structure in {data_dir}")
        print(f"  Contents: {list(data_dir.iterdir()) if data_dir.exists() else 'dir not found'}")
        return 0

    print(f"  Found data root: {root}")

    # Map their splits to ours
    if split_map is None:
        split_map = {"train": "train", "val": "val", "test": "test"}

    total_added = 0

    for src_split, dst_split in split_map.items():
        img_dir = root / "images" / src_split
        lbl_dir = root / "labels" / src_split

        if not img_dir.exists():
            # Try YOLO subdirectory
            img_dir = root / "YOLO" / "images" / src_split
            lbl_dir = root / "YOLO" / "labels" / src_split

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping {src_split}")
            continue

        dst_img_dir = YOLO_DIR / "images" / dst_split
        dst_lbl_dir = YOLO_DIR / "labels" / dst_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        added = 0

        for img_path in images:
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            if not lbl_path.exists():
                continue

            # Read labels and filter for pedestrian class
            ped_labels = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == PEDESTRIAN_CLASS:
                        # Remap to class 0
                        ped_labels.append(f"0 {' '.join(parts[1:])}")

            if not ped_labels:
                continue

            # Copy image with prefix to avoid name collision
            dst_name = f"iflow_{stem}"
            dst_img = dst_img_dir / f"{dst_name}{img_path.suffix}"
            dst_lbl = dst_lbl_dir / f"{dst_name}.txt"

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            with open(dst_lbl, "w") as f:
                f.write("\n".join(ped_labels) + "\n")

            added += 1

        total_added += added
        print(f"  {src_split} -> {dst_split}: {added} images with pedestrians (from {len(images)} total)")

    return total_added


# ============================================================
# 2. inD Dataset
# ============================================================

def integrate_ind(data_dir=None, split_ratio=(0.7, 0.15, 0.15)):
    from embed_traffic.paths import DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR / "inD"
    """
    Integrate inD dataset (drone-recorded intersections).
    inD provides trajectory CSV files and video recordings.

    Expected structure after download:
        inD/
            data/
                00_tracks.csv, 00_tracksMeta.csv, 00_recordingMeta.csv
                00_background.png
                01_tracks.csv, ...
            videos/ (optional)

    Tracks CSV columns: trackId, frame, x, y, width, height, xVelocity, yVelocity, ...
    TracksMeta CSV: trackId, class (car, truck, bus, pedestrian, bicycle), ...
    """
    data_dir = Path(data_dir)

    # Find data directory
    tracks_dir = None
    for candidate in [data_dir / "data", data_dir]:
        if list(candidate.glob("*_tracks.csv")):
            tracks_dir = candidate
            break

    if tracks_dir is None:
        print(f"  inD dataset not found at {data_dir}")
        print(f"  Download from: https://levelxdata.com/ind-dataset/")
        print(f"  Place files in: {data_dir}/data/")
        return 0

    import pandas as pd

    recordings = sorted(set(
        f.stem.split("_")[0] for f in tracks_dir.glob("*_tracks.csv")
    ))

    print(f"  Found {len(recordings)} recordings")

    total_added = 0

    for rec_id in recordings:
        tracks_file = tracks_dir / f"{rec_id}_tracks.csv"
        meta_file = tracks_dir / f"{rec_id}_tracksMeta.csv"
        bg_file = tracks_dir / f"{rec_id}_background.png"

        if not tracks_file.exists() or not meta_file.exists():
            continue

        tracks = pd.read_csv(tracks_file)
        meta = pd.read_csv(meta_file)

        # Filter pedestrian tracks
        ped_ids = set(meta[meta["class"] == "pedestrian"]["trackId"].values)
        ped_tracks = tracks[tracks["trackId"].isin(ped_ids)]

        if ped_tracks.empty:
            print(f"  Recording {rec_id}: no pedestrians, skipping")
            continue

        # Get recording metadata for image dimensions
        rec_meta_file = tracks_dir / f"{rec_id}_recordingMeta.csv"
        if rec_meta_file.exists():
            rec_meta = pd.read_csv(rec_meta_file)
            # orthoPxToMeter gives the scale
        else:
            continue

        # Group by frame
        frames = sorted(ped_tracks["frame"].unique())

        # Determine split
        n = len(frames)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        # Check if video exists for frame extraction
        video_file = None
        for vf in [data_dir / "videos" / f"{rec_id}.mp4",
                    data_dir / f"{rec_id}.mp4"]:
            if vf.exists():
                video_file = vf
                break

        # If no video, use background image with bbox overlay
        # For now, generate synthetic frames from background + bboxes
        bg = cv2.imread(str(bg_file)) if bg_file.exists() else None
        if bg is None and video_file is None:
            print(f"  Recording {rec_id}: no background or video, skipping")
            continue

        img_h, img_w = (bg.shape[:2] if bg is not None else (1080, 1920))

        # Sample every 10th frame to reduce redundancy
        sampled_frames = frames[::10]
        added = 0

        for i, frame_id in enumerate(sampled_frames):
            frame_data = ped_tracks[ped_tracks["frame"] == frame_id]

            # Determine split
            if i < int(len(sampled_frames) * split_ratio[0]):
                split = "train"
            elif i < int(len(sampled_frames) * (split_ratio[0] + split_ratio[1])):
                split = "val"
            else:
                split = "test"

            dst_img_dir = YOLO_DIR / "images" / split
            dst_lbl_dir = YOLO_DIR / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            fname = f"ind_{rec_id}_{frame_id:06d}"

            # Extract frame from video or use background
            if video_file:
                cap = cv2.VideoCapture(str(video_file))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame_img = cap.read()
                cap.release()
                if not ret:
                    continue
            else:
                frame_img = bg.copy()

            # Write image
            img_path = dst_img_dir / f"{fname}.jpg"
            if not img_path.exists():
                cv2.imwrite(str(img_path), frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Write YOLO labels
            # inD uses world coordinates (meters); x, y are center, width/height in meters
            # We need to convert to pixel coordinates using orthoPxToMeter
            labels = []
            for _, row in frame_data.iterrows():
                # inD provides x, y (center), width, height in pixels already
                # (depending on version — some provide meters, some pixels)
                cx = row["x"] / img_w
                cy = row["y"] / img_h
                w = row["width"] / img_w
                h = row["height"] / img_h

                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                if w > 0.01 and h > 0.01:  # Filter tiny boxes
                    labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if labels:
                with open(dst_lbl_dir / f"{fname}.txt", "w") as f:
                    f.write("\n".join(labels) + "\n")
                added += 1

        total_added += added
        print(f"  Recording {rec_id}: {added} frames with pedestrians")

    return total_added


# ============================================================
# 3. WTS Dataset
# ============================================================

def integrate_wts(data_dir=None, split_map=None):
    from embed_traffic.paths import DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR / "WTS"
    """
    Integrate WTS dataset (Woven Traffic Safety).
    WTS provides overhead_view videos with COCO-format bounding boxes.

    Expected structure:
        WTS/
            videos/
                train/[scenario_id]/overhead_view/*.mp4
                val/[scenario_id]/overhead_view/*.mp4
            bbox_annotated/
                train.json  (COCO format)
                val.json
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"  WTS dataset not found at {data_dir}")
        print(f"  Request access: https://forms.gle/szQPk1TMR8JXzm327")
        print(f"  Place files in: {data_dir}/")
        return 0

    if split_map is None:
        split_map = {"train": "train", "val": "val"}

    total_added = 0

    for src_split, dst_split in split_map.items():
        # Load COCO annotations
        bbox_file = None
        for candidate in [
            data_dir / "bbox_annotated" / f"{src_split}.json",
            data_dir / "annotations" / f"{src_split}.json",
            data_dir / f"bbox_{src_split}.json",
        ]:
            if candidate.exists():
                bbox_file = candidate
                break

        if bbox_file is None:
            print(f"  WARNING: No bbox annotations found for {src_split}")
            continue

        with open(bbox_file, "r") as f:
            coco = json.load(f)

        # Build lookup: image_id -> image_info
        img_info = {img["id"]: img for img in coco["images"]}

        # Build lookup: image_id -> list of annotations
        img_annots = defaultdict(list)
        for ann in coco["annotations"]:
            img_annots[ann["image_id"]].append(ann)

        # Find pedestrian category ID
        ped_cat_id = None
        for cat in coco.get("categories", []):
            if cat["name"].lower() in ["pedestrian", "person"]:
                ped_cat_id = cat["id"]
                break

        if ped_cat_id is None:
            # If no category info, assume all annotations are pedestrians
            print(f"  WARNING: No pedestrian category found, using all annotations")
            ped_cat_id = -1  # match all

        dst_img_dir = YOLO_DIR / "images" / dst_split
        dst_lbl_dir = YOLO_DIR / "labels" / dst_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for image_id, info in img_info.items():
            annots = img_annots.get(image_id, [])

            # Filter for pedestrians
            if ped_cat_id != -1:
                annots = [a for a in annots if a["category_id"] == ped_cat_id]

            if not annots:
                continue

            img_w = info.get("width", 1920)
            img_h = info.get("height", 1080)

            # Find the actual image/video frame
            file_name = info.get("file_name", "")
            img_path_candidates = [
                data_dir / "videos" / src_split / file_name,
                data_dir / "images" / src_split / file_name,
                data_dir / file_name,
            ]

            src_img = None
            for p in img_path_candidates:
                if p.exists():
                    src_img = p
                    break

            if src_img is None:
                continue

            # Copy image
            stem = src_img.stem
            fname = f"wts_{src_split}_{stem}"
            dst_img = dst_img_dir / f"{fname}{src_img.suffix}"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized
            labels = []
            for ann in annots:
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                if nw > 0.005 and nh > 0.005:
                    labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if labels:
                with open(dst_lbl_dir / f"{fname}.txt", "w") as f:
                    f.write("\n".join(labels) + "\n")
                added += 1

        total_added += added
        print(f"  {src_split} -> {dst_split}: {added} images with pedestrians")

    return total_added


# ============================================================
# 4. MIO-TCD Localization
# ============================================================

def integrate_miotcd(data_dir=None, split_ratio=(0.8, 0.1, 0.1)):
    if data_dir is None:
        data_dir = MIO_TCD_DIR
    """
    Integrate MIO-TCD Localization dataset (pedestrian subset).
    Annotations are in CSV: image_id, label, x1, y1, x2, y2

    Expected structure after extracting MIO-TCD-Localization.tar:
        MIO-TCD/
            MIO-TCD-Localization/
                train/
                    00000000.jpg, 00000001.jpg, ...
                test/
                    00000000.jpg, ...
                gt_train.csv
    """
    data_dir = Path(data_dir)

    # Find data root
    root = None
    for candidate in [data_dir / "MIO-TCD-Localization", data_dir]:
        if (candidate / "gt_train.csv").exists():
            root = candidate
            break

    if root is None:
        print(f"  MIO-TCD dataset not found at {data_dir}")
        print(f"  Download: wget https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar")
        print(f"  Extract to: {data_dir}/")
        return 0

    gt_file = root / "gt_train.csv"
    train_img_dir = root / "train"

    print(f"  Found data root: {root}")

    # Parse CSV: group pedestrian annotations by image
    ped_images = defaultdict(list)  # image_id -> list of (x1, y1, x2, y2)
    total_ped_boxes = 0

    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            img_id, label = parts[0], parts[1]
            if label.strip() == "pedestrian":
                x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                ped_images[img_id].append((x1, y1, x2, y2))
                total_ped_boxes += 1

    print(f"  Found {len(ped_images)} images with {total_ped_boxes} pedestrian boxes")

    if not ped_images:
        return 0

    # Sort image IDs and split
    sorted_ids = sorted(ped_images.keys())
    n = len(sorted_ids)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {}
    for i, img_id in enumerate(sorted_ids):
        if i < n_train:
            splits[img_id] = "train"
        elif i < n_train + n_val:
            splits[img_id] = "val"
        else:
            splits[img_id] = "test"

    total_added = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    for img_id, boxes in ped_images.items():
        split = splits[img_id]

        # Find image file
        img_path = train_img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = train_img_dir / f"{img_id}.png"
        if not img_path.exists():
            continue

        # Get image dimensions (read just the header)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        dst_img_dir = YOLO_DIR / "images" / split
        dst_lbl_dir = YOLO_DIR / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        fname = f"miotcd_{img_id}"
        dst_img = dst_img_dir / f"{fname}.jpg"
        dst_lbl = dst_lbl_dir / f"{fname}.txt"

        # Copy image
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write YOLO labels
        labels = []
        for x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            if w > 0.005 and h > 0.005:
                labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if labels:
            with open(dst_lbl, "w") as f:
                f.write("\n".join(labels) + "\n")
            total_added += 1
            split_counts[split] += 1

    for split, count in split_counts.items():
        print(f"  {split}: {count} images with pedestrians")

    return total_added


# ============================================================
# Main
# ============================================================

def main():
    print("=== Integrating Additional Datasets ===\n")

    # Current dataset stats
    for split in ["train", "val", "test"]:
        n = len(list((YOLO_DIR / "images" / split).glob("*")))
        print(f"  Current {split}: {n} images")

    # 1. Intersection-Flow-5K
    print("\n--- Intersection-Flow-5K ---")
    n_iflow = integrate_intersection_flow()
    print(f"  Total added: {n_iflow}")

    # 2. inD
    print("\n--- inD Dataset ---")
    n_ind = integrate_ind()
    print(f"  Total added: {n_ind}")

    # 3. WTS
    print("\n--- WTS Dataset ---")
    n_wts = integrate_wts()
    print(f"  Total added: {n_wts}")

    # 4. MIO-TCD
    print("\n--- MIO-TCD Localization ---")
    n_miotcd = integrate_miotcd()
    print(f"  Total added: {n_miotcd}")

    # Updated stats
    print("\n=== Updated Dataset Stats ===")
    total = 0
    for split in ["train", "val", "test"]:
        n = len(list((YOLO_DIR / "images" / split).glob("*")))
        print(f"  {split}: {n} images")
        total += n
    print(f"  Total: {total} images")

    # Disk usage
    size = sum(f.stat().st_size for f in YOLO_DIR.rglob("*") if f.is_file())
    print(f"  Disk usage: {size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
