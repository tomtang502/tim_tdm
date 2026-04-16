"""Visualize sample data with annotations from each dataset.

1. PIE - dashcam with pedestrian bboxes
2. JAAD - dashcam with pedestrian bboxes
3. Intersection-Flow-5K - fixed intersection camera
4. MIO-TCD - fixed traffic surveillance camera
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from embed_traffic.paths import (
    INTERSECTION_FLOW_DIR,
    MIO_TCD_DIR,
    OUTPUTS_DIR,
)

OUTPUT_DIR = OUTPUTS_DIR / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COLORS = {
    "crossing": (0, 0, 255),
    "not-crossing": (0, 255, 0),
    "unknown": (255, 200, 0),
}


def make_slideshow(images, labels_per_img, output_path, fps=2, title=""):
    """Create an MP4 slideshow from annotated images."""
    if not images:
        print(f"  No images for {title}, skipping")
        return

    # Resize all to same size
    target_h, target_w = 1080, 1920
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

    for img, labels in zip(images, labels_per_img):
        h, w = img.shape[:2]
        scale_x = target_w / w
        scale_y = target_h / h
        frame = cv2.resize(img, (target_w, target_h))

        for lbl in labels:
            x1 = int(lbl["bbox"][0] * scale_x)
            y1 = int(lbl["bbox"][1] * scale_y)
            x2 = int(lbl["bbox"][2] * scale_x)
            y2 = int(lbl["bbox"][3] * scale_y)
            color = COLORS.get(lbl.get("intent", "unknown"), (255, 200, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = lbl.get("text", "pedestrian")
            cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Title bar
        cv2.rectangle(frame, (0, 0), (target_w, 40), (0, 0, 0), -1)
        cv2.putText(frame, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show each image for multiple frames (slow slideshow)
        for _ in range(fps * 2):  # 2 seconds per image
            writer.write(frame)

    writer.release()
    print(f"  Wrote {len(images)} samples to {output_path}")


# ============================================================
# 1. PIE
# ============================================================
def sample_pie():
    from embed_traffic.data.loader import UnifiedDataLoader
    loader = UnifiedDataLoader()
    samples = loader.get_pie_samples(split="train")

    # Pick frames with close pedestrians from different videos
    frame_groups = defaultdict(list)
    for s in samples:
        h = s.bbox[3] - s.bbox[1]
        if h > 150:  # close pedestrians only
            frame_groups[(s.video_id, s.frame_id)].append(s)

    # Pick 10 diverse frames from different videos
    seen_videos = set()
    selected = []
    for (vid, fid), samps in sorted(frame_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        selected.append((vid, fid, samps))
        if len(selected) >= 10:
            break

    images = []
    labels_list = []
    for vid, fid, samps in selected:
        frame = loader.get_frame("pie", vid, fid)
        labels = []
        for s in samps:
            intent = {1: "crossing", 0: "not-crossing", -1: "unknown"}.get(s.crossing_intent, "unknown")
            action = "walk" if s.action == 1 else "stand"
            labels.append({
                "bbox": s.bbox,
                "intent": intent,
                "text": f"{s.ped_id} {intent} {action}",
            })
        images.append(frame)
        labels_list.append(labels)

    loader.release_all()
    make_slideshow(images, labels_list,
                   OUTPUT_DIR / "sample_dataset_pie.mp4", fps=1,
                   title="PIE Dataset - Dashcam, Toronto")


# ============================================================
# 2. JAAD
# ============================================================
def sample_jaad():
    from embed_traffic.data.loader import UnifiedDataLoader
    loader = UnifiedDataLoader()
    samples = loader.get_jaad_samples(split="train")

    frame_groups = defaultdict(list)
    for s in samples:
        h = s.bbox[3] - s.bbox[1]
        if h > 100:
            frame_groups[(s.video_id, s.frame_id)].append(s)

    seen_videos = set()
    selected = []
    for (vid, fid), samps in sorted(frame_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        selected.append((vid, fid, samps))
        if len(selected) >= 10:
            break

    images = []
    labels_list = []
    for vid, fid, samps in selected:
        frame = loader.get_frame("jaad", vid, fid)
        labels = []
        for s in samps:
            intent = {1: "crossing", 0: "not-crossing", -1: "unknown"}.get(s.crossing_intent, "unknown")
            action = "walk" if s.action == 1 else "stand"
            labels.append({
                "bbox": s.bbox,
                "intent": intent,
                "text": f"{s.ped_id} {intent} {action}",
            })
        images.append(frame)
        labels_list.append(labels)

    loader.release_all()
    make_slideshow(images, labels_list,
                   OUTPUT_DIR / "sample_dataset_jaad.mp4", fps=1,
                   title="JAAD Dataset - Dashcam, Multi-city")


# ============================================================
# 3. Intersection-Flow-5K
# ============================================================
def sample_intersection_flow():
    data_root = INTERSECTION_FLOW_DIR / "Intersection-Flow-5K"
    img_dir = data_root / "images" / "train"
    lbl_dir = data_root / "labels" / "train"

    if not img_dir.exists():
        print("  Intersection-Flow-5K not found, skipping")
        return

    PEDESTRIAN_CLASS = 3

    # Find images with pedestrian annotations
    selected = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        ped_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == PEDESTRIAN_CLASS:
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if h > 0.05:  # skip very small
                        ped_boxes.append((cx, cy, w, h))
        if len(ped_boxes) >= 2:
            selected.append((lbl_path.stem, ped_boxes))
        if len(selected) >= 10:
            break

    images = []
    labels_list = []
    for stem, boxes in selected:
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        labels = []
        for cx, cy, w, h in boxes:
            x1 = (cx - w / 2) * iw
            y1 = (cy - h / 2) * ih
            x2 = (cx + w / 2) * iw
            y2 = (cy + h / 2) * ih
            labels.append({
                "bbox": [x1, y1, x2, y2],
                "intent": "unknown",
                "text": "pedestrian",
            })
        images.append(img)
        labels_list.append(labels)

    make_slideshow(images, labels_list,
                   OUTPUT_DIR / "sample_dataset_iflow.mp4", fps=1,
                   title="Intersection-Flow-5K - Fixed Camera, Intersections")


# ============================================================
# 4. MIO-TCD
# ============================================================
def sample_miotcd():
    data_root = MIO_TCD_DIR / "MIO-TCD-Localization"
    gt_file = data_root / "gt_train.csv"
    img_dir = data_root / "train"

    if not gt_file.exists():
        print("  MIO-TCD not found, skipping")
        return

    # Parse pedestrian annotations
    ped_images = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6 and parts[1].strip() == "pedestrian":
                img_id = parts[0]
                x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                h = y2 - y1
                if h > 30:  # skip very small
                    ped_images[img_id].append((x1, y1, x2, y2))

    # Pick images with multiple pedestrians
    sorted_imgs = sorted(ped_images.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    images = []
    labels_list = []
    for img_id, boxes in sorted_imgs:
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        labels = []
        for x1, y1, x2, y2 in boxes:
            labels.append({
                "bbox": [x1, y1, x2, y2],
                "intent": "unknown",
                "text": "pedestrian",
            })
        images.append(img)
        labels_list.append(labels)

    make_slideshow(images, labels_list,
                   OUTPUT_DIR / "sample_dataset_miotcd.mp4", fps=1,
                   title="MIO-TCD Localization - Traffic Surveillance Camera")


def main():
    print("=== Generating dataset sample videos ===\n")

    print("1. PIE")
    sample_pie()

    print("\n2. JAAD")
    sample_jaad()

    print("\n3. Intersection-Flow-5K")
    sample_intersection_flow()

    print("\n4. MIO-TCD")
    sample_miotcd()

    print("\nDone! Check sample_dataset_*.mp4 files.")


if __name__ == "__main__":
    main()
