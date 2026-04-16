"""Prepare PIE+JAAD data in YOLO format for fine-tuning.

Writes frames (as JPEGs) and YOLO-format label files to `data/yolo_dataset/`.
Only exports annotated frames; samples every Nth frame per pedestrian track to
reduce redundancy.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from embed_traffic.data.loader import PedestrianSample, UnifiedDataLoader
from embed_traffic.paths import YOLO_DATASET_DIR

FRAME_SAMPLE_RATE = 5
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def prepare_split(
    loader: UnifiedDataLoader,
    dataset: str,
    samples: list[PedestrianSample],
    split_name: str,
    output_dir: Path = YOLO_DATASET_DIR,
) -> int:
    """Export frames + YOLO labels for one dataset/split. Returns frames written."""
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    frame_groups: dict[tuple[str, int], list[PedestrianSample]] = defaultdict(list)
    for s in samples:
        frame_groups[(s.video_id, s.frame_id)].append(s)

    all_keys = sorted(frame_groups.keys())
    sampled_keys = all_keys[::FRAME_SAMPLE_RATE]

    print(
        f"  {split_name}: {len(all_keys)} total frames -> {len(sampled_keys)} sampled"
    )

    written = 0
    skipped = 0
    for i, (video_id, frame_id) in enumerate(sampled_keys):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sampled_keys)} frames...")

        vid_safe = video_id.replace("/", "_")
        fname = f"{dataset}_{vid_safe}_{frame_id:06d}"
        img_path = img_dir / f"{fname}.jpg"
        lbl_path = lbl_dir / f"{fname}.txt"

        if img_path.exists() and lbl_path.exists():
            written += 1
            continue

        try:
            frame = loader.get_frame(dataset, video_id, frame_id)
        except RuntimeError:
            skipped += 1
            continue

        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        annotations = frame_groups[(video_id, frame_id)]
        with open(lbl_path, "w") as f:
            for s in annotations:
                if s.occlusion >= 2:
                    continue
                x1, y1, x2, y2 = s.bbox
                x_center = ((x1 + x2) / 2) / IMG_WIDTH
                y_center = ((y1 + y2) / 2) / IMG_HEIGHT
                w = abs(x2 - x1) / IMG_WIDTH
                h = abs(y2 - y1) / IMG_HEIGHT
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        written += 1

    print(f"  Written: {written}, Skipped: {skipped}")
    return written


def write_dataset_yaml(output_dir: Path = YOLO_DATASET_DIR) -> Path:
    config_path = output_dir / "dataset.yaml"
    config_path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n  0: pedestrian\n"
    )
    return config_path


def main() -> None:
    loader = UnifiedDataLoader()

    print("Loading annotations...")
    pie_train = loader.get_pie_samples(split="train")
    pie_val = loader.get_pie_samples(split="val")
    pie_test = loader.get_pie_samples(split="test")
    jaad_train = loader.get_jaad_samples(split="train")
    jaad_val = loader.get_jaad_samples(split="val")
    jaad_test = loader.get_jaad_samples(split="test")

    print(f"\nPIE:  train={len(pie_train)}, val={len(pie_val)}, test={len(pie_test)}")
    print(
        f"JAAD: train={len(jaad_train)}, val={len(jaad_val)}, test={len(jaad_test)}"
    )

    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Preparing train split ---")
    n_train = 0
    n_train += prepare_split(loader, "pie", pie_train, "train")
    n_train += prepare_split(loader, "jaad", jaad_train, "train")

    print("\n--- Preparing val split ---")
    n_val = 0
    n_val += prepare_split(loader, "pie", pie_val, "val")
    n_val += prepare_split(loader, "jaad", jaad_val, "val")

    print("\n--- Preparing test split ---")
    n_test = 0
    n_test += prepare_split(loader, "pie", pie_test, "test")
    n_test += prepare_split(loader, "jaad", jaad_test, "test")

    print("\n=== Summary ===")
    print(f"Train: {n_train}")
    print(f"Val:   {n_val}")
    print(f"Test:  {n_test}")

    config_path = write_dataset_yaml()
    print(f"\nDataset config written to {config_path}")

    total_size = sum(
        f.stat().st_size for f in YOLO_DATASET_DIR.rglob("*") if f.is_file()
    )
    print(f"Total disk usage: {total_size / 1e9:.1f} GB")

    loader.release_all()


if __name__ == "__main__":
    main()
