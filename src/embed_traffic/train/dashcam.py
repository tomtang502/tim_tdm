"""Fine-tune YOLO detector for dashcam view (PIE + JAAD).

Creates/refreshes `data/yolo_dataset_dashcam/` from `data/yolo_dataset/` by
symlinking only PIE- and JAAD-prefixed images, then runs ultralytics training.
Checkpoints land in `checkpoints/{run_name}/weights/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from embed_traffic.paths import (
    CHECKPOINTS_DIR,
    YOLO_DATASET_DASHCAM_DIR,
    YOLO_DATASET_DIR,
    detector_weights,
)
from embed_traffic.train.config import AUGMENTATION, OPTIMIZER, TRAIN_DEFAULTS

PREFIXES = ("pie_", "jaad_")


def prepare_subset(
    src: Path = YOLO_DATASET_DIR,
    dst: Path = YOLO_DATASET_DASHCAM_DIR,
    prefixes: tuple[str, ...] = PREFIXES,
) -> None:
    """Build a YOLO dataset directory containing only images whose basename
    starts with one of `prefixes`, using symlinks (no disk duplication)."""
    for split in ("train", "val", "test"):
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = src / "images" / split
        src_lbl_dir = src / "labels" / split
        if not src_img_dir.exists():
            print(f"  {split}: source dir {src_img_dir} not found, skipping")
            continue

        count = 0
        for img_path in src_img_dir.iterdir():
            name = img_path.stem
            if not any(name.startswith(p) for p in prefixes):
                continue
            dst_img = img_dir / img_path.name
            dst_lbl = lbl_dir / f"{name}.txt"
            lbl_path = src_lbl_dir / f"{name}.txt"

            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())
            if lbl_path.exists() and not dst_lbl.exists():
                dst_lbl.symlink_to(lbl_path.resolve())
            count += 1
        print(f"  {split}: {count} images")

    config = dst / "dataset.yaml"
    config.write_text(
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n  0: pedestrian\n"
    )
    print(f"  Config: {config}")


def main(
    run_name: str,
    pretrained: str = "yolo26x.pt",
    batch: int | None = None,
    workers: int | None = None,
) -> None:
    print(f"=== Preparing dashcam dataset (PIE + JAAD) for run '{run_name}' ===")
    prepare_subset()

    dataset_yaml = str((YOLO_DATASET_DASHCAM_DIR / "dataset.yaml").resolve())

    overrides = dict(TRAIN_DEFAULTS)
    if batch is not None:
        overrides["batch"] = batch
    if workers is not None:
        overrides["workers"] = workers

    model = YOLO(pretrained)
    model.train(
        data=dataset_yaml,
        project=str(CHECKPOINTS_DIR),
        name=run_name,
        device=0,
        **overrides,
        **AUGMENTATION,
        **OPTIMIZER,
    )

    print("\n=== Evaluating dashcam model on test set ===")
    best_model = YOLO(str(detector_weights(run_name)))
    metrics = best_model.val(
        data=dataset_yaml,
        split="test",
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=16,
        device=0,
        half=True,
    )

    print("\nDashcam Test Results:")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:      {metrics.box.mp:.4f}")
    print(f"  Recall:         {metrics.box.mr:.4f}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for dashcam view.")
    p.add_argument("--run-name", required=True, help="Checkpoint directory name.")
    p.add_argument("--pretrained", default="yolo26x.pt", help="Pretrained weights.")
    p.add_argument("--batch", type=int, default=None, help="Global batch size (overrides TRAIN_DEFAULTS).")
    p.add_argument("--workers", type=int, default=None, help="Dataloader workers (overrides TRAIN_DEFAULTS).")
    args = p.parse_args()
    main(args.run_name, args.pretrained, args.batch, args.workers)


if __name__ == "__main__":
    cli()
