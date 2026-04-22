"""Fine-tune YOLO detector for traffic-light (fixed) view.

Uses only Intersection-Flow-5K and MIO-TCD images, which share a fixed overhead
camera perspective. Checkpoints land in `checkpoints/{run_name}/weights/`.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from embed_traffic.paths import (
    CHECKPOINTS_DIR,
    YOLO_DATASET_DIR,
    YOLO_DATASET_TRAFFIC_LIGHT_DIR,
    detector_weights,
)
from embed_traffic.train.config import AUGMENTATION, OPTIMIZER, TRAIN_DEFAULTS
from embed_traffic.train.dashcam import prepare_subset

PREFIXES = ("iflow_", "miotcd_")


def main(
    run_name: str,
    pretrained: str = "yolo26x.pt",
    batch: int | None = None,
    workers: int | None = None,
) -> None:
    print(f"=== Preparing traffic-light dataset (IFlow + MIO-TCD) for run '{run_name}' ===")
    prepare_subset(
        src=YOLO_DATASET_DIR,
        dst=YOLO_DATASET_TRAFFIC_LIGHT_DIR,
        prefixes=PREFIXES,
    )

    dataset_yaml = str((YOLO_DATASET_TRAFFIC_LIGHT_DIR / "dataset.yaml").resolve())

    model = YOLO(pretrained)
    # Traffic-light dataset is small enough to fully cache in RAM
    overrides = dict(TRAIN_DEFAULTS)
    overrides["cache"] = True
    if batch is not None:
        overrides["batch"] = batch
    if workers is not None:
        overrides["workers"] = workers
    model.train(
        data=dataset_yaml,
        project=str(CHECKPOINTS_DIR),
        name=run_name,
        device=0,
        **overrides,
        **AUGMENTATION,
        **OPTIMIZER,
    )

    print("\n=== Evaluating traffic-light model on test set ===")
    best_model = YOLO(str(detector_weights(run_name)))
    metrics = best_model.val(
        data=dataset_yaml,
        split="test",
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=16,
        device=0,
        half=True,
    )

    print("\nTraffic Light Test Results:")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:      {metrics.box.mp:.4f}")
    print(f"  Recall:         {metrics.box.mr:.4f}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for traffic-light view.")
    p.add_argument("--run-name", required=True, help="Checkpoint directory name.")
    p.add_argument("--pretrained", default="yolo26x.pt", help="Pretrained weights.")
    p.add_argument("--batch", type=int, default=None, help="Global batch size (overrides TRAIN_DEFAULTS).")
    p.add_argument("--workers", type=int, default=None, help="Dataloader workers (overrides TRAIN_DEFAULTS).")
    args = p.parse_args()
    main(args.run_name, args.pretrained, args.batch, args.workers)


if __name__ == "__main__":
    cli()
