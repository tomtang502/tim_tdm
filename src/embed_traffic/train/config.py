"""Shared training hyperparameters for YOLO fine-tuning."""

from __future__ import annotations

# Strong augmentation profile used by both view-specific detectors.
AUGMENTATION = dict(
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.15,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    erasing=0.5,
    copy_paste=0.1,
)

# Optimizer profile — lr0=0.005 + cosine decay to 0.005 * 0.01 = 0.00005.
OPTIMIZER = dict(
    optimizer="AdamW",
    lr0=0.005,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
)

# Default training schedule (view-specific scripts may override imgsz, batch, nbs).
TRAIN_DEFAULTS = dict(
    epochs=80,
    imgsz=1280,
    batch=20,
    nbs=60,
    workers=16,
    patience=15,
    close_mosaic=30,
    amp=True,
    compile=True,
    cache="disk",
    single_cls=True,
    save=True,
    exist_ok=True,
)
