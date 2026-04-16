"""
Step 2.1-2.2: Load pretrained YOLOv8 and evaluate zero-shot pedestrian detection on PIE/JAAD.
Uses COCO pretrained model — class 0 = "person".
Computes mAP@0.5 and mAP@0.5:0.95.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
from embed_traffic.data.loader import UnifiedDataLoader, export_yolo_labels  # noqa: F401
import time


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        prec_at_r = mpre[mrec >= r]
        ap += (prec_at_r.max() if len(prec_at_r) > 0 else 0.0)
    return ap / 101.0


def evaluate_detections(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_threshold=0.5):
    """
    Compute precision, recall, and AP at a given IoU threshold.
    all_gt_boxes: list of lists of [x1,y1,x2,y2]
    all_pred_boxes: list of lists of [x1,y1,x2,y2]
    all_pred_scores: list of lists of confidence scores
    """
    # Flatten all predictions with image index
    preds = []
    for img_idx, (boxes, scores) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        for box, score in zip(boxes, scores):
            preds.append((img_idx, score, box))

    # Sort by confidence descending
    preds.sort(key=lambda x: x[1], reverse=True)

    total_gt = sum(len(gt) for gt in all_gt_boxes)
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    # Track which GT boxes have been matched
    matched = [np.zeros(len(gt), dtype=bool) for gt in all_gt_boxes]

    tp_list = []
    fp_list = []

    for img_idx, score, pred_box in preds:
        gt_boxes = all_gt_boxes[img_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and not matched[img_idx][best_gt_idx]:
            tp_list.append(1)
            fp_list.append(0)
            matched[img_idx][best_gt_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap(precisions, recalls)

    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0

    return ap, final_precision, final_recall


def evaluate_dataset(loader, model, dataset, samples, max_frames=200):
    """Evaluate YOLOv8 on a subset of frames from a dataset."""
    # Group samples by (video_id, frame_id)
    frame_groups = defaultdict(list)
    for s in samples:
        frame_groups[(s.video_id, s.frame_id)].append(s)

    # Sample frames evenly
    all_keys = sorted(frame_groups.keys())
    if len(all_keys) > max_frames:
        indices = np.linspace(0, len(all_keys) - 1, max_frames, dtype=int)
        all_keys = [all_keys[i] for i in indices]

    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []

    print(f"  Evaluating {len(all_keys)} frames...")
    t0 = time.time()

    for i, (video_id, frame_id) in enumerate(all_keys):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(all_keys)} frames processed...")

        try:
            frame = loader.get_frame(dataset, video_id, frame_id)
        except RuntimeError:
            continue

        # Ground truth boxes
        gt_boxes = [s.bbox for s in frame_groups[(video_id, frame_id)]
                    if s.occlusion < 2]  # skip fully occluded
        all_gt_boxes.append(gt_boxes)

        # YOLOv8 predictions — class 0 = person
        results = model(frame, verbose=False, classes=[0])
        pred_boxes = []
        pred_scores = []
        for r in results:
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                pred_boxes.append(box.tolist())
                pred_scores.append(float(conf))
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)

    elapsed = time.time() - t0
    fps = len(all_keys) / elapsed
    print(f"  Done in {elapsed:.1f}s ({fps:.1f} fps)")

    # Compute mAP at multiple IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou_thresh in iou_thresholds:
        ap, _, _ = evaluate_detections(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_thresh)
        aps.append(ap)

    ap50, prec50, rec50 = evaluate_detections(all_gt_boxes, all_pred_boxes, all_pred_scores, 0.5)
    ap50_95 = np.mean(aps)

    return {
        "mAP@0.5": ap50,
        "mAP@0.5:0.95": ap50_95,
        "precision@0.5": prec50,
        "recall@0.5": rec50,
        "fps": fps,
        "num_frames": len(all_keys),
        "total_gt": sum(len(gt) for gt in all_gt_boxes),
        "total_preds": sum(len(p) for p in all_pred_boxes),
    }


def main():
    print("Loading YOLOv8m pretrained model...")
    model = YOLO("yolov8m.pt")

    loader = UnifiedDataLoader()

    # Evaluate on JAAD test set
    print("\n=== JAAD Test Set ===")
    jaad_test = loader.get_jaad_samples(split="test")
    jaad_results = evaluate_dataset(loader, model, "jaad", jaad_test, max_frames=200)
    print(f"  mAP@0.5:      {jaad_results['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {jaad_results['mAP@0.5:0.95']:.4f}")
    print(f"  Precision:     {jaad_results['precision@0.5']:.4f}")
    print(f"  Recall:        {jaad_results['recall@0.5']:.4f}")
    print(f"  FPS:           {jaad_results['fps']:.1f}")
    print(f"  GT boxes:      {jaad_results['total_gt']}")
    print(f"  Predictions:   {jaad_results['total_preds']}")

    # Evaluate on PIE test set
    print("\n=== PIE Test Set ===")
    pie_test = loader.get_pie_samples(split="test")
    pie_results = evaluate_dataset(loader, model, "pie", pie_test, max_frames=200)
    print(f"  mAP@0.5:      {pie_results['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {pie_results['mAP@0.5:0.95']:.4f}")
    print(f"  Precision:     {pie_results['precision@0.5']:.4f}")
    print(f"  Recall:        {pie_results['recall@0.5']:.4f}")
    print(f"  FPS:           {pie_results['fps']:.1f}")
    print(f"  GT boxes:      {pie_results['total_gt']}")
    print(f"  Predictions:   {pie_results['total_preds']}")

    loader.release_all()


if __name__ == "__main__":
    main()
