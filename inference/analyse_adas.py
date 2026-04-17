"""
analyse_adas.py
---------------
Post-hoc analysis of ADAS detection results:
  1. Confidence threshold sweep — mAP@0.5 vs threshold for ADAS classes
  2. Per-class Precision-Recall curves
  3. Summary JSON + figures saved to results/

Requires that inference/detect.py has already been run so the model weights
are cached; this script re-runs inference at multiple thresholds rather than
relying on stored detections (scores are needed for the PR curve).

Usage:
    cd /Users/arjunprakash/ADAS-with-FPGAs
    python inference/analyse_adas.py [--device auto|cpu|mps]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ssd_detector import SSDDetector, ADAS_CLASSES
from inference.detect import load_annotations, compute_iou, compute_ap

IMAGES_DIR   = Path(__file__).parent.parent / "datasets" / "coco_val" / "images"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
ADAS_IDS     = list(ADAS_CLASSES.keys())   # [1, 3, 6, 8]


# ─────────────────────────────────────────────────────────────
# Run inference at a given threshold, return all_detections dict
# ─────────────────────────────────────────────────────────────

def run_inference(detector, image_paths, threshold):
    from PIL import Image
    all_dets = {}
    for img_path in image_paths:
        from PIL import Image as PILImage
        image = PILImage.open(img_path).convert("RGB")
        boxes, labels, scores = detector.detect(image, threshold=threshold)
        all_dets[img_path.name] = (boxes, labels, scores)
    return all_dets


# ─────────────────────────────────────────────────────────────
# Compute per-class AP from raw detections
# ─────────────────────────────────────────────────────────────

def compute_adas_map(all_detections, gt_by_file, iou_threshold=0.5):
    """Compute per-class AP and ADAS mAP (averaged over 4 ADAS classes only)."""
    class_tp    = {cid: [] for cid in ADAS_IDS}
    class_n_gt  = {cid: 0  for cid in ADAS_IDS}

    for fname, gt_boxes in gt_by_file.items():
        for gt in gt_boxes:
            cid = gt["category_id"]
            if cid in class_n_gt:
                class_n_gt[cid] += 1

    for fname, (boxes, labels, scores) in all_detections.items():
        gt_boxes = gt_by_file.get(fname, [])
        matched  = [False] * len(gt_boxes)
        order    = scores.argsort(descending=True)

        for idx in order:
            det_box   = boxes[idx].tolist()
            det_label = int(labels[idx])
            det_score = float(scores[idx])

            if det_label not in class_tp:
                continue   # ignore non-ADAS detections

            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if gt["category_id"] != det_label:
                    continue
                iou = compute_iou(det_box, gt["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                class_tp[det_label].append((det_score, True))
                matched[best_j] = True
            else:
                class_tp[det_label].append((det_score, False))

    per_class_ap = {}
    for cid in ADAS_IDS:
        per_class_ap[cid] = compute_ap(class_tp[cid], class_n_gt[cid])

    map_score = sum(per_class_ap.values()) / len(ADAS_IDS)
    return per_class_ap, map_score


# ─────────────────────────────────────────────────────────────
# PR curve data for a single class
# ─────────────────────────────────────────────────────────────

def pr_curve_data(all_detections, gt_by_file, class_id, iou_threshold=0.5):
    """Return (precisions, recalls) arrays for a class, sorted by recall."""
    tp_list = []
    n_gt    = sum(
        1 for gt_boxes in gt_by_file.values()
        for gt in gt_boxes if gt["category_id"] == class_id
    )

    for fname, (boxes, labels, scores) in all_detections.items():
        gt_boxes = gt_by_file.get(fname, [])
        matched  = [False] * len(gt_boxes)
        order    = scores.argsort(descending=True)

        for idx in order:
            if int(labels[idx]) != class_id:
                continue
            det_box   = boxes[idx].tolist()
            det_score = float(scores[idx])

            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if gt["category_id"] != class_id:
                    continue
                iou = compute_iou(det_box, gt["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                tp_list.append((det_score, True))
                matched[best_j] = True
            else:
                tp_list.append((det_score, False))

    if not tp_list or n_gt == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])

    tp_list.sort(key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    precs, recs = [], []
    for _, is_tp in tp_list:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precs.append(tp_cum / (tp_cum + fp_cum))
        recs.append(tp_cum / n_gt)

    return np.array(precs), np.array(recs)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_threshold_sweep(thresholds, map_scores, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, map_scores, 'b-o', markersize=5)
    best_idx = int(np.argmax(map_scores))
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', linewidth=1.2,
               label=f'Best threshold = {thresholds[best_idx]:.2f} '
                     f'(mAP={map_scores[best_idx]:.3f})')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('mAP@0.5 (ADAS classes)')
    ax.set_title('SSD300 mAP vs. Detection Confidence Threshold')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, max(map_scores) * 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def ap_from_pr(precs: np.ndarray, recs: np.ndarray) -> float:
    """11-point interpolation AP directly from precision/recall arrays."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recs >= t
        ap += precs[mask].max() if mask.any() else 0.0
    return ap / 11.0


def plot_pr_curves(all_detections, gt_by_file, out_path):
    colors = {1: '#e74c3c', 3: '#3498db', 6: '#f39c12', 8: '#2ecc71'}
    fig, ax = plt.subplots(figsize=(7, 5))

    for cid, name in ADAS_CLASSES.items():
        precs, recs = pr_curve_data(all_detections, gt_by_file, cid)
        ap = ap_from_pr(precs, recs)
        ax.plot(recs, precs, color=colors[cid], linewidth=2,
                label=f'{name} (AP={ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — ADAS Classes (SSD300, IoU=0.5)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    detector = SSDDetector()
    detector.load_model(args.device)

    # Ground truth
    gt_by_file = load_annotations()
    if not gt_by_file:
        print("No ground truth found. Run datasets/download_coco.py first.")
        sys.exit(1)

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Images: {len(image_paths)}")

    # ── 1. Threshold sweep ─────────────────────────────────────
    print("\n── Threshold sweep ──────────────────────────────")
    thresholds  = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]
    map_scores  = []
    per_class_by_thresh = []

    for t in thresholds:
        dets = run_inference(detector, image_paths, threshold=t)
        per_cls, mAP = compute_adas_map(dets, gt_by_file, args.iou_threshold)
        map_scores.append(round(mAP, 4))
        per_class_by_thresh.append({str(k): round(v, 4) for k, v in per_cls.items()})
        print(f"  threshold={t:.2f}  mAP={mAP:.4f}  "
              + "  ".join(f"{ADAS_CLASSES[c]}={per_cls[c]:.3f}" for c in ADAS_IDS))

    best_idx   = int(np.argmax(map_scores))
    best_thresh = thresholds[best_idx]
    best_map    = map_scores[best_idx]
    print(f"\nBest threshold : {best_thresh}  →  mAP={best_map:.4f}")

    # ── 2. PR curves at best threshold ────────────────────────
    print("\n── PR curves at best threshold ──────────────────")
    best_dets = run_inference(detector, image_paths, threshold=best_thresh)
    per_cls_best, _ = compute_adas_map(best_dets, gt_by_file, args.iou_threshold)

    # ── 3. Save figures ────────────────────────────────────────
    print("\n── Saving figures ───────────────────────────────")
    plot_threshold_sweep(thresholds, map_scores,
                         RESULTS_DIR / "adas_threshold_sweep.png")
    plot_pr_curves(best_dets, gt_by_file,
                   RESULTS_DIR / "adas_pr_curves.png")

    # ── 4. Save JSON ────────────────────────────────────────────
    analysis = {
        "model": "SSD300-VGG16",
        "dataset": "COCO val2017 (100 ADAS images)",
        "iou_threshold": args.iou_threshold,
        "adas_classes": {str(k): v for k, v in ADAS_CLASSES.items()},
        "default_threshold": {
            "value": 0.5,
            "map_at_50": map_scores[thresholds.index(0.5)] if 0.5 in thresholds else None,
            "per_class_ap": per_class_by_thresh[thresholds.index(0.5)]
                            if 0.5 in thresholds else {},
        },
        "best_threshold": {
            "value": best_thresh,
            "map_at_50": best_map,
            "per_class_ap": {str(k): round(v, 4) for k, v in per_cls_best.items()},
        },
        "threshold_sweep": [
            {"threshold": t, "map_at_50": m, "per_class_ap": pc}
            for t, m, pc in zip(thresholds, map_scores, per_class_by_thresh)
        ],
    }

    json_path = RESULTS_DIR / "adas_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── 5. Print paper-ready summary ────────────────────────────
    print()
    print("=" * 55)
    print("  PAPER NUMBERS — ADAS SSD300 baseline")
    print("=" * 55)
    print(f"  Default (t=0.50)  mAP@0.5 = "
          f"{map_scores[thresholds.index(0.5)] if 0.5 in thresholds else 'N/A':.4f}")
    print(f"  Best    (t={best_thresh:.2f})  mAP@0.5 = {best_map:.4f}")
    print()
    print("  Per-class AP at best threshold:")
    for cid in ADAS_IDS:
        print(f"    {ADAS_CLASSES[cid]:<8} {per_cls_best[cid]:.4f}")


if __name__ == '__main__':
    main()
