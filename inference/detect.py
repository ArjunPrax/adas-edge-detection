"""
Run SSD300 detection on all 100 COCO images and evaluate against ground truth.

Usage:
    python inference/detect.py [--device auto|cpu|mps|cuda] [--threshold 0.5]

Output:
    results/detections/      — annotated images with bounding boxes
    results/detection_summary.json — mAP, avg latency, per-class stats
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ssd_detector import SSDDetector, label_name, label_color

IMAGES_DIR = Path(__file__).parent.parent / "datasets" / "coco_val" / "images"
ANNOTATIONS_PATH = Path(__file__).parent.parent / "datasets" / "coco_val" / "annotations.json"
DETECTIONS_DIR = Path(__file__).parent.parent / "results" / "detections"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "detection_summary.json"

# Font size for labels (fallback to default PIL font if truetype unavailable)
FONT_SIZE = 14
BOX_LINE_WIDTH = 2


def load_annotations() -> dict:
    """Load ground truth annotations keyed by filename."""
    if not ANNOTATIONS_PATH.exists():
        print(f"Warning: annotations not found at {ANNOTATIONS_PATH}")
        print("Run datasets/download_coco.py first to get ground truth.")
        return {}

    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    # Build image_id -> filename mapping
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    # Build filename -> list of {bbox, category_id} ground truth boxes
    gt_by_file = {}
    for ann in data["annotations"]:
        fname = id_to_filename.get(ann["image_id"])
        if fname is None:
            continue
        if fname not in gt_by_file:
            gt_by_file[fname] = []
        # COCO bbox format: [x, y, width, height] — convert to [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        gt_by_file[fname].append({
            "box": [x, y, x + w, y + h],
            "category_id": ann["category_id"],
        })
    return gt_by_file


def get_font(size: int = FONT_SIZE):
    """Return a PIL font, falling back to default if TrueType is unavailable."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (IOError, AttributeError):
        pass
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, AttributeError):
        pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, boxes, labels, scores) -> Image.Image:
    """
    Draw bounding boxes with class labels and confidence scores on the image.

    Returns a new annotated PIL image.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(FONT_SIZE)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(label)
        color = label_color(class_id)
        name = label_name(class_id)
        text = f"{name} {score:.2f}"

        # Draw box border
        for offset in range(BOX_LINE_WIDTH):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Measure text for label background
        try:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(text) * 7, 12

        padding = 3
        label_x1 = x1
        label_y1 = max(0, y1 - th - 2 * padding)
        label_x2 = x1 + tw + 2 * padding
        label_y2 = y1

        # Filled label background
        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
        # White text
        draw.text(
            (label_x1 + padding, label_y1 + padding),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    return annotated


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_ap(tp_list: list, n_gt: int) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.

    tp_list: list of (score, is_tp) sorted by descending score
    n_gt: total number of ground truth boxes for this class
    """
    if n_gt == 0:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for _, is_tp in sorted(tp_list, key=lambda x: -x[0]):
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        prec = tp_cumsum / (tp_cumsum + fp_cumsum)
        rec = tp_cumsum / n_gt
        precisions.append(prec)
        recalls.append(rec)

    # 11-point interpolation
    ap = 0.0
    for t in [r / 10 for r in range(11)]:
        ps = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(ps) if ps else 0.0
    return ap / 11.0


def compute_map(
    all_detections: dict,
    gt_by_file: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute per-class AP and mean AP.

    all_detections: {filename: (boxes, labels, scores)}
    gt_by_file:     {filename: [{box, category_id}, ...]}

    Returns dict with per_class_ap and map.
    """
    # Gather per-class detection scores and GT counts
    class_tp = {}   # class_id -> [(score, is_tp), ...]
    class_n_gt = {} # class_id -> int

    for fname, gt_boxes in gt_by_file.items():
        for gt in gt_boxes:
            cid = gt["category_id"]
            class_n_gt[cid] = class_n_gt.get(cid, 0) + 1

    for fname, (boxes, labels, scores) in all_detections.items():
        gt_boxes = gt_by_file.get(fname, [])
        # Track which GT boxes have been matched
        matched = [False] * len(gt_boxes)

        # Sort detections by score descending
        order = scores.argsort(descending=True)
        for idx in order:
            det_box = boxes[idx].tolist()
            det_label = int(labels[idx])
            det_score = float(scores[idx])

            if det_label not in class_tp:
                class_tp[det_label] = []

            # Find best matching GT box of same class
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_boxes):
                if gt["category_id"] != det_label:
                    continue
                iou = compute_iou(det_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                class_tp[det_label].append((det_score, True))
                matched[best_j] = True
            else:
                class_tp[det_label].append((det_score, False))

    # Compute AP per class
    per_class_ap = {}
    for cid, tp_list in class_tp.items():
        n_gt = class_n_gt.get(cid, 0)
        per_class_ap[cid] = compute_ap(tp_list, n_gt)

    map_score = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    return {"per_class_ap": per_class_ap, "map": map_score}


def main():
    parser = argparse.ArgumentParser(description="Run ADAS detection on COCO images")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    args = parser.parse_args()

    print("=== ADAS SSD300 Detection ===\n")

    # Load model
    detector = SSDDetector()
    detector.load_model(args.device)

    # Load ground truth
    gt_by_file = load_annotations()

    # Find images
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        print("Run datasets/download_coco.py first.")
        sys.exit(1)
    print(f"Found {len(image_paths)} images.\n")

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_detections = {}
    total_latency = 0.0
    total_detections = 0

    print("Running detection ...")
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        boxes, labels, scores = detector.detect(image, threshold=args.threshold)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        total_latency += latency_ms
        total_detections += len(boxes)

        all_detections[img_path.name] = (boxes, labels, scores)

        # Draw and save annotated image
        annotated = draw_detections(image, boxes, labels, scores)
        out_path = DETECTIONS_DIR / img_path.name
        annotated.save(out_path, quality=95)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(image_paths)}] {img_path.name} "
                  f"— {len(boxes)} detections, {latency_ms:.1f} ms")

    avg_latency_ms = total_latency / len(image_paths)
    print(f"\nDetection complete.")
    print(f"  Avg latency : {avg_latency_ms:.1f} ms/image")
    print(f"  Total dets  : {total_detections}")

    # Compute mAP
    if gt_by_file:
        print("\nComputing mAP ...")
        metrics = compute_map(all_detections, gt_by_file)
        map_score = metrics["map"]
        per_class = metrics["per_class_ap"]
        print(f"  mAP@0.5 : {map_score:.4f}")
        for cid, ap in sorted(per_class.items()):
            from models.ssd_detector import label_name as ln
            print(f"    {ln(cid):12s} AP = {ap:.4f}")
    else:
        map_score = None
        per_class = {}
        print("Skipping mAP (no ground truth annotations found).")

    # Save summary
    summary = {
        "num_images": len(image_paths),
        "num_detections": total_detections,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "threshold": args.threshold,
        "device": str(detector.device),
        "map_at_50": round(map_score, 4) if map_score is not None else None,
        "per_class_ap": {
            str(cid): round(ap, 4) for cid, ap in per_class.items()
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnnotated images : {DETECTIONS_DIR}")
    print(f"Summary          : {SUMMARY_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
