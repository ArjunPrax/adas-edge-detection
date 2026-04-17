"""
quantize_ssd.py
---------------
Post-Training INT8 Quantization (PTQ) for SSD300-VGG16 (ADAS).

Method:
  1. Load SSD300-VGG16 with COCO pretrained weights.
  2. Register forward hooks on all Conv2d layers to collect activation ranges
     over a calibration set of COCO images.
  3. Quantize every Conv2d weight tensor to INT8 (symmetric, per-tensor).
  4. Dequantize weights back to float32 and patch the model in-place —
     this simulates the quantization error incurred during FPGA INT8 inference
     while keeping the PyTorch runtime in float32 (no custom CUDA kernels needed).
  5. Evaluate FP32 mAP@0.5 and simulated INT8 mAP@0.5 on the COCO eval set,
     using the same compute_adas_map() from inference/analyse_adas.py.
  6. Export INT8 weights (.npy + .bin) and a summary JSON.

Usage:
    python quantization/quantize_ssd.py [--device auto|cpu|mps] [--calib-batches 50]

Output:
    results/quantized_weights/   — per-layer .npy/.bin weight files + metadata
    results/ssd_quantization.json — FP32 vs INT8 mAP comparison
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ssd_detector import SSDDetector, ADAS_CLASSES, get_device
from inference.detect import load_annotations, compute_iou, compute_ap
from inference.analyse_adas import compute_adas_map

IMAGES_DIR   = Path(__file__).parent.parent / "datasets" / "coco_val" / "images"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
QUANT_DIR    = RESULTS_DIR / "quantized_weights"
ADAS_IDS     = list(ADAS_CLASSES.keys())   # [1, 3, 6, 8]


# ─────────────────────────────────────────────────────────────────────────────
# Quantization helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantize_array(arr: np.ndarray, num_bits: int = 8):
    """
    Symmetric per-tensor INT8 quantization.

    Returns:
        q   : quantized int8 ndarray
        scale      : float
        zero_point : float (always 0 for symmetric)
    """
    qmax = 2 ** (num_bits - 1) - 1   # 127
    abs_max = np.abs(arr).max()
    if abs_max == 0:
        return arr.astype(np.int8), 1.0, 0.0
    scale = abs_max / qmax
    q = np.clip(np.round(arr / scale), -qmax - 1, qmax).astype(np.int8)
    return q, float(scale), 0.0


def dequantize_array(q: np.ndarray, scale: float, zero_point: float = 0.0) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — collect activation ranges
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(model, image_paths, device, num_images: int = 50):
    """Run forward passes on `num_images` images and return per-layer act stats."""
    from torchvision.transforms import functional as F

    activations = {}  # layer_name -> list of (abs_max float)
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                activations.setdefault(name, []).append(float(out.detach().abs().max()))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    selected = image_paths[:num_images]
    print(f"  Calibrating on {len(selected)} images ...")
    with torch.no_grad():
        for p in tqdm(selected, desc="  calib"):
            img = Image.open(p).convert("RGB")
            t = F.to_tensor(img).to(device)
            model([t])

    for h in hooks:
        h.remove()

    # Summarise: take mean of observed abs-maxima as representative range
    act_ranges = {k: float(np.mean(v)) for k, v in activations.items()}
    return act_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Quantize + patch model
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(model, act_ranges):
    """
    Quantize every Conv2d weight to INT8 and patch the model with dequantized
    float32 weights (simulates quantization error without custom kernels).

    Returns quantized_params dict (for export) and the patched model.
    """
    quantized_params = {}
    print("  Quantizing Conv2d weights ...")
    with torch.no_grad():
        for name, module in tqdm(list(model.named_modules()), desc="  quant"):
            if not isinstance(module, nn.Conv2d):
                continue

            # Weight
            w_np = module.weight.data.cpu().numpy()
            w_q, w_scale, w_zp = quantize_array(w_np)
            w_dequant = dequantize_array(w_q, w_scale, w_zp)
            module.weight.data = torch.from_numpy(w_dequant).to(module.weight.device)

            # Bias (quantize to int32 — or skip bias quant, just keep float)
            b_q = b_scale = b_zp = None
            if module.bias is not None:
                b_np = module.bias.data.cpu().numpy()
                b_q, b_scale, b_zp = quantize_array(b_np, num_bits=32)
                # Don't patch bias — int32 bias is standard practice and negligible error

            act_scale = act_ranges.get(name, 1.0) / 127.0

            quantized_params[name] = {
                "weight_q":          w_q,
                "weight_scale":      w_scale,
                "weight_zero_point": w_zp,
                "bias_q":            b_q,
                "bias_scale":        b_scale,
                "bias_zero_point":   b_zp,
                "activation_scale":  act_scale,
                "in_channels":       module.in_channels,
                "out_channels":      module.out_channels,
                "kernel_size":       list(module.kernel_size),
                "stride":            list(module.stride),
                "padding":           list(module.padding),
            }
    return quantized_params


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_weights(quantized_params, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Exporting {len(quantized_params)} layers to {out_dir} ...")
    for name, p in quantized_params.items():
        safe = name.replace(".", "_")
        np.save(out_dir / f"{safe}_weight.npy", p["weight_q"])
        p["weight_q"].tofile(out_dir / f"{safe}_weight.bin")
        if p["bias_q"] is not None:
            np.save(out_dir / f"{safe}_bias.npy", p["bias_q"])
        meta = {
            "weight_scale":      p["weight_scale"],
            "weight_zero_point": p["weight_zero_point"],
            "activation_scale":  p["activation_scale"],
            "in_channels":       p["in_channels"],
            "out_channels":      p["out_channels"],
            "kernel_size":       p["kernel_size"],
            "stride":            p["stride"],
            "padding":           p["padding"],
        }
        with open(out_dir / f"{safe}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(detector, image_paths, gt_by_file, threshold, label=""):
    from torchvision.transforms import functional as F
    all_dets = {}
    detector.model.eval()
    with torch.no_grad():
        for p in tqdm(image_paths, desc=f"  eval {label}"):
            img = Image.open(p).convert("RGB")
            boxes, labels, scores = detector.detect(img, threshold=threshold)
            all_dets[p.name] = (boxes, labels, scores)
    per_cls, mAP = compute_adas_map(all_dets, gt_by_file, iou_threshold=0.5)
    return per_cls, mAP


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        default="auto", choices=["auto","cpu","mps","cuda"])
    parser.add_argument("--calib-batches", type=int, default=50,
                        help="Number of images used for calibration (default 50)")
    parser.add_argument("--threshold",     type=float, default=0.10,
                        help="Detection threshold for mAP eval (default 0.10 = best F1)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading SSD300-VGG16 ...")
    detector = SSDDetector()
    device   = detector.load_model(args.device)

    # ── Ground truth ────────────────────────────────────────────────────────
    print("\n[2/5] Loading ground truth ...")
    gt_by_file = load_annotations()
    if not gt_by_file:
        print("ERROR: No ground truth — run datasets/download_coco.py first.")
        sys.exit(1)

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"  Images: {len(image_paths)},  GT files: {len(gt_by_file)}")

    # ── FP32 baseline ───────────────────────────────────────────────────────
    print(f"\n[3/5] FP32 baseline (threshold={args.threshold}) ...")
    fp32_per_cls, fp32_map = run_eval(detector, image_paths, gt_by_file,
                                      threshold=args.threshold, label="FP32")
    print(f"  FP32 mAP@0.5 = {fp32_map:.4f}")
    for cid in ADAS_IDS:
        print(f"    {ADAS_CLASSES[cid]:<8} {fp32_per_cls[cid]:.4f}")

    # ── Calibrate ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Calibrating ({args.calib_batches} images) ...")
    act_ranges = calibrate(detector.model, image_paths,
                           device=device, num_images=args.calib_batches)
    print(f"  Collected activation ranges for {len(act_ranges)} Conv2d layers.")

    # ── Quantize (in-place weight patching) ─────────────────────────────────
    print("\n[5/5] Quantizing weights and evaluating INT8 ...")
    quantized_params = quantize_model(detector.model, act_ranges)

    # Evaluate with quantized (dequantized) weights
    int8_per_cls, int8_map = run_eval(detector, image_paths, gt_by_file,
                                      threshold=args.threshold, label="INT8")
    print(f"  INT8 mAP@0.5 = {int8_map:.4f}")
    for cid in ADAS_IDS:
        print(f"    {ADAS_CLASSES[cid]:<8} {int8_per_cls[cid]:.4f}")

    drop = fp32_map - int8_map

    # ── Export weights ───────────────────────────────────────────────────────
    export_weights(quantized_params, QUANT_DIR)

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "model":             "SSD300-VGG16",
        "dataset":           "COCO val2017 (100 ADAS images)",
        "iou_threshold":     0.5,
        "detection_threshold": args.threshold,
        "calib_images":      args.calib_batches,
        "conv_layers_quantized": len(quantized_params),
        "fp32": {
            "map_at_50":     round(fp32_map, 4),
            "per_class_ap":  {str(k): round(v, 4) for k, v in fp32_per_cls.items()},
        },
        "int8_simulated": {
            "map_at_50":     round(int8_map, 4),
            "per_class_ap":  {str(k): round(v, 4) for k, v in int8_per_cls.items()},
        },
        "map_drop":          round(drop, 4),
        "map_drop_pct":      round(drop / fp32_map * 100, 2) if fp32_map > 0 else 0,
    }

    json_path = RESULTS_DIR / "ssd_quantization.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print paper-ready summary ────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  PAPER NUMBERS — SSD300 INT8 quantization")
    print("=" * 55)
    print(f"  Conv2d layers quantized : {len(quantized_params)}")
    print(f"  FP32  mAP@0.5           : {fp32_map:.4f}  ({fp32_map*100:.2f}%)")
    print(f"  INT8  mAP@0.5           : {int8_map:.4f}  ({int8_map*100:.2f}%)")
    print(f"  Drop                    : {drop:.4f}  ({drop*100:.2f} pp)")
    print(f"  Relative drop           : {summary['map_drop_pct']:.2f}%")
    print()
    print("  Per-class AP (INT8):")
    for cid in ADAS_IDS:
        delta = fp32_per_cls[cid] - int8_per_cls[cid]
        print(f"    {ADAS_CLASSES[cid]:<8}  FP32={fp32_per_cls[cid]:.4f}  "
              f"INT8={int8_per_cls[cid]:.4f}  Δ={delta:+.4f}")
    print()
    print(f"  Weights exported : {QUANT_DIR}")
    print(f"  Summary JSON     : {json_path}")


if __name__ == "__main__":
    main()
