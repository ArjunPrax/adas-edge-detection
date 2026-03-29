# ADAS Edge Detection on Xilinx ZCU104

FPGA-accelerated ADAS (Advanced Driver Assistance Systems) object detection pipeline using SSD300-VGG16 on the Xilinx ZCU104 evaluation board.

Part of a URECA research project on **FPGA-accelerated edge vision** at Nanyang Technological University (NTU).

---

## Project Overview

This project benchmarks and evaluates SSD300-VGG16 (pretrained on COCO) for ADAS-relevant object categories — **person**, **car**, **bus**, and **truck** — across three hardware targets:

| Target | Device | Notes |
|--------|--------|-------|
| Mac (Apple Silicon) | MPS | Development and baseline |
| Mac (Intel) | CPU | Fallback |
| Xilinx ZCU104 | ARM Cortex-A53 (CPU) | Edge deployment target |

---

## Project Structure

```
adas-edge-detection/
├── models/
│   └── ssd_detector.py       # SSD300-VGG16 wrapper
├── datasets/
│   └── download_coco.py      # Download 100 COCO val images
├── benchmarks/
│   └── benchmark_adas.py     # Latency/FPS benchmark
├── inference/
│   └── detect.py             # Run detection, visualize, compute mAP
├── results/
│   └── detections/           # Annotated output images (after running detect.py)
├── requirements.txt          # Mac / Linux development
├── requirements-board.txt    # ZCU104 ARM aarch64 (torch==1.13.1)
└── README.md
```

---

## Setup

### Mac / Linux (development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Xilinx ZCU104 (ARM aarch64)

```bash
pip install -r requirements-board.txt
```

> `requirements-board.txt` pins `torch==1.13.1` and `torchvision==0.14.1`, which have pre-built `aarch64` wheels available on PyPI.

---

## Usage

### 1. Download COCO Data

Download 100 COCO 2017 validation images containing persons, cars, buses, or trucks:

```bash
python datasets/download_coco.py
```

This saves images to `datasets/coco_val/images/` and ground-truth annotations to `datasets/coco_val/annotations.json`.

> Raw images are gitignored (too large). Re-run this script on any machine.

---

### 2. Benchmark Latency

```bash
# Auto-detect best device (MPS on Apple Silicon, CPU otherwise)
python benchmarks/benchmark_adas.py

# Force CPU (for ZCU104 comparison)
python benchmarks/benchmark_adas.py --device cpu
```

Runs 20 warmup + 50 timed iterations on a dummy `(3, 300, 300)` input.

**Output:**
```
Device         : mps
Avg latency    : 42.318 ms
Min latency    : 38.201 ms
Max latency    : 61.874 ms
FPS            : 23.63
```

Results saved to `results/benchmark_results.json`.

---

### 3. Run Detection + Evaluate mAP

```bash
python inference/detect.py
```

Runs SSD300 on all 100 COCO images, draws annotated bounding boxes, and computes mAP@0.5 against ground truth.

**Options:**
```bash
python inference/detect.py --device cpu --threshold 0.4
```

**Output:**
- `results/detections/` — annotated images (boxes, class labels, confidence scores)
- `results/detection_summary.json` — mAP, avg latency, detection counts

---

## Results

*Populated after running the scripts above.*

| Metric | Mac MPS | ZCU104 CPU |
|--------|---------|------------|
| Avg latency (ms) | — | — |
| FPS | — | — |
| mAP@0.5 | — | — |

---

## Hardware

**Xilinx ZCU104** — Zynq UltraScale+ MPSoC EV
- Quad-core ARM Cortex-A53 @ 1.2 GHz
- Mali-400 MP2 GPU (not used here)
- Programmable Logic: 504K LUTs, 1728 DSPs
- Target for future FPGA acceleration via Vitis AI

---

## Research Context

This work is part of a URECA (Undergraduate Research Experience on Campus Award) project at **Nanyang Technological University (NTU)** investigating FPGA-accelerated edge vision systems for ADAS applications. The PyTorch baseline established here will serve as the reference for quantized and hardware-accelerated implementations on the ZCU104's programmable logic fabric.

---

## License

MIT
