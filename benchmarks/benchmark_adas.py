"""
Latency and FPS benchmark for SSD300 on ADAS hardware targets.

Targets:
  - Mac (MPS / Apple Silicon)
  - Mac (CPU)
  - ZCU104 ARM (CPU only)

Usage:
    python benchmarks/benchmark_adas.py [--device auto|cpu|mps|cuda]

Output:
    Prints a summary table and saves results/benchmark_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ssd_detector import SSDDetector, get_device

WARMUP_ITERS = 20
TIMED_ITERS = 50
RESULTS_PATH = Path(__file__).parent.parent / "results" / "benchmark_results.json"


def synchronize(device: torch.device) -> None:
    """Block until all device operations are complete (for accurate timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU: no-op (operations are already synchronous)


def run_benchmark(device_str: str) -> dict:
    """
    Run the benchmark on the specified device.

    Returns a dict with keys: device, avg_latency_ms, min_latency_ms, fps
    """
    detector = SSDDetector()
    device = detector.load_model(device_str)

    # Dummy input mimicking a 300x300 RGB image (SSD300 input size)
    dummy_input = [torch.randn(3, 300, 300).to(device)]

    print(f"\n--- Benchmarking on {device} ---")

    # Warmup
    print(f"Warmup: {WARMUP_ITERS} iterations ...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = detector.model(dummy_input)
    synchronize(device)

    # Timed iterations
    print(f"Timing: {TIMED_ITERS} iterations ...")
    latencies = []
    with torch.no_grad():
        for i in range(TIMED_ITERS):
            synchronize(device)
            t0 = time.perf_counter()
            _ = detector.model(dummy_input)
            synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    fps = 1000.0 / avg_ms

    result = {
        "device": str(device),
        "avg_latency_ms": round(avg_ms, 3),
        "min_latency_ms": round(min_ms, 3),
        "max_latency_ms": round(max_ms, 3),
        "fps": round(fps, 2),
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "torch_version": torch.__version__,
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"  Device         : {result['device']}")
    print(f"  Avg latency    : {result['avg_latency_ms']:.3f} ms")
    print(f"  Min latency    : {result['min_latency_ms']:.3f} ms")
    print(f"  Max latency    : {result['max_latency_ms']:.3f} ms")
    print(f"  FPS            : {result['fps']:.2f}")
    print(f"  PyTorch        : {result['torch_version']}")
    print("=" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark SSD300 for ADAS")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to benchmark on (default: auto — picks best available)",
    )
    args = parser.parse_args()

    print("=== ADAS SSD300 Benchmark ===")
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available : {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    result = run_benchmark(args.device)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to append (useful when running multiple devices)
    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_results = existing
            else:
                all_results = [existing]

    # Replace entry for same device or append
    all_results = [r for r in all_results if r.get("device") != result["device"]]
    all_results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
