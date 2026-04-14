"""
benchmark_cpu.py — Profile CPU inference speed for Kaggle submission.

Simulates the Kaggle inference environment (CPU-only, no AMP).
Run this before finalizing a model to confirm it fits within the 90-minute budget.

Usage:
    python scripts/benchmark_cpu.py
    python scripts/benchmark_cpu.py --config configs/infer_kaggle.yaml --n-windows 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/infer_kaggle.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--n-windows", type=int, default=200,
                   help="Number of windows to benchmark over")
    p.add_argument("--checkpoint", default=None,
                   help="Optional: load a real checkpoint for accurate timing")
    return p.parse_args()


def main():
    args = parse_args()
    from src.utils import load_config, setup_logger
    from src.model import build_model, load_checkpoint
    from src.infer_engine import benchmark_inference

    logger = setup_logger("benchmark")
    infer_cfg = load_config(args.config)
    train_cfg = load_config(args.train_config)

    logger.info(f"Building model (EfficientNet-B0, CPU)...")
    model = build_model(train_cfg)
    model.eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device="cpu")
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.info("Using random weights (sufficient for timing benchmark)")

    result = benchmark_inference(model, infer_cfg, n_windows=args.n_windows)

    print("\n" + "=" * 60)
    print("CPU INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"  Torch version:          {__import__('torch').__version__}")
    print(f"  Batch size:             {result['batch_size']}")
    print(f"  Windows tested:         {result['n_windows_tested']}")
    print(f"  Time per window:        {result['time_per_window_ms']:.1f} ms")
    print(f"  Time per 60s soundscape: {result['time_per_soundscape_60s_sec']:.2f} s")
    print(f"  Max soundscapes in 90min: {result['max_soundscapes_in_90min']}")
    print("=" * 60)

    # Kaggle exit criterion
    limit = infer_cfg["runtime"]["time_limit_seconds"]
    per_sc = result["time_per_soundscape_60s_sec"]
    max_sc = int(limit / (per_sc + 0.1))

    if max_sc >= 200:
        print(f"\n  STATUS: PASS — fits {max_sc} soundscapes in budget (need ~50–200)")
    elif max_sc >= 50:
        print(f"\n  STATUS: WARN — fits {max_sc} soundscapes; monitor during submission")
    else:
        print(f"\n  STATUS: FAIL — only {max_sc} soundscapes in budget. "
              "Reduce model size or export to ONNX.")


if __name__ == "__main__":
    main()
