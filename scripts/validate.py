"""
validate.py — Load an OOF NPZ and print per-taxon AUC breakdown.

Usage:
    # From OOF file
    python scripts/validate.py --oof artifacts/oof/baseline_v1_oof.npz

    # From a checkpoint + val fold (runs inference)
    python scripts/validate.py \
        --checkpoint artifacts/checkpoints/baseline_v1_fold0_best.pt \
        --fold 0 \
        --config configs/train.yaml

Outputs to stdout and optionally to a JSON file.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (load_config, get_taxonomy_df, build_label_encoder,
                       compute_metrics, format_metrics, setup_logger)

logger = setup_logger("validate")


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--oof", help="Path to OOF NPZ file from train.py")
    g.add_argument("--checkpoint",
                   help="Path to checkpoint .pt file (also requires --fold --config)")
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index to validate (used with --checkpoint)")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--out-json", default=None,
                   help="Optional: write metrics JSON to this path")
    return p.parse_args()


def validate_from_oof(oof_path: str, taxonomy_csv: str) -> dict:
    """Load saved OOF predictions and compute metrics."""
    logger.info(f"Loading OOF from {oof_path}")
    data = np.load(oof_path, allow_pickle=True)
    y_pred = data["preds"]
    y_true = data["targets"]
    class_labels = data["class_labels"]
    fold_ids = data.get("fold_indices", np.zeros(len(y_pred), dtype=int))

    logger.info(f"OOF shape: {y_pred.shape} | folds: {np.unique(fold_ids)}")
    taxonomy_df = get_taxonomy_df(taxonomy_csv)
    metrics = compute_metrics(y_true, y_pred, taxonomy_df)

    # Per-fold AUC
    logger.info("\nPer-fold breakdown:")
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        fold_metrics = compute_metrics(y_true[mask], y_pred[mask], taxonomy_df)
        logger.info(f"  Fold {fold}: macro_auc={fold_metrics['macro_auc']:.4f} "
                    f"| {fold_metrics['n_classes_scored']} classes")

    return metrics


def validate_from_checkpoint(checkpoint_path: str, fold: int,
                               config_path: str) -> dict:
    """Run inference on a validation fold using a saved checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import build_clip_windows, build_soundscape_windows, merge_folds, WindowDataset
    from src.model import build_model, load_checkpoint

    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    taxonomy_df = get_taxonomy_df(cfg["paths"]["taxonomy_csv"])
    encoder = build_label_encoder(taxonomy_df)
    n_classes = cfg["competition"]["n_classes"]

    # Build window index (same as train.py)
    train_df = pd.read_csv(cfg["paths"]["train_csv"])
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    clip_windows = build_clip_windows(
        train_df, cfg["paths"]["train_audio"], encoder, n_classes
    )

    ss_labels_df = pd.read_csv(cfg["paths"]["soundscape_labels_csv"])
    ss_windows = build_soundscape_windows(
        ss_labels_df, cfg["paths"]["train_soundscapes"], encoder, n_classes
    )

    folds_df = pd.read_csv(cfg["paths"]["folds_csv"])
    clip_windows = merge_folds(clip_windows, folds_df)
    ss_windows = merge_folds(ss_windows, folds_df)
    all_windows = pd.concat([clip_windows, ss_windows], ignore_index=True)
    all_windows = all_windows[all_windows["fold"].notna()].copy()
    all_windows["fold"] = all_windows["fold"].astype(int)

    val_win = all_windows[all_windows["fold"] == fold].reset_index(drop=True)
    logger.info(f"Val fold {fold}: {len(val_win)} windows")

    val_ds = WindowDataset(val_win, cfg, augmenter=None, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = build_model(cfg, n_classes=n_classes).to(device)
    ckpt_meta = load_checkpoint(model, checkpoint_path, device=device)
    logger.info(f"Loaded checkpoint from epoch {ckpt_meta.get('epoch', '?')}")

    from src.train_engine import validate_one_epoch
    metrics = validate_one_epoch(model, val_loader, device, taxonomy_df)
    return metrics


def main():
    args = parse_args()

    if args.oof:
        cfg = load_config(args.config)
        metrics = validate_from_oof(args.oof, cfg["paths"]["taxonomy_csv"])
    else:
        if args.fold is None:
            logger.error("--fold required when using --checkpoint")
            sys.exit(1)
        metrics = validate_from_checkpoint(args.checkpoint, args.fold, args.config)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(format_metrics(metrics))
    print("=" * 60)

    # Worst-performing classes (bottom 10)
    per_class = metrics.get("per_class_auc", {})
    if per_class:
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        print("\nBottom 10 classes by AUC:")
        for label, auc in sorted_classes[:10]:
            print(f"  {label:<20} {auc:.4f}")
        print("\nTop 10 classes by AUC:")
        for label, auc in sorted_classes[-10:][::-1]:
            print(f"  {label:<20} {auc:.4f}")

    # Exit criterion check (Phase 0)
    macro_auc = metrics["macro_auc"]
    print(f"\nPhase 0 exit criterion check:")
    print(f"  macro_auc >= 0.65: {'PASS' if macro_auc >= 0.65 else 'FAIL'} "
          f"({macro_auc:.4f})")
    for taxon, auc in metrics.get("per_taxon_auc", {}).items():
        status = "PASS" if auc >= 0.60 else "WARN"
        print(f"  {taxon:<12} AUC >= 0.60: {status} ({auc:.4f})")

    if args.out_json:
        # Remove large arrays before serialising
        out = {k: v for k, v in metrics.items()
               if k not in ("y_pred", "y_true")}
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Metrics written to {args.out_json}")


if __name__ == "__main__":
    main()
