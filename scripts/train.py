"""
train.py — Main training entrypoint for BirdCLEF 2026.

Usage:
    # Full 5-fold CV
    python scripts/train.py --config configs/train.yaml

    # Single fold (for fast iteration)
    python scripts/train.py --config configs/train.yaml --fold 0

    # Dry run (1 epoch, small batch)
    python scripts/train.py --config configs/train.yaml --dry-run

Requirements:
    pip install torch torchaudio timm scikit-learn soundfile pyyaml tqdm

Phase 0 exit criterion:
    OOF macro AUC > 0.65 AND inference time < 60 min on CPU.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (seed_everything, load_config, get_taxonomy_df,
                       build_label_encoder, compute_metrics, format_metrics,
                       setup_logger, ensure_dirs, save_oof)
from src.features import LogMelExtractor
from src.dataset import (build_clip_windows, build_soundscape_windows,
                          merge_folds, WindowDataset)
from src.model import build_model, count_parameters
from src.losses import get_loss
from src.augment import Augmenter
from src.train_engine import run_training, OOFAccumulator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--fold", type=int, default=None,
                   help="Run a single fold only (None = all 5 folds)")
    p.add_argument("--dry-run", action="store_true",
                   help="1 epoch, 2 batches — verify pipeline without full training")
    p.add_argument("--device", default=None,
                   help="Override device (cuda/cpu)")
    return p.parse_args()


def build_optimizer_scheduler(model, config, steps_per_epoch: int):
    opt_cfg = config["optimizer"]
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    lr = float(opt_cfg["lr"])
    bb_lr = lr * float(opt_cfg.get("backbone_lr_multiplier", 0.1))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": bb_lr},
        {"params": head_params, "lr": lr},
    ], weight_decay=wd)

    sched_cfg = config["scheduler"]
    epochs = config["training"]["epochs"]
    warmup = sched_cfg.get("warmup_epochs", 2)
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return cosine * (1 - min_lr / lr) + min_lr / lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def get_sampler(window_df: pd.DataFrame, config: dict):
    """Optional weighted sampler for rare species oversampling (Phase 2+)."""
    # Phase 0: uniform sampling
    return None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_name = cfg.get("experiment_name", "baseline_v1")

    # Setup
    seed_everything(cfg.get("seed", 42))
    log_dir = cfg["logging"]["log_dir"]
    ensure_dirs(log_dir, cfg["checkpoint"]["save_dir"],
                os.path.join(cfg["paths"]["artifacts"], "oof"))
    logger = setup_logger("train",
                          log_file=f"{log_dir}/{exp_name}.log")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device
    logger.info(f"Experiment: {exp_name} | device: {device}")
    logger.info(f"Config: {args.config}")

    if args.dry_run:
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 8
        logger.info("DRY RUN: 1 epoch, batch_size=8")

    # ---------------------------------------------------------------------------
    # Data setup
    # ---------------------------------------------------------------------------
    taxonomy_df = get_taxonomy_df(cfg["paths"]["taxonomy_csv"])
    encoder = build_label_encoder(taxonomy_df)
    n_classes = cfg["competition"]["n_classes"]
    window_sec = cfg["competition"]["window_seconds"]
    sec_mode = cfg["data"].get("secondary_labels", "ignore")

    logger.info("Building clip window index...")
    train_df = pd.read_csv(cfg["paths"]["train_csv"])
    train_df["primary_label"] = train_df["primary_label"].astype(str)

    # Optional: filter by minimum rating
    min_rating = cfg["data"].get("min_rating", 0.0)
    if min_rating > 0:
        before = len(train_df)
        train_df = train_df[
            (train_df["rating"].isna()) | (train_df["rating"] >= min_rating)
        ]
        logger.info(f"Rating filter (>= {min_rating}): {before} → {len(train_df)} clips")

    clip_windows = build_clip_windows(
        train_df,
        audio_root=cfg["paths"]["train_audio"],
        encoder=encoder,
        n_classes=n_classes,
        window_sec=window_sec,
        energy_gate_db=-45.0,
        secondary_mode=sec_mode,
    )

    if cfg["data"].get("use_train_soundscapes", True):
        logger.info("Building soundscape window index...")
        ss_labels_df = pd.read_csv(cfg["paths"]["soundscape_labels_csv"])
        ss_windows = build_soundscape_windows(
            ss_labels_df,
            soundscape_root=cfg["paths"]["train_soundscapes"],
            encoder=encoder,
            n_classes=n_classes,
        )
    else:
        ss_windows = pd.DataFrame()

    # Merge fold assignments
    folds_path = cfg["paths"]["folds_csv"]
    if not Path(folds_path).exists():
        logger.error(f"folds.csv not found at {folds_path}. "
                     "Run: python scripts/make_folds.py")
        sys.exit(1)
    folds_df = pd.read_csv(folds_path)

    clip_windows = merge_folds(clip_windows, folds_df)
    if not ss_windows.empty:
        ss_windows = merge_folds(ss_windows, folds_df)
        all_windows = pd.concat([clip_windows, ss_windows], ignore_index=True)
    else:
        all_windows = clip_windows

    # Drop windows with no fold assignment
    no_fold = all_windows["fold"].isna()
    if no_fold.sum() > 0:
        logger.warning(f"Dropping {no_fold.sum()} windows without fold assignment")
        all_windows = all_windows[~no_fold].copy()
    all_windows["fold"] = all_windows["fold"].astype(int)

    n_folds = folds_df["fold"].nunique()
    fold_range = [args.fold] if args.fold is not None else list(range(n_folds))
    logger.info(f"Running folds: {fold_range}")

    # OOF accumulator
    oof = OOFAccumulator(len(all_windows), n_classes)

    # ---------------------------------------------------------------------------
    # Cross-validation loop
    # ---------------------------------------------------------------------------
    for fold in fold_range:
        logger.info(f"\n{'='*60}\nStarting fold {fold}\n{'='*60}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        train_mask = all_windows["fold"] != fold
        val_mask = all_windows["fold"] == fold
        train_win = all_windows[train_mask].reset_index(drop=True)
        val_win = all_windows[val_mask].reset_index(drop=True)
        val_indices = all_windows[val_mask].index.values

        logger.info(f"Fold {fold}: train={len(train_win)} | val={len(val_win)}")

        augmenter = Augmenter(cfg, background_pool=[], window_df=train_win)

        train_ds = WindowDataset(train_win, cfg, augmenter=augmenter,
                                 is_train=True)
        val_ds = WindowDataset(val_win, cfg, augmenter=None, is_train=False)

        bs = cfg["training"]["batch_size"]
        nw = cfg["training"].get("num_workers", 4)
        if args.dry_run:
            nw = 0  # avoid fork overhead in dry run

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=(device == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=(device == "cuda"),
        )

        model = build_model(cfg, n_classes=n_classes).to(device)
        params = count_parameters(model)
        logger.info(f"Model params: {params['trainable']:,} trainable / "
                    f"{params['total']:,} total")

        optimizer, scheduler = build_optimizer_scheduler(
            model, cfg, steps_per_epoch=len(train_loader)
        )
        loss_fn = get_loss(cfg, device=device)

        if args.dry_run:
            # Override train loader to just 2 batches
            from itertools import islice
            class TwoStepLoader:
                def __init__(self, loader): self._loader = loader
                def __iter__(self): return islice(iter(self._loader), 2)
                def __len__(self): return 2
            train_loader = TwoStepLoader(train_loader)

        best_state, best_metrics = run_training(
            model, train_loader, val_loader,
            optimizer, scheduler, loss_fn,
            taxonomy_df, cfg, fold,
            augmenter=augmenter,
        )

        # Accumulate OOF
        if best_state is not None:
            model.load_state_dict(best_state)
            oof.update(
                val_indices,
                best_metrics.get("y_pred", np.zeros((len(val_win), n_classes))),
                best_metrics.get("y_true", np.zeros((len(val_win), n_classes))),
                fold,
            )

        logger.info(f"Fold {fold} best macro AUC: {best_metrics.get('macro_auc', 0):.4f}")

    # ---------------------------------------------------------------------------
    # OOF summary
    # ---------------------------------------------------------------------------
    if oof.is_complete():
        oof_preds, oof_targets, fold_ids = oof.get_arrays()
        oof_metrics = compute_metrics(oof_targets, oof_preds, taxonomy_df)
        logger.info(f"\nFull OOF metrics:\n{format_metrics(oof_metrics)}")

        oof_path = save_oof(
            oof_preds, oof_targets, fold_ids, taxonomy_df,
            out_dir=os.path.join(cfg["paths"]["artifacts"], "oof"),
            experiment_name=exp_name,
        )
        logger.info(f"OOF saved to {oof_path}")

    logger.info("\nTraining complete.")


if __name__ == "__main__":
    main()
