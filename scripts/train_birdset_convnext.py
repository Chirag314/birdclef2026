"""
train_birdset_convnext.py — Fine-tune BirdSET ConvNeXt-Base-XCL on BirdCLEF 2026.

BirdSET ConvNeXt-Base is pre-trained on Xeno-Canto Large (9736 species).
We replace the 9736-class head with a 234-class head and fine-tune.

Input:  (B, 1, 128, 334) mel spectrogram — 5s at 32kHz, 128 mels
        Using our existing LogMelExtractor for consistency.

Writes:
  artifacts/birdset/fold{k}_best.pt
  artifacts/birdset/metrics.json

Usage:
  cd /data/birdclef2026
  .venv/bin/python scripts/train_birdset_convnext.py --fold 0 --epochs 20
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.features import LogMelExtractor
from src.losses import AsymmetricLoss
from src.dataset import WindowDataset as TrainDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_DIR    = pathlib.Path(__file__).parent.parent
BIRDSET_DIR = REPO_DIR / "artifacts" / "birdset"
OUT_DIR     = BIRDSET_DIR
LE_PATH     = REPO_DIR / "artifacts" / "exports" / "phase4_ls" / "label_encoder.json"


def build_convnext_model(n_classes: int, model_path: str, freeze_epochs: int = 5):
    """
    Load BirdSET ConvNeXt-Base backbone (ignore original classification head),
    add 234-class head.
    """
    from transformers import ConvNextModel
    import torch.nn as nn

    # Load backbone only (no classification head)
    backbone = ConvNextModel.from_pretrained(
        str(BIRDSET_DIR),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    hidden_size = backbone.config.hidden_sizes[-1]  # 1024 for ConvNeXt-Base
    logger.info(f"ConvNeXt-Base hidden_size={hidden_size}")

    class BirdSETModel(nn.Module):
        def __init__(self, backbone, hidden_size, n_classes):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, n_classes),
            )

        def forward(self, x):
            # x: (B, 1, H, W) — single channel mel spectrogram
            # BirdSET ConvNeXt was fine-tuned on 1-channel spectrograms
            out = self.backbone(pixel_values=x)
            feat = out.last_hidden_state  # (B, hidden, H', W')
            pooled = self.pool(feat).flatten(1)  # (B, hidden)
            return self.head(pooled)

    model = BirdSETModel(backbone, hidden_size, n_classes)
    return model


def compute_macro_auc(labels, preds):
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",        type=int,   default=0)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Epochs to freeze backbone, train head only")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(LE_PATH) as f:
        le = json.load(f)
    class_list = le["classes"]
    n_classes  = len(class_list)

    # Load the base config to build LogMelExtractor
    import yaml
    with open(REPO_DIR / "configs" / "train_phase4_ls.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.utils import load_config
    cfg = load_config(str(REPO_DIR / "configs" / "train_phase4_ls.yaml"))

    # Build model
    logger.info("Loading BirdSET ConvNeXt-Base...")
    model = build_convnext_model(n_classes, str(BIRDSET_DIR / "model.safetensors"),
                                  args.freeze_epochs)
    model = model.to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Build window index using the same pipeline as train.py
    from src.dataset import build_clip_windows, merge_folds
    from src.utils import get_taxonomy_df, build_label_encoder

    taxonomy_df = get_taxonomy_df(cfg["paths"]["taxonomy_csv"])
    label_encoder = build_label_encoder(taxonomy_df)

    folds_df = pd.read_csv(REPO_DIR / "data" / "folds.csv")
    audio_root = cfg["paths"]["train_audio"]
    n_classes  = len(label_encoder)
    windows    = build_clip_windows(folds_df, audio_root, label_encoder,
                                    n_classes, window_sec=5.0)
    windows    = merge_folds(windows, folds_df)

    val_mask   = windows["fold"] == args.fold
    train_win  = windows[~val_mask].reset_index(drop=True)
    val_win    = windows[val_mask].reset_index(drop=True)

    logger.info(f"Fold {args.fold}: train={len(train_win)} val={len(val_win)}")

    train_ds = TrainDataset(train_win, cfg, is_train=True)
    val_ds   = TrainDataset(val_win,   cfg, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

    criterion = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)

    # Phase 1: freeze backbone, train head only
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auc   = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            logger.info(f"Epoch {epoch}: unfreezing backbone")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6)

        model.train()
        tr_loss = 0.0
        for log_mel, labels in train_loader:
            log_mel, labels = log_mel.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(log_mel)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(log_mel)
        scheduler.step()
        tr_loss /= len(train_ds)

        # Validate every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for log_mel, labels in val_loader:
                    logits = model(log_mel.to(device))
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(labels.numpy())
            preds  = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            auc = compute_macro_auc(labels, preds)
            logger.info(f"Ep {epoch:2d}/{args.epochs} | loss={tr_loss:.4f} | val_auc={auc:.4f}")

            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save
    ckpt_path = OUT_DIR / f"fold{args.fold}_best.pt"
    torch.save({"model_state_dict": best_state, "best_auc": best_auc,
                "fold": args.fold, "args": vars(args)}, str(ckpt_path))
    logger.info(f"Saved → {ckpt_path} | best_auc={best_auc:.4f}")

    # Update metrics
    metrics_path = OUT_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    metrics[str(args.fold)] = best_auc
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
