"""
train_perch_mlp.py — 5-fold training of PerchMLP on precomputed embeddings.

Reads:
  artifacts/perch/embeddings/clips_embeddings.npy
  artifacts/perch/embeddings/clips_metadata.csv
  artifacts/perch/embeddings/ss_embeddings.npy
  artifacts/perch/embeddings/ss_metadata.csv
  artifacts/exports/phase4_ls/label_encoder.json
  data/folds.csv

Writes:
  artifacts/perch/mlp/fold{k}_best.pt    (model_state_dict, val_auc, epoch)
  artifacts/perch/mlp/oof_preds.npy      (N_clips, n_classes) OOF predictions
  artifacts/perch/mlp/oof_meta.csv
  artifacts/perch/mlp/metrics.json

Usage:
  cd /data/birdclef2026
  .venv/bin/python scripts/train_perch_mlp.py [--folds 0,1,2,3,4] [--epochs 60] [--lr 3e-3]
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.perch_mlp import PerchMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_DIR = pathlib.Path(__file__).parent.parent
EMB_DIR  = REPO_DIR / "artifacts" / "perch" / "embeddings"
OUT_DIR  = REPO_DIR / "artifacts" / "perch" / "mlp"
LE_PATH  = REPO_DIR / "artifacts" / "exports" / "phase4_ls" / "label_encoder.json"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_embeddings():
    """Load and merge clip + soundscape embeddings with multi-hot labels."""
    with open(LE_PATH) as f:
        le = json.load(f)
    class_list = le["classes"]
    class_idx  = {c: i for i, c in enumerate(class_list)}
    n_classes  = len(class_list)

    # --- Clips ---
    clip_embs  = np.load(str(EMB_DIR / "clips_embeddings.npy"))   # (N, 1536)
    clip_meta  = pd.read_csv(EMB_DIR / "clips_metadata.csv")       # filename, primary_label, fold, source

    # Single-label → multi-hot
    clip_labels = np.zeros((len(clip_meta), n_classes), dtype=np.float32)
    for i, label in enumerate(clip_meta["primary_label"]):
        for lbl in str(label).split(";"):
            ci = class_idx.get(lbl.strip())
            if ci is not None:
                clip_labels[i, ci] = 1.0

    folds = clip_meta["fold"].values.astype(int)

    # --- Soundscapes (no fold assignment — include in all training sets) ---
    ss_emb_path  = EMB_DIR / "ss_embeddings.npy"
    ss_meta_path = EMB_DIR / "ss_metadata.csv"
    ss_embs, ss_labels = None, None
    if ss_emb_path.exists() and ss_meta_path.exists():
        ss_embs  = np.load(str(ss_emb_path))
        ss_meta  = pd.read_csv(ss_meta_path)
        ss_labels = np.zeros((len(ss_meta), n_classes), dtype=np.float32)
        for i, label in enumerate(ss_meta["primary_label"]):
            for lbl in str(label).split(";"):
                ci = class_idx.get(lbl.strip())
                if ci is not None:
                    ss_labels[i, ci] = 1.0
        logger.info(f"Soundscape embeddings: {ss_embs.shape}")

    return clip_embs, clip_labels, folds, ss_embs, ss_labels, class_list


# ------------------------------------------------------------------
# Training loop for one fold
# ------------------------------------------------------------------

def smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Binary label smoothing: 1→(1-eps), 0→eps."""
    return labels * (1 - smoothing) + smoothing / labels.shape[1]


def train_fold(fold_idx: int,
               X_train: np.ndarray, Y_train: np.ndarray,
               X_val:   np.ndarray, Y_val:   np.ndarray,
               class_list: list,
               args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Fold {fold_idx} | train={len(X_train)} val={len(X_val)} | device={device}")

    n_classes = len(class_list)
    model = PerchMLP(n_classes=n_classes, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_auc  = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            Y_smooth = smooth_labels(Y_b, args.label_smoothing)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, Y_smooth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        scheduler.step()
        train_loss /= len(X_train)

        # Validate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_b, _ in val_loader:
                    logits = model(X_b.to(device))
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_preds = np.concatenate(val_preds, axis=0)

            # Macro ROC-AUC (only on classes with positive examples in val)
            try:
                active = Y_val.sum(axis=0) > 0
                auc = roc_auc_score(Y_val[:, active], val_preds[:, active], average="macro")
            except Exception:
                auc = 0.0

            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | val_auc={auc:.4f} | lr={lr_now:.2e}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 5
                if patience_count >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

    # Return best model's val predictions for OOF
    model.load_state_dict(best_state)
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            logits = model(X_b.to(device))
            val_preds.append(torch.sigmoid(logits).cpu().numpy())
    val_preds = np.concatenate(val_preds, axis=0)

    return {"model_state_dict": best_state, "val_auc": best_auc, "val_preds": val_preds}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",         type=str,  default="0,1,2,3,4")
    parser.add_argument("--epochs",        type=int,  default=60)
    parser.add_argument("--batch-size",    type=int,  default=512)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience",      type=int,  default=20)
    args = parser.parse_args()

    fold_list = [int(x) for x in args.folds.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings...")
    clip_embs, clip_labels, clip_folds, ss_embs, ss_labels, class_list = load_embeddings()
    logger.info(f"Clips: {clip_embs.shape}  Labels: {clip_labels.shape}")
    if ss_embs is not None:
        logger.info(f"Soundscapes: {ss_embs.shape}")

    oof_preds = np.zeros_like(clip_labels)
    fold_aucs = {}

    for fold_idx in fold_list:
        val_mask   = (clip_folds == fold_idx)
        train_mask = ~val_mask

        X_val,   Y_val   = clip_embs[val_mask],   clip_labels[val_mask]
        X_train, Y_train = clip_embs[train_mask], clip_labels[train_mask]

        # Append soundscapes to training (never in validation)
        if ss_embs is not None:
            X_train = np.concatenate([X_train, ss_embs], axis=0)
            Y_train = np.concatenate([Y_train, ss_labels], axis=0)
            logger.info(f"  Fold {fold_idx}: train after SS concat = {len(X_train)}")

        result = train_fold(fold_idx, X_train, Y_train, X_val, Y_val, class_list, args)

        oof_preds[val_mask] = result["val_preds"]
        fold_aucs[fold_idx] = result["val_auc"]

        ckpt_path = OUT_DIR / f"fold{fold_idx}_best.pt"
        torch.save({
            "model_state_dict": result["model_state_dict"],
            "val_auc": result["val_auc"],
            "epoch":   args.epochs,
            "args":    vars(args),
        }, str(ckpt_path))
        logger.info(f"Fold {fold_idx} saved → {ckpt_path} | val_auc={result['val_auc']:.4f}")

    # OOF summary
    n_folds_done = len(fold_list)
    if n_folds_done > 1:
        # Macro AUC over folds that were run
        mask = np.zeros(len(clip_labels), dtype=bool)
        for k in fold_list:
            mask |= (clip_folds == k)
        active = clip_labels[mask].sum(axis=0) > 0
        try:
            oof_auc = roc_auc_score(clip_labels[mask][:, active], oof_preds[mask][:, active], average="macro")
        except Exception:
            oof_auc = 0.0
        logger.info(f"OOF AUC ({n_folds_done} folds): {oof_auc:.4f}")
        fold_aucs["oof"] = float(oof_auc)

    # Save OOF
    val_indices = np.where(np.isin(clip_folds, fold_list))[0]
    np.save(str(OUT_DIR / "oof_preds.npy"), oof_preds[val_indices])

    folds_df = pd.read_csv(REPO_DIR / "data" / "folds.csv")
    folds_df[folds_df["fold"].isin(fold_list)].to_csv(OUT_DIR / "oof_meta.csv", index=False)

    # Metrics
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(fold_aucs, f, indent=2)
    logger.info(f"Metrics: {fold_aucs}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
