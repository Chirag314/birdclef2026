"""
train_perch_mlp_v2.py — Perch MLP with soundscape holdout validation.

Key difference from v1: instead of dumping all 1478 SS windows into every
fold's training set, we hold out ~20% of SS windows for validation. This gives
honest OOF signal on the soundscape domain (proxy for test performance).

Writes:
  artifacts/perch/mlp_v2/fold{k}_best.pt
  artifacts/perch/mlp_v2/oof_ss_preds.npy   (N_ss_val, n_classes)
  artifacts/perch/mlp_v2/oof_ss_meta.csv
  artifacts/perch/mlp_v2/metrics.json

Usage:
  .venv/bin/python scripts/train_perch_mlp_v2.py [--epochs 80] [--lr 3e-3]
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
OUT_DIR  = REPO_DIR / "artifacts" / "perch" / "mlp_v2"
LE_PATH  = REPO_DIR / "artifacts" / "exports" / "phase4_ls" / "label_encoder.json"


def load_data():
    with open(LE_PATH) as f:
        le = json.load(f)
    class_list = le["classes"]
    class_idx  = {c: i for i, c in enumerate(class_list)}
    n_classes  = len(class_list)

    def to_multihot(labels_series):
        out = np.zeros((len(labels_series), n_classes), dtype=np.float32)
        for i, lbl in enumerate(labels_series):
            for l in str(lbl).split(";"):
                ci = class_idx.get(l.strip())
                if ci is not None:
                    out[i, ci] = 1.0
        return out

    clip_embs   = np.load(str(EMB_DIR / "clips_embeddings.npy"))
    clip_meta   = pd.read_csv(EMB_DIR / "clips_metadata.csv")
    clip_labels = to_multihot(clip_meta["primary_label"])
    clip_folds  = clip_meta["fold"].values.astype(int)

    ss_embs   = np.load(str(EMB_DIR / "ss_embeddings.npy"))
    ss_meta   = pd.read_csv(EMB_DIR / "ss_metadata.csv")
    ss_labels = to_multihot(ss_meta["primary_label"])

    logger.info(f"Clips: {clip_embs.shape}  SS: {ss_embs.shape}")
    return clip_embs, clip_labels, clip_folds, ss_embs, ss_labels, ss_meta, class_list


def smooth_labels(labels, eps):
    return labels * (1 - eps) + eps / labels.shape[1]


def compute_macro_auc(labels, preds):
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def train_fold(fold_idx, X_tr, Y_tr, X_val, Y_val, X_ss_val, Y_ss_val, n_classes, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Fold {fold_idx} | clip_train={len(X_tr)} clip_val={len(X_val)} ss_val={len(X_ss_val)} | {device}")

    model = PerchMLP(n_classes=n_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float()),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    best_auc   = 0.0
    best_state = None
    patience   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            Ys = smooth_labels(Yb, args.label_smoothing)
            optimizer.zero_grad()
            loss = criterion(model(Xb), Ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        scheduler.step()
        tr_loss /= len(X_tr)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                # Clip OOF AUC
                clip_preds = torch.sigmoid(
                    model(torch.from_numpy(X_val).float().to(device))
                ).cpu().numpy()
                clip_auc = compute_macro_auc(Y_val, clip_preds)

                # SS validation AUC (our honest soundscape-domain signal)
                ss_preds = torch.sigmoid(
                    model(torch.from_numpy(X_ss_val).float().to(device))
                ).cpu().numpy()
                ss_auc = compute_macro_auc(Y_ss_val, ss_preds)

            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"  Ep {epoch:3d}/{args.epochs} | loss={tr_loss:.4f} | "
                        f"clip_auc={clip_auc:.4f} | ss_auc={ss_auc:.4f} | lr={lr_now:.2e}")

            # Use SS AUC as the model selection criterion
            target_auc = ss_auc if len(X_ss_val) > 0 else clip_auc
            if target_auc > best_auc:
                best_auc   = target_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience  += 5
                if patience >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        clip_val_preds = torch.sigmoid(
            model(torch.from_numpy(X_val).float().to(device))
        ).cpu().numpy()
        ss_val_preds = torch.sigmoid(
            model(torch.from_numpy(X_ss_val).float().to(device))
        ).cpu().numpy() if len(X_ss_val) > 0 else np.empty((0, n_classes))

    return {
        "model_state_dict": best_state,
        "best_auc": best_auc,
        "clip_val_preds": clip_val_preds,
        "ss_val_preds": ss_val_preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",           type=str,  default="0,1,2,3,4")
    parser.add_argument("--epochs",          type=int,  default=80)
    parser.add_argument("--batch-size",      type=int,  default=512)
    parser.add_argument("--lr",              type=float, default=3e-3)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience",        type=int,  default=25)
    parser.add_argument("--ss-val-frac",     type=float, default=0.2,
                        help="Fraction of SS windows held out for validation")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_list = [int(x) for x in args.folds.split(",")]

    clip_embs, clip_labels, clip_folds, ss_embs, ss_labels, ss_meta, class_list = load_data()
    n_classes = len(class_list)

    # Split soundscapes into train/val by FILE (not by window) to avoid leakage.
    # Group windows by source soundscape filename, hold out 20% of files.
    ss_files = ss_meta["filename"].unique()
    rng = np.random.default_rng(42)
    n_val_files = max(1, int(len(ss_files) * args.ss_val_frac))
    val_files = set(rng.choice(ss_files, size=n_val_files, replace=False))
    ss_val_mask = ss_meta["filename"].isin(val_files).values
    ss_tr_mask  = ~ss_val_mask

    X_ss_tr,  Y_ss_tr  = ss_embs[ss_tr_mask],  ss_labels[ss_tr_mask]
    X_ss_val, Y_ss_val = ss_embs[ss_val_mask], ss_labels[ss_val_mask]
    logger.info(f"SS split: train={ss_tr_mask.sum()} val={ss_val_mask.sum()} "
                f"(from {len(ss_files)} files, {n_val_files} held out)")

    # Count zero-clip classes present in SS val set
    zc_in_val = int((Y_ss_val.sum(axis=0) > 0).sum())
    logger.info(f"Classes with positives in SS val: {zc_in_val}/{n_classes}")

    oof_clip_preds = np.zeros_like(clip_labels)
    all_ss_val_preds, all_ss_val_meta = [], []
    fold_aucs = {}

    for fold_idx in fold_list:
        val_mask   = (clip_folds == fold_idx)
        train_mask = ~val_mask

        X_tr = np.concatenate([clip_embs[train_mask], X_ss_tr], axis=0)
        Y_tr = np.concatenate([clip_labels[train_mask], Y_ss_tr], axis=0)
        X_val = clip_embs[val_mask]
        Y_val = clip_labels[val_mask]

        result = train_fold(fold_idx, X_tr, Y_tr, X_val, Y_val,
                            X_ss_val, Y_ss_val, n_classes, args)

        oof_clip_preds[val_mask] = result["clip_val_preds"]
        all_ss_val_preds.append(result["ss_val_preds"])
        all_ss_val_meta.append(ss_meta[ss_val_mask])
        fold_aucs[fold_idx] = result["best_auc"]

        ckpt_path = OUT_DIR / f"fold{fold_idx}_best.pt"
        torch.save({
            "model_state_dict": result["model_state_dict"],
            "best_auc":  result["best_auc"],
            "args":      vars(args),
        }, str(ckpt_path))
        logger.info(f"Fold {fold_idx} saved → {ckpt_path} | best_ss_auc={result['best_auc']:.4f}")

    # OOF summary
    if len(fold_list) > 1:
        clip_auc = compute_macro_auc(
            clip_labels[np.isin(clip_folds, fold_list)],
            oof_clip_preds[np.isin(clip_folds, fold_list)],
        )
        ss_preds_all = np.concatenate(all_ss_val_preds, axis=0) if all_ss_val_preds else None
        ss_auc = compute_macro_auc(Y_ss_val, all_ss_val_preds[-1]) if all_ss_val_preds else 0.0
        logger.info(f"OOF clip AUC ({len(fold_list)} folds): {clip_auc:.4f}")
        logger.info(f"SS holdout AUC (fold 0 val): {ss_auc:.4f}")
        fold_aucs["oof_clip"] = float(clip_auc)
        fold_aucs["ss_val"]   = float(ss_auc)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(fold_aucs, f, indent=2)

    logger.info(f"Metrics: {fold_aucs}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
