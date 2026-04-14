"""
train_engine.py — training and validation loop for BirdCLEF 2026.

Key design choices:
- Per-taxon AUC is reported after every validation epoch (not just macro).
- OOF predictions are accumulated across folds and saved to NPZ at the end.
- Mixed precision (fp16) is supported via torch.cuda.amp.
- Early stopping on val_macro_roc_auc with configurable patience.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single epoch: train
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: str,
                    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
                    grad_accumulation: int = 1,
                    log_every: int = 50,
                    augmenter=None) -> Dict:
    """
    Train model for one epoch. Returns dict with 'train_loss'.

    Args:
        model:             BirdCLEFModel (or any nn.Module)
        loader:            DataLoader yielding (log_mel, label_vec) batches
        optimizer:         Configured optimizer
        loss_fn:           Loss function (logits, targets) → scalar
        device:            'cuda' or 'cpu'
        scaler:            GradScaler for fp16 (None → fp32)
        grad_accumulation: Number of steps before optimizer.step()
        log_every:         Log loss every N batches
        augmenter:         Augmenter instance for spec_augment (post-mel)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    t0 = time.time()

    for batch_idx, (log_mel, labels) in enumerate(loader):
        log_mel = log_mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Optional spec augmentation (applied after mel, before model)
        if augmenter is not None and augmenter.spec_aug_enabled:
            log_mel = augmenter.apply_spec(log_mel)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(log_mel)
                loss = loss_fn(logits, labels) / grad_accumulation
            scaler.scale(loss).backward()
        else:
            logits = model(log_mel)
            loss = loss_fn(logits, labels) / grad_accumulation
            loss.backward()

        if (batch_idx + 1) % grad_accumulation == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accumulation
        n_batches += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            logger.info(f"  batch {batch_idx+1}/{len(loader)} | "
                        f"loss={total_loss/n_batches:.4f} | "
                        f"{elapsed:.0f}s elapsed")

    return {"train_loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Single epoch: validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       loader: DataLoader,
                       device: str,
                       taxonomy_df: "pd.DataFrame") -> Dict:
    """
    Run validation and compute macro ROC-AUC per taxon.

    Returns dict with: val_loss (if labels provided), macro_auc, per_taxon_auc,
    and accumulated y_pred/y_true arrays for OOF saving.
    """
    from src.utils import compute_metrics, format_metrics

    model.eval()
    all_preds = []
    all_targets = []

    for log_mel, labels in loader:
        log_mel = log_mel.to(device, non_blocking=True)
        logits = model(log_mel)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.numpy()
        all_preds.append(probs)
        all_targets.append(targets)

    y_pred = np.concatenate(all_preds, axis=0)   # (N, 234)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 234)

    metrics = compute_metrics(y_true, y_pred, taxonomy_df)
    logger.info(f"Validation results:\n{format_metrics(metrics)}")

    metrics["y_pred"] = y_pred
    metrics["y_true"] = y_true
    return metrics


# ---------------------------------------------------------------------------
# Full training run (one fold)
# ---------------------------------------------------------------------------

def run_training(model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_fn: nn.Module,
                 taxonomy_df: "pd.DataFrame",
                 config: dict,
                 fold: int,
                 augmenter=None) -> Tuple[nn.Module, Dict]:
    """
    Full training loop for one fold.

    Returns: (best_model_state_dict, best_metrics)
    """
    from src.model import save_checkpoint

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = config["training"]["epochs"]
    patience = config["validation"].get("early_stopping_patience", 5)
    use_amp = (config["training"].get("mixed_precision", False)
               and device == "cuda")
    grad_accum = config["training"].get("gradient_accumulation", 1)
    log_every = config["logging"].get("log_every_n_steps", 50)
    ckpt_dir = config["checkpoint"]["save_dir"]
    exp_name = config.get("experiment_name", "baseline_v1")

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_auc = 0.0
    best_state = None
    best_metrics = {}
    patience_counter = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold} | {epochs} epochs | device={device} | amp={use_amp}")
    logger.info(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler,
            grad_accumulation=grad_accum,
            log_every=log_every,
            augmenter=augmenter,
        )

        val_metrics = validate_one_epoch(model, val_loader, device, taxonomy_df)

        # Step scheduler
        if scheduler is not None:
            if hasattr(scheduler, "step"):
                scheduler.step()

        macro_auc = val_metrics["macro_auc"]
        elapsed = time.time() - epoch_start
        logger.info(f"  train_loss={train_metrics['train_loss']:.4f} | "
                    f"val_macro_auc={macro_auc:.4f} | {elapsed:.0f}s")

        # Save best checkpoint
        if macro_auc > best_auc:
            best_auc = macro_auc
            best_state = {k: v.cpu().clone() for k, v in
                          model.state_dict().items()}
            best_metrics = val_metrics.copy()
            patience_counter = 0
            ckpt_path = f"{ckpt_dir}/{exp_name}_fold{fold}_best.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, config, ckpt_path)
            logger.info(f"  *** New best: {best_auc:.4f} → saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping (patience={patience})")
                break

    logger.info(f"\nFold {fold} complete. Best val macro AUC: {best_auc:.4f}")
    return best_state, best_metrics


# ---------------------------------------------------------------------------
# OOF accumulation helper
# ---------------------------------------------------------------------------

class OOFAccumulator:
    """Collects OOF predictions across folds."""

    def __init__(self, n_total: int, n_classes: int):
        self.preds = np.zeros((n_total, n_classes), dtype=np.float32)
        self.targets = np.zeros((n_total, n_classes), dtype=np.float32)
        self.fold_ids = np.full(n_total, -1, dtype=np.int32)
        self.filled = np.zeros(n_total, dtype=bool)

    def update(self, indices: np.ndarray, preds: np.ndarray,
               targets: np.ndarray, fold: int):
        self.preds[indices] = preds
        self.targets[indices] = targets
        self.fold_ids[indices] = fold
        self.filled[indices] = True

    def is_complete(self) -> bool:
        return self.filled.all()

    def get_arrays(self):
        mask = self.filled
        return self.preds[mask], self.targets[mask], self.fold_ids[mask]
