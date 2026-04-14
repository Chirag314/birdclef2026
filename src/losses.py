"""
losses.py — loss functions for BirdCLEF 2026.

Phase 0: Binary Cross-Entropy with logits (multi-label).
Phase 1+: Focal loss and Asymmetric loss available as alternatives.

All losses accept:
    logits:  (B, n_classes) — raw model output (no sigmoid applied)
    targets: (B, n_classes) — float32 in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard BCE
# ---------------------------------------------------------------------------

class BCEWithLogitsLoss(nn.Module):
    """
    Vanilla BCE with logits. Supports optional per-class pos_weight for
    handling class imbalance.

    Args:
        pos_weight: None or (n_classes,) tensor. If None, no reweighting.
        label_smoothing: Smooth hard targets toward 0.5. 0.0 = off.
    """

    def __init__(self, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )


# ---------------------------------------------------------------------------
# Focal loss (Phase 1+ option)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.
    Reduces relative loss on well-classified examples.

    Args:
        gamma: Focusing parameter. 0 = BCE. Typical: 2.0.
        alpha: Positive class weight scalar (None = no weighting).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_t

        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Asymmetric loss (Phase 1+ option, good for multi-label with many negatives)
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) from Ben-Baruch et al. 2021.
    Decouples positive/negative focusing: down-weights easy negatives.

    Args:
        gamma_pos: Focusing for positives (typically 0).
        gamma_neg: Focusing for negatives (typically 4).
        clip:      Probability margin to shift negatives below (0.0 = off).
    """

    def __init__(self, gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Probability shift: clip easy negatives
        probs_neg = probs
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)

        # BCE for positives and negatives separately
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Focal weighting
        loss_pos = loss_pos * (1 - probs) ** self.gamma_pos
        loss_neg = loss_neg * probs_neg ** self.gamma_neg

        return -(loss_pos + loss_neg).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loss(config: dict, device: str = "cpu") -> nn.Module:
    """Build loss function from config loss.name."""
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "bce_with_logits")
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    pos_weight = None
    pw_cfg = loss_cfg.get("pos_weight", None)
    if pw_cfg is not None:
        import numpy as np
        pw = torch.tensor(np.array(pw_cfg), dtype=torch.float32).to(device)
        pos_weight = pw

    if name == "bce_with_logits":
        return BCEWithLogitsLoss(pos_weight=pos_weight,
                                 label_smoothing=label_smoothing)
    elif name == "focal_loss":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma)
    elif name == "asymmetric_loss":
        return AsymmetricLoss(
            gamma_pos=float(loss_cfg.get("gamma_pos", 0.0)),
            gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
            clip=float(loss_cfg.get("clip", 0.05)),
        )
    else:
        raise ValueError(f"Unknown loss: {name}. "
                         "Choose from: bce_with_logits, focal_loss, asymmetric_loss")
