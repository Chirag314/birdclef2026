"""
losses.py — loss functions for BirdCLEF 2026.

Primary losses (classification head):
    BCEWithLogitsLoss   — vanilla multi-label BCE, with optional label smoothing
    FocalLoss           — binary focal loss, down-weights easy examples
    AsymmetricLoss      — ASL (Ben-Baruch et al. 2021), good for many negatives

Auxiliary loss (feature alignment):
    PerchAlignmentLoss  — MSE between projected backbone features and
                          precomputed 1536-dim Perch embeddings.
                          Both sides are L2-normalised before MSE so the
                          loss operates in cosine space (scale-invariant).

Composite:
    CombinedLoss        — any primary loss + optional PerchAlignmentLoss.
                          Backward-compatible: if features/perch_embs are
                          not supplied, behaves as the primary loss only.

Signatures
----------
Primary losses:
    forward(logits, targets) -> scalar

CombinedLoss:
    forward(logits, targets,
            features=None,      # (B, feat_dim) pooled backbone features
            perch_embs=None,    # (B, 1536) precomputed Perch embeddings
    ) -> scalar

Factory:
    get_loss(config, device, feat_dim=None) -> nn.Module
    Pass feat_dim=model.get_feature_dim() to enable Perch alignment.
    If feat_dim is None or loss.perch_weight == 0, Perch alignment is off.

Config keys (loss.*):
    name              : str   — bce_with_logits | focal_loss | asymmetric_loss
    label_smoothing   : float — 0.0 = off
    pos_weight        : list  — per-class positive weight (BCE only)
    focal_gamma       : float — focal gamma (default 2.0)
    gamma_pos/neg     : float — ASL gammas
    clip              : float — ASL probability clip
    perch_weight      : float — auxiliary loss coefficient (0.0 = off)
    perch_dim         : int   — Perch embedding dimension (default 1536)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard BCE
# ---------------------------------------------------------------------------

class BCEWithLogitsLoss(nn.Module):
    """
    Vanilla BCE with logits. Supports optional per-class pos_weight and
    label smoothing.

    Args:
        pos_weight:      None or (n_classes,) tensor.
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
# Focal loss
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
# Asymmetric loss
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) from Ben-Baruch et al. 2021.
    Decouples positive/negative focusing; down-weights easy negatives.

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

        probs_neg = probs
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        loss_pos = loss_pos * (1 - probs) ** self.gamma_pos
        loss_neg = loss_neg * probs_neg ** self.gamma_neg

        return -(loss_pos + loss_neg).mean()


# ---------------------------------------------------------------------------
# Perch feature-alignment loss
# ---------------------------------------------------------------------------

class PerchAlignmentLoss(nn.Module):
    """
    Auxiliary MSE loss that aligns EfficientNet pooled features with
    precomputed 1536-dimensional Perch target embeddings.

    A single learned linear projection maps backbone features into the
    Perch embedding space.  Both the projected features and the target
    embeddings are L2-normalised before computing MSE, so the loss
    operates in cosine space and is insensitive to embedding scale.

    Args:
        feat_dim:  Backbone pooling output dimension (e.g. 1280 for B0).
        perch_dim: Perch embedding dimension (default 1536).

    Forward:
        features   : (B, feat_dim) — pooled backbone features, pre-dropout.
        perch_embs : (B, perch_dim) — precomputed Perch target embeddings.

    Returns scalar MSE in the normalised embedding space.
    """

    PERCH_DIM: int = 1536

    def __init__(self, feat_dim: int, perch_dim: int = PERCH_DIM):
        super().__init__()
        self.projector = nn.Linear(feat_dim, perch_dim, bias=False)
        nn.init.xavier_uniform_(self.projector.weight)

    def forward(self,
                features: torch.Tensor,
                perch_embs: torch.Tensor) -> torch.Tensor:
        projected = self.projector(features)                   # (B, perch_dim)
        projected = F.normalize(projected, p=2, dim=-1)
        targets   = F.normalize(perch_embs.to(features.dtype), p=2, dim=-1)
        return F.mse_loss(projected, targets)


# ---------------------------------------------------------------------------
# Combined classification + Perch alignment loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Wraps any primary classification loss with an optional Perch
    feature-alignment term:

        total = cls_loss(logits, targets)
              + perch_weight * PerchAlignmentLoss(features, perch_embs)

    The Perch term is skipped (zero cost) when features or perch_embs
    are None, making this backward-compatible with the existing training
    loop which calls loss_fn(logits, targets).

    Args:
        cls_loss     : Any primary loss module (BCE, focal, ASL …).
        feat_dim     : Backbone pooling output dimension.
        perch_dim    : Perch embedding dimension (default 1536).
        perch_weight : Coefficient λ for the alignment term (default 1.0).

    Forward:
        logits     : (B, n_classes)
        targets    : (B, n_classes)
        features   : (B, feat_dim)   optional — pooled backbone features
        perch_embs : (B, perch_dim)  optional — Perch target embeddings
    """

    def __init__(self,
                 cls_loss: nn.Module,
                 feat_dim: int,
                 perch_dim: int = PerchAlignmentLoss.PERCH_DIM,
                 perch_weight: float = 1.0):
        super().__init__()
        self.cls_loss     = cls_loss
        self.perch_weight = perch_weight
        self.perch_loss   = PerchAlignmentLoss(feat_dim, perch_dim)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                features: torch.Tensor = None,
                perch_embs: torch.Tensor = None) -> torch.Tensor:
        loss = self.cls_loss(logits, targets)
        if (features is not None
                and perch_embs is not None
                and self.perch_weight > 0.0):
            loss = loss + self.perch_weight * self.perch_loss(features, perch_embs)
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loss(config: dict,
             device: str = "cpu",
             feat_dim: int = None) -> nn.Module:
    """
    Build a loss module from config.

    Args:
        config   : Full training config dict (reads config['loss']).
        device   : Target device for pos_weight tensor.
        feat_dim : Backbone pooling output dimension, required to enable
                   Perch alignment.  Obtain via model.get_feature_dim().
                   If None or if loss.perch_weight == 0, Perch alignment
                   is disabled and the plain primary loss is returned.

    Returns:
        CombinedLoss  when feat_dim is provided and perch_weight > 0,
        primary loss  otherwise.
    """
    loss_cfg = config.get("loss", {})
    name             = loss_cfg.get("name", "bce_with_logits")
    label_smoothing  = float(loss_cfg.get("label_smoothing", 0.0))
    perch_weight     = float(loss_cfg.get("perch_weight", 0.0))
    perch_dim        = int(loss_cfg.get("perch_dim", PerchAlignmentLoss.PERCH_DIM))

    pos_weight = None
    pw_cfg = loss_cfg.get("pos_weight", None)
    if pw_cfg is not None:
        import numpy as np
        pos_weight = torch.tensor(
            np.array(pw_cfg), dtype=torch.float32
        ).to(device)

    # Build primary loss
    if name == "bce_with_logits":
        primary = BCEWithLogitsLoss(
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
    elif name == "focal_loss":
        primary = FocalLoss(gamma=float(loss_cfg.get("focal_gamma", 2.0)))
    elif name == "asymmetric_loss":
        primary = AsymmetricLoss(
            gamma_pos=float(loss_cfg.get("gamma_pos", 0.0)),
            gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
            clip=float(loss_cfg.get("clip", 0.05)),
        )
    else:
        raise ValueError(
            f"Unknown loss '{name}'. "
            "Choose from: bce_with_logits, focal_loss, asymmetric_loss"
        )

    # Wrap with Perch alignment if requested and feat_dim is known
    if perch_weight > 0.0 and feat_dim is not None:
        return CombinedLoss(
            cls_loss=primary,
            feat_dim=feat_dim,
            perch_dim=perch_dim,
            perch_weight=perch_weight,
        )

    return primary
