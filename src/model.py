"""
model.py — BirdCLEF 2026 model definition.

Architecture: EfficientNet-B0 (or other timm backbone) + GeM pooling + linear head.
Input:  (batch, 1, n_mels, time_frames) — single-channel log-mel spectrogram
Output: (batch, n_classes)              — raw logits (no sigmoid)

GeM pooling (Generalized Mean):
    Pools (C, H, W) → C using element-wise power-mean.
    p=1 → average pooling; p→∞ → max pooling.
    p=3 is the standard competition default and works well for audio.

Design note:
    EfficientNet-B0 expects 3-channel input by default. We set in_chans=1 in timm,
    which replaces the first conv layer with a single-channel equivalent.
    All other weights are unchanged (pretrained ImageNet features still transfer
    reasonably well — spectrograms share low-level edge statistics with images).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Applies learned or fixed power-mean over spatial dimensions.

    Args:
        p:         Initial power. Trainable if p is a Parameter.
        eps:       Small value to avoid numerical issues with p < 1.
        trainable: If True, p is learned during training.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → output: (B, C)
        p = self.p
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(p),
            kernel_size=(x.size(-2), x.size(-1)),
        ).pow(1.0 / p).squeeze(-1).squeeze(-1)

    def extra_repr(self) -> str:
        p_val = self.p.item() if isinstance(self.p, nn.Parameter) else self.p
        return f"p={p_val:.1f}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BirdCLEFModel(nn.Module):
    """
    BirdCLEF 2026 classification model.

    Pipeline:
        log-mel (B,1,128,500) → backbone feature map → GeM pool → dropout → linear

    Args:
        config: Full config dict (uses model.* keys)
        n_classes: Number of output classes (234)
    """

    def __init__(self, config: dict, n_classes: int = 234):
        super().__init__()
        import timm

        model_cfg = config["model"]
        backbone_name = model_cfg.get("backbone", "efficientnet_b0")
        pretrained = model_cfg.get("pretrained", True)
        in_channels = model_cfg.get("in_channels", 1)
        dropout_rate = model_cfg.get("dropout", 0.2)
        gem_p = model_cfg.get("gem_p", 3.0)

        # Create backbone — in_chans=1 replaces first conv layer
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,        # remove default classifier head
            global_pool="",       # remove default pooling (we use GeM)
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 500)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1]

        self.pool = GeMPooling(p=gem_p, trainable=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(feat_dim, n_classes)

        self._init_head()

    def _init_head(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor,
                return_features: bool = False):
        """
        Args:
            x:               (B, 1, n_mels, time_frames) log-mel spectrogram
            return_features: When True, also return pre-dropout pooled features
                             for use with PerchAlignmentLoss.

        Returns:
            logits           when return_features=False  (B, n_classes)
            (logits, pooled) when return_features=True   pooled: (B, feat_dim)
        """
        features = self.backbone(x)          # (B, C, H', W')
        pooled   = self.pool(features)       # (B, C)  pre-dropout
        logits   = self.head(self.dropout(pooled))  # (B, n_classes)
        if return_features:
            return logits, pooled
        return logits

    def get_feature_dim(self) -> int:
        return self.head.in_features


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: dict, n_classes: int = 234) -> BirdCLEFModel:
    """Build model from config. Handles device placement."""
    model = BirdCLEFModel(config, n_classes=n_classes)
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    device: str = "cpu") -> dict:
    """Load a checkpoint and return the saved metadata dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                    metrics: dict, config: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)
