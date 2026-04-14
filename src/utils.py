"""
utils.py — shared utilities for BirdCLEF 2026.

Covers: config loading, seed setting, label encoding, metric computation.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, base_path: Optional[str] = None) -> dict:
    """
    Load a YAML config file, merging with base.yaml from the same directory.
    Values in config_path override base.yaml values.
    """
    config_path = Path(config_path)
    if base_path is None:
        base_path = config_path.parent / "base.yaml"
    base_path = Path(base_path)

    config: dict = {}
    if base_path.exists():
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    return _deep_merge(config, override)


# ---------------------------------------------------------------------------
# Taxonomy / label encoding
# ---------------------------------------------------------------------------

def get_taxonomy_df(taxonomy_csv: str) -> pd.DataFrame:
    """Load taxonomy.csv and normalise primary_label to string."""
    df = pd.read_csv(taxonomy_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    return df.reset_index(drop=True)


def build_label_encoder(taxonomy_df: pd.DataFrame) -> Dict[str, int]:
    """
    Return {primary_label: class_index} using taxonomy_df row order.
    This order must match sample_submission.csv column order.
    """
    return {str(row["primary_label"]): idx
            for idx, row in taxonomy_df.iterrows()}


def labels_to_vec(primary_label: str,
                  secondary_labels,
                  encoder: Dict[str, int],
                  n_classes: int,
                  secondary_mode: str = "ignore") -> np.ndarray:
    """
    Build a float32 binary label vector of length n_classes.

    secondary_mode:
        "ignore"      — only primary label is set
        "soft_label"  — secondary labels set to 0.5
        "include"     — secondary labels set to 1.0
    """
    vec = np.zeros(n_classes, dtype=np.float32)
    pl = str(primary_label)
    if pl in encoder:
        vec[encoder[pl]] = 1.0

    if secondary_mode != "ignore" and secondary_labels:
        if isinstance(secondary_labels, str):
            secondary_labels = (secondary_labels.strip("[]")
                                .replace("'", "").split(","))
            secondary_labels = [s.strip() for s in secondary_labels if s.strip()]
        secondary_val = 0.5 if secondary_mode == "soft_label" else 1.0
        for sl in secondary_labels:
            sl = str(sl).strip()
            if sl in encoder:
                vec[encoder[sl]] = max(vec[encoder[sl]], secondary_val)

    return vec


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    taxonomy_df: pd.DataFrame) -> Dict:
    """
    Compute macro ROC-AUC overall and by taxonomic class.

    Args:
        y_true: (N, 234) binary float32
        y_pred: (N, 234) float32 scores
        taxonomy_df: DataFrame[primary_label, class_name]

    Returns dict with: macro_auc, n_classes_scored, per_class_auc, per_taxon_auc
    """
    from sklearn.metrics import roc_auc_score

    n_classes = y_true.shape[1]
    per_class_auc: Dict[str, float] = {}

    for i in range(n_classes):
        label = str(taxonomy_df.iloc[i]["primary_label"])
        if y_true[:, i].sum() == 0 or (1 - y_true[:, i]).sum() == 0:
            continue  # mirrors competition scoring rule
        per_class_auc[label] = float(roc_auc_score(y_true[:, i], y_pred[:, i]))

    macro_auc = (float(np.mean(list(per_class_auc.values())))
                 if per_class_auc else 0.0)

    per_taxon_auc: Dict[str, float] = {}
    for taxon in taxonomy_df["class_name"].unique():
        rows = taxonomy_df[taxonomy_df["class_name"] == taxon]
        scores = [per_class_auc[str(r["primary_label"])]
                  for _, r in rows.iterrows()
                  if str(r["primary_label"]) in per_class_auc]
        if scores:
            per_taxon_auc[taxon] = float(np.mean(scores))

    return {
        "macro_auc": macro_auc,
        "n_classes_scored": len(per_class_auc),
        "per_class_auc": per_class_auc,
        "per_taxon_auc": per_taxon_auc,
    }


def format_metrics(metrics: Dict) -> str:
    """Human-readable metric summary for logging."""
    lines = [
        f"  Macro AUC:        {metrics['macro_auc']:.4f}"
        f"  ({metrics['n_classes_scored']} classes scored)",
    ]
    for taxon, auc in sorted(metrics["per_taxon_auc"].items()):
        lines.append(f"  {taxon:<12} AUC: {auc:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_oof(oof_preds: np.ndarray,
             oof_targets: np.ndarray,
             fold_indices: np.ndarray,
             taxonomy_df: pd.DataFrame,
             out_dir: str,
             experiment_name: str) -> str:
    """Save OOF predictions and targets to NPZ for later analysis."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_name}_oof.npz")
    np.savez_compressed(
        path,
        preds=oof_preds.astype(np.float32),
        targets=oof_targets.astype(np.float32),
        fold_indices=fold_indices,
        class_labels=taxonomy_df["primary_label"].values,
    )
    return path
