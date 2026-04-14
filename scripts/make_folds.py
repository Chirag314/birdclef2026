"""
make_folds.py — Generate site-stratified cross-validation folds.

Strategy:
  Labeled soundscapes (66 files, 9 sites): GroupKFold on site code.
  Train clips (35,549 files): StratifiedKFold on primary_label.

Both groups get a fold column (0–4). During training, for fold k:
  - val set  = rows where fold == k
  - train set = rows where fold != k

Output: data/folds.csv
Columns: filename, fold, source (clip|soundscape), site_code, primary_label

Usage:
    python scripts/make_folds.py
    python scripts/make_folds.py --config configs/folds.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/folds.yaml")
    p.add_argument("--base-config", default="configs/base.yaml")
    return p.parse_args()


def load_config(folds_path: str, base_path: str) -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}
    with open(folds_path) as f:
        override = yaml.safe_load(f) or {}
    cfg.update(override)
    return cfg


def parse_site_from_filename(filename: str) -> str:
    """Extract site code like 'S08' from soundscape filename."""
    import re
    m = re.search(r"_(S\d+)_", filename)
    return m.group(1) if m else "UNKNOWN"


def make_soundscape_folds(soundscape_labels_df: pd.DataFrame,
                           n_folds: int = 5) -> pd.DataFrame:
    """
    Assign folds to labeled soundscapes using GroupKFold on site_code.
    Returns DataFrame with columns: filename, site_code, fold, source='soundscape'
    """
    from sklearn.model_selection import GroupKFold

    df = soundscape_labels_df[["filename"]].drop_duplicates().copy()
    df["site_code"] = df["filename"].apply(parse_site_from_filename)
    df["source"] = "soundscape"
    df["fold"] = -1

    groups = df["site_code"].values
    X = np.arange(len(df))

    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(X, groups=groups)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    # Log distribution
    logger.info("Soundscape fold distribution:")
    for fold in range(n_folds):
        fold_df = df[df["fold"] == fold]
        sites = fold_df["site_code"].unique()
        logger.info(f"  Fold {fold}: {len(fold_df)} files | sites: {sorted(sites)}")

    return df


def make_clip_folds(train_df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Assign folds to training clips using StratifiedKFold on primary_label.

    Note: This does NOT prevent author leakage (same author in train+val is possible).
    This is the Phase 0 baseline. Author-stratified CV is a Phase 1 improvement.
    """
    from sklearn.model_selection import StratifiedKFold

    df = train_df[["filename", "primary_label"]].copy()
    df["primary_label"] = df["primary_label"].astype(str)
    df["source"] = "clip"
    df["site_code"] = "N/A"
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    X = np.arange(len(df))
    y = df["primary_label"].values

    for fold_idx, (_, val_idx) in enumerate(skf.split(X, y)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    # Verify all folds assigned
    assert (df["fold"] >= 0).all(), "Some rows have no fold assignment"

    # Log class coverage per fold
    logger.info("Clip fold distribution:")
    for fold in range(n_folds):
        n = (df["fold"] == fold).sum()
        n_classes = df[df["fold"] == fold]["primary_label"].nunique()
        logger.info(f"  Fold {fold}: {n} clips | {n_classes} distinct species in val")

    return df


def check_rare_class_coverage(folds_df: pd.DataFrame,
                               n_folds: int,
                               min_val_samples: int = 1) -> None:
    """
    Warn if any class has zero validation samples in any fold.
    These classes will produce NaN AUC (skipped in macro average).
    """
    rare_warnings = []
    for fold in range(n_folds):
        val = folds_df[folds_df["fold"] == fold]
        label_counts = val["primary_label"].value_counts()
        missing = set(folds_df["primary_label"].unique()) - set(label_counts.index)
        if missing:
            rare_warnings.append(f"  Fold {fold}: {len(missing)} classes absent from val")

    if rare_warnings:
        logger.warning("Rare class coverage issues:\n" + "\n".join(rare_warnings))
    else:
        logger.info("All classes present in at least one validation fold.")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.base_config)

    n_folds = cfg.get("n_folds", 5)
    output_path = cfg.get("output_path", "data/folds.csv")
    train_csv = cfg["paths"]["train_csv"]
    soundscape_labels_csv = cfg["paths"]["soundscape_labels_csv"]

    logger.info(f"Loading train data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    logger.info(f"  {len(train_df)} clips, {train_df['primary_label'].nunique()} species")

    logger.info(f"Loading soundscape labels from {soundscape_labels_csv}")
    ss_labels_df = pd.read_csv(soundscape_labels_csv)
    logger.info(f"  {ss_labels_df['filename'].nunique()} labeled soundscape files")

    # Build folds
    clip_folds = make_clip_folds(train_df, n_folds=n_folds)
    ss_folds = make_soundscape_folds(ss_labels_df, n_folds=n_folds)

    # Soundscapes: primary_label = first label in the semicolon list
    ss_folds["primary_label"] = ss_labels_df.drop_duplicates("filename").set_index("filename").loc[
        ss_folds["filename"].values, "primary_label"
    ].apply(lambda x: str(x).split(";")[0].strip()).values

    folds_df = pd.concat([clip_folds, ss_folds], ignore_index=True)

    # Sanity checks
    assert folds_df["fold"].between(0, n_folds - 1).all(), \
        "Some rows have invalid fold values"
    logger.info(f"\nTotal rows in folds.csv: {len(folds_df)}")
    logger.info(f"  clips:       {(folds_df['source']=='clip').sum()}")
    logger.info(f"  soundscapes: {(folds_df['source']=='soundscape').sum()}")

    check_rare_class_coverage(folds_df, n_folds)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(output_path, index=False)
    logger.info(f"\nFolds saved to {output_path}")

    # Quick summary table
    summary = folds_df.groupby(["fold", "source"]).size().unstack(fill_value=0)
    logger.info(f"\nFold summary:\n{summary.to_string()}")


if __name__ == "__main__":
    main()
