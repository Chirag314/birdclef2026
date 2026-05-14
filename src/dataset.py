"""
dataset.py — PyTorch datasets for BirdCLEF 2026.

Three components:
1. build_window_index()  — pre-compute all (file, start_sec, label_vec) rows
2. WindowDataset         — __getitem__ loads and extracts one 5-second window
3. InferenceDataset      — loads a full soundscape and yields all non-overlapping
                           windows for Kaggle submission

Label assignment rules (Phase 0):
- Clip windows: inherit clip-level primary_label; energy gate removes silent windows.
- Soundscape windows: use exact per-window labels from train_soundscapes_labels.csv.
- Secondary labels: controlled by config data.secondary_labels (default: ignore).
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window index construction
# ---------------------------------------------------------------------------

def build_clip_windows(train_df: pd.DataFrame,
                       audio_root: str,
                       encoder: Dict[str, int],
                       n_classes: int,
                       window_sec: float = 5.0,
                       energy_gate_db: float = -45.0,
                       secondary_mode: str = "ignore") -> pd.DataFrame:
    """
    Enumerate all non-overlapping 5-second windows from training clips.

    For each clip, we compute duration from soundfile metadata (no full decode),
    then create one row per window. Silent windows are flagged and excluded from
    training (label kept as zeros so they can optionally be used as negatives).

    Returns DataFrame with columns:
        filepath, start_sec, end_sec, primary_label, label_vec_idx (list),
        is_silent, fold (NaN until folds.csv is merged), source='clip'
    """
    import soundfile as sf
    from src.utils import labels_to_vec

    rows = []
    n_clips = len(train_df)
    unknown_labels = set()

    for i, row in train_df.iterrows():
        filepath = Path(audio_root) / row["filename"]
        if not filepath.exists():
            logger.warning(f"Missing clip: {filepath}")
            continue

        try:
            info = sf.info(str(filepath))
            duration = info.duration
        except Exception as e:
            logger.warning(f"Cannot read {filepath}: {e}")
            continue

        pl = str(row["primary_label"])
        if pl not in encoder:
            unknown_labels.add(pl)

        sec_labels = row.get("secondary_labels", "[]")
        label_vec = labels_to_vec(pl, sec_labels, encoder, n_classes, secondary_mode)

        # Enumerate non-overlapping windows
        n_windows = max(1, int(duration // window_sec))
        for w in range(n_windows):
            start = w * window_sec
            end = start + window_sec
            rows.append({
                "filepath": str(filepath),
                "start_sec": start,
                "end_sec": end,
                "primary_label": pl,
                "label_vec": label_vec,
                "source": "clip",
                "fold": np.nan,
            })

    if unknown_labels:
        logger.warning(f"{len(unknown_labels)} unknown primary_labels not in taxonomy: "
                       f"{list(unknown_labels)[:10]}")
    logger.info(f"Built {len(rows)} clip windows from {n_clips} clips")
    return pd.DataFrame(rows)


def _parse_sec(t) -> float:
    """Accept 'HH:MM:SS' strings or numeric seconds."""
    try:
        return float(t)
    except (ValueError, TypeError):
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)


def build_soundscape_windows(soundscape_labels_df: pd.DataFrame,
                              soundscape_root: str,
                              encoder: Dict[str, int],
                              n_classes: int) -> pd.DataFrame:
    """
    Build window rows from labeled train soundscapes.

    soundscape_labels.csv columns: filename, start, end, primary_label
    primary_label here is semicolon-separated: "22961;23158;24321;517063;65380"

    These are strong labels — used directly, no energy gating needed.
    """
    rows = []
    unknown_labels = set()

    for _, row in soundscape_labels_df.iterrows():
        filepath = Path(soundscape_root) / row["filename"]
        if not filepath.exists():
            logger.warning(f"Missing soundscape: {filepath}")
            continue

        raw_labels = str(row["primary_label"])
        label_strs = [s.strip() for s in raw_labels.split(";") if s.strip()]

        label_vec = np.zeros(n_classes, dtype=np.float32)
        for ls in label_strs:
            if ls in encoder:
                label_vec[encoder[ls]] = 1.0
            else:
                unknown_labels.add(ls)

        rows.append({
            "filepath": str(filepath),
            "start_sec": _parse_sec(row["start"]),
            "end_sec": _parse_sec(row["end"]),
            "primary_label": label_strs[0] if label_strs else "",
            "label_vec": label_vec,
            "source": "soundscape",
            "fold": np.nan,
        })

    if unknown_labels:
        logger.warning(f"Unknown soundscape labels: {list(unknown_labels)[:10]}")
    logger.info(f"Built {len(rows)} soundscape windows from "
                f"{soundscape_labels_df['filename'].nunique()} files")
    return pd.DataFrame(rows)


def merge_folds(window_df: pd.DataFrame,
                folds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign fold numbers to window_df rows by joining on filepath basename.

    folds_df columns: filename (basename without dir), fold, source
    """
    folds_df = folds_df.copy()
    folds_df["filepath_base"] = folds_df["filename"].apply(lambda f: Path(f).name)
    window_df["filepath_base"] = window_df["filepath"].apply(lambda f: Path(f).name)

    merged = window_df.merge(
        folds_df[["filepath_base", "fold"]],
        on="filepath_base",
        how="left",
        suffixes=("_old", ""),
    )
    # Use new fold column if present, otherwise keep original
    if "fold_old" in merged.columns:
        merged["fold"] = merged["fold"].fillna(merged["fold_old"])
        merged = merged.drop(columns=["fold_old"])
    merged = merged.drop(columns=["filepath_base"])

    n_no_fold = merged["fold"].isna().sum()
    if n_no_fold > 0:
        logger.warning(f"{n_no_fold} windows have no fold assignment — will be excluded from CV")
    return merged


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class WindowDataset:
    """
    PyTorch Dataset that loads and returns one 5-second spectrogram per item.

    Each row in window_df is one (file, start_sec, label_vec) tuple.
    Audio loading and mel extraction happen lazily in __getitem__.

    Args:
        window_df:      DataFrame from build_clip_windows / build_soundscape_windows
        config:         Full training config dict
        augmenter:      Optional Augmenter instance (None for validation)
        energy_gate_db: Skip silent windows below this dBFS (training only)
        is_train:       If True, apply augmentation and energy gating
    """

    def __init__(self,
                 window_df: pd.DataFrame,
                 config: dict,
                 augmenter=None,
                 energy_gate_db: float = -45.0,
                 is_train: bool = True):
        from src.features import LogMelExtractor
        self.df = window_df.reset_index(drop=True)
        self.config = config
        self.augmenter = augmenter
        self.energy_gate_db = energy_gate_db
        self.is_train = is_train
        self.window_samples = int(config["audio"]["sample_rate"] *
                                  config["competition"]["window_seconds"])
        self.extractor = LogMelExtractor(config)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch
        from src.features import load_audio, pad_or_trim, peak_normalize, is_silent

        row = self.df.iloc[idx]
        waveform = load_audio(
            row["filepath"],
            target_sr=self.config["audio"]["sample_rate"],
            offset=float(row["start_sec"]),
            duration=self.config["competition"]["window_seconds"],
        )
        waveform = pad_or_trim(waveform, self.window_samples)
        waveform = peak_normalize(waveform)

        label_vec = row["label_vec"].copy()

        # Energy gating during training: zero-out labels for silent windows
        if self.is_train and is_silent(waveform, self.energy_gate_db):
            label_vec = np.zeros_like(label_vec)

        # Augmentation (phase 0: all disabled)
        if self.is_train and self.augmenter is not None:
            waveform, label_vec = self.augmenter(waveform, label_vec)

        log_mel = self.extractor(waveform)  # (1, n_mels, T)

        return log_mel, torch.from_numpy(label_vec)


# ---------------------------------------------------------------------------
# Inference dataset (Kaggle submission)
# ---------------------------------------------------------------------------

class InferenceDataset:
    """
    Dataset for a single test soundscape at inference time.

    Yields non-overlapping 5-second windows and their row_ids.
    row_id format: {filename_stem}_{end_second}
    """

    def __init__(self, filepath: str, config: dict):
        from src.features import LogMelExtractor, load_audio, pad_or_trim, peak_normalize
        import soundfile as sf

        self.filepath = filepath
        self.config = config
        self.window_sec = config["competition"]["window_seconds"]
        self.sr = config["audio"]["sample_rate"]
        self.window_samples = int(self.sr * self.window_sec)
        self.extractor = LogMelExtractor(config)

        info = sf.info(filepath)
        self.total_duration = info.duration
        stem = Path(filepath).stem
        self.stem = stem

        self._windows: List[Tuple[float, float, str]] = []
        t = 0.0
        while t + self.window_sec <= self.total_duration + 1e-3:
            end_sec = int(round(t + self.window_sec))
            row_id = f"{stem}_{end_sec}"
            self._windows.append((t, t + self.window_sec, row_id))
            t += self.window_sec

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        import torch
        from src.features import load_audio, pad_or_trim, peak_normalize

        start, end, row_id = self._windows[idx]
        waveform = load_audio(
            self.filepath,
            target_sr=self.sr,
            offset=start,
            duration=self.window_sec,
        )
        waveform = pad_or_trim(waveform, self.window_samples)
        waveform = peak_normalize(waveform)
        log_mel = self.extractor(waveform)
        return log_mel, row_id

    @property
    def row_ids(self) -> List[str]:
        return [w[2] for w in self._windows]
