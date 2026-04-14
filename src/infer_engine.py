"""
infer_engine.py — Kaggle CPU inference engine for BirdCLEF 2026.

Constraints:
  - CPU-only, 90-minute hard limit (5400s → we stop at 5100s)
  - No GPU, no mixed precision
  - EfficientNet-B0 target: ~2–3 ms per 5-second window on Kaggle CPU
  - Batch processing is essential for throughput

Usage (see kaggle_notebook/inference.py for full submission notebook):
    engine = InferenceEngine(model, config)
    submission_df = engine.run(test_dir, taxonomy_df, sample_submission)
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Runs inference over all test soundscapes and builds submission DataFrame.

    Args:
        model:      Trained BirdCLEFModel, already loaded and eval()
        config:     Inference config dict (from infer_kaggle.yaml)
        device:     'cpu' (default for Kaggle submission)
    """

    def __init__(self, model: nn.Module, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.batch_size = config["inference"].get("batch_size", 32)
        self.time_limit = config["runtime"].get("time_limit_seconds", 5100)
        self.fill_value = config["submission"].get(
            "fill_value", 1.0 / 234
        )

    def run(self,
            test_dir: str,
            taxonomy_df: pd.DataFrame,
            sample_submission: pd.DataFrame,
            time_budget: Optional[float] = None) -> pd.DataFrame:
        """
        Iterate over all test soundscapes, run inference, build submission.

        Args:
            test_dir:          Directory containing test soundscape OGG files
            taxonomy_df:       DataFrame with primary_label column (234 rows, ordered)
            sample_submission: sample_submission.csv DataFrame (defines column order)
            time_budget:       Override time limit in seconds (None = use config)

        Returns:
            submission DataFrame with row_id column and 234 probability columns
        """
        from src.dataset import InferenceDataset

        if time_budget is not None:
            self.time_limit = time_budget

        start_time = time.time()
        class_cols = [c for c in sample_submission.columns if c != "row_id"]
        n_classes = len(class_cols)

        test_files = sorted(Path(test_dir).glob("*.ogg"))
        logger.info(f"Found {len(test_files)} test soundscapes in {test_dir}")

        all_row_ids = []
        all_probs = []

        for file_idx, filepath in enumerate(test_files):
            elapsed = time.time() - start_time
            if elapsed > self.time_limit:
                logger.warning(f"Time limit reached at file {file_idx}/{len(test_files)}")
                break

            try:
                probs, row_ids = self._process_soundscape(str(filepath))
                all_row_ids.extend(row_ids)
                all_probs.append(probs)
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                # Fill with prior probability rather than crash
                ds = _safe_row_ids(str(filepath), self.config)
                for rid in ds:
                    all_row_ids.append(rid)
                    all_probs.append(
                        np.full((1, n_classes), self.fill_value, dtype=np.float32)
                    )

            if (file_idx + 1) % 10 == 0:
                logger.info(f"  {file_idx+1}/{len(test_files)} files | "
                             f"{time.time()-start_time:.0f}s elapsed")

        if all_probs:
            all_probs = np.concatenate(all_probs, axis=0)
        else:
            all_probs = np.zeros((0, n_classes), dtype=np.float32)

        submission = pd.DataFrame(all_probs, columns=class_cols)
        submission.insert(0, "row_id", all_row_ids)

        # Reindex to match sample_submission row order if possible
        expected_row_ids = sample_submission["row_id"].tolist()
        if set(all_row_ids) >= set(expected_row_ids):
            submission = (submission
                          .set_index("row_id")
                          .reindex(expected_row_ids)
                          .fillna(self.fill_value)
                          .reset_index()
                          .rename(columns={"index": "row_id"}))

        total_time = time.time() - start_time
        logger.info(f"Inference complete: {len(submission)} rows | {total_time:.0f}s total")
        return submission

    @torch.no_grad()
    def _process_soundscape(self, filepath: str):
        """
        Process one soundscape and return (probs_array, row_ids).

        probs_array: (n_windows, n_classes) float32
        """
        from src.dataset import InferenceDataset
        from torch.utils.data import DataLoader

        ds = InferenceDataset(filepath, self.config)
        if len(ds) == 0:
            return np.zeros((0, 234), dtype=np.float32), []

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # must be 0 on Kaggle CPU (fork overhead)
            pin_memory=False,
        )

        probs_list = []
        row_ids = []
        for log_mel_batch, batch_row_ids in loader:
            log_mel_batch = log_mel_batch.to(self.device)
            logits = self.model(log_mel_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            row_ids.extend(batch_row_ids)

        return np.concatenate(probs_list, axis=0), row_ids


def _safe_row_ids(filepath: str, config: dict) -> List[str]:
    """Fallback: generate expected row_ids without loading audio."""
    import soundfile as sf
    stem = Path(filepath).stem
    window_sec = config["competition"]["window_seconds"]
    try:
        info = sf.info(filepath)
        duration = info.duration
    except Exception:
        duration = 60.0  # assume 60s soundscape
    row_ids = []
    t = 0.0
    while t + window_sec <= duration + 1e-3:
        end_sec = int(round(t + window_sec))
        row_ids.append(f"{stem}_{end_sec}")
        t += window_sec
    return row_ids


def benchmark_inference(model: nn.Module, config: dict,
                        n_windows: int = 100) -> dict:
    """
    Estimate per-window and per-soundscape inference time on CPU.
    Run this before submission to verify 90-minute budget is safe.

    A 60-second soundscape = 12 windows.
    Target: < 0.5s per soundscape (generous margin for 200 soundscapes in 90 min).
    """
    import time
    model.eval()
    batch_size = config["inference"].get("batch_size", 32)
    n_mels = config["audio"]["n_mels"]
    window_samples = int(config["audio"]["sample_rate"] *
                         config["competition"]["window_seconds"])
    frames = int(window_samples / config["audio"]["hop_length"])

    dummy = torch.zeros(batch_size, 1, n_mels, frames)
    n_batches = n_windows // batch_size

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            model(dummy)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_batches):
            model(dummy)
    elapsed = time.time() - t0

    time_per_window = elapsed / n_windows
    time_per_soundscape_60s = time_per_window * 12
    max_soundscapes_in_budget = int(5100 / (time_per_soundscape_60s + 0.1))

    result = {
        "time_per_window_ms": time_per_window * 1000,
        "time_per_soundscape_60s_sec": time_per_soundscape_60s,
        "max_soundscapes_in_90min": max_soundscapes_in_budget,
        "batch_size": batch_size,
        "n_windows_tested": n_windows,
    }
    logger.info(
        f"Inference benchmark:\n"
        f"  {time_per_window*1000:.1f} ms/window | "
        f"  {time_per_soundscape_60s:.2f} s/soundscape | "
        f"  est. {max_soundscapes_in_budget} soundscapes in 90 min"
    )
    return result
