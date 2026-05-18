"""
BirdCLEF 2026 — Inference Notebook
baseline_v1: EfficientNet-B0, 5-fold ensemble, CPU inference

To run on Kaggle:
  - Attach dataset: birdclef2026-model  → /kaggle/input/birdclef2026-model
  - Attach dataset: birdclef-2026       → /kaggle/input/birdclef-2026
  - Internet: OFF
  - Runtime: CPU
"""

import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths — adjust if dataset names differ
# ---------------------------------------------------------------------------
def _find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise RuntimeError(f"None of these paths exist: {candidates}")

MODEL_DIR = _find_dir([
    "/kaggle/input/birdclef2026-model",
    "/kaggle/input/datasets/cid007/birdclef2026-model",
])
DATA_DIR = _find_dir([
    "/kaggle/input/birdclef-2026",
    "/kaggle/input/competitions/birdclef-2026",
])
OUTPUT = Path("/kaggle/working/submission.csv")

print(f"MODEL_DIR: {MODEL_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print("MODEL_DIR files:", [p.name for p in MODEL_DIR.iterdir()])

# src/ may be a directory or a zip (kaggle datasets version --dir-mode zip creates src.zip)
import tempfile
_src_dir = MODEL_DIR / "src"
_src_zip = MODEL_DIR / "src.zip"
if _src_dir.is_dir():
    sys.path.insert(0, str(MODEL_DIR))
elif _src_zip.exists():
    _tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(_src_zip) as _z:
        _z.extractall(_tmpdir)
    sys.path.insert(0, _tmpdir)
    print(f"Extracted src.zip to {_tmpdir}")
else:
    raise RuntimeError(f"src/ not found in {MODEL_DIR}. Files: {[p.name for p in MODEL_DIR.iterdir()]}")

TEST_DIR        = DATA_DIR / "test_soundscapes"
TAXONOMY_CSV    = DATA_DIR / "taxonomy.csv"
SAMPLE_SUB      = DATA_DIR / "sample_submission.csv"
TRAIN_CONFIG    = MODEL_DIR / "train_config.yaml"
LABEL_ENCODER   = MODEL_DIR / "label_encoder.json"

TIME_LIMIT = 5100  # seconds (90 min = 5400; stop at 5100 for safety)
BATCH_SIZE = 64
N_THREADS  = 4

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

torch.set_num_threads(N_THREADS)

from src.features import LogMelExtractor
from src.model import build_model
from src.dataset import InferenceDataset

# ---------------------------------------------------------------------------
# Load config and metadata
# ---------------------------------------------------------------------------
with open(TRAIN_CONFIG) as f:
    cfg = yaml.safe_load(f)

# Inject competition section if missing (for InferenceDataset compatibility)
cfg.setdefault("competition", {"window_seconds": 5})

with open(LABEL_ENCODER) as f:
    le = json.load(f)
class_cols = le["classes"]
n_classes = len(class_cols)
logger.info(f"Classes: {n_classes}")

taxonomy_df = pd.read_csv(TAXONOMY_CSV)
sample_sub  = pd.read_csv(SAMPLE_SUB)
logger.info(f"Sample submission: {len(sample_sub)} rows")

# ---------------------------------------------------------------------------
# Load all fold checkpoints
# ---------------------------------------------------------------------------
def load_fold_model(fold: int) -> torch.nn.Module:
    ckpt_path = MODEL_DIR / f"fold{fold}_best.pt"
    # pretrained=False: no HuggingFace download; we load weights from checkpoint
    infer_cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(infer_cfg, n_classes=n_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

fold_models = []
fold = 0
while (MODEL_DIR / f"fold{fold}_best.pt").exists():
    m = load_fold_model(fold)
    fold_models.append(m)
    logger.info(f"Loaded fold {fold} checkpoint")
    fold += 1

logger.info(f"Ensemble size: {len(fold_models)} folds")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
start_time = time.time()
logger.info(f"DATA_DIR contents: {[p.name for p in DATA_DIR.iterdir()]}")
test_files = sorted(TEST_DIR.glob("*.ogg"))
logger.info(f"Test soundscapes: {len(test_files)}")

all_row_ids = []
all_probs   = []
fill_value  = 1.0 / n_classes

for file_idx, filepath in enumerate(test_files):
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        logger.warning(f"Time limit hit at file {file_idx}/{len(test_files)}")
        break

    try:
        ds = InferenceDataset(str(filepath), cfg)
        if len(ds) == 0:
            continue

        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

        file_probs = []
        for log_mel_batch, _ in loader:
            fold_preds = []
            with torch.no_grad():
                for model in fold_models:
                    logits = model(log_mel_batch)
                    probs  = torch.sigmoid(logits).numpy()
                    fold_preds.append(probs)
            avg = np.mean(fold_preds, axis=0)  # (batch, n_classes)
            file_probs.append(avg)

        file_probs = np.concatenate(file_probs, axis=0)
        all_probs.append(file_probs)
        all_row_ids.extend(ds.row_ids)

    except Exception as e:
        logger.error(f"Error on {filepath.name}: {e}")
        try:
            import soundfile as sf
            duration = sf.info(str(filepath)).duration
        except Exception:
            duration = 60.0
        window_sec = cfg["competition"]["window_seconds"]
        t = 0.0
        while t + window_sec <= duration + 1e-3:
            end_sec = int(round(t + window_sec))
            all_row_ids.append(f"{filepath.stem}_{end_sec}")
            all_probs.append(np.full((1, n_classes), fill_value, dtype=np.float32))
            t += window_sec

    if (file_idx + 1) % 20 == 0:
        logger.info(f"  {file_idx+1}/{len(test_files)} | {time.time()-start_time:.0f}s")

# ---------------------------------------------------------------------------
# Build submission
# ---------------------------------------------------------------------------
sub_cols = sample_sub.columns.tolist()
if not all_probs:
    logger.warning("No soundscapes processed — filling all rows with prior probability")
    submission = sample_sub.copy()
    for c in class_cols:
        submission[c] = fill_value
    submission.to_csv(OUTPUT, index=False)
    logger.info(f"Saved {len(submission)} prior-filled rows → {OUTPUT}")
    raise SystemExit(0)

all_probs = np.concatenate(all_probs, axis=0)
submission = pd.DataFrame(all_probs, columns=class_cols)
submission.insert(0, "row_id", all_row_ids)

# Align to sample submission column order
sub_cols = sample_sub.columns.tolist()
missing_cols = [c for c in sub_cols if c != "row_id" and c not in submission.columns]
if missing_cols:
    logger.warning(f"{len(missing_cols)} classes missing — filling with prior")
    for c in missing_cols:
        submission[c] = fill_value

submission = submission[sub_cols]

# Fill any missing rows
expected = sample_sub["row_id"].tolist()
predicted = set(submission["row_id"].tolist())
missing_rows = [r for r in expected if r not in predicted]
if missing_rows:
    logger.warning(f"{len(missing_rows)} missing rows — filling with prior")
    filler = pd.DataFrame([[r] + [fill_value] * n_classes for r in missing_rows],
                          columns=sub_cols)
    submission = pd.concat([submission, filler], ignore_index=True)

submission.to_csv(OUTPUT, index=False)
total_time = time.time() - start_time
logger.info(f"Saved {len(submission)} rows → {OUTPUT} | {total_time:.0f}s total")
logger.info("Done.")
