"""
BirdCLEF 2026 — Perch MLP Inference Notebook
perch_mlp_v1: Perch v2 ONNX feature extractor + 5-fold MLP ensemble, CPU inference

Datasets required (attach to Kaggle kernel):
  cid007/birdclef2026-perch  → /kaggle/input/birdclef2026-perch
  birdclef-2026              → /kaggle/input/birdclef-2026

Runtime: CPU, Internet OFF
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Debug: show what datasets are mounted
import os
_input = Path("/kaggle/input")
print("=== /kaggle/input contents ===")
if _input.exists():
    for p in sorted(_input.iterdir()):
        print(f"  {p.name}/")
else:
    print("  /kaggle/input does NOT exist!")

# onnxruntime is not pre-installed in Kaggle CPU env
# Find the wheel anywhere under /kaggle/input
_wheels = sorted(Path("/kaggle/input").rglob("onnxruntime*.whl"))
print(f"onnxruntime wheels found: {[str(w) for w in _wheels]}")
if _wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", str(_wheels[0]), "--quiet"], check=True)
    print(f"Installed {_wheels[0].name}")
else:
    # No wheel found — try pip (may fail with internet=off but worth trying)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime", "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Cannot install onnxruntime. pip stderr: {r.stderr[:500]}\n"
            f"Available datasets: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'N/A'}"
        )
    print("Installed onnxruntime via pip")

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise RuntimeError(f"None of these paths exist: {candidates}")

PERCH_DIR = _find_dir([
    "/kaggle/input/birdclef2026-perch",
    "/kaggle/input/datasets/cid007/birdclef2026-perch",
])
DATA_DIR = _find_dir([
    "/kaggle/input/birdclef-2026",
    "/kaggle/input/competitions/birdclef-2026",
])
OUTPUT = Path("/kaggle/working/submission.csv")

print(f"PERCH_DIR: {PERCH_DIR}")
print(f"DATA_DIR:  {DATA_DIR}")
print(f"DATA_DIR contents: {[p.name for p in DATA_DIR.iterdir()]}")

TEST_DIR          = DATA_DIR / "test_soundscapes"
SAMPLE_SUB        = DATA_DIR / "sample_submission.csv"
ONNX_PATH         = PERCH_DIR / "perch_v2.onnx"
LE_PATH           = PERCH_DIR / "label_encoder.json"
SITE_HOUR_PRIOR   = PERCH_DIR / "site_hour_prior.npz"

# src/ in dataset may be flat or zip-extracted (double-nested)
_src_flat   = PERCH_DIR / "src" / "perch_mlp.py"
_src_nested = PERCH_DIR / "src" / "src" / "perch_mlp.py"
if _src_flat.exists():
    sys.path.insert(0, str(PERCH_DIR))
elif _src_nested.exists():
    sys.path.insert(0, str(PERCH_DIR / "src"))
else:
    raise RuntimeError("perch_mlp.py not found in dataset")

from src.perch_mlp import PerchMLP

TIME_LIMIT  = 5100   # seconds
BATCH_SIZE  = 32
N_THREADS   = 4
TARGET_SR   = 32000
WIN_LEN     = 160000  # 5s × 32kHz

# Multi-window aggregation: mix each window's prediction with the file-level
# max-pool across all windows. Helps rare/brief species (macro AUC weights
# all 234 classes equally). 0=pure file max, 1=pure per-window (baseline).
AGGREGATION_ALPHA = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
torch.set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# Load label encoder
# ---------------------------------------------------------------------------
with open(LE_PATH) as f:
    le = json.load(f)
class_cols = le["classes"]
n_classes  = len(class_cols)
logger.info(f"Classes: {n_classes}")

sample_sub = pd.read_csv(SAMPLE_SUB)
fill_value = 1.0 / n_classes

# Site×Hour prior DISABLED: prior built from 9 training sites; test sites differ,
# global fallback zeros 159/234 classes → -0.083 LB regression confirmed.
site_hour_prior = None

# ---------------------------------------------------------------------------
# Load Perch ONNX session
# ---------------------------------------------------------------------------
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = N_THREADS
sess_opts.inter_op_num_threads = N_THREADS
perch_sess = ort.InferenceSession(str(ONNX_PATH), sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])
logger.info("Perch ONNX session ready")

# ---------------------------------------------------------------------------
# Load MLP fold models
# ---------------------------------------------------------------------------
def load_mlp_fold(fold: int) -> torch.nn.Module:
    ckpt_path = PERCH_DIR / f"fold{fold}_mlp_best.pt"
    model = PerchMLP(n_classes=n_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

mlp_models = []
fold = 0
while (PERCH_DIR / f"fold{fold}_mlp_best.pt").exists():
    mlp_models.append(load_mlp_fold(fold))
    logger.info(f"Loaded MLP fold {fold}")
    fold += 1
logger.info(f"Ensemble: {len(mlp_models)} MLP folds")

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_window(path: str, start_sec: float) -> np.ndarray:
    info = sf.info(path)
    native_sr = info.samplerate
    start_frame = int(start_sec * native_sr)
    frames_need = int(5.0 * native_sr) + 1
    data, sr = sf.read(path, dtype="float32", always_2d=False,
                       start=start_frame, frames=frames_need)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import resampy
        data = resampy.resample(data, sr, TARGET_SR)
    if len(data) < WIN_LEN:
        data = np.pad(data, (0, WIN_LEN - len(data)))
    return data[:WIN_LEN].astype(np.float32)

# ---------------------------------------------------------------------------
# Embed + classify a batch of windows
# ---------------------------------------------------------------------------
def predict_windows(audio_batch: list) -> np.ndarray:
    """audio_batch: list of (WIN_LEN,) float32 → (B, n_classes) probabilities."""
    X = np.stack(audio_batch)  # (B, WIN_LEN)
    embeddings = perch_sess.run(["embedding"], {"inputs": X})[0]  # (B, 1536)
    emb_tensor = torch.from_numpy(embeddings).float()

    fold_preds = []
    with torch.no_grad():
        for model in mlp_models:
            logits = model(emb_tensor)
            fold_preds.append(torch.sigmoid(logits).numpy())

    return np.mean(fold_preds, axis=0)  # (B, n_classes)

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
start_time = time.time()
test_files = sorted(TEST_DIR.glob("*.ogg"))
logger.info(f"Test soundscapes: {len(test_files)}")

all_row_ids = []
all_probs   = []

for file_idx, filepath in enumerate(test_files):
    if time.time() - start_time > TIME_LIMIT:
        logger.warning(f"Time limit hit at file {file_idx}/{len(test_files)}")
        break

    try:
        info = sf.info(str(filepath))
        duration = info.duration
        window_sec = 5.0
        starts = [i * window_sec for i in range(int(duration // window_sec))]
        if not starts:
            continue

        # Build row_ids: filename_stem_<end_sec>
        row_ids = [f"{filepath.stem}_{int(s + window_sec)}" for s in starts]

        # Batch the windows
        batch_audio, batch_ids = [], []
        file_probs = []

        for s, rid in zip(starts, row_ids):
            audio = load_window(str(filepath), start_sec=s)
            batch_audio.append(audio)
            batch_ids.append(rid)

            if len(batch_audio) == BATCH_SIZE:
                preds = predict_windows(batch_audio)
                file_probs.append(preds)
                batch_audio, batch_ids = [], []

        if batch_audio:
            preds = predict_windows(batch_audio)
            file_probs.append(preds)

        if file_probs:
            file_arr = np.concatenate(file_probs, axis=0)  # (N_windows, 234)

            # Multi-window aggregation: mix each window with file-level max-pool
            if AGGREGATION_ALPHA < 1.0 and len(file_arr) > 1:
                file_max = file_arr.max(axis=0, keepdims=True)  # (1, 234)
                file_arr = AGGREGATION_ALPHA * file_arr + (1 - AGGREGATION_ALPHA) * file_max

            all_probs.append(file_arr)
            all_row_ids.extend(row_ids)

    except Exception as e:
        logger.error(f"Error on {filepath.name}: {e}")
        # Fill with uniform prior
        info2 = sf.info(str(filepath))
        n_windows = max(1, int(info2.duration // 5))
        for i in range(n_windows):
            all_row_ids.append(f"{filepath.stem}_{(i+1)*5}")
            all_probs.append(np.full((1, n_classes), fill_value, dtype=np.float32))

    if (file_idx + 1) % 20 == 0:
        elapsed = time.time() - start_time
        logger.info(f"  {file_idx+1}/{len(test_files)} | {elapsed:.0f}s")

# ---------------------------------------------------------------------------
# Build submission
# ---------------------------------------------------------------------------
sub_cols = sample_sub.columns.tolist()
if not all_probs:
    logger.warning("No files processed — filling all rows with prior")
    sub = sample_sub.copy()
    for c in class_cols:
        sub[c] = fill_value
    sub.to_csv(OUTPUT, index=False)
    raise SystemExit(0)

all_probs = np.concatenate(all_probs, axis=0)

# Apply Site×Hour prior
if site_hour_prior is not None:
    all_probs = site_hour_prior.apply(all_probs, all_row_ids)
    logger.info("Site×Hour prior applied.")

submission = pd.DataFrame(all_probs, columns=class_cols)
submission.insert(0, "row_id", all_row_ids)

# Align columns and fill missing rows
missing_cols = [c for c in sub_cols if c != "row_id" and c not in submission.columns]
for c in missing_cols:
    submission[c] = fill_value
submission = submission[sub_cols]

expected  = set(sample_sub["row_id"].tolist())
predicted = set(submission["row_id"].tolist())
missing_rows = expected - predicted
if missing_rows:
    logger.warning(f"{len(missing_rows)} missing rows — filling with prior")
    filler = pd.DataFrame([[r] + [fill_value] * n_classes for r in missing_rows],
                          columns=sub_cols)
    submission = pd.concat([submission, filler], ignore_index=True)

submission.to_csv(OUTPUT, index=False)
total_time = time.time() - start_time
logger.info(f"Saved {len(submission)} rows → {OUTPUT} | {total_time:.0f}s total")
logger.info("Done.")
