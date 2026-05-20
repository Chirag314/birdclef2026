"""
BirdCLEF 2026 — Ensemble Inference: Perch MLP + EfficientNet-B0 (phase4)

Datasets required:
  cid007/birdclef2026-perch  → Perch ONNX + 5-fold MLP weights
  cid007/birdclef2026-model  → EfficientNet-B0 5-fold weights + src/

Runtime: CPU, Internet OFF
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

# ---------------------------------------------------------------------------
# Install onnxruntime from bundled wheel
# ---------------------------------------------------------------------------
_input = Path("/kaggle/input")
print("=== /kaggle/input ===", [p.name for p in _input.iterdir()])

_wheels = sorted(_input.rglob("onnxruntime*.whl"))
if _wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", str(_wheels[0]), "--quiet"], check=True)
    print(f"Installed {_wheels[0].name}")
else:
    raise RuntimeError("onnxruntime wheel not found")

import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise RuntimeError(f"None exist: {candidates}")

PERCH_DIR = _find_dir([
    "/kaggle/input/birdclef2026-perch",
    "/kaggle/input/datasets/cid007/birdclef2026-perch",
])
MODEL_DIR = _find_dir([
    "/kaggle/input/birdclef2026-model",
    "/kaggle/input/datasets/cid007/birdclef2026-model",
])
DATA_DIR = _find_dir([
    "/kaggle/input/birdclef-2026",
    "/kaggle/input/competitions/birdclef-2026",
])
OUTPUT = Path("/kaggle/working/submission.csv")

print(f"PERCH_DIR: {PERCH_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"DATA_DIR:  {DATA_DIR}")

TEST_DIR   = DATA_DIR / "test_soundscapes"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Set up src/ imports — MODEL_DIR provides features/model, PERCH_DIR provides
# perch_mlp. Both extract to src/src/ (double-nested zip). No name conflicts.
# ---------------------------------------------------------------------------
def _add_src(dataset_dir):
    if (dataset_dir / "src" / "features.py").exists() or \
       (dataset_dir / "src" / "perch_mlp.py").exists():
        sys.path.insert(0, str(dataset_dir))
    elif (dataset_dir / "src" / "src").exists():
        sys.path.insert(0, str(dataset_dir / "src"))
    else:
        raise RuntimeError(f"src/ not found in {dataset_dir}")

_add_src(MODEL_DIR)   # provides src.features, src.model, src.dataset
_add_src(PERCH_DIR)   # provides src.perch_mlp

from src.features import LogMelExtractor, load_audio, pad_or_trim, peak_normalize
from src.model import build_model
from src.perch_mlp import PerchMLP

TIME_LIMIT  = 5100
BATCH_SIZE  = 32
N_THREADS   = 4
SR          = 32000
WIN_LEN     = 160000   # 5s × 32kHz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
torch.set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# Load configs and label encoder (from MODEL_DIR)
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "train_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("competition", {"window_seconds": 5})

with open(MODEL_DIR / "label_encoder.json") as f:
    le = json.load(f)
class_cols = le["classes"]
n_classes  = len(class_cols)
logger.info(f"Classes: {n_classes}")

sample_sub = pd.read_csv(SAMPLE_SUB)
fill_value = 1.0 / n_classes

# ---------------------------------------------------------------------------
# Load EfficientNet fold models
# ---------------------------------------------------------------------------
def load_enf_fold(fold: int) -> torch.nn.Module:
    ckpt_path = MODEL_DIR / f"fold{fold}_best.pt"
    infer_cfg  = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(infer_cfg, n_classes=n_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

enf_models = []
for fold in range(10):
    if not (MODEL_DIR / f"fold{fold}_best.pt").exists():
        break
    enf_models.append(load_enf_fold(fold))
    logger.info(f"Loaded EfficientNet fold {fold}")
logger.info(f"EfficientNet ensemble: {len(enf_models)} folds")

# ---------------------------------------------------------------------------
# Load Perch ONNX + MLP fold models
# ---------------------------------------------------------------------------
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = N_THREADS
sess_opts.inter_op_num_threads = N_THREADS
perch_sess = ort.InferenceSession(
    str(PERCH_DIR / "perch_v2.onnx"),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)
logger.info("Perch ONNX session ready")

# Load Perch label encoder (may differ from MODEL_DIR's — use MODEL_DIR's as canonical)
# MLP n_classes must match what it was trained on (same 234 classes)
def load_mlp_fold(fold: int) -> torch.nn.Module:
    ckpt_path = PERCH_DIR / f"fold{fold}_mlp_best.pt"
    model = PerchMLP(n_classes=n_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

mlp_models = []
for fold in range(10):
    if not (PERCH_DIR / f"fold{fold}_mlp_best.pt").exists():
        break
    mlp_models.append(load_mlp_fold(fold))
    logger.info(f"Loaded MLP fold {fold}")
logger.info(f"Perch MLP ensemble: {len(mlp_models)} folds")

# ---------------------------------------------------------------------------
# Feature extractor for EfficientNet
# ---------------------------------------------------------------------------
log_mel_extractor = LogMelExtractor(cfg)

# ---------------------------------------------------------------------------
# Predict a batch: raw audio (B, WIN_LEN) → averaged probs (B, n_classes)
# ---------------------------------------------------------------------------
def predict_batch(audio_batch: list) -> np.ndarray:
    """
    audio_batch: list of (WIN_LEN,) float32 numpy arrays
    Returns: (B, n_classes) float32 averaged across both model families
    """
    B = len(audio_batch)
    X_raw = np.stack(audio_batch)  # (B, WIN_LEN)

    # --- Perch MLP ---
    embeddings = perch_sess.run(["embedding"], {"inputs": X_raw})[0]  # (B, 1536)
    emb_t = torch.from_numpy(embeddings).float()
    mlp_fold_preds = []
    with torch.no_grad():
        for model in mlp_models:
            mlp_fold_preds.append(torch.sigmoid(model(emb_t)).numpy())
    perch_probs = np.mean(mlp_fold_preds, axis=0)  # (B, n_classes)

    # --- EfficientNet ---
    # Compute log-mel for each window, stack into batch tensor
    log_mels = []
    for audio in audio_batch:
        audio_norm = peak_normalize(pad_or_trim(audio, WIN_LEN))
        lm = log_mel_extractor(audio_norm)   # (1, n_mels, T)
        log_mels.append(lm)
    log_mel_batch = torch.stack(log_mels, dim=0)  # (B, 1, n_mels, T)

    enf_fold_preds = []
    with torch.no_grad():
        for model in enf_models:
            logits = model(log_mel_batch)
            enf_fold_preds.append(torch.sigmoid(logits).numpy())
    enf_probs = np.mean(enf_fold_preds, axis=0)  # (B, n_classes)

    # --- Ensemble average ---
    return (perch_probs + enf_probs) / 2.0

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
        logger.warning(f"Time limit at file {file_idx}/{len(test_files)}")
        break

    try:
        info = sf.info(str(filepath))
        duration = info.duration
        window_sec = 5.0
        starts = [i * window_sec for i in range(int(duration // window_sec))]
        if not starts:
            continue

        row_ids = [f"{filepath.stem}_{int(s + window_sec)}" for s in starts]

        # Load all windows for this file
        native_sr = info.samplerate
        file_probs = []
        batch_audio, batch_ids = [], []

        for s, rid in zip(starts, row_ids):
            start_frame = int(s * native_sr)
            frames_need = int(window_sec * native_sr) + 1
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=False,
                               start=start_frame, frames=frames_need)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SR:
                import resampy
                data = resampy.resample(data, sr, SR)
            if len(data) < WIN_LEN:
                data = np.pad(data, (0, WIN_LEN - len(data)))
            batch_audio.append(data[:WIN_LEN].astype(np.float32))
            batch_ids.append(rid)

            if len(batch_audio) == BATCH_SIZE:
                file_probs.append(predict_batch(batch_audio))
                batch_audio, batch_ids = [], []

        if batch_audio:
            file_probs.append(predict_batch(batch_audio))

        if file_probs:
            all_probs.append(np.concatenate(file_probs, axis=0))
            all_row_ids.extend(row_ids)

    except Exception as e:
        import traceback
        logger.error(f"Error on {filepath.name}: {e}")
        logger.error(traceback.format_exc())
        try:
            n_windows = max(1, int(sf.info(str(filepath)).duration // 5))
        except Exception:
            n_windows = 12
        for i in range(n_windows):
            all_row_ids.append(f"{filepath.stem}_{(i+1)*5}")
            all_probs.append(np.full((1, n_classes), fill_value, dtype=np.float32))

    if (file_idx + 1) % 20 == 0:
        logger.info(f"  {file_idx+1}/{len(test_files)} | {time.time()-start_time:.0f}s")

# ---------------------------------------------------------------------------
# Build submission
# ---------------------------------------------------------------------------
sub_cols = sample_sub.columns.tolist()
if not all_probs:
    logger.warning("No soundscapes processed — filling with prior")
    sub = sample_sub.copy()
    for c in class_cols:
        sub[c] = fill_value
    sub.to_csv(OUTPUT, index=False)
    raise SystemExit(0)

all_probs  = np.concatenate(all_probs, axis=0)
submission = pd.DataFrame(all_probs, columns=class_cols)
submission.insert(0, "row_id", all_row_ids)

missing_cols = [c for c in sub_cols if c != "row_id" and c not in submission.columns]
for c in missing_cols:
    submission[c] = fill_value
submission = submission[sub_cols]

expected  = set(sample_sub["row_id"].tolist())
predicted = set(submission["row_id"].tolist())
missing_rows = expected - predicted
if missing_rows:
    logger.warning(f"{len(missing_rows)} missing rows — filling")
    filler = pd.DataFrame([[r] + [fill_value] * n_classes for r in missing_rows],
                          columns=sub_cols)
    submission = pd.concat([submission, filler], ignore_index=True)

submission.to_csv(OUTPUT, index=False)
total_time = time.time() - start_time
logger.info(f"Saved {len(submission)} rows → {OUTPUT} | {total_time:.0f}s")
logger.info(f"Ensemble: {len(enf_models)} EfficientNet folds + {len(mlp_models)} Perch MLP folds")
logger.info("Done.")
