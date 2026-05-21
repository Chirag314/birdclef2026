"""
BirdCLEF 2026 — Perch MLP + HGNet Ensemble Inference

Combines:
  Perch v2 MLP  — raw waveform → 1536-dim Perch embedding → 5-fold MLP
  HGNet 256x512 — mel spectrogram (10s context, 2.5s shift) → 4-fold HGNet

Datasets required:
  cid007/birdclef2026-perch              → Perch ONNX + MLP weights + ort wheel
  nischaydnk/birdclef-tawara-hgnet-lb-4fold → HGNet 256x512 ONNX weights
  birdclef-2026                          → competition data

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
import torch.nn.functional as F
import torchaudio

# Install onnxruntime from bundled wheel
_input = Path("/kaggle/input")
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

PERCH_DIR  = _find_dir(["/kaggle/input/birdclef2026-perch",
                         "/kaggle/input/datasets/cid007/birdclef2026-perch"])
HGNET_DIR  = _find_dir(["/kaggle/input/birdclef-tawara-hgnet-lb-4fold",
                         "/kaggle/input/datasets/nischaydnk/birdclef-tawara-hgnet-lb-4fold"])
DATA_DIR   = _find_dir(["/kaggle/input/birdclef-2026",
                         "/kaggle/input/competitions/birdclef-2026"])
OUTPUT     = Path("/kaggle/working/submission.csv")

print(f"PERCH_DIR: {PERCH_DIR}")
print(f"HGNET_DIR: {HGNET_DIR}")
print(f"DATA_DIR:  {DATA_DIR}")

TEST_DIR   = DATA_DIR / "test_soundscapes"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_LIMIT  = 5100
N_THREADS   = 4
SR          = 32000
WIN_SEC     = 5
WIN_LEN     = SR * WIN_SEC           # 160000 — Perch input
SHIFT_LEN   = int(SR * 2.5)          # 80000  — HGNet context padding each side
CTX_LEN     = WIN_LEN + 2*SHIFT_LEN  # 320000 — HGNet input (10s)
BATCH_SIZE  = 12                     # fixed by HGNet ONNX
PERCH_FOLDS = 5
HGNET_FOLDS = 4
ENSEMBLE_W  = 0.5                    # weight for Perch (1-w for HGNet)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
torch.set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# src/ import for PerchMLP
# ---------------------------------------------------------------------------
_src_flat   = PERCH_DIR / "src" / "perch_mlp.py"
_src_nested = PERCH_DIR / "src" / "src" / "perch_mlp.py"
if _src_flat.exists():
    sys.path.insert(0, str(PERCH_DIR))
elif _src_nested.exists():
    sys.path.insert(0, str(PERCH_DIR / "src"))
else:
    raise RuntimeError("perch_mlp.py not found in dataset")
from src.perch_mlp import PerchMLP

# ---------------------------------------------------------------------------
# Label setup (both models use taxonomy order = submission order)
# ---------------------------------------------------------------------------
with open(PERCH_DIR / "label_encoder.json") as f:
    le = json.load(f)
class_cols = le["classes"]
n_classes  = len(class_cols)
sample_sub = pd.read_csv(SAMPLE_SUB)
sub_cols   = sample_sub.columns.tolist()
fill_value = 1.0 / n_classes
logger.info(f"Classes: {n_classes}")

# ---------------------------------------------------------------------------
# ONNX session options
# ---------------------------------------------------------------------------
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = N_THREADS
sess_opts.inter_op_num_threads = N_THREADS

# ---------------------------------------------------------------------------
# Load Perch ONNX + MLP folds
# ---------------------------------------------------------------------------
perch_sess = ort.InferenceSession(str(PERCH_DIR / "perch_v2.onnx"),
                                   sess_options=sess_opts,
                                   providers=["CPUExecutionProvider"])
logger.info("Perch ONNX ready")

mlp_models = []
for fold in range(PERCH_FOLDS):
    ckpt_path = PERCH_DIR / f"fold{fold}_mlp_best.pt"
    if not ckpt_path.exists():
        break
    model = PerchMLP(n_classes=n_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mlp_models.append(model)
    logger.info(f"Loaded Perch MLP fold {fold}")
logger.info(f"Perch MLP: {len(mlp_models)} folds")

# ---------------------------------------------------------------------------
# Load HGNet ONNX folds (256x512 = 10s context)
# ---------------------------------------------------------------------------
hgnet_sessions = []
for fold in range(HGNET_FOLDS):
    onnx_path = HGNET_DIR / f"best_model_fold{fold}_256x512.onnx"
    if not onnx_path.exists():
        logger.warning(f"HGNet fold {fold} not found: {onnx_path}")
        continue
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    hgnet_sessions.append(sess)
    logger.info(f"Loaded HGNet fold {fold}")
logger.info(f"HGNet: {len(hgnet_sessions)} folds")

hgnet_inp = hgnet_sessions[0].get_inputs()[0].name

# ---------------------------------------------------------------------------
# HGNet spectrogram transform (Tawara params)
# ---------------------------------------------------------------------------
_mel_t = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000, n_fft=2048, win_length=626, hop_length=313,
    f_min=20, n_mels=256, power=2.0, center=True, pad_mode="reflect",
    norm="slaney", mel_scale="htk")
_db_t  = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def ctx_to_lms(ctx_batch: np.ndarray) -> np.ndarray:
    """(B, 320000) → (B, 1, 256, 512) normalised"""
    x   = torch.from_numpy(ctx_batch)
    lms = _db_t(_mel_t(x)).unsqueeze(1)
    lms = F.interpolate(lms, size=(256, 512), mode="bilinear", align_corners=False)
    B   = lms.shape[0]
    flat = lms.view(B, -1)
    mn = flat.min(1)[0].view(B, 1, 1, 1)
    mx = flat.max(1)[0].view(B, 1, 1, 1)
    return ((lms - mn) / (mx - mn + 1e-7)).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Predict a batch: (B≤12) windows → (B, n_classes) ensemble probs
# ---------------------------------------------------------------------------
def predict_batch(win_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
    """
    win_batch: (B, 160000) float32 — 5s for Perch
    ctx_batch: (B, 320000) float32 — 10s for HGNet
    Returns:   (B, n_classes) float32 ensemble probs
    """
    B = len(win_batch)

    # --- Perch path ---
    emb = perch_sess.run(["embedding"], {"inputs": win_batch})[0]   # (B, 1536)
    emb_t = torch.from_numpy(emb).float()
    perch_preds = []
    with torch.no_grad():
        for m in mlp_models:
            perch_preds.append(torch.sigmoid(m(emb_t)).numpy())
    perch_avg = np.mean(perch_preds, axis=0)   # (B, n_classes)

    # --- HGNet path — pad to batch=12 ---
    if B < BATCH_SIZE:
        pad_ctx = np.zeros((BATCH_SIZE - B, CTX_LEN), dtype=np.float32)
        ctx_padded = np.concatenate([ctx_batch, pad_ctx], axis=0)
    else:
        ctx_padded = ctx_batch
    lms = ctx_to_lms(ctx_padded)               # (12, 1, 256, 512)
    hgnet_preds = []
    for sess in hgnet_sessions:
        logits = sess.run(None, {hgnet_inp: lms})[0][:B]  # slice to real B
        hgnet_preds.append(1.0 / (1.0 + np.exp(-logits)))
    hgnet_avg = np.mean(hgnet_preds, axis=0)   # (B, n_classes)

    # --- Ensemble ---
    return ENSEMBLE_W * perch_avg + (1 - ENSEMBLE_W) * hgnet_avg


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
start_time = time.time()
test_files = sorted(TEST_DIR.glob("*.ogg"))
logger.info(f"Test soundscapes: {len(test_files)}")

all_row_ids, all_probs = [], []

for file_idx, filepath in enumerate(test_files):
    if time.time() - start_time > TIME_LIMIT:
        logger.warning(f"Time limit at {file_idx}/{len(test_files)}")
        break

    try:
        info = sf.info(str(filepath))
        dur  = info.duration
        native_sr = info.samplerate
        starts = [i * WIN_SEC for i in range(int(dur // WIN_SEC))]
        if not starts:
            continue
        row_ids = [f"{filepath.stem}_{int(s + WIN_SEC)}" for s in starts]

        # Load full file once
        raw, sr_read = sf.read(str(filepath), dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if sr_read != SR:
            import resampy
            raw = resampy.resample(raw, sr_read, SR)

        # Pad for HGNet context
        raw_padded = np.pad(raw, (SHIFT_LEN, SHIFT_LEN))

        # Build per-window arrays
        win_list, ctx_list = [], []
        for s in starts:
            s_sr = int(s * SR)
            # 5s window for Perch
            win = raw[s_sr:s_sr + WIN_LEN]
            if len(win) < WIN_LEN:
                win = np.pad(win, (0, WIN_LEN - len(win)))
            win_list.append(win[:WIN_LEN].astype(np.float32))
            # 10s context for HGNet
            ctx = raw_padded[s_sr:s_sr + CTX_LEN]
            if len(ctx) < CTX_LEN:
                ctx = np.pad(ctx, (0, CTX_LEN - len(ctx)))
            ctx_list.append(ctx[:CTX_LEN].astype(np.float32))

        # Batch inference (BATCH_SIZE=12 due to HGNet)
        file_probs = []
        for i in range(0, len(win_list), BATCH_SIZE):
            wb = np.stack(win_list[i:i + BATCH_SIZE])
            cb = np.stack(ctx_list[i:i + BATCH_SIZE])
            file_probs.append(predict_batch(wb, cb))

        all_probs.append(np.concatenate(file_probs, axis=0))
        all_row_ids.extend(row_ids)

    except Exception as e:
        import traceback
        logger.error(f"Error on {filepath.name}: {e}\n{traceback.format_exc()}")
        try:
            n_win = max(1, int(sf.info(str(filepath)).duration // WIN_SEC))
        except Exception:
            n_win = 12
        for i in range(n_win):
            all_row_ids.append(f"{filepath.stem}_{(i+1)*WIN_SEC}")
            all_probs.append(np.full((1, n_classes), fill_value, dtype=np.float32))

    if (file_idx + 1) % 20 == 0:
        logger.info(f"  {file_idx+1}/{len(test_files)} | {time.time()-start_time:.0f}s")

# ---------------------------------------------------------------------------
# Build submission
# ---------------------------------------------------------------------------
if not all_probs:
    logger.warning("No files processed — filling with prior")
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

expected  = set(sample_sub["row_id"])
predicted = set(submission["row_id"])
if expected - predicted:
    logger.warning(f"{len(expected-predicted)} missing rows — filling")
    filler = pd.DataFrame([[r] + [fill_value]*n_classes for r in (expected-predicted)],
                          columns=sub_cols)
    submission = pd.concat([submission, filler], ignore_index=True)

submission.to_csv(OUTPUT, index=False)
total = time.time() - start_time
logger.info(f"Saved {len(submission)} rows → {OUTPUT} | {total:.0f}s")
logger.info(f"Ensemble: {len(mlp_models)} Perch MLP + {len(hgnet_sessions)} HGNet folds | w={ENSEMBLE_W:.2f}/{1-ENSEMBLE_W:.2f}")
logger.info("Done.")
