"""
BirdCLEF 2026 — HGNet Inference
Tawara HGNetV2-B0 4-fold ensemble using ONNX runtime.

Datasets required:
  nischaydnk/birdclef-tawara-hgnet-lb-4fold  → HGNet ONNX models
  cid007/birdclef2026-perch                  → onnxruntime wheel
  birdclef-2026                              → competition data

Runtime: CPU, Internet OFF
Spectrogram params from: ttahara/birdclef-2026-hgnetv2-b0-baseline-inference
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

# Install onnxruntime from our bundled wheel
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

HGNET_DIR = _find_dir([
    "/kaggle/input/birdclef-tawara-hgnet-lb-4fold",
    "/kaggle/input/datasets/nischaydnk/birdclef-tawara-hgnet-lb-4fold",
])
DATA_DIR = _find_dir([
    "/kaggle/input/birdclef-2026",
    "/kaggle/input/competitions/birdclef-2026",
])
OUTPUT = Path("/kaggle/working/submission.csv")

print(f"HGNET_DIR: {HGNET_DIR}")
print(f"DATA_DIR:  {DATA_DIR}")
print(f"HGNET contents: {[p.name for p in HGNET_DIR.iterdir() if '256x256' in p.name or p.suffix=='.txt']}")

TEST_DIR   = DATA_DIR / "test_soundscapes"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

TIME_LIMIT = 5100
N_THREADS  = 4
SR         = 32000
WIN_SEC    = 5
WIN_LEN    = SR * WIN_SEC          # 160000 samples
BATCH_SIZE = 12                    # fixed by ONNX model
N_FOLDS    = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectrogram transform (exact Tawara params)
# ---------------------------------------------------------------------------
_MEL_PARAMS = dict(
    sample_rate=32000, n_fft=2048, win_length=626, hop_length=313,
    f_min=20, n_mels=256, power=2.0, center=True, pad_mode="reflect",
    norm="slaney", mel_scale="htk",
)
_mel_transform = torchaudio.transforms.MelSpectrogram(**_MEL_PARAMS)
_db_transform  = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def audio_to_lms(wave_batch: np.ndarray) -> np.ndarray:
    """
    wave_batch: (B, WIN_LEN) float32
    Returns:    (B, 1, 256, 256) float32, min-max normalised per sample
    """
    x = torch.from_numpy(wave_batch)          # (B, samples)
    mel = _mel_transform(x)                   # (B, 256, T)
    lms = _db_transform(mel)                  # (B, 256, T)
    lms = lms.unsqueeze(1)                    # (B, 1, 256, T)
    # Resize time axis to 256
    lms = F.interpolate(lms, size=(256, 256), mode="bilinear", align_corners=False)
    # Min-max normalise per sample
    B = lms.shape[0]
    flat = lms.view(B, -1)
    mn = flat.min(dim=1)[0].view(B, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(B, 1, 1, 1)
    lms = (lms - mn) / (mx - mn + 1e-7)
    return lms.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Load taxonomy for label order (HGNet outputs in taxonomy order)
# ---------------------------------------------------------------------------
taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
SPECIES   = taxonomy["primary_label"].tolist()   # 234 species in taxonomy order
n_classes = len(SPECIES)

sample_sub = pd.read_csv(SAMPLE_SUB)
# Map HGNet output (taxonomy order) to submission order
sub_cols = sample_sub.columns.tolist()
class_cols = [c for c in sub_cols if c != "row_id"]

# Build reindex: hgnet outputs in SPECIES order, submission expects class_cols order
hgnet_idx = {s: i for i, s in enumerate(SPECIES)}
sub_order = [hgnet_idx[c] for c in class_cols]  # indices to reorder output

fill_value = 1.0 / n_classes
logger.info(f"Classes: {n_classes}")

# ---------------------------------------------------------------------------
# Load ONNX sessions
# ---------------------------------------------------------------------------
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = N_THREADS
sess_opts.inter_op_num_threads = N_THREADS

sessions = []
for fold in range(N_FOLDS):
    onnx_path = HGNET_DIR / f"best_model_fold{fold}_256x256.onnx"
    if not onnx_path.exists():
        logger.warning(f"Missing fold {fold} ONNX: {onnx_path}")
        continue
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    sessions.append(sess)
    logger.info(f"Loaded fold {fold} ONNX")
logger.info(f"HGNet ensemble: {len(sessions)} folds")

inp_name = sessions[0].get_inputs()[0].name


def predict_batch(wave_batch: np.ndarray) -> np.ndarray:
    """
    wave_batch: (B≤12, WIN_LEN) → (B, n_classes) sigmoid probabilities
    Pads to batch=12, slices back to B.
    """
    B = len(wave_batch)
    # Pad to BATCH_SIZE=12
    if B < BATCH_SIZE:
        pad = np.zeros((BATCH_SIZE - B, WIN_LEN), dtype=np.float32)
        wave_batch = np.concatenate([wave_batch, pad], axis=0)

    lms = audio_to_lms(wave_batch)  # (12, 1, 256, 256)

    fold_preds = []
    for sess in sessions:
        logits = sess.run(None, {inp_name: lms})[0]  # (12, 234)
        probs  = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
        fold_preds.append(probs[:B])                  # slice to real B

    avg = np.mean(fold_preds, axis=0)  # (B, 234)
    # Reorder from taxonomy order to submission order
    return avg[:, sub_order]


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
        logger.warning(f"Time limit at {file_idx}/{len(test_files)}")
        break

    try:
        info    = sf.info(str(filepath))
        dur     = info.duration
        starts  = [i * WIN_SEC for i in range(int(dur // WIN_SEC))]
        if not starts:
            continue
        row_ids = [f"{filepath.stem}_{int(s + WIN_SEC)}" for s in starts]
        native_sr = info.samplerate

        # Load all windows
        windows = []
        for s in starts:
            start_frame = int(s * native_sr)
            frames_need = int(WIN_SEC * native_sr) + 1
            data, sr_read = sf.read(str(filepath), dtype="float32", always_2d=False,
                                     start=start_frame, frames=frames_need)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr_read != SR:
                import resampy
                data = resampy.resample(data, sr_read, SR)
            if len(data) < WIN_LEN:
                data = np.pad(data, (0, WIN_LEN - len(data)))
            windows.append(data[:WIN_LEN].astype(np.float32))

        # Batch inference
        file_probs = []
        for i in range(0, len(windows), BATCH_SIZE):
            batch = np.stack(windows[i:i + BATCH_SIZE])
            file_probs.append(predict_batch(batch))

        file_arr = np.concatenate(file_probs, axis=0)
        all_probs.append(file_arr)
        all_row_ids.extend(row_ids)

    except Exception as e:
        import traceback
        logger.error(f"Error on {filepath.name}: {e}")
        logger.error(traceback.format_exc())
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
    filler = pd.DataFrame(
        [[r] + [fill_value] * n_classes for r in (expected - predicted)],
        columns=sub_cols)
    submission = pd.concat([submission, filler], ignore_index=True)

submission.to_csv(OUTPUT, index=False)
total = time.time() - start_time
logger.info(f"Saved {len(submission)} rows → {OUTPUT} | {total:.0f}s")
logger.info(f"HGNet {len(sessions)}-fold ensemble done.")
