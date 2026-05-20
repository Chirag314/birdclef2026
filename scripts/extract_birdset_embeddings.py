"""
extract_birdset_embeddings.py — Extract 1024-dim BirdSET ConvNeXt-Base features.

BirdSET ConvNeXt-Base was pretrained on 9736 bird species from Xeno-Canto Large.
Its frozen features are highly relevant for BirdCLEF 2026's 234 species.

Same pattern as extract_perch_embeddings.py — cache embeddings once, train
lightweight MLP on top.

Produces:
  artifacts/birdset/embeddings/clips_embeddings.npy    (35549, 1024)
  artifacts/birdset/embeddings/clips_metadata.csv
  artifacts/birdset/embeddings/ss_embeddings.npy       (1478, 1024)
  artifacts/birdset/embeddings/ss_metadata.csv

Usage:
  cd /data/birdclef2026
  .venv/bin/python scripts/extract_birdset_embeddings.py [--batch-size 64] [--resume]
  Estimated time: ~45 min (faster than Perch ONNX — GPU available)
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ConvNextModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_DIR  = pathlib.Path(__file__).parent.parent
DATA_DIR  = pathlib.Path("/data/birdclef_2026/data/raw/birdclef-2026")
AUDIO_DIR = DATA_DIR / "train_audio"
SS_DIR    = DATA_DIR / "train_soundscapes"
SS_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
FOLDS_CSV = REPO_DIR / "data" / "folds.csv"
MODEL_DIR = REPO_DIR / "artifacts" / "birdset"
OUT_DIR   = REPO_DIR / "artifacts" / "birdset" / "embeddings"

SR      = 32000
WIN_LEN = 160000  # 5s

# BirdSET spectrogram params (from README: 128 mels, 32kHz, 5s → (1, 128, 334))
_MEL_PARAMS = dict(
    sample_rate=32000, n_fft=1024, hop_length=320, f_min=0,
    n_mels=128, power=2.0, center=True,
)
_MEL_NORM_MEAN = -4.268
_MEL_NORM_STD  =  4.569


def build_mel_transform():
    mel_t = torchaudio.transforms.MelSpectrogram(**_MEL_PARAMS)
    db_t  = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return mel_t, db_t

_mel_t, _db_t = build_mel_transform()


def audio_to_spec(wave_batch: np.ndarray) -> torch.Tensor:
    """(B, WIN_LEN) float32 → (B, 1, 128, 334) normalized"""
    x = torch.from_numpy(wave_batch)
    mel = _mel_t(x)          # (B, 128, T)
    lms = _db_t(mel)         # (B, 128, T)
    # Resize to (128, 334) to match BirdSET training
    lms = F.interpolate(lms.unsqueeze(1), size=(128, 334),
                        mode="bilinear", align_corners=False)  # (B, 1, 128, 334)
    lms = (lms - _MEL_NORM_MEAN) / _MEL_NORM_STD
    return lms


@torch.no_grad()
def extract_features(model: torch.nn.Module, spec: torch.Tensor, device) -> np.ndarray:
    """(B, 1, 128, 334) → (B, 1024) features via global avg pool"""
    spec = spec.to(device)
    out = model(pixel_values=spec)
    feat = out.last_hidden_state  # (B, 1024, H', W')
    pooled = feat.mean(dim=[-2, -1])  # (B, 1024)
    return pooled.cpu().numpy().astype(np.float32)


def load_window(path: str, start_sec: float = 0.0) -> np.ndarray:
    info = sf.info(str(path))
    sr = info.samplerate
    start_frame = int(start_sec * sr)
    frames_need = int(5.0 * sr) + 1
    data, sr_read = sf.read(str(path), dtype="float32", always_2d=False,
                            start=start_frame, frames=frames_need)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_read != SR:
        import resampy
        data = resampy.resample(data, sr_read, SR)
    if len(data) < WIN_LEN:
        data = np.pad(data, (0, WIN_LEN - len(data)))
    return data[:WIN_LEN].astype(np.float32)


def run_extraction(model, device, paths, meta_rows, emb_path, meta_path,
                   batch_size, resume, label):
    N = len(paths)
    start = 0
    prev_embs, prev_meta = [], []
    if resume and emb_path.exists() and meta_path.exists():
        prev_embs = [np.load(str(emb_path))]
        prev_meta = [pd.read_csv(meta_path)]
        start = len(prev_meta[0])
        logger.info(f"Resuming {label} from {start}/{N}")
    if start >= N:
        logger.info(f"{label} already complete.")
        return

    logger.info(f"Extracting {N - start} {label} (batch={batch_size})")
    t0 = time.time()
    all_embs, all_metas = list(prev_embs), list(prev_meta)
    buf_audio, buf_rows = [], []

    def flush():
        if not buf_audio:
            return
        batch = np.stack(buf_audio)
        spec = audio_to_spec(batch)
        embs = extract_features(model, spec, device)
        all_embs.append(embs)
        all_metas.append(pd.DataFrame(buf_rows))
        buf_audio.clear(); buf_rows.clear()

    for i, (path, start_sec) in enumerate(paths[start:], start=start):
        try:
            audio = load_window(str(path), start_sec=start_sec)
        except Exception as e:
            logger.warning(f"Load error {path}: {e}")
            audio = np.zeros(WIN_LEN, dtype=np.float32)
        buf_audio.append(audio)
        buf_rows.append(meta_rows[i])

        if len(buf_audio) == batch_size:
            flush()

        if (i - start + 1) % 2000 == 0:
            flush()
            elapsed = time.time() - t0
            rate = (i - start + 1) / elapsed
            eta = (N - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(f"  {i+1}/{N} | {elapsed/60:.1f}min | ETA {eta:.1f}min")
            np.save(str(emb_path), np.concatenate(all_embs, axis=0))
            pd.concat(all_metas, ignore_index=True).to_csv(str(meta_path), index=False)

    flush()
    final_embs = np.concatenate(all_embs, axis=0)
    final_meta = pd.concat(all_metas, ignore_index=True)
    np.save(str(emb_path), final_embs)
    final_meta.to_csv(str(meta_path), index=False)
    elapsed = time.time() - t0
    logger.info(f"{label} done: {final_embs.shape} | {elapsed/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--clips-only", action="store_true")
    parser.add_argument("--ss-only",    action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading BirdSET ConvNeXt-Base from {MODEL_DIR} on {device}")
    model = ConvNextModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True, ignore_mismatched_sizes=True
    ).eval().to(device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    if not args.ss_only:
        folds_df = pd.read_csv(FOLDS_CSV)
        clips_df = folds_df[folds_df["source"] == "clip"].reset_index(drop=True)
        paths    = [(AUDIO_DIR / row["filename"], 0.0) for _, row in clips_df.iterrows()]
        meta     = [{"filename": row["filename"], "primary_label": row["primary_label"],
                     "fold": row["fold"], "source": "clip"}
                    for _, row in clips_df.iterrows()]
        run_extraction(model, device, paths, meta,
                       OUT_DIR / "clips_embeddings.npy", OUT_DIR / "clips_metadata.csv",
                       args.batch_size, args.resume, "clips")

    if not args.clips_only:
        ssl  = pd.read_csv(SS_LABELS)
        paths, meta = [], []
        for _, row in ssl.iterrows():
            parts = str(row["start"]).split(":")
            sec   = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
            paths.append((SS_DIR / row["filename"], float(sec)))
            meta.append({"filename": row["filename"], "start": row["start"],
                         "primary_label": row["primary_label"]})
        run_extraction(model, device, paths, meta,
                       OUT_DIR / "ss_embeddings.npy", OUT_DIR / "ss_metadata.csv",
                       args.batch_size, args.resume, "soundscapes")

    logger.info("All done.")


if __name__ == "__main__":
    main()
