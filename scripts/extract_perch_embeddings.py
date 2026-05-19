"""
extract_perch_embeddings.py — extract 1536-dim Perch v2 embeddings for all training data.

Produces:
  artifacts/perch/embeddings/clips_embeddings.npy    (35549, 1536) float32
  artifacts/perch/embeddings/clips_metadata.csv      filename, primary_label, fold, source
  artifacts/perch/embeddings/ss_embeddings.npy       (1478, 1536) float32
  artifacts/perch/embeddings/ss_metadata.csv         filename, start, primary_label

Usage:
  cd /data/birdclef2026
  .venv/bin/python scripts/extract_perch_embeddings.py [--batch-size 32] [--workers 8] [--resume]
  Estimated time: ~1h for clips, ~10min for soundscapes
"""

import argparse
import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_DIR  = pathlib.Path(__file__).parent.parent
DATA_DIR  = pathlib.Path("/data/birdclef_2026/data/raw/birdclef-2026")
AUDIO_DIR = DATA_DIR / "train_audio"
SS_DIR    = DATA_DIR / "train_soundscapes"
SS_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
FOLDS_CSV = REPO_DIR / "data" / "folds.csv"
ONNX_PATH = REPO_DIR / "artifacts" / "perch" / "perch_v2.onnx"
OUT_DIR   = REPO_DIR / "artifacts" / "perch" / "embeddings"

TARGET_SR = 32000
WIN_LEN   = 160000  # 5 s × 32 kHz


def load_window(path: str, start_sec: float = 0.0) -> np.ndarray:
    """Load 5s window from audio file, resampling to 32kHz as needed."""
    info = sf.info(str(path))
    native_sr = info.samplerate
    start_frame = int(start_sec * native_sr)
    frames_needed = int(5.0 * native_sr) + 1  # small extra to handle rounding
    data, sr = sf.read(str(path), dtype="float32", always_2d=False,
                       start=start_frame, frames=frames_needed)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import resampy
        data = resampy.resample(data, sr, TARGET_SR)
    need = WIN_LEN
    if len(data) < need:
        data = np.pad(data, (0, need - len(data)))
    return data[:need].astype(np.float32)


def _load_clip_row(args):
    path, start_sec = args
    try:
        return load_window(str(path), start_sec=start_sec)
    except Exception as e:
        logger.warning(f"Load error {path}: {e}")
        return np.zeros(WIN_LEN, dtype=np.float32)


def embed_batch(sess: ort.InferenceSession, batch: list) -> np.ndarray:
    X = np.stack(batch)
    return sess.run(["embedding"], {"inputs": X})[0].astype(np.float32)


def run_extraction(
    sess: ort.InferenceSession,
    paths: list,         # list of (path, start_sec) tuples
    meta_rows: list,     # list of dicts for metadata
    emb_path: pathlib.Path,
    meta_path: pathlib.Path,
    batch_size: int,
    n_workers: int,
    resume: bool,
    label: str = "items",
) -> None:
    N = len(paths)

    start = 0
    prev_embs, prev_meta = [], []
    if resume and emb_path.exists() and meta_path.exists():
        prev_embs = [np.load(str(emb_path))]
        prev_meta_df = pd.read_csv(meta_path)
        start = len(prev_meta_df)
        prev_meta = [prev_meta_df]
        logger.info(f"Resuming {label} from {start}/{N}")
    if start >= N:
        logger.info(f"{label} already complete.")
        return

    logger.info(f"Extracting {N - start} {label} (batch={batch_size}, workers={n_workers})")
    t0 = time.time()

    all_embs = list(prev_embs)
    all_metas = list(prev_meta)
    buf_embs, buf_metas = [], []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for batch_start in range(start, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_paths = paths[batch_start:batch_end]
            batch_rows  = meta_rows[batch_start:batch_end]

            # Parallel audio loading
            audios = list(pool.map(_load_clip_row, batch_paths))

            # ONNX inference
            embs = embed_batch(sess, audios)
            buf_embs.append(embs)
            buf_metas.extend(batch_rows)

            done = batch_end - start
            total = N - start
            if done % 2000 < batch_size or batch_end == N:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate / 60 if rate > 0 else 0
                logger.info(f"  {batch_end}/{N} | {elapsed/60:.1f}min elapsed | ETA {eta:.1f}min")
                # Checkpoint
                all_embs_now = all_embs + buf_embs
                all_metas_now = all_metas + [pd.DataFrame(buf_metas)]
                np.save(str(emb_path), np.concatenate(all_embs_now, axis=0))
                pd.concat(all_metas_now, ignore_index=True).to_csv(str(meta_path), index=False)

    all_embs_now = all_embs + buf_embs
    all_metas_now = all_metas + [pd.DataFrame(buf_metas)]
    final_embs = np.concatenate(all_embs_now, axis=0)
    final_meta = pd.concat(all_metas_now, ignore_index=True)
    np.save(str(emb_path), final_embs)
    final_meta.to_csv(str(meta_path), index=False)

    elapsed = time.time() - t0
    logger.info(f"{label} done: {final_embs.shape} | "
                f"range=[{final_embs.min():.4f},{final_embs.max():.4f}] | {elapsed/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--clips-only", action="store_true")
    parser.add_argument("--ss-only",    action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    logger.info(f"ONNX ready: {ONNX_PATH.name}")

    if not args.ss_only:
        folds_df = pd.read_csv(FOLDS_CSV)
        clips_df = folds_df[folds_df["source"] == "clip"].reset_index(drop=True)
        paths = [(AUDIO_DIR / row["filename"], 0.0) for _, row in clips_df.iterrows()]
        meta_rows = [
            {"filename": row["filename"], "primary_label": row["primary_label"],
             "fold": row["fold"], "source": "clip"}
            for _, row in clips_df.iterrows()
        ]
        run_extraction(sess, paths, meta_rows,
                       OUT_DIR / "clips_embeddings.npy",
                       OUT_DIR / "clips_metadata.csv",
                       args.batch_size, args.workers, args.resume,
                       label="clips")

    if not args.clips_only:
        ssl = pd.read_csv(SS_LABELS)
        paths, meta_rows = [], []
        for _, row in ssl.iterrows():
            parts = str(row["start"]).split(":")
            start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            paths.append((SS_DIR / row["filename"], float(start_sec)))
            meta_rows.append({
                "filename":      row["filename"],
                "start":         row["start"],
                "primary_label": row["primary_label"],
            })
        run_extraction(sess, paths, meta_rows,
                       OUT_DIR / "ss_embeddings.npy",
                       OUT_DIR / "ss_metadata.csv",
                       args.batch_size, args.workers, args.resume,
                       label="soundscapes")

    logger.info("All done.")


if __name__ == "__main__":
    main()
