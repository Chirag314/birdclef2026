# BirdCLEF 2026 — Phase 0 Report

> Converted from `phase0_report.html` for GitHub rendering. Original HTML preserved in this folder.

# BirdCLEF 2026

Phase 0 Report — Infrastructure & Baseline

---

Generated 2026-04-14 | Deadline Jun 3, 2026 (~7 weeks remaining)

- **Phase:** Phase 0 — Infrastructure
- **Status:** Code Complete — Ready to Run
- **Exit Criterion:** OOF Macro AUC > 0.65 · CPU Infer < 60 min
- **Files Written:** 11 Python modules (0 lines → complete)

[§1 Status](#status)
[§2 What Was Built](#what-was-built)
[§3 How to Run](#how-to-run)
[§4 Design Decisions](#design)
[§5 Exit Criterion](#exit)
[§6 Risks](#risks)
[§7 → Phase 1](#next)

## §1 — Phase 0 Status Audit

Phase 0 target dates: Apr 14–26 | All code written Apr 14. Ready to execute on GPU machine.

### Component checklist

- **✓configs/base.yaml** (configs/base.yaml) — Shared defaults: paths, audio params, seed. Complete.
- **✓configs/train.yaml** (configs/train.yaml) — EfficientNet-B0, GeM, 30 epochs, cosine LR, all augmentations OFF. Complete.
- **✓configs/folds.yaml** (configs/folds.yaml) — Site-stratified 5-fold strategy. Complete.
- **✓configs/infer_kaggle.yaml** (configs/infer_kaggle.yaml) — CPU inference config, 90-min budget, batch_size=32. Complete.
- **✓src/utils.py** (src/utils.py — 185 lines) — Seed, config loading, label encoder, per-taxon ROC-AUC, OOF save.
- **✓src/features.py** (src/features.py — 120 lines) — load_audio (soundfile + torchaudio resample), peak_normalize, energy gate, LogMelExtractor.
- **✓src/dataset.py** (src/dataset.py — 210 lines) — Clip windowing + energy gating, soundscape window loader, InferenceDataset for Kaggle.
- **✓src/model.py** (src/model.py — 110 lines) — GeMPooling (p=3, fixed), EfficientNet-B0 backbone via timm, in_chans=1, 234-class linear head.
- **✓src/losses.py** (src/losses.py — 120 lines) — BCEWithLogitsLoss (Phase 0), FocalLoss, AsymmetricLoss (Phase 1+ options).
- **✓src/augment.py** (src/augment.py — 195 lines) — Full augmentation pipeline (all OFF for Phase 0): background mixing, waveform mixup, SpecAugment, saturation.
- **✓src/train_engine.py** (src/train_engine.py — 180 lines) — train_one_epoch, validate_one_epoch (per-taxon AUC), run_training loop, OOFAccumulator.
- **✓src/infer_engine.py** (src/infer_engine.py — 160 lines) — InferenceEngine, benchmark_inference CPU timer, safe fallback for failed files.
- **✓scripts/make_folds.py** (scripts/make_folds.py — 145 lines) — GroupKFold on soundscape site codes, StratifiedKFold on primary_label for clips. Writes data/folds.csv.
- **✓scripts/train.py** (scripts/train.py — 190 lines) — Full 5-fold CV loop. Accumulates OOF, saves per-fold checkpoints. --dry-run mode.
- **✓scripts/validate.py** (scripts/validate.py — 145 lines) — Loads OOF NPZ or checkpoint, prints per-taxon AUC, Phase 0 exit criterion check.
- **✗data/folds.csv** (data/folds.csv — currently empty) — Needs sklearn, run:python scripts/make_folds.py
- **✗Training run** (artifacts/oof/, artifacts/checkpoints/) — No OOF results yet. Needs GPU environment with torch+timm installed.

**Summary:** All code is written and syntax-verified. 13 of 15 Phase 0 deliverables are complete.
The two remaining items (`data/folds.csv` and the training run itself) require a GPU environment
with `torch`, `torchaudio`, `timm`, and `scikit-learn` installed.

## §2 — What Was Built

### Data pipeline

| Step | Code | Detail |
| --- | --- | --- |
| Audio loading | `src/features.py:load_audio` | soundfile OGG decode → mono → torchaudio resample to 32kHz. Supports offset+duration for windowed reads. |
| Peak normalization | `src/features.py:peak_normalize` | Scales peak to 0.9. Applied universally to every window. Removes 600× loudness variance (EDA §E3). |
| Windowing | `src/dataset.py:build_clip_windows` | Non-overlapping 5s windows. Short clips padded with zeros. Window index pre-computed (metadata only, no audio load at index time). |
| Energy gating | `src/features.py:is_silent` + `WindowDataset.__getitem__` | If RMS < −45 dBFS, label vector is zeroed out. Label noise from silent windows eliminated. Threshold configurable. |
| Soundscape windows | `src/dataset.py:build_soundscape_windows` | Parses semicolon-separated per-window labels from soundscape\_labels.csv. Strong labels used directly. |
| Log-mel extraction | `src/features.py:LogMelExtractor` | torchaudio MelSpectrogram → AmplitudeToDB. n\_mels=128, hop=320, fmin=20Hz, fmax=16kHz. Output: (1,128,500). |

### Label encoding

The taxonomy.csv row order defines class indices 0–233.
Labels in `train.csv` use the same `primary_label` string keys.
Soundscape labels are semicolon-separated strings of the same keys.
The label encoder (`src/utils.py:build_label_encoder`) maps each key to its taxonomy index.
This ordering matches the `sample_submission.csv` column order exactly.

### Model

| Component | Choice | Rationale |
| --- | --- | --- |
| Backbone | EfficientNet-B0 (timm) | ~5M params; feasible on Kaggle CPU in 90 min; standard BC baseline |
| Input | (B, 1, 128, 500) | in\_chans=1 via timm; single-channel log-mel |
| Pooling | GeM (p=3, fixed) | Standard audio competition choice; better than avg pooling for species detection |
| Head | Dropout(0.2) + Linear(feat→234) | Flat 234-class head; multi-taxon head tested in Phase 3 |
| Output | Raw logits | Sigmoid applied at inference only; BCE loss operates on logits directly |
| Optimizer | AdamW, backbone LR = 0.1× head LR | Differential LR preserves pretrained features |
| Scheduler | Linear warmup (2 ep) + cosine decay | Standard; avoids destructive early updates |

### Validation

**Soundscapes:** GroupKFold on site code. 9 labeled sites → ~1–2 sites per val fold.
Prevents site-acoustic leakage from soundscape training data.

**Clips:** StratifiedKFold on primary\_label (Phase 0 baseline).
Does not prevent author leakage — this is known and will be improved in Phase 2
(author-stratified CV). The gap between random and author-stratified OOF will reveal
the magnitude of recorder fingerprint leakage.

Metrics reported: **macro ROC-AUC overall** and separately for each of the 5 taxon classes
(Aves, Amphibia, Insecta, Mammalia, Reptilia). Non-bird AUC below 0.60 is a red flag.

## §3 — How to Run Phase 0

### Step 1 — Install dependencies

```
pip install torch torchaudio timm scikit-learn soundfile pyyaml tqdm numpy pandas
```

### Step 2 — Generate folds

```
cd /data/birdclef2026
python scripts/make_folds.py
# Output: data/folds.csv (~35,615 rows)
# Verify: tail -3 data/folds.csv
```

### Step 3 — Dry run (verify pipeline end-to-end)

```
python scripts/train.py --config configs/train.yaml --fold 0 --dry-run
# Expected: 1 epoch, 2 batches, no crash
# Takes ~2 min on CPU, ~30s on GPU
```

### Step 4 — Full training

```
# All 5 folds (recommended)
python scripts/train.py --config configs/train.yaml

# Single fold (for fast iteration / debugging)
python scripts/train.py --config configs/train.yaml --fold 0
```

### Step 5 — Validate OOF

```
python scripts/validate.py \
    --oof artifacts/oof/baseline_v1_oof.npz \
    --out-json artifacts/oof/baseline_v1_metrics.json
```

### Step 6 — Benchmark inference

```
python -c "
import torch
from src.utils import load_config
from src.model import build_model
from src.infer_engine import benchmark_inference

cfg = load_config('configs/infer_kaggle.yaml')
model = build_model(load_config('configs/train.yaml'))
result = benchmark_inference(model, cfg, n_windows=200)
print(result)
"
# Target: < 250 ms/soundscape (12 windows × <20ms/window)
```

### Expected runtime

| Hardware | Per epoch | 30 epochs × 5 folds |
| --- | --- | --- |
| Kaggle GPU (T4) | ~8–12 min | ~20–30 hours |
| A100 80GB | ~3–5 min | ~8–12 hours |
| CPU only | ~60+ min | Not practical for training |

**Note:** Train on GPU. The 90-minute CPU constraint applies only to Kaggle *inference* notebooks —
not to training. Training is done locally or on a cloud GPU.

## §4 — Key Design Decisions

### Why energy gating in the dataset rather than pre-filtering?

Pre-filtering would require loading all 35K audio files once to compute RMS, which takes ~2 hours.
Lazy per-window gating in `__getitem__` adds ~0.5ms per window but requires no pre-computation.
The gate zeros the label vector rather than skipping the sample — the silent window still provides
a valid negative training signal for all species. The threshold (−45 dBFS) is configurable and
will be swept in Phase 1.

### Why soundfile rather than torchaudio for loading?

soundfile is already installed on this machine and handles OGG natively without additional codecs.
torchaudio (when available) is used for resampling and mel extraction, where its GPU acceleration helps.
This keeps audio loading portable.

### Why StratifiedKFold on primary\_label for clips (not GroupKFold on author)?

Author-stratified CV is more correct (prevents recorder fingerprint leakage) but requires
a careful author-cleaning step first. The Phase 0 goal is a runnable baseline with a
*known* limitation. The gap between Phase 0 OOF and Phase 2 author-stratified OOF
will directly measure the magnitude of recorder leakage — which is valuable diagnostic information.

### Why pad-to-exactly-5s rather than zero-filling longer?

The prediction unit is exactly 5 seconds. Padding short clips < 5s matches the test scoring unit.
Clips shorter than 2.5s (very rare based on EDA) will be mostly padding — the energy gate will
catch these and zero their labels if the audio energy is below threshold.

### Why no ONNX export yet?

ONNX export is a Phase 5 (final polish) concern. For Phase 0, getting a working training loop
and a measurable OOF AUC is the priority. ONNX will be validated against the PyTorch output
before any submission.

## §5 — Exit Criterion Check

Phase 0 exit requirements (from modeling\_hypotheses.md):

| Criterion | Target | Status | How to check |
| --- | --- | --- | --- |
| OOF Macro AUC | > 0.65 | Pending training run | `python scripts/validate.py --oof artifacts/oof/baseline_v1_oof.npz` |
| Non-bird taxon AUC | > 0.60 per taxon | Pending training run | Per-taxon breakdown printed by validate.py |
| CPU inference time | < 60 min for expected test set | Pending benchmark | `benchmark_inference(model, cfg, n_windows=200)` |
| Pipeline end-to-end | Dry run completes without error | Pending GPU run | `python scripts/train.py --dry-run --fold 0` |
| All code syntax clean | No import errors | PASS | `python -m py_compile src/*.py scripts/*.py` — all OK |

**What to expect from the baseline:**
BC2024 baselines with EfficientNet-B0 + no augmentation typically land at 0.60–0.70 macro AUC on val
depending on the validation scheme. With a fully-leaky random split, expect ~0.72–0.78 (inflated).
With site-stratified splits, expect ~0.60–0.68. Amphibia and Insecta AUC will likely be below 0.60
at baseline — this is expected and motivates Phase 1 augmentations.

### If exit criterion is not met

- **AUC < 0.55:** Check data pipeline. Is label encoding correct? Are soundscape windows loading properly? Run on fold 0 alone with `--dry-run` and inspect first batch shapes and label sums.
- **AUC 0.55–0.65:** Expected territory. Proceed to Phase 1 augmentations — they should add +0.04–0.08.
- **AUC > 0.65 but non-bird < 0.60:** Phase 1 mixup and background mixing will address. Proceed to Phase 1 with this flagged.
- **Inference timeout:** Reduce batch\_size or switch to EfficientNet-B0 ONNX. Do not proceed to Phase 2 until inference budget is met.

## §6 — Risks for Phase 0

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Label mismatch between train.csv and taxonomy.csv primary\_label format | Medium | Both normalised to string. Check: `set(train_df.primary_label) - set(taxonomy_df.primary_label)` should be empty. |
| Soundscape files not decodable with soundfile (codec issue) | Medium | load\_audio has try/except; corrupt files are skipped with a warning. Pre-scan: count warnings in dry run. |
| Fold assignment: some species absent from all val folds | Low | make\_folds.py logs rare class warnings. Absent-from-val species get NaN AUC (skipped in macro) — same as competition. |
| GPU OOM with batch\_size=64 | Low | Reduce batch\_size to 32 or 16 in train.yaml. EfficientNet-B0 with (64, 1, 128, 500) input uses ~2GB GPU memory. |
| Inflated OOF AUC due to StratifiedKFold on clips (author leakage) | Medium — Known | Known limitation of Phase 0. Report OOF AUC with caveat. Phase 2 adds author-stratified CV for true estimate. |
| Insect sonotype classes (25 zero-clip species) produce AUC = 0.5 | Expected | Only 66 labeled soundscape windows exist for these. H11 (two-stage detection) addresses in Phase 4. Accept for Phase 0. |

## §7 — Transition to Phase 1

Phase 1 starts as soon as Phase 0 produces a valid OOF AUC (even below 0.65).
The baseline AUC is needed to measure Phase 1 gains accurately.

### Phase 1 config changes (no code changes needed)

All augmentations are already implemented and flag-controlled. Phase 1 is a config edit:

```
# configs/train_phase1.yaml — inherits train.yaml, enables P1 augmentations
experiment_name: phase1_bg_mixup

augmentation:
  background_noise:
    enabled: true
    snr_db_range: [-5.0, 10.0]  # H1: Pantanal background mixing

  mixup:
    enabled: true
    alpha: 0.4
    prob: 0.5                   # H2: waveform mixup (50% of batches)

# Phase 1 also needs the background pool list:
# background_pool: paths to all 10,592 unlabeled soundscapes
```

### Additional Phase 1 steps

1. Build unlabeled soundscape path list (all 10,592 OGG files not in soundscape\_labels.csv)
2. Pre-screen soundscapes for extreme artifacts (rain, wind — EDA §E)
3. Enable background\_noise + mixup, retrain baseline fold 0 only, compare OOF AUC
4. If +0.03+ improvement: run full 5-fold Phase 1 run
5. Check rare-class AUC — mixup sometimes hurts classes with very few positives

Full Phase 1 plan: <phase1_report.html>

BirdCLEF 2026 — Phase 0 Report | Generated 2026-04-14 | reports/phase0\_report.html
