# BirdCLEF 2026

Kaggle competition workspace — structured, reproducible, medal-focused.

**Competition:** [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)
**Metric:** Macro ROC-AUC across 234 species
**Inference constraint:** CPU-only, 90-minute runtime on Kaggle

---

## Current Status

| Phase | Status | Notes |
|---|---|---|
| Competition understanding | Done | `competition_memo.md` |
| Deep EDA | Done | `eda/`, `reports/` |
| Validation design | Done | GroupKFold by site, `data/folds.csv` |
| Baseline training (5-fold) | Done | `baseline_v1`, macro AUC 0.9248 |
| Phase 1a — Background noise aug | Done | macro AUC **0.9412** ← best so far |
| Phase 1b — Mixup aug | Done | macro AUC 0.9223 (below baseline) |
| Phase 1c — BG noise + Mixup | **Running** | Combined augmentation |
| Kaggle inference notebook | Done | `kaggle_notebook/` |
| Submission | Pending | After phase1c |

---

## Repository Structure

```
birdclef2026/
├── CLAUDE.md                  # Competition strategy and working rules
├── competition_memo.md        # Competition understanding summary
├── configs/
│   ├── base.yaml              # Shared defaults for all experiments
│   ├── train.yaml             # Full baseline config
│   ├── train_phase1a.yaml     # Phase 1a: background noise only
│   ├── train_phase1b.yaml     # Phase 1b: mixup only
│   ├── infer_kaggle.yaml      # Kaggle inference settings
│   └── folds.yaml             # Fold generation logic
├── data/
│   ├── folds.csv              # GroupKFold fold assignments
│   └── processed/
├── eda/                       # EDA notebooks
├── reports/                   # Analysis reports
│   ├── eda_summary.md
│   ├── risks_and_hypotheses.md
│   └── ...
├── src/
│   ├── dataset.py             # Dataset and window construction
│   ├── features.py            # Log-mel spectrogram extraction
│   ├── augment.py             # Mixup, background noise, SpecAugment
│   ├── model.py               # EfficientNet-B0 + GEM pooling
│   ├── losses.py              # BCE with label smoothing
│   ├── train_engine.py        # Training loop (AMP, early stopping)
│   ├── infer_engine.py        # Inference helpers
│   └── utils.py               # Config loading, fold utils
├── scripts/
│   ├── train.py               # Main training entrypoint
│   ├── make_folds.py          # Generate folds.csv
│   ├── validate.py            # Local evaluation
│   ├── export_for_kaggle.py   # Package artifacts for Kaggle
│   └── benchmark_cpu.py       # CPU runtime feasibility check
├── artifacts/
│   ├── checkpoints/           # Best model weights per fold/experiment
│   ├── oof/                   # OOF predictions and metrics
│   └── logs/                  # Training logs and config snapshots
└── kaggle_notebook/
    ├── notebook.ipynb         # Kaggle inference notebook
    ├── inference.py           # Standalone inference code
    └── kernel-metadata.json
```

---

## Model

- **Backbone:** EfficientNet-B0 (pretrained ImageNet)
- **Pooling:** GEM (p=3.0)
- **Head:** Linear (234 classes)
- **Input:** 128-mel log spectrogram, 5-second windows
- **Loss:** BCE with logits
- **Optimizer:** AdamW, cosine warmup schedule
- **Mixed precision:** Yes (AMP)

---

## Experiments

### Baseline (`baseline_v1`)
5-fold GroupKFold, no augmentation. OOF saved to `artifacts/oof/`.

### Phase 1 — Augmentation Isolation
Goal: determine which augmentation drives CV gain before combining.

| Experiment | Augmentation | Macro AUC |
|---|---|---|
| `baseline_v1` | None | 0.9248 |
| `phase1a_bg_only` | Background noise (SNR -5 to 10 dB) | **0.9412** |
| `phase1b_mixup_only` | Waveform mixup (α=0.4, p=0.5) | 0.9223 |
| `phase1c_bg_mixup` | BG noise + Mixup combined | Running |

**Finding:** Background noise augmentation alone gives the biggest gain (+1.6% over baseline). Mixup alone underperforms. Phase1c tests whether combining both helps further.

---

## Training

```bash
# Full 5-fold baseline
python scripts/train.py --config configs/train.yaml

# Single fold (fast iteration)
python scripts/train.py --config configs/train.yaml --fold 0

# Phase 1a (background noise only)
python scripts/train.py --config configs/train_phase1a.yaml

# Phase 1b (mixup only)
python scripts/train.py --config configs/train_phase1b.yaml

# Dry run (2 batches, sanity check)
python scripts/train.py --config configs/train.yaml --dry-run
```

**Requirements:** GPU strongly recommended. PyTorch must be installed with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Key Findings

- **28 species** have zero focal training clips — covered only via soundscapes
- **Domain shift** is the main risk: global focal recordings vs Pantanal soundscapes
- **Validation:** random row split leaks badly; GroupKFold by site is the safe choice
- **Background noise augmentation** bridges clip↔soundscape domain gap
- **Kaggle inference is CPU-only** — model must run in <90 min; EfficientNet-B0 fits comfortably

---


## Public LB Result

| Submission | CV AUC | Public LB | Gap |
|---|---|---|---|
| phase1a_bg_only (5-fold) | 0.9412 | **0.789** | -0.152 |

**Root cause of gap:** domain shift. Training is 35,549 focal clips (206 species), test is Pantanal soundscapes. Only 12 species overlap between clip-training and soundscape-label domains. CV on clips is optimistic by ~0.15.

**New direction:** soundscape-aware training — oversample soundscape windows, soundscape-based validation, fine-tune on soundscape distribution.

## Next Steps

1. ~~Augmentation isolation~~ Done — phase1a bg noise wins (CV 0.9412)
2. ~~Submit~~ Done — public LB 0.789 reveals domain shift gap
3. **Soundscape-aware training** — oversample soundscapes 10x, soundscape-based CV
4. Fine-tune on soundscapes, filter low-quality clips (rating=0)
5. Re-submit with soundscape-aware model
