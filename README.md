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
| Baseline training (5-fold) | Done | `baseline_v1`, OOF saved |
| Phase 1a — Background noise aug | **Done** | Reptilia AUC 0.91, fold 0 |
| Phase 1b — Mixup aug | **Running** | Est. finish ~13:53 today |
| Kaggle inference notebook | Done | `kaggle_notebook/` |
| Phase 1 comparison & decision | Pending | After phase1b finishes |
| Submission | Pending | After augmentation decision |

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

| Experiment | Augmentation | Status | AUC |
|---|---|---|---|
| `phase1a_bg_only` | Background noise (SNR -5 to 10 dB) | Done | Reptilia 0.91 |
| `phase1b_mixup_only` | Waveform mixup (α=0.4, p=0.5) | Running | — |

Next: compare phase1a vs phase1b vs baseline OOF AUC, then decide combination strategy.

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

## Next Steps

1. Wait for phase1b to finish (~13:53)
2. Compare OOF AUC: baseline vs phase1a vs phase1b
3. Decide augmentation strategy (combine? pick one? tune SNR?)
4. Train combined augmentation model
5. Export best checkpoint and submit
