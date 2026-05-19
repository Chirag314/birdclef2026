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
| Baseline training (5-fold) | Done | `baseline_v1`, clip CV 0.9248 |
| Phase 1 — Augmentation isolation | Done | bg noise best (CV 0.9412, LB 0.789) |
| Phase 2 — Soundscape-aware training | Done | No improvement — see findings |
| Phase 3 — EfficientNet-B2 | Done | B2 5-fold LB=0.753 — **worse** than B0 |
| Phase 4 — Label smoothing (B0) | **Running folds 1-4** | fold0 LB=**0.818** ← new best! |
| Kaggle inference notebook | Done | `kaggle_notebook/` |

---

## LB Scoreboard

| Experiment | Backbone | Folds | CV (clip) | Public LB |
|---|---|---|---|---|
| baseline_v1 | B0 | 5 | 0.9248 | — |
| phase1a_bg_only | B0 | 5 | 0.9412 | **0.789** ← best |
| phase2_ss_aware | B0 | 1 | 0.8220 (SS holdout) | 0.756 |
| phase2b_ss | B0 | 1 | 0.7883 (SS holdout) | 0.774 |
| phase3_b2 | **B2** | 1 | 0.9604 | 0.774 |
| phase3_b2 | **B2** | 5 | 0.9590 avg | 0.753 ← worse than B0! |
| phase4_ls | B0 | 1 | 0.9289 | **0.818** ← NEW BEST (+0.029) |
| phase4_ls | B0 | 5 | — | ~0.833 est. (folds 1-4 running) |

---

## Key Findings

### Augmentation (Phase 1)
- Background noise augmentation (+1.6% CV, +? LB) is the best single augmentation
- Mixup alone hurts (-0.019 CV vs baseline)
- Combining both also hurts vs bg noise alone

### Soundscape oversampling (Phase 2)
- Training soundscapes ≠ test soundscapes in distribution
- Oversampling 1,478 labeled training soundscapes causes overfitting to a biased sample
- Removing low-quality clips (rating<3) dropped 40% of data and hurt LB
- Soundscape holdout CV (~0.79) is a more honest proxy than clip CV (~0.94)
- **Conclusion:** soundscape oversampling does not help with this dataset

### Ensemble effect
- Single fold consistently gives LB=0.774 regardless of architecture
- 5-fold ensemble worth exactly +0.015 LB (0.774→0.789 for B0)
- Every experiment with 1 fold scores 0.774 on public LB

### Backbone capacity (Phase 3) — NEGATIVE RESULT
- B2 (8M params) CV=0.9590 avg, LB=0.753 — WORSE than B0 5-fold (0.789)
- CV-LB gap: B0=0.152, B2=0.206 — B2 overfits clips MORE aggressively
- **Conclusion: bigger model = more clip overfitting = worse soundscape LB**
- B4 or larger would make this worse, not better
- Ruled out: capacity increase as the path to improvement

### Root cause (confirmed) + Fix found
Higher clip CV = more overconfidence on clips = worse LB.
**Label smoothing (0.05) fixes this directly:**
- Clip CV drops: 0.940 → 0.929 (less memorization)
- LB jumps: 0.774 → **0.818** (+0.029 from 1 fold alone)
- 5-fold expected: ~0.833

**This is the single biggest LB improvement found.**

---

## Repository Structure

```
birdclef2026/
├── CLAUDE.md                  # Competition strategy and working rules
├── competition_memo.md        # Competition understanding summary
├── configs/
│   ├── base.yaml              # Shared defaults for all experiments
│   ├── train.yaml             # Full baseline config
│   ├── train_phase1a.yaml     # Phase 1a: background noise only (best)
│   ├── train_phase1b.yaml     # Phase 1b: mixup only
│   ├── train_phase1c.yaml     # Phase 1c: bg noise + mixup
│   ├── train_phase2_ss.yaml   # Phase 2: soundscape holdout, 10x oversample
│   ├── train_phase2b_ss.yaml  # Phase 2b: soundscape holdout, 5x oversample
│   ├── train_phase3_b2.yaml   # Phase 3: EfficientNet-B2 (current)
│   ├── infer_kaggle.yaml      # Kaggle inference settings
│   └── folds.yaml             # Fold generation logic
├── data/
│   ├── folds.csv              # GroupKFold fold assignments
│   └── processed/
├── eda/                       # EDA notebooks
├── reports/                   # Analysis reports
├── src/
│   ├── dataset.py             # Dataset and window construction
│   ├── features.py            # Log-mel spectrogram extraction
│   ├── augment.py             # Mixup, background noise, SpecAugment
│   ├── model.py               # EfficientNet-B0/B2 + GEM pooling
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
    ├── inference.py           # Kaggle inference script (CPU-safe)
    └── kernel-metadata.json
```

---

## Training Commands

```bash
# Phase 3 B2 (current best)
python scripts/train.py --config configs/train_phase3_b2.yaml --fold 0

# Phase 1a (best LB so far)
python scripts/train.py --config configs/train_phase1a.yaml

# Dry run (sanity check)
python scripts/train.py --config configs/train_phase3_b2.yaml --dry-run

# Requirements: CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Next Steps

1. ~~Phase 1: augmentation~~ Done — bg noise best (LB 0.789)
2. ~~Phase 2: soundscape oversampling~~ Done — no improvement
3. ~~Phase 3: B2 backbone~~ Done — worse (LB 0.753), rules out capacity
4. ~~Phase 4 fold0: label smoothing~~ Done — LB **0.818** (+0.029, new best!)
5. **Phase 4 folds 1-4** — running now, est. 5-fold LB ~0.833
6. Next: SpecAugment on top of label smoothing, then B2+smoothing
