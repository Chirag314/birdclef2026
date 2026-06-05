# BirdCLEF 2026

Kaggle competition workspace — structured, reproducible, medal-focused.

**Competition:** [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)
**Metric:** Macro ROC-AUC across 234 species
**Inference constraint:** CPU-only, 90-minute runtime on Kaggle

---

## ⚑ FINAL RESULTS — Competition ended 2026-06-03

| Metric | Public LB | Private LB |
|---|---|---|
| Score | 0.95097 | **0.94138** |
| Rank | 293 / 4,085 (top 7.2%) | 865 / 4,085 (top 21.2%) |
| Medal | Bronze (inside cutoff) | **No medal** |

Selected submissions: `birdnet_sidecar_v2` (ref 53205912) + `tax_blend v2` (ref 53126671) — both 0.941 private.
**Root cause of miss:** BirdNET correction caps overfit the 34% public test set. See `solutions_analysis.md`.

---

## Experiment Phases

| Phase | Status | Notes |
|---|---|---|
| Competition understanding | Done | `competition_memo.md` |
| Deep EDA | Done | `eda/`, `reports/` |
| Validation design | Done | GroupKFold by site, `data/folds.csv` |
| Baseline training (5-fold) | Done | `baseline_v1`, clip CV 0.9248 |
| Phase 1 — Augmentation isolation | Done | bg noise best (CV 0.9412, LB 0.789) |
| Phase 2 — Soundscape-aware training | Done | No improvement — see findings |
| Phase 3 — EfficientNet-B2 | Done | B2 5-fold LB=0.753 — **worse** than B0 |
| Phase 4 — Label smoothing (B0, 5-fold) | Done | LB=**0.833** |
| Phase 5 — Perch MLP (5-fold) | Done | OOF 0.977, LB=**0.873** (+0.040) |
| Phase 5 — ProtoSSM pipeline | Done | LB=**0.944** (+0.071) |
| Phase 5 — EoS pipeline | Done | LB=**0.949** |
| Phase 6 — EoS dial tuning (exp017–023) | Done | All dials exhausted at 0.949 |
| Phase 6 — Tax blend (Model mix + TAX) | Done | LB=**0.950** |
| Phase 6 — BirdNET sidecar v1/v2/v3 | Done | LB=**0.950** (private: 0.941, overfit) |
| Phase 7 — FINAL | **Complete** | No medal — public/private gap 0.009 |

---

## Full LB Scoreboard

| Experiment | Public LB | Private LB | Notes |
|---|---|---|---|
| baseline_v1 (B0 5-fold) | — | — | CV 0.9248 |
| phase1a_bg_noise | 0.789 | — | |
| phase3_b2 | 0.753 | — | bigger = worse |
| phase4_label_smooth | 0.833 | — | |
| Perch MLP 5-fold | 0.873 | — | OOF 0.977 |
| ProtoSSM | 0.944 | — | |
| EoS pipeline | 0.949 | — | correction_weight=0.10 |
| EoS 80% + ProtoSSM 20% | 0.948 | — | -0.001, ruled out |
| EoS dial tuning (exp017–023) | 0.949 | — | ceiling confirmed |
| Tax blend (eos9 + TAX_SMOOTH) | 0.950 | — | diversity hedge (slot 2) |
| BirdNET sidecar v1 | 0.950 | — | rank +200 |
| BirdNET sidecar v2 | **0.950** | **0.941** | rank +250, selected slot 1 |
| BirdNET sidecar v3 | 0.950 | — | no gain over v2 |
| **Final (best private)** | **0.950** | **0.941** | rank 920, no medal |

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
- 5-fold: ~0.833

### Perch v2 embeddings — massive domain gap fix (Phase 5)
Perch v2 (1536-dim, trained on deployment-style audio) bridges the focal-clip → soundscape gap:
- Perch MLP 5-fold: LB **0.873** (+0.040 over phase4)
- ProtoSSM head on Perch: LB **0.944** (+0.071)
- EoS variant (correction_weight=0.10): LB **0.949** ← current best
- ProtoSSM parameter tuning (power, gates): confirmed ceiling at 0.944, no path forward
- EoS+ProtoSSM blend (80/20): pending — expected 0.950+

**LB progression: 0.789 → 0.833 → 0.873 → 0.944 → 0.949**

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

## Lessons for BirdCLEF 2027

1. **Never tune postprocessing against the public test set.** BirdNET correction caps dialled to 4-decimal public precision cost ~400 private ranks.
2. **Diversity of model architectures matters more than single-model dial tuning.** Top teams used Perch + distilled SED + custom CNN — we stayed on Perch the whole time.
3. **Pseudo-labeling on soundscapes is the unlock.** Nikita Babych (2025 + 2026 winner) built on noisy student self-training. We never attempted it.
4. **OOF validation > public LB sniping.** Tune postprocessing on OOF predictions, not LB scores.
5. **Domain shift is real and large.** Train-test gap confirmed: focal clips (global iNat/XC) vs Pantanal soundscapes. Soundscape-aware training + pseudo-labels on test-domain audio is the structural fix.

See `solutions_analysis.md` for detailed comparison with medal solutions.
