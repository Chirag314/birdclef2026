# BirdCLEF 2026 — Phase 1 Report

> Converted from `phase1_report.html` for GitHub rendering. Original HTML preserved in this folder.

# BirdCLEF 2026

Phase 1 Report — Core Augmentation (H1, H2, H3, H7)

---

Generated 2026-04-14 | Target dates: Apr 26 – May 3

- **Phase:** Phase 1 — Core Augmentation
- **Entry Condition:** Phase 0 OOF AUC measured (any value)
- **Exit Criterion:** +0.03 vs Phase 0 · Non-bird AUC > 0.60
- **Hypotheses Tested:** H1 · H2 · H3 · H7 (all P1 Must-Have)

[§1 Goal](#goal)
[§2 Augmentations](#aug-plan)
[§3 Config Changes](#config)
[§4 Background Pool](#background-pool)
[§5 Experiment Order](#experiment-order)
[§6 Expected Results](#expected-results)
[§7 Risks](#risks)
[§8 Exit Criterion](#exit)

## §1 — Phase 1 Goal

Phase 1 target: Apr 26 – May 3 | Must produce +0.03 macro AUC vs Phase 0 baseline.
If Phase 0 AUC = 0.63, Phase 1 must reach ≥ 0.66. All 4 P1 augmentations are
already implemented in src/augment.py — Phase 1 is config changes + background pool setup.

Phase 0 trains a clean baseline with **zero augmentation**.
Phase 1 enables the four **"P1 Must Have"** interventions identified in EDA:

| # | Hypothesis | EDA Evidence | Expected Gain |
| --- | --- | --- | --- |
| H1 | **Background mixing** with unlabeled Pantanal soundscapes | §G5, §F1: clip→soundscape domain gap is primary risk | +0.04–0.08 AUC |
| H2 | **Waveform mixup** (λ~Beta(0.4,0.4)) with soft labels | §F2, §C: 87% of test windows have 2+ co-active species; training is single-label | +0.02–0.05 AUC |
| H3 | **Energy gating threshold tuning** | §E3, §F3: 14% mean silence ratio; clip labels on silent windows = false positives | Precision improvement (not directly AUC but matters for rare classes) |
| H7 | **Peak normalization** (already in Phase 0 as default) | §E3: 600× loudness variance | Correctness fix, already enabled |

**Important:** Test H1 (background mixing) and H2 (mixup) independently on fold 0
before running a full 5-fold Phase 1 run. This prevents wasting 20+ GPU-hours on a combination
that doesn't help.

## §2 — Augmentation Details

#### H1 — Background Mixing P1 Must Have

Code: src/augment.py:background\_mixing
Config key: augmentation.background\_noise.enabled
Phase 0 state: disabled

**What it does:** For each training window, sample a random 5-second window
from an unlabeled Pantanal soundscape and mix it into the signal at a random SNR.
The background label is all-zeros (the background carries no species labels).

**Why it's the highest-ROI intervention:** The test set consists of passive ARU
soundscape recordings in the Pantanal. The training clips are close-mic XC recordings from
around the world. Background mixing directly bridges this domain gap by overlaying
real Pantanal acoustic environments onto training clips — making the model learn species
calls on top of the exact noise distribution it will encounter at test time.

**SNR range:** −5 to +10 dB. At −5 dB, the background is slightly louder
than the signal (challenging); at +10 dB, the signal is clearly dominant. This range
covers the realistic SNR spectrum seen in Pantanal ARU recordings.

**Background pool pre-screening:** Before mixing, pre-screen soundscapes for
extreme artifacts (heavy rain, wind-dominated files). Files with mean RMS below −55 dBFS
(essentially silent) or peak saturation > 5% of frames should be excluded from the pool.

#### H2 — Waveform Mixup P1 Must Have

Code: src/augment.py:waveform\_mixup
Config key: augmentation.mixup.enabled
Phase 0 state: disabled

**What it does:** With probability 0.5, sample a second training window,
blend waveforms as x = λ·x₁ + (1−λ)·x₂ where λ ~ Beta(0.4, 0.4), and blend labels
as y = λ·y₁ + (1−λ)·y₂. The loss function must accept soft (float) labels — BCE
with logits does this naturally.

**Why it matters:** 87% of expert-labeled test soundscape windows contain
2+ co-active species. Training clips are almost entirely single-label. This mismatch
means the baseline model has never been trained to predict multiple simultaneous species —
mixup directly simulates this co-presence during training.

**Warning — rare class risk:** Mixup blends labels, so a rare species
(very few positives) can end up with diluted label values (e.g., 0.3 instead of 1.0).
Monitor per-class AUC before and after enabling mixup for classes with <10 clips.
If rare-class AUC drops significantly, reduce mixup\_prob to 0.3 or reduce alpha to 0.2.

#### H3 — Energy Gating Threshold Sweep P1 Must Have

Code: src/features.py:is\_silent + WindowDataset.\_\_getitem\_\_
Config key: energy\_gate\_db (in WindowDataset constructor)
Phase 0 state: threshold = −45 dBFS (default)

**What to test:** Compare OOF AUC on fold 0 with thresholds −40, −45, −50 dBFS.
The optimal threshold balances:

- Too lenient (−35 dBFS): many noisy windows get labels → false positives
- Too aggressive (−55 dBFS): valid but quiet calls are labeled as background → false negatives for distant species

**How to sweep:** This is a one-line change in the `WindowDataset`
constructor call in train.py. Run fold 0 only (not full 5-fold) for each threshold value.
Takes ~1/5 the compute of a full run.

#### Rare-species Oversampling P2 High Value

Code: scripts/train.py:get\_sampler (currently returns None)
Phase 0 state: uniform sampling

**What it does:** Use `torch.utils.data.WeightedRandomSampler`
where the sampling weight for each window is proportional to
1/sqrt(class\_frequency of primary\_label). This ensures rare species appear in every epoch.

**When to enable:** After H1+H2 are validated. Oversampling is not an
augmentation for Phase 1 — it is a Phase 2 enhancement that works best when combined
with background mixing (so oversampled rare clips still land in the Pantanal domain).

**Implementation in get\_sampler():** Compute per-window weights from
`window_df.primary_label.map(label_freq_inv)`. Pass to WeightedRandomSampler.
Drop `shuffle=True` from the DataLoader (sampler handles it).

## §3 — Config Changes for Phase 1

All Phase 1 augmentations are already implemented. Phase 1 requires creating a new
`configs/train_phase1.yaml` that inherits from `train.yaml` and
overrides only the augmentation section. No code changes.

```
# configs/train_phase1.yaml
# Phase 1: Background Mixing + Waveform Mixup (H1 + H2)
# Run: python scripts/train.py --config configs/train_phase1.yaml

experiment_name: phase1_bg_mixup

augmentation:
  background_noise:
    enabled: true
    snr_db_range: [-5.0, 10.0]   # H1: Pantanal soundscape overlay

  mixup:
    enabled: true
    alpha: 0.4                    # H2: λ ~ Beta(0.4, 0.4)
    prob: 0.5                     # 50% of training windows get mixup
```

**Background pool setup (required):** Before running Phase 1, build the list of
unlabeled soundscape paths and add it to the Augmenter call in scripts/train.py.
See §4 below.

### Phase 1a — test background mixing alone (fold 0 only)

```
# configs/train_phase1a.yaml — only H1
experiment_name: phase1a_bg_only

augmentation:
  background_noise:
    enabled: true
    snr_db_range: [-5.0, 10.0]
  mixup:
    enabled: false
```

### Phase 1b — test mixup alone (fold 0 only)

```
# configs/train_phase1b.yaml — only H2
experiment_name: phase1b_mixup_only

augmentation:
  background_noise:
    enabled: false
  mixup:
    enabled: true
    alpha: 0.4
    prob: 0.5
```

### Decision tree

1. Run Phase1a (fold 0). Compare fold 0 AUC vs Phase 0 fold 0 AUC.
2. Run Phase1b (fold 0). Compare fold 0 AUC vs Phase 0 fold 0 AUC.
3. If both help: enable both in Phase1 full run.
4. If only one helps: only enable that one.
5. If neither helps on fold 0: diagnose before proceeding (check background pool quality, check label encoding with mixup).

## §4 — Background Pool Setup

### Build the pool

```
# scripts/build_background_pool.py
# Run once before Phase 1 training.
# Identifies all unlabeled soundscapes for background mixing.

import pandas as pd
from pathlib import Path

SOUNDSCAPE_DIR = "/data/birdclef_2026/data/raw/birdclef-2026/train_soundscapes"
LABELS_CSV = "/data/birdclef_2026/data/raw/birdclef-2026/train_soundscapes_labels.csv"
OUT_PATH = "/data/birdclef2026/data/background_pool.txt"

labels_df = pd.read_csv(LABELS_CSV)
labeled_files = set(labels_df["filename"].unique())

all_files = list(Path(SOUNDSCAPE_DIR).glob("*.ogg"))
unlabeled = [str(f) for f in all_files
             if f.name not in labeled_files]

print(f"Total soundscapes: {len(all_files)}")
print(f"Labeled (excluded from pool): {len(labeled_files)}")
print(f"Unlabeled (in pool): {len(unlabeled)}")

with open(OUT_PATH, "w") as f:
    f.write("\n".join(unlabeled))
print(f"Pool saved to {OUT_PATH}")
```

Expected output: ~10,592 unlabeled soundscape paths.

### Pre-screen the pool (optional but recommended)

```
# Quick quality check — flag problematic files
import numpy as np
import soundfile as sf
from pathlib import Path

POOL_PATH = "/data/birdclef2026/data/background_pool.txt"
paths = open(POOL_PATH).read().strip().split("\n")

flagged = []
for p in paths[:500]:  # sample 500 for speed
    try:
        audio, sr = sf.read(p, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        clip_frac = (np.abs(audio) > 0.99).mean()
        if rms_db < -55 or clip_frac > 0.05:
            flagged.append((p, rms_db, clip_frac))
    except Exception as e:
        flagged.append((p, -999, -1))

print(f"Flagged {len(flagged)}/500 sampled files")
print("Extreme examples:", flagged[:5])
```

### Load pool in train.py

```
# In scripts/train.py, add before Augmenter creation:
pool_path = "/data/birdclef2026/data/background_pool.txt"
if Path(pool_path).exists():
    background_pool = open(pool_path).read().strip().split("\n")
    logger.info(f"Background pool: {len(background_pool)} soundscapes")
else:
    background_pool = []
    logger.warning("Background pool not found — H1 disabled")

augmenter = Augmenter(cfg, background_pool=background_pool, window_df=train_win)
```

## §5 — Experiment Order

| Step | Run | Folds | Purpose | Est. GPU time |
| --- | --- | --- | --- | --- |
| 1 | Phase 0 baseline (from Phase 0) | All 5 | Reference AUC for all comparisons | ~20–30 h (T4) |
| 2 | Phase 1a: H1 only (bg mixing) | Fold 0 | Isolate H1 contribution | ~4–6 h |
| 3 | Phase 1b: H2 only (mixup) | Fold 0 | Isolate H2 contribution | ~4–6 h |
| 4 | Phase 1: H1 + H2 combined | Fold 0 | Check for interaction (usually additive) | ~4–6 h |
| 5 | Energy gate sweep (−40/−45/−50 dBFS) | Fold 0 | Find optimal silence threshold (H3) | ~4 h × 3 = 12 h |
| 6 | Phase 1 full run (best config) | All 5 | Official Phase 1 OOF AUC | ~20–30 h |

**Shortcut option:** If time is tight, skip steps 2 and 3 and run H1+H2 together on fold 0
(step 4). If it shows +0.04+, run full 5-fold immediately (step 6). The individual contribution
decomposition is scientifically informative but not strictly necessary to proceed.

## §6 — Expected Results

| Experiment | Expected OOF Macro AUC | Key signal |
| --- | --- | --- |
| Phase 0 baseline | 0.60–0.68 (site-stratified) | Amphibia and Insecta likely below 0.60 |
| +H1 (background mixing) | +0.04–0.08 vs baseline | Largest single gain expected. All taxa benefit. |
| +H2 (mixup) | +0.02–0.04 vs baseline | Aves and Amphibia benefit most. Check rare-class AUC. |
| H1 + H2 combined | +0.05–0.10 vs baseline | Usually near-additive. May be subadditive if bg mixing already handles domain gap. |
| Phase 1 exit target | > Phase 0 + 0.03 · All taxon AUC > 0.60 | If target not met after both H1+H2: diagnose. Do not proceed to Phase 2 blindly. |

### What "does not help" looks like

- **H1 not helping:** Background pool has too many silent or clipped files → pre-screen pool and retry
- **H2 not helping:** Rare-class AUC is dropping → reduce mixup\_prob to 0.3 or alpha to 0.2
- **Both not helping:** Check that augmenter is actually being called (add logging to Augmenter.\_\_call\_\_). Verify mix output RMS is not near-zero (waveform cancellation).

## §7 — Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Mixup dilutes rare-class labels → rare-class AUC drops | Medium | Monitor per-class AUC. For species with <10 clips: if AUC drops >0.02, reduce mixup\_prob to 0.3. |
| Background pool contains test soundscapes (data leakage) | High if present | Pool is built from *train* soundscapes only. Test soundscapes are in a separate directory and never touched. |
| Background mixing SNR too low → signal inaudible → model learns noise | Medium | Minimum SNR = −5 dB (background slightly louder than signal). Lower bound is intentional — test soundscapes have species at low SNR. |
| Augmented training is slower (double audio load for mixup) | Low | Mixup partner is loaded in \_\_getitem\_\_ from the window\_df. This adds ~2ms per window. Acceptable overhead. |
| Energy gate removes distant-species calls (very quiet valid signals) | Medium | Sweep threshold in Phase 1 (step 5 in experiment order). Do not go below −50 dBFS. |

## §8 — Exit Criterion

| Criterion | Target | How to check |
| --- | --- | --- |
| OOF Macro AUC improvement | +0.03 vs Phase 0 | `python scripts/validate.py --oof artifacts/oof/phase1_bg_mixup_oof.npz` |
| All taxon AUC | > 0.60 for all 5 taxon classes | Per-taxon breakdown in validate.py output |
| Rare-class AUC not degraded | No class with <10 clips drops > 0.02 AUC vs Phase 0 | Compare per\_class\_auc in both OOF NPZ files |
| Energy gate threshold locked | Best threshold from sweep selected and fixed for remaining phases | Record in experiment log |

**→ Phase 2 trigger:** Once Phase 1 OOF shows +0.03 improvement and non-bird AUC > 0.60,
proceed to Phase 2 (data expansion: BC2025 data + Neotropical XC for zero-clip species).
Full Phase 2 plan: [modeling\_hypotheses.md §H5, §H6](modeling_hypotheses.md).

### Quick comparison script

```
# Compare Phase 0 vs Phase 1 OOF
import numpy as np
from src.utils import load_config, get_taxonomy_df, compute_metrics, format_metrics

taxonomy_df = get_taxonomy_df("configs/../data/../../birdclef_2026/data/raw/birdclef-2026/taxonomy.csv")

for name, path in [
    ("Phase 0", "artifacts/oof/baseline_v1_oof.npz"),
    ("Phase 1", "artifacts/oof/phase1_bg_mixup_oof.npz"),
]:
    data = np.load(path, allow_pickle=True)
    m = compute_metrics(data["targets"], data["preds"], taxonomy_df)
    print(f"\n{name}:")
    print(format_metrics(m))
    delta = ""
print(f"\nDelta: {m2['macro_auc'] - m1['macro_auc']:+.4f}")
```

BirdCLEF 2026 — Phase 1 Report | Generated 2026-04-14 | reports/phase1\_report.html
