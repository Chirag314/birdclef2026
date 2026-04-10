# BirdCLEF 2026 — Competition Memo

**Date written:** 2026-04-09  
**Data path:** `/data/birdclef_2026/data/raw/birdclef-2026/` (16 GB)  
**Status:** File-derived analysis + web search. Evaluation metric sourced from multiple community references — treat as high-confidence but verify against official Kaggle page directly.

---

## 1. What Is Given

| File / Directory | Rows / Count | Purpose |
|---|---|---|
| `train.csv` | 35,549 | Per-clip metadata + labels (XC + iNat sources) |
| `train_metadata.csv` | 35,549 | **Identical to train.csv** — fully redundant |
| `taxonomy.csv` | 234 rows | Target species list with iNat IDs and class names |
| `train_soundscapes/` | 10,658 ogg | Full deployment-domain soundscape recordings |
| `train_soundscapes_labels.csv` | 1,478 windows | Expert 5-second window labels for 66 of those soundscapes |
| `test_soundscapes/` | readme only | Hidden. Test scored against these files. |
| `sample_submission.csv` | 3 rows | Submission format — 234 species columns + row_id |
| `recording_location.txt` | — | Pantanal, Brazil. Lat −16.5 to −21.6, Lon −55.9 to −57.6 |

**Critical finding:** `train.csv` and `train_metadata.csv` are byte-for-byte identical. One column is `primary_label` (string iNat taxon ID) and another is `inat_taxon_id` (integer) — these encode the same information. Only one file should be used.

---

## 2. What Is Expected

- **Prediction unit:** One row per 5-second window of a test soundscape
- **Row ID format:** `{soundscape_filename_stem}_{end_second}` — e.g., `BC2026_Test_0001_S05_20250227_010002_5` = window ending at second 5
- **Output:** Probability (0–1) for each of 234 species per window
- **Submission:** Wide format — 234 species columns plus `row_id`
- **Test soundscapes:** Dated 2025-02-27 based on sample submission filenames; site S05 visible

---

## 3. Taxonomy — This Is NOT a Pure Bird Competition

| Class | Target Species |
|---|---|
| Aves (birds) | 162 |
| Amphibia (frogs) | 35 |
| Insecta (insects) | 28 |
| Mammalia (mammals) | 8 |
| Reptilia (caiman) | 1 |
| **Total** | **234** |

This is a **multi-taxon bioacoustics** competition. Frogs, insects, mammals, and caiman are all scored alongside birds. Models and feature extraction choices must not be optimized solely for bird calls.

---

## 4. Training Data — Two Very Different Signal Types

### 4A. Individual Clips (`train_audio/`, described in `train.csv`)

- 35,549 clips across 206 species directories
- Sources: XC (Xeno-Canto): 23,043 clips | iNat: 12,506 clips
- **Geographic origin: GLOBAL** — latitude spans −54.86 to +69.58, longitude −159.66 to +175.32
- **Target domain: Pantanal, Brazil only** — severe deployment domain shift
- Labels are **whole-clip labels**, not time-aligned within the clip
- 4,372 clips (12.3%) have secondary labels
- Rating: mean 2.6/5, but **12,849 clips (36%) have rating = 0.0** (unrated quality)

### 4B. Train Soundscapes (`train_soundscapes/`, labels in `train_soundscapes_labels.csv`)

- 10,658 soundscape recordings from 23 deployment sites in Pantanal
- Recordings span 2014–2025; concentrated in 2021–2024
- Month distribution: heavy in Oct–Jan (southern hemisphere spring/summer), very sparse May–Aug
- **Only 66 of 10,658 soundscapes (0.6%) have expert 5-second window labels**
- Those 66 span 9 of 23 sites; site S22 dominates (64% of labeled windows)
- Labels per window: 1–10 co-occurring species (median ~4–5)
- The other 10,592 soundscapes are **unlabeled** — potential for semi-supervised use

---

## 5. The 28 Species With No Training Clips

These appear in `taxonomy.csv` and in the submission, but have **zero files** in `train_audio/`:

| Category | Count | Species |
|---|---|---|
| Insect sonotypes | 25 | `47158son01` through `47158son25` — morphologically similar insects from parent iNat taxon 47158, split into acoustic sonotypes |
| Amphibia | 3 | Guaraní leaf-litter frog (1491113), Chiasmocleis mehelyi (25073), Southern Orange-legged Leaf Frog (517063) |

**Implication:** For 28/234 target classes, the only training signal comes from the 66 labeled soundscape windows. These are the hardest classes to learn. The 25 insect sonotypes are internally distinguished by sound type alone — they are effectively unsupervised acoustic clusters.

---

## 6. Class Imbalance in Train Clips

```
Species with fewest clips:
  116570 (Southern Spectacled Caiman):  1 clip
  23150:                                 1 clip
  516975:                                1 clip
  23724:                                 1 clip
  209233:                                2 clips
  24321:                                 2 clips

Species with most clips:
  rubthr1 (Ruby-throated Hummingbird): 499 clips
  banana:                              498 clips
  fepowl:                              497 clips
```

- Mean clips per species: 172
- Median: 125
- Standard deviation: 155
- Ratio most/least: 499:1

Imbalance is severe. The rarest species also tend to be non-bird taxa where soundscape-based learning is the only option.

---

## 7. Label Format — Hybrid Numeric + eBird Codes

The submission header mixes two identifier types:
- **47 species** identified by **numeric iNat taxon IDs** (e.g., `1161364`, `22961`)
- **187 species** identified by **eBird/XC short codes** (e.g., `ashgre1`, `houspa`, `osprey`)

The `primary_label` column in `train.csv` stores the iNat taxon ID as a string. The `taxonomy.csv` provides the mapping between both systems. Always use `taxonomy.csv` as the canonical label map when building submission columns — do not rely on the order in `sample_submission.csv` alone.

---

## 8. Evaluation Metric

**Macro-averaged ROC-AUC, skipping classes with no true positives in the test set.**

This is consistent with BirdCLEF 2024 and 2025 and confirmed by multiple community sources.

### What this means in practice

- **ROC-AUC per class:** For each of the 234 species, compute the area under the ROC curve treating that species as a binary classification problem (present vs. absent in each 5-second window).
- **Macro average:** Average the per-class AUC scores equally across all classes.
- **Skipping:** Classes where the hidden test set has zero positive windows are excluded from the average. This means rare species that happen to be absent in test contribute nothing to the score — but you still need to score well on those that do appear.

### Critical implications

1. **This is a ranking metric**, not a threshold metric. Calibration of raw probability values matters less than the relative ordering of predictions. You do not need perfectly calibrated probabilities — you need good discrimination.
2. **All classes are equally weighted** in the macro average, regardless of how rare they are in training. A species with 1 training clip has the same weight as one with 499, if it appears in the test set.
3. **Domain shift hurts rare classes most.** If a rare species only appears in a few test windows and your model has poor representation for it, that class tanks your macro average disproportionately.
4. **Threshold-free at evaluation time.** But during training, loss functions with implicit thresholds (BCE, focal loss) are fine — the scoring does not penalize or reward threshold choices.
5. **You cannot know which classes are "skipped"** — that depends on the hidden test set. Do not assume any class is safely skippable.

### What NOT to optimize for

- Do not tune decision thresholds for this metric — it doesn't help.
- Do not sacrifice rare-class performance to boost common-class performance — the macro average penalizes this equally.

---

## 9. Submission Format

```
row_id,species_1,species_2,...,species_234
BC2026_Test_0001_S05_20250227_010002_5,0.004,0.004,...
BC2026_Test_0001_S05_20250227_010002_10,0.004,...
```

- Each row is a 5-second window ending at the given second
- Values are continuous probabilities (0–1)
- 234 species columns must appear in exact taxonomy order

---

## 10. Key Risks and What Is Unusual About This Competition

### Risk 1: Severe domain shift (HIGH)
Train clips are from global XC/iNat sources. Test soundscapes are from Pantanal soundscape recorders. Call acoustics, SNR, background noise, and co-occurrence patterns will be different. Models trained purely on XC clips may not generalize.

### Risk 2: Weak supervision in soundscapes (HIGH)
Only 66/10,658 soundscapes have labels. The 10,592 unlabeled soundscapes must be either ignored or treated as unlabeled data. Pseudo-labeling or semi-supervised use of unlabeled soundscapes is a major design decision.

### Risk 3: Zero-clip species (HIGH)
28 species have no individual training clips. The only training signal for them is the 66 labeled soundscape windows. Learning these species reliably requires either soundscape-based training or transfer from parent taxa.

### Risk 4: Insect sonotypes (HIGH)
25 insect "sonotypes" derived from iNat taxon 47158 are defined by acoustic pattern alone, with no per-sonotype training clips. These are effectively sub-type clusters. Their boundaries are not biologically validated.

### Risk 5: Rating-zero clips (MEDIUM)
36% of train clips have quality rating 0.0 (unrated, not zero quality). These may include clean or noisy recordings. Naively training on all clips may hurt if many rating-zero clips are poor quality.

### Risk 6: Whole-clip vs. window-level alignment (HIGH)
Train clips are labeled at the clip level. Test is scored at 5-second windows. Naively assigning clip-level labels to all 5-second windows will introduce false positives (species may not be audible throughout). This is the standard BirdCLEF label alignment problem.

### Risk 7: Seasonal mismatch (MEDIUM)
Soundscapes are concentrated in Oct–Jan (Pantanal wet season). May–Aug coverage is very sparse. Test soundscapes (Feb 2025) are from the start of the dry season transition. If model performance varies seasonally, this matters.

### Risk 8: Site imbalance in labeled soundscapes (MEDIUM)
Site S22 contributes 64% of the labeled windows. If site acoustics differ, the labeled soundscape distribution is not representative of the full 23-site deployment.

---

## 11. What The Competition Is Really Testing

1. **Soundscape-domain generalization** — can you close the gap between globally-sourced clip recordings and Pantanal soundscape recorders?
2. **Multi-taxon detection** — not just birds; frogs, insects, mammals with different spectral/temporal characteristics
3. **Weak supervision handling** — can you extract useful signal from 10K unlabeled soundscapes?
4. **Rare class learning** — 28 species with no clips, many with < 5 clips total
5. **5-second window prediction** — temporal alignment of clip-level labels to sub-clip windows

---

## 12. Runtime Constraints

- **CPU-only notebook on Kaggle**
- **90-minute hard time limit** for the full inference run
- No GPU available at inference time
- This is the same constraint as BirdCLEF 2025

### What this means for model design

The test set contains soundscapes from multiple sites. Each soundscape must be sliced into 5-second windows and scored for all 234 species. With 90 minutes of CPU time:

- Heavy models (large ViT, EfficientNet-L) will time out
- Model quantization, ONNX export, or smaller architectures are necessary
- Mel spectrogram extraction must be fast (avoid real-time librosa calls per window — pre-batch)
- Multi-threading within the notebook may help but must be tested
- There is a known discussion thread about potential runtime incompatibilities (`discussion/684693`) — worth monitoring

### Practical Bottlenecks

- **Inference speed:** CPU-only 90-min cap is the binding constraint on model size
- **Label alignment:** how you slice clips into windows and assign labels is critical — wrong alignment wrecks training signal
- **Soundscape pseudo-labeling:** using 10K unlabeled soundscapes requires a reliable pseudo-label model — if baseline is weak, noise gets amplified
- **Insect sonotype handling:** these 25 classes likely need a dedicated strategy (or accept near-zero performance on them early on)

---

## 13. Competition Timeline

| Date | Event |
|---|---|
| March 11, 2026 | Competition start |
| May 27, 2026 | Entry and team merger deadline |
| **June 3, 2026** | **Final submission deadline** |
| June 17, 2026 | Working note deadline ($5,000 in publication prizes) |

**~8 weeks remain from today (April 9).**

---

## 14. Next Actions (Ranked by ROI)

1. **Confirm evaluation metric from competition page** — everything depends on this
2. **Inspect audio quality across rating groups** — listen to some rating-0 clips vs rating-4+ clips
3. **EDA on soundscape temporal/geographic structure** — site-level distributions, season effects
4. **Label-space analysis** — class frequency, co-occurrence in soundscape labels, rare class audit
5. **Soundscape label coverage audit** — which of the 234 species actually appear in the 66 labeled files?
6. **Geographic mismatch visualization** — plot train clip locations vs soundscape deployment locations
7. **Audio sample rate / duration distribution** — across both clip types and soundscapes
8. **Inspect a few insect sonotype examples** — understand what these sound like and how distinct they are
