# BirdCLEF 2026 — Modeling Report

> Converted from `modeling_report.html` for GitHub rendering. Original HTML preserved in this folder.

# BirdCLEF 2026

Modeling Hypotheses & Strategy Report

Synthesis of EDA Sections A–I  |  Phase 6 Deliverable

---

- **Target Species:** 234 (162 birds · 35 frogs · 28 insects · 8 mammals · 1 reptile)
- **Training Clips:** 35,549 — 206 of 234 species have clips
- **Zero-Clip Species:** 28 — supervised only via 66 labeled soundscape windows
- **Train Soundscapes:** 10,658 total — only 0.6% have expert window labels
- **Evaluation Metric:** Macro-averaged ROC-AUC (skip classes absent in test)
- **Inference Constraint:** CPU-only Kaggle notebook · 90-minute hard limit
- **Domain Gap:** Global XC/iNat clips → Pantanal passive ARU recorders (SEVERE)
- **Deadline:** June 3, 2026 — ~8 weeks remaining

[§1 Structure](#s1)
[§2 Hypotheses](#s2)
[§3 Pipeline](#s3)
[§4 Validation](#s4)
[§5 Augmentation](#s5)
[§6 Prior Data](#s6)
[§7 Risk Register](#s7)
[§8 Experiment Plan](#s8)
[§9 Open Questions](#s9)

# Contents

- [§1 Competition Structure — What the Model Must Actually Do](#s1)
- [§2 Ranked Modeling Hypotheses (Top 12)](#s2)
- [§3 Data Pipeline Architecture](#s3)
- [§4 Validation Strategy](#s4)
- [§5 Augmentation Plan (Evidence-Based)](#s5)
- [§6 Prior Data Decision](#s6)
- [§7 Risk Register](#s7)
- [§8 Phased Experiment Plan](#s8)
- [§9 Open Questions & Gaps](#s9)

# §1 — Competition Structure: What the Model Must Actually Do

Before any modeling hypothesis can be tested, the prediction contract must be understood precisely. Misunderstanding the supervision unit vs. the scoring unit is the most common source of wasted effort in BirdCLEF-style competitions.

## 1.1 — Prediction Contract (Train vs. Score)

| Axis | Training Side | Scoring Side |
| --- | --- | --- |
| Unit | Whole clip (3–120 s) | 5-second window |
| Label type | Clip-level primary + secondary | Per-window presence / absence |
| Label count | Mostly 1; 12.3% have secondary labels | 3–5 species per window (EDA §F) |
| Supervision | 35K clips + 1,478 expert soundscape windows | Hidden test soundscapes |
| Geography | Global (lat −55 to +70) | Pantanal, Brazil only |
| Recorder | Close-mic / phone / field recorder | Passive ARU deployment |

**Core tension:** The model is trained on close-mic clips with whole-clip labels and must generalize to passive soundscape recorders predicting at 5-second resolution. This is a *three-way mismatch*: recording device, label granularity, and geographic domain. Every design decision must close at least one of these gaps.

## 1.2 — Metric Implications

- **Macro ROC-AUC** — each of 234 classes gets equal weight regardless of frequency. A species appearing in 2 test windows matters as much as one appearing in 2,000.
- Metric is **threshold-free** — calibration of absolute probability values does not matter. Relative ranking within each class is what counts.
- **Classes absent in test are skipped** — but you cannot know which ones in advance. Do not deprioritize rare species.
- Improving common-species AUC by sacrificing rare-species AUC is net-neutral at best and net-negative under macro averaging.
- This metric strongly rewards models that *discriminate* within rare classes, not just models that avoid false positives overall.

## 1.3 — CPU Inference Constraint (90-min, no GPU)

- Hard limit on model size. EfficientNet-B4 and below are feasible; B6+ and large ViTs will time out.
- Mel spectrogram extraction must be batched — per-window librosa calls add ~2× overhead.
- ONNX export or `torch.compile` will likely be necessary for competitive inference speed.
- Top BC2024/2025 solutions used EfficientNet-B0 to B2 or small ConvNext variants with ONNX export.
- Plan model architecture around inference budget first, then optimize within that envelope.

# §2 — Ranked Modeling Hypotheses (Top 12)

Each hypothesis is grounded in a specific EDA finding. Priority is assigned by **expected AUC impact × implementation feasibility**. Every hypothesis includes: the claim, the EDA evidence, the expected impact, and the test approach.

## Tier 1 — Critical (must test first)

These four hypotheses address the dominant risks identified in EDA. None of the other hypotheses should be tested before these are validated on OOF CV.

| # | Hypothesis | EDA Evidence | Expected Impact | How to Test |
| --- | --- | --- | --- | --- |
| H1 | **Background mixing with unlabeled Pantanal soundscapes closes the clip→soundscape domain gap more than any other single intervention.** | Section G5, F1: clip-to-soundscape domain gap is the primary risk. 10,592 unlabeled soundscapes provide real Pantanal noise backgrounds at no label cost. | Large | Baseline OOF AUC vs. +background mixing. Expected: +0.04–0.08 AUC. Monitor per-taxon AUC change. |
| H2 | **Waveform mixup (λ~Beta(0.4,0.4)) with soft multi-label targets teaches the model co-presence of 3–5 species, matching the test window structure.** | Section F2, C: 87% of expert-labeled windows have 2+ co-active species. Training clips are mostly single-label. This gap hurts multi-label discrimination. | High | Baseline OOF AUC vs. +waveform mixup. Expected: +0.02–0.05. Check that rare-class AUC does not drop. |
| H3 | **Energy gating — discarding silent 5s windows as unlabeled — reduces label noise more than any augmentation.** | Section E3, F3: 14% mean silence ratio; some clips 90% silent. Assigning clip-level labels to silent windows creates systematic false positives. | High | Train with and without energy gating. Compare OOF precision on silent-window positives. Threshold sweep: −40, −45, −50 dBFS. |
| H4 | **Site-stratified CV avoids inflated OOF AUC caused by site-level acoustic leakage from recorder fingerprints and location clustering.** | Section G1, G2: top-5 authors = 40%+ of clips; site-level recorder fingerprints are learnable. Random CV splits leak site characteristics. | High | Compare random 5-fold OOF vs. site-stratified 5-fold OOF. Gap > 0.05 = leakage confirmed. Use site-stratified as the primary metric. |

## Tier 2 — High Priority (test in first 2 weeks after baseline)

| # | Hypothesis | EDA Evidence | Expected Impact | How to Test |
| --- | --- | --- | --- | --- |
| H5 | **BirdCLEF 2025 Pantanal clips and soundscapes are the highest-value external data — adding them directly expands labeled Pantanal coverage.** | Section H: BC2025 covers the same Pantanal region, same ARU recorder type, same multi-taxon scope. Highest domain match of any prior competition. | Med–High | Add BC2025 data (after URL dedup). Measure OOF AUC change. Flag: check if BC2025 soundscape sites overlap with BC2026 test sites. |
| H6 | **Backbone pre-training on all BirdCLEF clip audio (2021–2025), then fine-tuning on 2026+Pantanal data, gives better representations for rare species than training from scratch.** | Section H: prior competition clips share XC/iNat sources. Species-level acoustic features transfer across geographies even if domain differs. | Med–High | Compare scratch training vs. pre-trained backbone. Focus comparison on rare-class AUC (< 10 clips). Expected gain concentrated in under-represented species. |
| H7 | **Per-clip peak normalization before mel extraction removes the 600× loudness variance that confounds spectral features with recording level.** | Section E3: RMS distribution spans 600× dynamic range across recordings. Without normalization, the model partially learns loudness as a species feature. | Medium | Train with and without normalization. Mostly affects convergence speed and feature quality rather than final AUC. Low-risk, do by default. |
| H8 | **A multi-taxon head architecture (shared backbone, separate dense heads per taxon) outperforms a flat 234-class head by allocating capacity to each taxon's frequency range.** | Section A §3: frogs are 200–4000 Hz, insects 4000–16000 Hz, birds 1000–8000 Hz. A flat head must learn all three spectral regimes simultaneously with shared capacity. | Medium | Flat head baseline vs. 5-head multi-taxon model. Report per-taxon AUC separately. Expected: +0.03–0.06 on Amphibia/Insecta. |

## Tier 3 — Medium Priority (test after baseline is stable)

| # | Hypothesis | EDA Evidence | Expected Impact | How to Test |
| --- | --- | --- | --- | --- |
| H9 | **Semi-supervised learning on 10,592 unlabeled soundscapes provides additional Pantanal-domain signal for rare and zero-clip species.** | Section B: 10,592 soundscapes are unlabeled but come from the exact deployment domain. Even without labels, they carry domain-specific acoustic patterns. | Medium | Pseudo-label with baseline model (threshold > 0.5). Retrain on pseudo-labeled + real labeled. *Only start this after H1–H4 are validated.* |
| H10 | **Time-of-day conditioning (sin/cos hour encoding) improves prediction accuracy because species activity is time-dependent.** | Section E4: dawn chorus peak in soundscapes. Test row\_id encodes datetime. Section D: seasonal patterns visible in soundscape metadata. | Low–Med | Add `sin(2π·hour/24)`, `cos(2π·hour/24)` as input features. Free feature, low implementation cost. Expected: small gain on temporally-structured species. |
| H11 | **Two-stage detection for insect sonotypes (detect parent taxon → sub-classify sonotype) recovers near-zero AUC on all 25 insect sonotype classes.** | Section B §5: 25 insect sonotypes have zero training clips. Their only supervision is the 66 labeled soundscape windows. Flat 234-class model produces near-random scores for these classes. | Medium | First: visualize sonotype spectrograms — are they visually distinguishable? If yes: train parent-taxon detector, then sonotype sub-classifier. If no: accept low AUC; not worth engineering cost early. |
| H12 | **Post-processing by Pantanal species prevalence reduces systematic over-prediction of globally-common XC species that are locally rare or absent in Pantanal.** | Section G5: train clip geographic distribution is global; test is Pantanal. Model will over-score species abundant globally but Pantanal-absent. | Low–Med | Estimate species prevalence from labeled soundscape windows. Multiply raw scores by prior. Under ROC-AUC this only helps by re-ranking across windows; measure OOF before applying. |

**Priority order:** H1 → H2 → H3 → H4 → H7 → H5 → H6 → H8 → H9 → H10 → H11 → H12  
*Do not test H9–H12 until H1–H4 are verified on OOF CV. Stacking unvalidated interventions produces uninterpretable results.*

# §3 — Data Pipeline Architecture

## 3.1 — Input Sources (Ranked by In-Domain Value)

| Source | Size | In-Domain? | Labels? | Recommended Use | Risk |
| --- | --- | --- | --- | --- | --- |
| Train soundscapes (BC2026) | 10,658 ogg | Pantanal ARU | 66 labeled (1,478 windows) | Primary training + all as backgrounds | Site S22 dominates labeled subset |
| Train clips (BC2026) | 35,549 ogg | Partial — global | Clip-level | Training with windowing + energy gating | Global geography ≠ Pantanal domain |
| BC2025 clips + soundscapes | ~50K est. | High — same region | Clip + soundscape window | Add directly after URL dedup | Possible test-set site overlap |
| Prior BC clips 2021–2024 | Large | Low — diff regions | Clip-level | Backbone pre-training only | Hurts calibration if mixed directly |
| Additional Neotropical XC | Variable | Medium — regional | Clip-level | Rare/zero-clip species only | May introduce non-Pantanal recordings |
| Non-Pantanal soundscapes | Various | NO — wrong domain | Various | Do NOT use as domain training data | Confuses domain adaptation |

## 3.2 — Clip Processing Pipeline

| Step | Action | Why |
| --- | --- | --- |
| 1. Load & resample | Load OGG → resample to 32 kHz mono | All BC2026 clips already 32 kHz. Minimal overhead. |
| 2. Peak normalization | Normalize peak to 0.9 before any windowing | Removes 600× loudness variance. Apply universally. |
| 3. Window extraction | Slice into non-overlapping 5s windows. Short clips (<5s): pad with zeros and treat as single window. | Matches the 5s scoring unit exactly. |
| 4. Energy gating | Compute RMS of each window. If RMS < −45 dBFS → mark as `background_only` (no species label). Do NOT assign primary\_label to silent windows. | Removes most systematic false-positive labels from silence. |
| 5. Background mixing | Sample random 5s window from unlabeled Pantanal soundscape. Mix at SNR ~ Uniform(−5, +10) dB. Background label = all-zeros. | Highest-ROI step. Directly bridges clip→soundscape domain gap. |
| 6. Waveform mixup | With prob 0.5: sample second clip. Mix x = λ·x1 + (1-λ)·x2, λ ~ Beta(0.4, 0.4). Blend labels: y = λ·y1 + (1-λ)·y2 (soft labels, multi-label BCE). | Trains multi-label co-presence matching test windows. |
| 7. Mel spectrogram | n\_mels=128, hop\_length=320 (10ms), n\_fft=1024, fmin=50, fmax=14000 Hz. Log-amplitude. Result: 128×500 per 5s window. | Covers frog (200–4k), bird (1–8k), insect (4–16k) ranges. |
| 8. SpecAugment (P3 only) | Freq masking 1–2 bands (≤10% mel bins), time masking 1 block (≤8% frames). *Only after P1–P2 are validated.* | Breaks recorder-fingerprint memorization (author EQ patterns). |

## 3.3 — Label Construction

- **Clip windows:** assign primary\_label + secondary\_labels to all non-silent windows. Accept clip-level label noise; energy gating handles the worst cases.
- **Soundscape labeled windows:** use exact per-window expert labels as gold standard. Oversample these relative to clips in the training loop.
- **Unlabeled soundscape windows (as background mix source):** label as all-zeros. Do NOT infer labels from clip overlap.
- **Rare-species oversampling:** weight each labeled soundscape window by `1/sqrt(rarest_class_freq)` to ensure all 234 classes appear in every epoch.
- **Soft labels for mixup:** BCE loss must support continuous [0,1] targets. Confirm framework compatibility before training.

# §4 — Validation Strategy

A realistic validation scheme is the single most important infrastructure decision. An optimistic CV scheme means every experiment looks better than it actually is. BirdCLEF competitions are notorious for site-level acoustic leakage inflating OOF scores.

## 4.1 — Why Random K-Fold Is Wrong Here

- **Author leakage:** Top-5 authors = 40%+ of clips. Random splits put clips from the same recording session in both train and validation. The model memorizes recorder fingerprints.
- **Location leakage:** Nearby recordings share forest background noise. GPS-clustered clips in both train/val look like generalization but isn't.
- **Site leakage in soundscapes:** Site S22 = 64% of labeled windows. A random split across S22 windows leaks site acoustics — not real generalization.
- **Temporal leakage:** Same recorder makes multiple trips to the same site in the same year. Near-duplicate recording clusters (EDA §G3) leak across random splits.

## 4.2 — Recommended Validation Scheme

| Scheme | How to Split | When to Use |
| --- | --- | --- |
| Site-stratified 5-fold (clips) | Group clips by GPS cluster (K-means on lat/lon → ~20 clusters). Each fold holds out complete geographic clusters. | Primary CV for clip-based experiments |
| Author-stratified 5-fold (clips) | Group clips by author/recorder ID. Each fold holds out complete author groups. | Cross-check: should give similar result to site-stratified |
| Soundscape site holdout | Reserve 3 non-S22 soundscape sites as held-out evaluation. Train on remaining 20 sites. S22 should not be the holdout — it is too dominant. | Primary CV for soundscape-based experiments |
| Taxonomic-stratified sanity check | Ensure each fold contains examples from all 5 taxon classes. Verify Amphibia and Insecta are not concentrated in one fold. | Sanity check on all fold configurations |

**Key rule:** Always report OOF AUC separately per taxonomic class (Aves, Amphibia, Insecta, Mammalia, Reptilia). A high overall macro AUC driven by bird classes while Amphibia/Insecta AUC is near 0.5 is a false positive. The leaderboard will expose this.

# §5 — Augmentation Plan (Evidence-Based)

Every augmentation below is tied to a specific EDA finding. Generic augmentations not supported by evidence are excluded. The list is ordered by implementation priority.

## 5.1 — Recommended Augmentations (Ordered by Priority)

| Priority | Augmentation | Problem It Addresses | EDA Source | Caveat |
| --- | --- | --- | --- | --- |
| P1 Must Have | **Energy gating** (preprocessing) | 14% mean silence; some clips 90% silent. Clip-level labels assigned to silent windows → systematic false positives. | §E3, §F3 | Threshold tuning: test −45, −40, −50 dBFS on OOF before fixing. |
| P1 Must Have | **Peak normalization** (preprocessing) | 600× loudness variance. Without normalization, model learns recording level as a species cue. | §E3 | Apply per-clip before windowing. No hyperparameters. |
| P1 Must Have | **Background mixing** (soundscape overlay) | Clip→soundscape domain gap is the primary train/test mismatch. 10,592 unlabeled Pantanal soundscapes provide real noise backgrounds. | §G5, §F1 | Pre-screen soundscapes for extreme artifacts (rain, wind) first. SNR range: −5 to +10 dB. |
| P1 Must Have | **Waveform mixup** (λ~Beta(0.4,0.4)) | Training is mostly single-label; test windows have 3–5 co-active species. | §F2, §C | Requires soft-label BCE loss. Confirm framework supports float targets. |
| P2 High Value | **Rare-species oversampling** | 28 zero-clip species; many with <5 clips. Standard sampling skips them for entire epochs. | §B, §C | Weight = 1/sqrt(class\_freq). Watch for common-class degradation. |
| P3 Medium | **SpecAugment** (conservative) | Top-5 author recorder fingerprints = 40%+ of data. Frequency masking breaks EQ-pattern memorization. | §G2 | Max 10% freq bins, max 8% time frames. Validate on OOF before enabling permanently. |
| P3 Medium | **Soft saturation** (tanh clipping sim) | ~8% of clips have clipping artifacts. Model should be robust to saturation distortion. | §E3 | Apply to 20–30% of clips randomly only. |
| P3 Low | **Time-of-day feature** (not augmentation) | Dawn chorus peak; species activity is time-conditional. Test row\_id encodes datetime. | §E4 | Encode as sin(2π·hour/24), cos(2π·hour/24). Free, low-risk feature. |

## 5.2 — Augmentations Explicitly Excluded

| Excluded Augmentation | Reason |
| --- | --- |
| Large pitch shift (±4+ semitones) | Frog and insect calls are pitch-specific for species ID. Large shifts change species identity in non-bird taxa. Use ≤±2 semitones for birds only. |
| Time stretching >±20% | Temporal rhythm of calls (rate, duty cycle) is species-diagnostic, especially for insects. Heavy stretching distorts these cues. |
| Additive Gaussian / white noise | Pantanal background noise is structured (wind, rain, water, co-species) — not white. Use real soundscape backgrounds instead. |
| Spectrogram CutMix | Pasting spectrogram patches creates frequency-localized chimeras that are acoustically impossible. Waveform mixup is physically meaningful; spectrogram CutMix is not. |
| Random crop to <2.5s | Creates training examples shorter than half a prediction window. Label assignment becomes meaningless for sub-2.5s fragments. |
| Frequency shifting (full spectrum) | Harmonic structures are key species discriminators. Shifting destroys harmonic ratios. |

# §6 — Prior Data Decision

This section answers: which prior BirdCLEF competition data should be used, in what capacity, and what should be avoided. The answer is not binary — different datasets serve different purposes.

| Dataset | Geography | Taxon Overlap | Domain Match | Recommended Use | Risk |
| --- | --- | --- | --- | --- | --- |
| BC2025 clips + soundscapes | Pantanal, Brazil | High | Excellent | Add directly to training after URL dedup | Possible test-site overlap with BC2026 — check dates |
| BC2024 clips | Multiple regions | Moderate | Poor | Backbone pre-training only | Hurts calibration if mixed directly with BC2026 |
| BC2023 / BC2022 clips | Global / India / Africa | Low | Very poor | Backbone pre-training only if needed | Geographic and species mismatch |
| BC2021 clips | Cornell / North America | Very low | Very poor | Skip or very broad pre-training only | Unlikely to help; may hurt calibration |
| Non-Pantanal soundscapes (any year) | Various non-Pantanal | Low | Wrong domain | Do NOT use as domain training data | Confuses soundscape domain adaptation |

**Decision:** Use BC2025 as in-domain training data (high priority). Use BC2021–2024 clips for backbone pre-training only. Do NOT use non-Pantanal soundscapes as soundscape domain training data.  
  
**Mandatory before combining any prior data:** deduplicate by Xeno-Canto URL. Prior competitions draw from the same XC pool. Duplicate clips in both train and validation will inflate OOF AUC.

# §7 — Risk Register

| Risk | Severity | EDA Source | Mitigation |
| --- | --- | --- | --- |
| Clip→soundscape domain gap | CRITICAL | §G5, §F1 | H1: Background mixing. H4: Site-stratified CV. Monitor per-site AUC during training. |
| Label noise from clip windowing | HIGH | §E3, §F3 | H3: Energy gating. Discard silent windows. Accept residual noise for non-silent windows. |
| 28 zero-clip species (25 insect sonotypes + 3 frogs) | HIGH | §B §5, §C | Oversample labeled soundscape windows. Accept near-zero AUC on insect sonotypes initially. H11: Two-stage detection in Phase 4. |
| Site S22 dominance in labeled soundscapes (64% of windows) | HIGH | §G4 | Hold out non-S22 sites for evaluation. Do not train final model exclusively on S22-derived labels. |
| Author/recorder fingerprint memorization | MEDIUM | §G2 | H4: Author-stratified CV. P3 SpecAugment. Monitor gap between author-stratified and random CV OOF. |
| Seasonal mismatch (test ≈ Feb 2025, start of dry season) | MEDIUM | §D, §E4 | H10: Add time-of-day and month features. Weight training examples less from extreme wet-season months. |
| Rating-zero clips (36% of training clips, quality unknown) | MEDIUM | §A §4A | Manually audit 50 random rating-0 clips. If systematic noise: quality-stratified sampling. If clean: keep all clips. |
| XC duplicate clips across BC2025–BC2026 inflating OOF | MEDIUM | §H | Deduplicate by XC URL before combining datasets. |
| CPU inference timeout on complex models | HIGH | §A §12 | Design around ≤ EfficientNet-B2 or ConvNext-Tiny. Profile inference early. ONNX export for speed. |
| Macro AUC masking poor non-bird taxon performance | MEDIUM | §A §3 | Track OOF AUC per taxon separately. Target: all taxon classes ≥ 0.70. Non-bird AUC below 0.60 requires targeted intervention. |

# §8 — Phased Experiment Plan

Experiments are sequenced to maximize information gain per GPU-hour. Each phase must produce a validated OOF AUC before moving to the next. Do not stack unvalidated interventions.

| Phase | Target Dates | Goals | Exit Criterion |
| --- | --- | --- | --- |
| **Phase 0** Infrastructure | Weeks 1–2 (by Apr 26) | - Clip windowing + energy gating pipeline - Site-stratified 5-fold splits - Batched mel spectrogram extraction - Baseline EfficientNet-B0, flat 234-class - OOF AUC tracking per taxon | Reproducible baseline OOF AUC > 0.65 Inference time < 60 min on CPU |
| **Phase 1** Core Augmentation | Weeks 2–3 (by May 3) | - Peak normalization (H7) - Background mixing P1 (H1) - Waveform mixup P1 (H2) - Rare-species oversampling (H3) | +0.03 OOF AUC vs Phase 0 Non-bird taxon AUC > 0.60 |
| **Phase 2** Data Expansion | Weeks 3–4 (by May 10) | - Add BC2025 data + dedup (H5) - Add Neotropical XC for zero-clip species - SpecAugment P3 validation | +0.02 OOF AUC vs Phase 1 |
| **Phase 3** Architecture | Weeks 4–5 (by May 17) | - Multi-taxon head experiment (H8) - Backbone pre-training on BC2021–2024 (H6) - Time-of-day conditioning (H10) | +0.02 OOF AUC vs Phase 2 Decide: multi-taxon head yes/no |
| **Phase 4** Semi-supervised | Weeks 5–6 (by May 24) | - Pseudo-labeling on unlabeled soundscapes (H9) - Two-stage insect sonotype detection (H11) - Model ensemble (2–3 seeds minimum) | +0.01–0.02 OOF AUC Insect sonotype AUC > 0.55 |
| **Phase 5** Final Polish | Weeks 6–8 (by Jun 3) | - Inference optimization (ONNX / torch.compile) - Prevalence post-processing experiment (H12) - Final ensemble submission - Submission slot management | Submission time < 85 min Final ensemble > Phase 4 OOF |

**Checkpoint rule:** If a phase does not meet its exit criterion after 2 attempts, diagnose root cause before advancing. Do not move to the next phase on faith — each phase builds on validated results from the previous one.

# §9 — Open Questions & Gaps

These cannot be resolved from EDA alone and require either direct experimentation or information that becomes available later in the competition.

| Question | Why It Matters | How to Resolve |
| --- | --- | --- |
| Which species actually appear in the test soundscapes? | Classes absent in test are skipped from macro AUC. Knowing test class coverage would change rare-class priority. | Cannot know until final scoring. Treat all 234 classes as potentially present. |
| How many test soundscapes are there? | Determines inference time budget per soundscape and maximum architecture size. | Check sample\_submission row count. Estimate: ~50–200 soundscapes based on prior year competition sizes. |
| What quality are the rating-zero clips? | 36% of training clips are unrated. If many are low quality, they should be down-weighted or excluded. | Listen to 50 random rating-0 clips manually. Run SNR estimation script on all rating-0 clips. |
| Are BC2025 soundscapes from the same physical sites as BC2026 test soundscapes? | If yes, site-level acoustic leakage between BC2025 and BC2026 test is possible. | Compare site metadata. If site IDs overlap: exclude overlapping BC2025 soundscapes from domain training. |
| Are insect sonotypes acoustically distinguishable in practice? | If sonotype boundaries are visually/acoustically ambiguous even to experts, the 25-class insect block may be inherently unlearnable. | Spectrogram visualization of sonotype examples. If visually indistinguishable: accept low AUC; skip H11 engineering cost. |
| What is the test soundscape recording date distribution? | Feb 2025 is all we know from sample submission. If test spans multiple months, seasonal conditioning matters more. | Infer from final submission row\_ids as more test info becomes available. |
| Does BC2026 evaluation skip truly zero-positive classes or uses all 234? | The skip policy determines effective N for macro average. If most 234 classes appear, rare classes matter more. | Check competition FAQ / discussion forum. Assume all 234 classes are possible. |

### Three Headline Findings From This Analysis

- **The domain gap (clip → soundscape) is the #1 problem.**
  Background mixing with unlabeled Pantanal soundscapes (H1) is the highest-ROI single
  intervention. Do this before anything else.
- **Validation scheme determines everything.**
  Site-stratified 5-fold CV is mandatory (H4).
  Random splits produce inflated OOF that doesn't match leaderboard position.
- **Non-bird taxa are the leaderboard differentiator.**
  Frogs, insects, and mammals are 30% of targets.
  Track their AUC separately. Most competitors will under-optimize for them.

BirdCLEF 2026 — Modeling Hypotheses & Strategy Report  | 
Generated 2026-04-12  | 
Synthesized from EDA Sections A–I  | 
reports/modeling\_report.html
