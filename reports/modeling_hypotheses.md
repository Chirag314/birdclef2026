# BirdCLEF 2026 — Ranked Modeling Ideas & Hypotheses

**Synthesized from EDA Sections A–I**  
**Date:** 2026-04-14  
**Phase:** End of EDA / Start of Modeling Strategy

---

## Three Headline Findings

1. **Domain gap (clip → soundscape) is the #1 problem.** Background mixing with the 10,592 unlabeled Pantanal soundscapes is the highest-ROI single intervention.
2. **Validation scheme determines everything.** Random k-fold is wrong. Site-stratified CV is mandatory — random splits produce inflated OOF that doesn't reflect leaderboard position.
3. **Non-bird taxa are the leaderboard differentiator.** Frogs, insects, and mammals are 30% of targets. Most competitors will under-optimize for them. Per-taxon AUC tracking is essential.

---

## Ranked Hypotheses

### Tier 1 — Critical (must test before anything else)

| # | Hypothesis | EDA Grounding | Expected AUC Impact | Test |
|---|-----------|--------------|-------------------|------|
| H1 | **Background mixing** with unlabeled Pantanal soundscapes closes the clip→soundscape domain gap more than any other single intervention | §G5, §F1: severe domain shift confirmed; 10,592 unlabeled soundscapes provide real noise at no label cost | **Large (+0.04–0.08)** | Baseline OOF vs +background mixing |
| H2 | **Waveform mixup** (λ~Beta(0.4,0.4)) with soft multi-label targets trains co-presence of 3–5 species, matching test window structure | §F2, §C: 87% of expert-labeled windows have 2+ co-active species; training clips are mostly single-label | **High (+0.02–0.05)** | Baseline OOF vs +mixup; check rare-class AUC doesn't drop |
| H3 | **Energy gating** (discard silent 5s windows) reduces label noise more than any augmentation | §E3, §F3: 14% mean silence ratio; some clips 90% silent; clip-level labels on silent windows = systematic false positives | **High** | OOF precision on silent-window positives; threshold sweep: −40/−45/−50 dBFS |
| H4 | **Site-stratified 5-fold CV** avoids inflated OOF from site-level acoustic leakage | §G1, §G2: top-5 authors = 40%+ of clips; recorder fingerprints are learnable; random splits leak site characteristics | **High (infrastructure)** | Compare random 5-fold OOF vs site-stratified; gap >0.05 = leakage confirmed |

### Tier 2 — High Priority (first 2 weeks after baseline)

| # | Hypothesis | EDA Grounding | Expected AUC Impact | Test |
|---|-----------|--------------|-------------------|------|
| H5 | **BirdCLEF 2025 Pantanal data** is highest-value external data — same region, same ARU type, same multi-taxon scope | §H: BC2025 Pantanal match is highest of any prior year | **Med–High** | Add BC2025 after URL dedup; measure OOF AUC change; check site overlap with BC2026 test |
| H6 | **Backbone pre-training** on all BC clips 2021–2025, fine-tuned on 2026+Pantanal, improves rare-class representations | §H: XC/iNat clips share acoustic feature space; species features transfer even across geographies | **Med–High** (concentrated on rare classes) | Scratch vs pre-trained backbone; focus on species with <10 clips |
| H7 | **Per-clip peak normalization** removes the 600× loudness variance that confounds spectral features with recording level | §E3: RMS spans 600× dynamic range; model learns loudness as a species cue without normalization | **Medium** (correctness fix) | Train with/without; mostly affects convergence; low-risk, apply by default |
| H8 | **Multi-taxon head architecture** (shared backbone + separate head per taxon) outperforms flat 234-class head | §A §3: frogs 200–4000 Hz, insects 4000–16000 Hz, birds 1–8 kHz; flat head shares capacity across 3 incompatible spectral regimes | **Medium (+0.03–0.06 on non-birds)** | Flat head vs 5-head; report per-taxon AUC separately |

### Tier 3 — Medium Priority (after baseline is stable)

| # | Hypothesis | EDA Grounding | Expected AUC Impact | Test |
|---|-----------|--------------|-------------------|------|
| H9 | **Semi-supervised pseudo-labeling** on 10,592 unlabeled soundscapes adds Pantanal-domain signal for rare/zero-clip species | §B: unlabeled soundscapes come from exact deployment domain; even without labels they carry domain-specific patterns | **Medium** | Pseudo-label at threshold >0.5, retrain. **Only after H1–H4 are validated.** |
| H10 | **Time-of-day conditioning** (sin/cos hour encoding) improves accuracy because species activity is time-dependent | §E4: dawn chorus peak visible in soundscapes; test row_id encodes datetime | **Low–Med** | Add sin(2π·hour/24), cos(2π·hour/24) as input features; low cost |
| H11 | **Two-stage detection for insect sonotypes** (detect parent taxon → sub-classify sonotype) recovers near-zero AUC on 25 zero-clip insect classes | §B §5: 25 insect sonotypes have zero training clips; only 66 labeled soundscape windows for supervision | **Medium** (conditional on visual separability) | First: visualize sonotype spectrograms. If distinguishable → two-stage. If not → accept low AUC. |
| H12 | **Pantanal species prevalence post-processing** reduces over-prediction of globally-common species that are locally rare or absent | §G5: train clips are global; model will over-score species abundant globally but Pantanal-absent | **Low–Med** | Estimate prevalence from labeled soundscape windows; multiply raw scores by prior; validate on OOF first |

**Execution order:** H1 → H2 → H3 → H4 → H7 → H5 → H6 → H8 → H9 → H10 → H11 → H12

---

## Augmentation Plan (Evidence-Based)

| Priority | Augmentation | Problem It Addresses | EDA Source | Caveat |
|---------|-------------|---------------------|-----------|--------|
| P1 Must Have | Energy gating (preprocessing) | 14% mean silence; clip labels on silent windows → false positives | §E3, §F3 | Test −40/−45/−50 dBFS threshold |
| P1 Must Have | Peak normalization (preprocessing) | 600× loudness variance; model learns recording level as species cue | §E3 | Per-clip before windowing. No hyperparams. |
| P1 Must Have | Background mixing (soundscape overlay) | Clip→soundscape domain gap; 10,592 unlabeled Pantanal backgrounds | §G5, §F1 | Pre-screen for extreme artifacts (rain/wind). SNR: −5 to +10 dB |
| P1 Must Have | Waveform mixup λ~Beta(0.4,0.4) | Training single-label; test windows have 3–5 co-active species | §F2, §C | Requires soft-label BCE. Confirm framework supports float targets. |
| P2 High | Rare-species oversampling | 28 zero-clip species; many with <5 clips; skipped for full epochs | §B, §C | Weight = 1/sqrt(class_freq). Watch common-class degradation. |
| P3 Medium | SpecAugment (conservative) | Top-5 authors = 40%+ of data; frequency masking breaks EQ-pattern memorization | §G2 | Max 10% freq bins, 8% time frames. Validate on OOF. |
| P3 Medium | Soft saturation simulation | ~8% of clips have clipping artifacts | §E3 | Apply to 20–30% of clips randomly. |

### Augmentations Explicitly Excluded

| Excluded | Reason |
|---------|--------|
| Pitch shift >±2 semitones | Frog/insect calls are pitch-specific; large shifts change species identity |
| Time stretching >±20% | Temporal rhythm (rate, duty cycle) is species-diagnostic especially for insects |
| Additive Gaussian/white noise | Pantanal background is structured (wind, rain, water, co-species) — not white |
| Spectrogram CutMix | Creates acoustically impossible frequency-localized chimeras; waveform mixup is physically meaningful |
| Random crop to <2.5s | Label assignment becomes meaningless below half a prediction window |
| Full-spectrum frequency shifting | Destroys harmonic ratios that are key species discriminators |

---

## Prior Data Decision

| Dataset | Domain Match | Taxon Overlap | Recommended Use |
|---------|-------------|--------------|----------------|
| BC2025 clips + soundscapes | **Excellent** (Pantanal, same ARU) | High | Add directly after XC URL dedup. Check for test-site overlap. |
| BC2024 clips | Poor (multi-region) | Moderate | Backbone pre-training only |
| BC2023 / BC2022 clips | Very poor (Global/India/Africa) | Low | Backbone pre-training only if needed |
| BC2021 clips | Very poor (North America) | Very low | Skip or very broad pre-training |
| Non-Pantanal soundscapes (any year) | Wrong domain | Low | **Do NOT use** as domain training data |

**Mandatory before combining:** Deduplicate by Xeno-Canto URL. Prior competitions draw from the same XC pool. Duplicates in train+validation inflate OOF AUC.

---

## Validation Strategy

**Why random k-fold is wrong:**
- Top-5 authors = 40%+ of clips → recorder fingerprint leakage
- GPS-clustered recordings share forest background → location leakage
- Site S22 = 64% of labeled soundscape windows → site leakage
- Near-duplicate recording clusters exist (same recorder, same site, same season)

**Recommended schemes:**

| Scheme | How to Split | Use For |
|--------|-------------|---------|
| Site-stratified 5-fold | K-means on lat/lon → ~20 clusters; hold out complete geographic clusters | Primary CV for clip experiments |
| Author-stratified 5-fold | Group clips by author/recorder ID; hold out complete author groups | Cross-check (should match site-stratified) |
| Soundscape site holdout | Reserve 3 non-S22 sites; train on remaining 20 | Primary CV for soundscape experiments |

**Key rule:** Always report OOF AUC per taxon (Aves / Amphibia / Insecta / Mammalia / Reptilia). High macro AUC driven by birds while Amphibia/Insecta AUC is ~0.5 is a false positive that the leaderboard will expose.

---

## Phased Experiment Schedule

| Phase | Dates | Goals | Exit Criterion |
|-------|-------|-------|---------------|
| P0 Infrastructure | Apr 14–26 | Clip windowing + energy gating; site-stratified folds; batched mel extraction; baseline EfficientNet-B0; OOF tracking per taxon | OOF AUC > 0.65; inference < 60 min CPU |
| P1 Core Augmentation | Apr 26 – May 3 | Peak norm (H7), background mixing (H1), waveform mixup (H2), rare-species oversampling | +0.03 vs P0; non-bird taxon AUC > 0.60 |
| P2 Data Expansion | May 3–10 | Add BC2025 data + dedup (H5); Neotropical XC for zero-clip species; SpecAugment validation | +0.02 vs P1 |
| P3 Architecture | May 10–17 | Multi-taxon head (H8); backbone pre-training (H6); time-of-day conditioning (H10) | +0.02 vs P2; multi-taxon head decision |
| P4 Semi-supervised | May 17–24 | Pseudo-labeling on unlabeled soundscapes (H9); two-stage insect sonotype (H11); 2–3 seed ensemble | +0.01–0.02 vs P3; insect sonotype AUC > 0.55 |
| P5 Final Polish | May 24 – Jun 3 | ONNX/torch.compile inference; prevalence post-processing (H12); final ensemble | Submission < 85 min; final > P4 OOF |

**Checkpoint rule:** If a phase doesn't meet its exit criterion after 2 attempts, diagnose before advancing. Never stack unvalidated interventions.

---

## Top Risk Register

| Risk | Severity | Mitigation |
|------|---------|-----------|
| Clip→soundscape domain gap | **CRITICAL** | H1: background mixing; H4: site-stratified CV |
| Label noise from clip windowing | HIGH | H3: energy gating |
| 28 zero-clip species (25 insect sonotypes) | HIGH | Oversample soundscape windows; H11 two-stage if feasible |
| Site S22 dominance (64% of labeled soundscapes) | HIGH | Hold out non-S22 sites for evaluation |
| Author/recorder fingerprint memorization | MEDIUM | Author-stratified CV; SpecAugment |
| CPU inference timeout | HIGH | Design around ≤ EfficientNet-B2; profile early; ONNX export |
| Macro AUC masking poor non-bird performance | MEDIUM | Track per-taxon AUC throughout; target all ≥ 0.70 |
| XC duplicates inflating OOF when adding prior data | MEDIUM | Dedup by XC URL before any data combination |

---

## Open Questions

| Question | Why It Matters |
|---------|---------------|
| Which species appear in test soundscapes? | Classes absent are skipped from macro AUC — but cannot know in advance. Treat all 234 as potentially present. |
| How many test soundscapes? | Determines inference time budget per soundscape. Estimate: 50–200 based on prior years. Check sample_submission. |
| Quality of rating-0 clips? | 36% of training clips are unrated. Manually audit 50 samples + run SNR estimation. |
| BC2025 soundscape sites overlap BC2026 test sites? | If yes, site leakage between BC2025 and BC2026 is possible. Compare site metadata before combining. |
| Are insect sonotypes acoustically distinguishable? | If not, H11 engineering cost is not worth it. Visualize sonotype spectrograms first. |
| Test soundscape recording date distribution? | If multi-month, seasonal conditioning (H10) matters more. Infer from row_ids as info becomes available. |

---

*Full modeling report (HTML): `reports/modeling_report.html`*  
*Deadline: June 3, 2026*
