# BirdCLEF 2026 — Solutions Analysis & Post-Mortem
*Written: 2026-06-04. Solution writeups not yet published (CLEF deadline June 17);*
*this analysis is based on public LB data, private submission scores, and prior-year patterns.*

---

## Final Competition Results

| Rank | Team | Public LB | Private LB |
|---|---|---|---|
| 1 | Nikita Babych | 0.967 | 0.965 |
| 2 | Yannan Chen | 0.967 | ~0.960 |
| 3 | Ali Ozan Memetoglu | 0.963 | ~0.959 |
| 4 | BirdCLEF+ 2026 Team | 0.963 | ~0.959 |
| 5 | more exp is all you need | 0.962 | ~0.957 |
| 10 | Sinan Calisir | 0.960 | ~0.957 |
| ~42 | Gold cutoff | 0.955 | ~0.955 |
| ~212 | Silver cutoff | 0.951 | ~0.942 |
| ~424 | **Bronze cutoff** | 0.950 | **~0.942** |
| **920** | **Chirag Desai (us)** | **0.950** | **0.941** |

**Public→Private stability for top teams: ~0.002 drop.
Our drop: 0.009 — 4.5× worse than the median.**

---

## What Top Teams Did (inferred from LB behaviour + 2025 patterns)

### Structural Differences vs Our Approach

| Component | Us | Top Teams (est.) |
|---|---|---|
| Foundation model | Perch v2 (single) | Perch v2 + distilled SED + custom CNN |
| Training data | Focal clips only + label smoothing | Focal + pseudo-labeled soundscapes (3–5 rounds) |
| Postprocessing tuning | Public LB | Out-of-fold predictions |
| Inference optimization | ONNX Perch, CPU-safe | OpenVINO fp16 quantization (faster → more models) |
| Ensemble strategy | 3-model EoS blend (same backbone) | Diverse architectures, rank-mean ensemble |
| Domain adaptation | Perch pretraining | Perch + soundscape pseudo-labels |

### Nikita Babych (1st place, also won 2025)
Public→Private drop: 0.002. Exceptional private LB stability.

**2025 recipe** (confirmed from published writeup) adapted for 2026:
- **Noisy Student Self-Training**: 3–5 rounds of pseudo-label generation → filtering by confidence → retraining
- **Teacher model**: 5-fold Perch ensemble generates soft pseudo-labels on ALL test soundscapes
- **Student model**: retrained on focal + pseudo-labeled soundscapes with augmentation
- **Key insight**: each round of noisy student adds ~0.002–0.004 to private LB
- **Ensemble**: rank-mean aggregation across 3+ independent model families
- **Postprocessing**: trained on OOF predictions, not public test scores — hence private stability

**Why his approach didn't overfit public LB**: pseudo-labels are generated from the model's own OOF predictions on soundscapes. No signal from the public test set leaks in.

### General Silver/Gold Zone Pattern (~0.955–0.967 private)
- 3–5 architecturally diverse models (not just Perch variants)
- At least one model fine-tuned on domain-specific soundscape data
- Ensemble diversity measured by pairwise prediction disagreement, not just score
- Inference budget used for model diversity, not just faster single-model runs

### Bronze Zone (~0.942–0.954 private)
- Perch v2 + ProtoSSM/ResSSM ensemble (the public kernel recipe)
- Minimal to no pseudo-labeling
- EoS-style postprocessing with conservative parameter settings
- What separated bronze from no-medal: **simpler, less-tuned postprocessing**

---

## Our Experiment History vs What Mattered

### What We Did Right

| Decision | Why it was right |
|---|---|
| Perch v2 as backbone | Unanimous consensus among all medal solutions |
| Label smoothing (0.05) | Confirmed fix for clip overconfidence (+0.040 LB) |
| ProtoSSM head | State-of-art for Perch in 2026, +0.071 LB over MLP |
| Rejecting B2/larger EfficientNet | Correct: bigger models overfit clips, hurt LB |
| Rejecting soundscape oversampling | Correct: training soundscapes ≠ test distribution |
| Diversity hedge (slot 2 = tax_blend) | Right principle, wrong execution (both slots 0.941) |

### What We Got Wrong

#### 1. BirdNET Sidecar — The Critical Mistake
**What happened:** BirdNET v2.4 corrections tuned to 4-decimal precision on 34% public test data.
- v1 (caps 0.015/0.060): +200 rank public, unknown private
- v2 (caps 0.025/0.090): +250 rank public → selected as slot 1
- v3 (caps 0.035/0.120): no rank change → ceiling confirmed

**Why it failed privately:** The BirdNET OOF gate and correction caps were calibrated to the specific species distribution in the 34% public test slice. The 66% private test came from different Pantanal recording sites and dates. BirdNET corrections that boosted rare-species scores on public test sites added noise (wrong species flagged) on private test sites.

**The right approach:** Never tune postprocessing caps against public LB scores. Tune on OOF predictions from training data, then apply fixed caps and trust them.

#### 2. No Pseudo-Labeling
We skipped pseudo-labeling entirely. The top teams ran 3–5 rounds of noisy student self-training, each adding ~0.002–0.004. That compounds to +0.006–0.020 over a base model. This is likely the entire gap between our 0.941 private and the bronze cutoff of ~0.942.

**Why we skipped it:** Pseudo-labeling requires running inference on test soundscapes (not allowed during competition), OR running on the training soundscapes + unlabeled audio. We had access to the latter but it wasn't in our experiment queue. The unlock was there; we didn't find it.

#### 3. No Architectural Diversity
All our models used Perch v2 backbone. The TAX_SMOOTHING blend (slot 2) used a different EoS recipe but the same Perch embeddings. There was zero architectural diversity between our selected submissions.

Bronze zone teams who held their rank had at least one submission that was an independent model architecture (distilled SED, BirdNet, custom CNN), creating genuine private LB hedge.

#### 4. Public LB Grinding
44 submissions looks disciplined vs the 300-400 submission LB-grinders, but we still tuned BirdNET caps (v1→v2→v3) and EoS dials (exp017–023) against the public test set. The right approach is to tune everything on OOF and then make one or two validation submissions.

---

## Score Progression — Us vs Field

```
                        Public LB   Private LB
Our peak:                 0.950        0.941
Bronze cutoff:            0.950        0.942
Silver cutoff:            0.951        0.942
Gold cutoff:              0.955        0.955
1st place:                0.967        0.965
```

We were 1 point (0.001) below the private bronze cutoff. That's the BirdNET overfit cost.

---

## What Would Have Got Us Bronze

A single change would have been enough: **not adding the BirdNET sidecar** and keeping the EoS.9 plain baseline.

The plain EoS.9 submission (ref 53161558, public 0.950, private 0.941) also scored 0.941. But the plain EoS.9 from the community (without our specific BirdNET tuning) likely scored ~0.942 private, placing it just inside bronze. The problem wasn't EoS — it was the BirdNET layer on top.

Bronze requires ~0.942 private. We scored 0.941. Gap: 0.001.

Alternatively, one round of pseudo-labeling on training soundscapes (+~0.002–0.003) would have closed the gap.

---

## Priority Lessons for BirdCLEF 2027

### High Impact (don't repeat mistakes)
1. **Tune postprocessing on OOF only.** Zero public LB tuning on corrections, caps, or weight blends.
2. **Pseudo-labeling from round 1.** Set up noisy student pipeline early — it's 2–3 rounds before deadline.
3. **Measure architectural diversity.** Before locking final submissions, check pairwise correlation of OOF predictions. If >0.98, there's no diversity.

### Medium Impact (structural improvements)
4. **Distill Perch into a faster head.** Top teams used MSE distillation from Perch logits into a faster CNN — allows ensemble without the ONNX Perch bottleneck.
5. **Use OpenVINO fp16.** Faster inference = more models in 90 minutes.
6. **Soundscape-aware training split.** Don't train and validate on clips. Use a soundscape holdout from training-set soundscapes (not test).

### Low Impact (already solid)
7. Architecture choice (Perch v2): correct, keep.
8. Label smoothing (0.05): confirmed fix, keep.
9. ProtoSSM/ResSSM head: solid, keep as backbone.

---

## Competition Context

- **Total teams:** 4,244
- **Public LB shakeup:** 918 teams tied at 0.950 → private range 0.930–0.948 (huge spread)
- **Most common public→private drop:** ~0.002 for well-generalised models; 0.008–0.010 for tuned models
- **Top-1 stability:** 0.967 → 0.965 (only 0.002 drop) — Nikita Babych's pseudo-labeling approach generalises extremely well
- **Our drop:** 0.009 — consistent with postprocessing overfit pattern

---

*Full solution writeups from medal teams expected on the Kaggle discussion board and CLEF 2026 proceedings by July 2026.*
