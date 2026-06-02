# CLAUDE.md

## BirdCLEF 2026 — Current Phase
Phase 7 — **LOCKED**. Final submissions selected 2026-05-31.
SLOT 1: ref 53205912 birdnet_sidecar_v2 (0.950+, rank +250) | SLOT 2: ref 53126671 tax_blend v2 (diversity hedge).

## Main Objective
Help me work like a serious Kaggle competitor under token limits.
Prioritize high-ROI experiments, realistic validation, and efficient repo inspection.

## Hard Rules
- Do not rescan the whole repo unless I ask.
- Read only the files needed for the current task.
- Do not repeat competition background unless needed.
- Keep responses compact, practical, and action-oriented.
- Treat discussion/forum ideas as hypotheses to test, not facts.
- Prefer one experiment at a time.
- Before editing, summarize current state and exact files to change.
- Prefer minimal diffs over rewrites.
- Always rank next steps by expected LB gain, validation value, and implementation cost.

## Competition Facts To Keep In Mind
- 234 target classes across birds, frogs, insects, mammals, reptiles
- Macro ROC-AUC ranking task — all 234 classes weighted equally
- Hidden test is 5-second soundscape windows from Pantanal, Brazil
- Training labels: 35k focal clips (206 species) + 1,478 labeled soundscape windows (251 species)
- Only 12 species appear in BOTH focal clips AND soundscape labels
- ~25 insect/amphibian classes have zero focal clips — model is near-blind on these
- Strong domain shift: global iNat/XC clips vs Pantanal deployment soundscapes
- Label smoothing (0.05) is confirmed to reduce clip overconfidence and improve LB
- Perch v2 embeddings: 1536-dim, trained on deployment-style audio — bridges domain gap
- Kaggle inference: CPU-only, 90-minute limit — EfficientNet-B0 fits comfortably

## Current Priorities (medal sprint)
1. [DONE] Perch MLP 5-fold → 0.873 LB
2. [DONE] ProtoSSM pipeline → 0.944 LB
3. [DONE] ProtoSSM v2 (power/gate tuning) → 0.944 LB (no gain — ceiling hit)
4. [DONE] EoS pipeline (correction_weight=0.10) → **0.949 LB** ← current best
5. [DONE] EoS 80% + ProtoSSM 20% blend → 0.948 (hurt -0.001, ruled out)
6. [DONE] exp017: lambda_prior 0.4→0.5 → 0.949 (stacked into exp019)
7. [DONE] exp019: rank_aware_scaling.power 0.5→0.6 → **0.949** (confirmed)
8. [DONE] exp020: file_confidence_scale.power 0.4→0.5 → 0.948 (-0.001)
9. [DONE] exp021: file_confidence_scale.power 0.4→0.3 → 0.948 (-0.001, dial exhausted at 0.4)
10. [DONE] exp022: adaptive_delta_smooth.base_alpha 0.20→0.25 → 0.948 (-0.001, reversed)
11. [DONE] exp023: adaptive_delta_smooth.base_alpha 0.20→0.15 → 0.948 (-0.001, dial exhausted at 0.20)
12. [PIVOT] All EoS single-model dials exhausted at 0.949
13. [DONE] Community scan: 0.950 found via lambda_prior=0.65 + rank_power=0.65 + f_TAX_SMOOTHING_POSTPROC
14. [DONE] kaggle_kernel_eos_tax: lambda_prior=0.65 + rank_power=0.65 + TAX_SMOOTHING → 0.949 (no gain on single model)
15. [DONE] kaggle_kernel_tax_blend: fork karnak_tax → **0.950 LB** ← current best
16. [DONE] kaggle_kernel_eos9: fork EoS.9 → 0.950 (same ceiling, weight tweak made no difference)
17. [DONE] Community scan: no public kernel claims >0.950 LB as of 2026-05-29
18. [DONE] kaggle_kernel_pcen_sidecar: TIMED OUT (PCEN/ConvNeXt too slow for 90-min CPU limit)
19. [DONE] kaggle_kernel_birdnet_sidecar: EoS.9 + BirdNET (caps 0.015/0.060) → 0.950+ (rank +200)
20. [DONE] kaggle_kernel_birdnet_sidecar_v2: BirdNET caps 0.025/0.090 → **0.950+ (rank +250)** ← current best
21. [DONE] kaggle_kernel_birdnet_sidecar_v3: BirdNET caps 0.035/0.120 → 0.950 (no rank change, ceiling confirmed at v2)
22. [DONE] kaggle_kernel_taxrank_birdnet: EoS8 TaxRank + BirdNET → 0.950 same rank (TaxRank adds no diversity vs EoS9)
23. [LOCKED] All public approaches exhausted. Final slots: birdnet_sidecar_v2 (slot 1) + tax_blend v2 (slot 2)
10. Target: 0.950+ (top 200 gold medal)

## Experiment Philosophy
- Data/validation fixes before fancy models
- Soundscape-aware training before large ensembles
- Missing-class strategy matters
- Domain adaptation matters more than cosmetic augmentation
- Bigger model is not automatically better
- Ensemble only after single-model gains stabilize

## Default Workflow
For every task:
1. Read only relevant files
2. Summarize current state in 5 bullets max
3. Identify biggest bottleneck
4. Propose 3 next actions max
5. Edit only after approval unless explicitly told to proceed

## Output Style
Always return:
- Summary
- Why this matters
- Exact files to inspect/edit
- Recommended next command/prompt

## Token Efficiency Rules
- Avoid long explanations
- Avoid rereading unchanged files
- Reuse repo facts already established
- If context is growing, recommend starting a fresh session
- Prefer targeted prompts over broad planning prompts

## Current High-ROI Experiment Queue
1. [DONE] Phase 4 5-fold label smoothing — LB ~0.833
2. [DONE] Perch MLP 5-fold — OOF 0.977, LB 0.873
3. [DONE] ProtoSSM pipeline — LB 0.944
4. [DONE] ProtoSSM v2 (adaptive weights, power=0.5) — LB 0.944 (no gain)
5. [DONE] EoS pipeline (ONNX Perch, correction_weight=0.10) — LB **0.949**
6. [DONE] EoS 80% + ProtoSSM 20% blend → 0.948 (hurt -0.001, ruled out)
7. [DONE] exp017: lambda_prior 0.4→0.5 (stacked into exp019)
8. [DONE] exp019: rank_aware_scaling.power 0.5→0.6 → LB **0.949** confirmed
9. [DONE] exp020: file_confidence_scale.power 0.4→0.5 → 0.948 (-0.001)
10. [DONE] exp021: file_confidence_scale.power 0.4→0.3 → 0.948 (-0.001, dial exhausted at 0.4)
11. [NEXT] exp022: adaptive_delta_smooth.base_alpha 0.20→0.25
12. [NEXT] exp023: adaptive_delta_smooth.base_alpha 0.20→0.15 (if 0.25 hurts)
11. [DONE] Community scan: 0.950 via f_TAX_SMOOTHING_POSTPROC + lambda_prior=0.65 + rank_power=0.65
12. [NEXT] kaggle_kernel_eos_tax: reproduce 0.950
12. [FUTURE] Fine-tuned Perch head with SS-aware training

## EoS Postprocessing Dial State (as of exp019)
| Dial | Current | Tried | Status |
|---|---|---|---|
| `lambda_prior` | 0.5 | 0.4→0.5 | in exp019 baseline |
| `rank_aware_scaling.power` | 0.6 | 0.5→0.6 | +0 confirmed (exp019) |
| `file_confidence_scale.power` | 0.4 | 0.3→0.948, 0.5→0.948 | exhausted at 0.4 |
| `adaptive_delta_smooth.base_alpha` | 0.20 | 0.15→0.948, 0.25→0.948 | exhausted at 0.20 |
| `file_confidence_scale.top_k` | 2 | — | low priority |

## What We've Ruled Out
- **Site×Hour prior at inference**: -0.083 LB (0.873→0.790). Training SS cover 9 sites; test sites differ. Global fallback zeros 159/234 classes. DO NOT USE.
- **Ghost species LogReg correction**: -0.010 LB (0.949→0.939). OOF AUC≈1.0 was a trap — LogReg memorized site-level Perch features, not species acoustics. Training SS sites ≠ test sites → predictions are pure noise on new sites. DO NOT USE.
- **50/50 Perch+EfficientNet ensemble**: -0.019 LB (0.873→0.854). EfficientNet (~0.833 est.) too weak to help; drags Perch down.
- **EoS 80% + ProtoSSM 20% blend**: -0.001 LB (0.949→0.948). 0.005 gap too large — ProtoSSM adds noise. Rule: only blend models within ~0.002–0.003 of each other.
- **EoS postprocessing dial tuning** (exp020–023): all four dials exhausted at our search range — single-model ceiling = 0.949. Community later found lambda_prior=0.65 and rank_power=0.65 are better (we only tested up to 0.5 and 0.6 respectively).
- **f_TAX_SMOOTHING_POSTPROC alone (eos_tax)**: 0.949 — no gain on single model. Needs blend diversity.
- **tax_blend** (Model_22 2% + Model_51 1.3% + Model_74 96.7% + TAX_SMOOTHING): **0.950** ← confirmed recipe.
- **ProtoSSM parameter tuning** (power, gate, adaptive weights): confirmed 0.944 ceiling — no path to gain.
- **correction_weight=0.10 in our ProtoSSM**: -0.001 LB (0.944→0.943). EoS's 0.10 is specific to its full postprocessing chain; our pipeline optimum is ~0.35.
- Soundscape oversampling (phases 2/2b): hurts LB — training soundscapes ≠ test distribution
- Bigger backbone (phase 3 B2): higher clip CV = more overfitting = worse LB
- Mixup augmentation: hurts CV and LB
- SpecAugment: deprioritised — overconfidence was the real bottleneck, now fixed by label smoothing

## Anti-Patterns
- Don’t recommend generic BirdCLEF tricks without tying them to this repo
- Don’t assume old competition data helps without test design
- Don’t jump straight to Perch/large ensemble/transformers
- Don’t optimize public LB at the expense of trustworthy CV
- Don’t propose 10 experiments when 1 focused one is enough

## My Goal
I want to push as high as possible on the leaderboard with disciplined, efficient experimentation and minimal token waste.
