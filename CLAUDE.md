# CLAUDE.md

## BirdCLEF 2026 — Current Phase
Phase 4 (label smoothing, 5-fold) is locking in ~0.833 baseline. Phase 5 begins now.
Current goal: close the remaining CV→LB gap via domain alignment and inference calibration.

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

## Current Priorities (15-day medal sprint)
1. [DONE] Perch MLP 5-fold → 0.873 LB (new best, +0.055 over phase4)
2. Phase4 5-fold EfficientNet finishing (~0.833) — submit for LB confirmation
3. Ensemble Perch MLP + EfficientNet phase4 → expected +0.01–0.02
4. Site×Hour prior on Perch MLP inference → free gain, already built
5. Zero-clip class audit — 25+ species near-blind, fix before further tuning
6. Target: 0.949 tier (top 300 gold medal) — gap is 0.076

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
1. [DONE] Phase 4 fold0: label smoothing — LB 0.818 (+0.029)
2. [RUNNING] Phase 4 folds 1-4 — fold4 still training
3. [DONE] Site×Hour prior — src/postprocess.py built
4. [DONE] Perch v2 embeddings extracted — (35549, 1536) clips + (1478, 1536) soundscapes
5. [DONE] Perch MLP 5-fold trained — OOF 0.977, **LB 0.873** (new best)
6. [DONE] cid007/birdclef2026-perch dataset uploaded — self-contained, no 3rd-party deps
7. [NEXT] Ensemble Perch MLP + EfficientNet phase4 in inference notebook
8. [NEXT] Add Site×Hour prior to Perch MLP inference (already in src/postprocess.py)
9. [NEXT] Zero-clip class audit and fix
10. [FUTURE] Better Perch head (attention/SSM), fine-tuned Perch

## What We've Ruled Out
- **Site×Hour prior at inference**: -0.083 LB (0.873→0.790). Training SS cover 9 sites; test sites differ. Global fallback zeros 159/234 classes. DO NOT USE.
- **50/50 Perch+EfficientNet ensemble**: -0.019 LB (0.873→0.854). EfficientNet (~0.833 est.) too weak to help; drags Perch down. Only ensemble models within 0.01 of each other.
- Soundscape oversampling (phases 2/2b): hurts LB — training soundscapes ≠ test distribution
- Bigger backbone (phase 3 B2): higher clip CV = more overfitting = worse LB
- Mixup augmentation: hurts CV and LB
- SpecAugment: deprioritised — overconfidence was the real bottleneck, now fixed by label smoothing
- Rank-aware power scaling (p=0.6): unverified — our problem was OVERconfidence, squashing toward 1 would likely hurt

## Anti-Patterns
- Don’t recommend generic BirdCLEF tricks without tying them to this repo
- Don’t assume old competition data helps without test design
- Don’t jump straight to Perch/large ensemble/transformers
- Don’t optimize public LB at the expense of trustworthy CV
- Don’t propose 10 experiments when 1 focused one is enough

## My Goal
I want to push as high as possible on the leaderboard with disciplined, efficient experimentation and minimal token waste.
