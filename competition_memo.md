# BirdCLEF 2026 — Working Memo

## Task
Predict per-species probabilities for each 5-second soundscape window.

## Metric
Macro-style per-class ROC-AUC ranking metric.
Implication: ranking quality matters more than threshold tuning.

## Label Space
- 234 classes total
- Not bird-only: birds, frogs, insects, mammals, reptile
- 28 classes have zero focal training clips
- Some rare classes rely almost entirely on labeled soundscapes

## Data Structure
Two main training sources:
1. Global focal clips with clip-level labels
2. Pantanal soundscapes with limited 5-second labels

## Core Competition Difficulty
This is mainly a domain-shift + weak-supervision problem:
- global focal clips are not the deployment domain
- test is Pantanal soundscapes
- training supervision is partly clip-level, but scoring is 5-second window-level

## Biggest Risks
1. Site leakage / unrealistic CV
2. Clip-to-window label misalignment
3. Sparse or zero-clip classes
4. Soundscape site imbalance
5. Overfitting to clean focal clips instead of noisy soundscapes
6. CPU-only 90-minute Kaggle inference constraint

## What Matters Most
- trustworthy CV
- soundscape-aware learning
- time-aware pooling / detection behavior
- missing-class strategy
- efficient inference path

## Confirmed LB Results

| Experiment | LB | Notes |
|---|---|---|
| baseline_v1 B0 5-fold | — | clip CV 0.9248 |
| phase1a bg noise aug 5-fold | 0.789 | best before label smoothing |
| phase2 SS oversampling | 0.756–0.774 | hurt LB — wrong soundscapes |
| phase3 B2 backbone 5-fold | 0.753 | higher CV = more overfitting |
| **phase4 B0 label_smooth=0.05** | **0.818** | **+0.029 breakthrough** |
| phase4 5-fold (est.) | ~0.833 | in progress |

## Root Cause of CV→LB Gap (Confirmed)
The model was overconfident on easy focal clips (high clip CV) but this certainty
did not transfer to messy Pantanal soundscapes (low LB). Label smoothing directly
reduced overconfidence and improved LB by +0.029 from a single fold.

Higher clip CV reliably predicted worse LB throughout all experiments.

## Working Hypotheses — Updated
1. [CONFIRMED] Label smoothing fixes overconfidence → +0.029 LB from 1 fold
2. [CONFIRMED] Soundscape oversampling hurts — training SS ≠ test SS distribution
3. [CONFIRMED] Bigger backbone = more overfitting = worse LB (B2 proven)
4. [ACTIVE] Perch v2 distillation should bridge the remaining domain gap
5. [ACTIVE] Site×Hour prior at inference = free +LB from spatial-temporal context
6. [PENDING] Zero-clip class handling — 25+ insect/amphibian classes near-blind
7. [DEFERRED] Rank-aware power scaling — unverified against our distribution

## Current Experiment Order (15-day sprint)
1. Phase 4 5-fold baseline lock (~0.833 LB) — running
2. Site×Hour prior inference post-processing — implemented in src/postprocess.py
3. Perch v2 embedding precompute — 64GB RAM can hold all training embeddings
4. Perch distillation 1-fold LB check (PerchAlignmentLoss already in losses.py)
5. If Perch helps: 5-fold Perch distillation + combine with Site×Hour prior
6. Zero-clip class audit and soundscape-conditioning strategy

## Constraints
- local: RTX 3060 12GB, 64GB RAM
- Kaggle inference: CPU-only, 90-minute limit

## Decision Rule
Prefer experiments that improve:
1. validation realism
2. domain alignment
3. rare-class handling
4. inference feasibility
before spending time on complex model stacks
