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
| phase4 5-fold (est.) | ~0.833 | folds 0-3 done, fold4 running |
| **Perch MLP 5-fold (frozen embs)** | **0.873** | **+0.055 vs phase4 1-fold; new best** |
| Perch MLP + Site×Hour prior | 0.790 | **-0.083** — prior kills 159/234 zero-SS classes |

## Root Cause of CV→LB Gap (Confirmed)
The model was overconfident on easy focal clips (high clip CV) but this certainty
did not transfer to messy Pantanal soundscapes (low LB). Label smoothing directly
reduced overconfidence and improved LB by +0.029 from a single fold.

Higher clip CV reliably predicted worse LB throughout all experiments.

## Working Hypotheses — Updated
1. [CONFIRMED] Label smoothing fixes overconfidence → +0.029 LB from 1 fold
2. [CONFIRMED] Soundscape oversampling hurts — training SS ≠ test SS distribution
3. [CONFIRMED] Bigger backbone = more overfitting = worse LB (B2 proven)
4. [CONFIRMED] Perch frozen embeddings + MLP → 0.873 LB (+0.055 over phase4 single fold)
5. [RULED OUT] Site×Hour prior — kills 159 zero-SS classes, -0.083 LB on Perch MLP
6. [ACTIVE] Perch+EfficientNet ensemble — two uncorrelated architectures
7. [PENDING] Zero-clip class handling — 25+ insect/amphibian classes near-blind
8. [PENDING] Better Perch head — SSM/attention instead of MLP; or fine-tuned ONNX

## Gap to Top Tier
- Our best: 0.873 (Perch MLP 5-fold)
- Top tier: 0.949 (508 teams tied — likely fine-tuned Perch or better head)
- Gap: 0.076
- Main suspects: (1) frozen vs fine-tuned Perch, (2) zero-clip classes, (3) ensemble

## Current Experiment Order
1. Phase 4 5-fold EfficientNet — fold4 running, submit to confirm ~0.833 LB
2. Ensemble Perch MLP + EfficientNet phase4 5-fold → expected +0.01–0.02
3. Add Site×Hour prior to Perch MLP inference → free gain
4. Audit zero-clip class predictions — 25+ species near-blind in Perch MLP
5. Longer/better Perch MLP training (more epochs, attention head)
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
