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

## Current Baseline Status
- baseline exists and submits successfully
- rank is still low, so major gains likely come from validation, alignment, and domain adaptation rather than small tuning

## Working Hypotheses To Test
1. Better grouped CV will improve LB correlation
2. Time-aware heads should beat naive clip pooling
3. Soundscape-focused negatives / augmentation should reduce false positives
4. Sparse-class handling needs dedicated attention
5. Older BirdCLEF data may help selectively, but not by blind mixing
6. Heavy ensemble should wait until strong single models exist

## Current Experiment Order
1. CV refinement
2. Sparse / zero-clip class audit
3. Time-aware head experiment
4. Soundscape-domain augmentation / negatives
5. Distillation experiment
6. Selective ensemble experiment

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
