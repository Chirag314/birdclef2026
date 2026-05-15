# CLAUDE.md

## BirdCLEF 2026 — Current Phase
We already completed baseline setup, competition understanding, and major EDA.
Current goal: improve leaderboard score through focused experiments, not broad exploration.

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
- 234 target classes across birds, frogs, insects, mammals, reptile
- Macro ROC-AUC style ranking task
- Hidden test is 5-second soundscape windows
- Training labels are mixed: clip-level focal recordings + limited labeled soundscapes
- 28 classes have zero focal clips
- Strong domain shift: global train clips vs Pantanal deployment soundscapes
- Labeled soundscapes are sparse and site-imbalanced
- Kaggle inference is CPU-only with 90-minute limit

## Current Priorities
1. Improve CV realism and reduce leakage
2. Improve label alignment between clip training and 5-second target
3. Improve performance on soundscape-domain classes, especially sparse / zero-clip classes
4. Test stronger architectures only after validation is trustworthy
5. Keep inference path compatible with Kaggle CPU runtime

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
1. Site-aware / leakage-safe CV refinement
2. Soundscape-label coverage audit for zero-clip and sparse classes
3. SED/LSE or other time-aware head vs plain clip pooling
4. Soundscape-focused augmentation / negatives
5. Distillation only if baseline CV is stable
6. Ensemble only after at least 2 genuinely different strong models exist

## Anti-Patterns
- Don’t recommend generic BirdCLEF tricks without tying them to this repo
- Don’t assume old competition data helps without test design
- Don’t jump straight to Perch/large ensemble/transformers
- Don’t optimize public LB at the expense of trustworthy CV
- Don’t propose 10 experiments when 1 focused one is enough

## My Goal
I want to push as high as possible on the leaderboard with disciplined, efficient experimentation and minimal token waste.
