# BirdCLEF 2026

A structured, reproducible BirdCLEF 2026 competition workspace for:

- competition understanding
- deep EDA
- previous-competition comparison
- validation design
- training
- export to Kaggle
- CPU-safe Kaggle inference

This repo is built so the workflow stays clean:

1. understand the competition
2. inspect the data deeply
3. compare with prior BirdCLEF data
4. decide whether prior data helps
5. decide whether augmentation is needed
6. build validation and training pipelines
7. export artifacts cleanly to Kaggle
8. run inference and submission in Kaggle notebook

---

## Project Goals

This project is designed to answer the following questions before serious modeling begins:

- What is the competition really asking us to predict?
- What files are provided and how do they connect?
- What are the hidden bottlenecks?
- How noisy or weak are the labels?
- What leakage risks exist?
- How realistic should validation be?
- Can previous BirdCLEF data help?
- Is augmentation necessary, and if so, which kinds?

The goal is not just to train a model.  
The goal is to build a reliable competition pipeline with strong local validation and safe Kaggle deployment.

---

## Repository Structure

~~~text
birdclef2026/
├── CLAUDE.md
├── README.md
├── configs/
│   ├── base.yaml
│   ├── train.yaml
│   ├── infer_kaggle.yaml
│   └── folds.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── folds.csv
├── src/
│   ├── dataset.py
│   ├── features.py
│   ├── augment.py
│   ├── model.py
│   ├── losses.py
│   ├── train_engine.py
│   ├── infer_engine.py
│   └── utils.py
├── scripts/
│   ├── make_folds.py
│   ├── train.py
│   ├── validate.py
│   ├── export_for_kaggle.py
│   ├── package_kaggle_dataset.py
│   └── benchmark_cpu.py
├── artifacts/
│   ├── checkpoints/
│   ├── oof/
│   ├── logs/
│   └── exports/
└── kaggle_notebook/
    ├── notebook.ipynb
    ├── inference.py
    └── kernel-metadata.json
~~~

---

## Directory Purpose

### `CLAUDE.md`
Project instructions for Claude so analysis stays rigorous, skeptical, and EDA-first.

### `configs/`
All YAML config files for reproducible experiments and inference.

- `base.yaml` → shared defaults
- `train.yaml` → model/training setup
- `infer_kaggle.yaml` → Kaggle inference settings
- `folds.yaml` → fold-generation logic and grouping rules

### `data/`
Data storage.

- `raw/` → downloaded competition and prior-competition data
- `processed/` → derived tables, cached features, cleaned metadata
- `folds.csv` → fold assignments used by experiments

### `src/`
Reusable Python modules.

- `dataset.py` → dataset classes and sample construction
- `features.py` → audio feature extraction
- `augment.py` → augmentation functions
- `model.py` → model definitions
- `losses.py` → loss functions
- `train_engine.py` → training loop helpers
- `infer_engine.py` → inference helpers
- `utils.py` → common utilities

### `scripts/`
Executable entry points.

- `make_folds.py` → create fold assignments
- `train.py` → run training
- `validate.py` → local validation and evaluation
- `export_for_kaggle.py` → export model/config/metadata bundle
- `package_kaggle_dataset.py` → prepare Kaggle dataset packaging
- `benchmark_cpu.py` → estimate CPU inference feasibility

### `artifacts/`
Saved outputs.

- `checkpoints/` → trained model weights
- `oof/` → OOF predictions and fold metrics
- `logs/` → logs and experiment traces
- `exports/` → Kaggle-ready export bundles

### `kaggle_notebook/`
Files used inside Kaggle notebook submission workflow.

- `notebook.ipynb` → notebook version of final inference pipeline
- `inference.py` → clean standalone inference code
- `kernel-metadata.json` → Kaggle notebook metadata

---

## Recommended Workflow

### Phase 1 — Competition Understanding
Before doing anything else:

- read competition overview
- read rules carefully
- inspect evaluation metric
- inspect submission format
- understand runtime constraints
- identify what the competition is truly testing

Expected outputs:
- `competition_memo.md`
- notes on files, metric, constraints, and hidden bottlenecks

---

### Phase 2 — Deep EDA
Perform serious EDA on:

- metadata
- labels
- audio properties
- missing values
- duplicates
- outliers
- suspicious values
- leakage risks
- domain shift risks
- train-target alignment risks

Expected outputs:
- plots
- markdown summaries
- risk list
- modeling hypotheses

---

### Phase 3 — Previous Competition Comparison
Compare BirdCLEF 2026 with prior BirdCLEF data.

Questions to answer:
- what overlaps?
- what changed?
- what is missing?
- does prior data help representation?
- does prior data hurt calibration or realism?
- should prior data be used for pretraining, auxiliary training, or not at all?

Expected outputs:
- overlap tables
- label-space comparison
- schema comparison
- distribution comparison
- recommendation on prior-data use

---

### Phase 4 — Augmentation Decision
Only after EDA and comparison:

- decide whether augmentation is necessary
- tie augmentation choices to actual observed problems
- reject generic augmentation habits that are not supported by evidence

Expected outputs:
- augmentation decision memo
- priority-ranked augmentation list
- risks for each augmentation type

---

### Phase 5 — Validation Design
Build folds that reflect realistic generalization.

Possible risks to inspect:
- site leakage
- recorder leakage
- location leakage
- temporal leakage
- duplicate clip leakage
- background leakage
- weak-label leakage

Expected outputs:
- `data/folds.csv`
- fold summary tables
- leakage-risk notes

---

### Phase 6 — Training
After the analysis phase is complete:

- train baseline models
- track fold-level metrics
- keep configs reproducible
- save checkpoints and OOF predictions
- compare experiments using the same validation logic

Expected outputs:
- model checkpoints
- fold metrics
- OOF predictions
- training logs

---

### Phase 7 — Export for Kaggle
Prepare a clean export bundle containing:

- model weights
- config
- class mapping
- preprocessing parameters
- runtime notes

Expected outputs:
- files in `artifacts/exports/`
- Kaggle dataset-ready artifact bundle

---

### Phase 8 — Kaggle Inference
In Kaggle notebook:

- load exported artifacts
- reproduce preprocessing exactly
- run CPU-safe inference
- generate submission file
- verify runtime feasibility

Expected outputs:
- final submission
- runtime notes
- notebook-ready reproducible inference flow

---

## Suggested Early Deliverables

These files are not required immediately, but are good first targets:

~~~text
competition_memo.md
reports/eda_summary.md
reports/risks_and_hypotheses.md
eda/01_data_inventory.ipynb
eda/02_metadata_eda.ipynb
eda/03_label_eda.ipynb
eda/04_audio_eda.ipynb
eda/05_previous_comp_comparison.ipynb
~~~

You can create those later as the project grows.

---

## Guiding Principles

This repo follows these principles:

- evidence first
- reproducibility first
- leakage awareness
- alignment awareness
- realistic validation over leaderboard excitement
- prior data used carefully, not automatically
- augmentation only if justified by analysis
- clean local-to-Kaggle handoff

---

## What Not To Do Early

Avoid these before analysis is complete:

- blindly choosing a heavy model
- throwing in every augmentation
- trusting public leaderboard over local reasoning
- mixing prior BirdCLEF data without testing overlap and shift
- ignoring weak-label and time-window mismatch
- building fancy ensembles before a trustworthy baseline

---

## Minimal Setup Notes

Typical working flow on Linux:

~~~bash
cd ~/data/birdclef2026
code .
~~~

Then fill files in this order:

1. `CLAUDE.md`
2. `README.md`
3. `configs/base.yaml`
4. `configs/folds.yaml`
5. `configs/train.yaml`
6. `configs/infer_kaggle.yaml`

After that, start adding scripts and notebooks.

---

## Status

Current repo status:

- structure initialized
- analysis-first workflow defined
- ready for config setup
- ready for competition memo and EDA phase

---

## Next Recommended Step

Fill these files next:

- `configs/base.yaml`
- `configs/folds.yaml`
- `configs/train.yaml`
- `configs/infer_kaggle.yaml`

Then start with:

- competition memo
- file inventory
- metadata EDA
- label EDA
- audio EDA