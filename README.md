# HyperSpeech Pipeline

This repository is organized for a train-once, compare-many workflow.

## Notebook order

1. `notebooks/00_check_data.ipynb`
2. `notebooks/01_make_splits.ipynb`
3. One of:
   - `notebooks/02_train_baselines.ipynb` (all enabled baselines)
   - `notebooks/02a_train_classical.ipynb`
   - `notebooks/02b_train_foundation_and_fastdl.ipynb`
   - `notebooks/02c_train_transformers_and_others.ipynb`
4. `notebooks/03_train_hyperspeech.ipynb`
5. `notebooks/04_compare_and_plots.ipynb`

## Core principles

- Subject-aware, leakage-safe outer CV via `StratifiedGroupKFold`
- Fold-level artifact caching under `artifacts/`
- Post-hoc calibration (Platt / Isotonic)
- Two operating points: best F1 and recall-constrained screening

## Optional baselines

Some models require extra dependencies and implementation adapters:

- SAINT
- FT-Transformer
- TabTransformer
- DCNv2
- TabNet
- NODE
- CARTE

Use `src/models/wrappers_optional.py` to connect your preferred package implementation for each.

Default wiring in this repo:

- SAINT, FT-Transformer, TabTransformer, DCNv2, NODE: `pytorch-tabular`
- TabNet: `pytorch-tabnet`
- CARTE: adapter expects a package exposing `CARTEClassifier`
