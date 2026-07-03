# Scripts

This folder contains the main entry-point scripts for the paper.

If you are new to the repository, start here.

## Before you run anything

1. Activate the project environment.
2. Run all commands from the repository root.
3. If you want to use the `Hc` GNN workflow, make sure the `GNN_Uncertainty` submodule has been downloaded with `bash setup.sh`.

## Quick examples

Curie-temperature workflows that run with the included dataset:

```bash
python scripts/train_model_TC_bnn.py
python scripts/train_model_TC_gpr.py
python scripts/train_model_TC_rfr_bagging.py
python scripts/evaluate_uncertainty_TC_bnn.py
python scripts/evaluate_uncertainty_TC_gpr.py
python scripts/evaluate_uncertainty_TC_rfr_bagging.py
```

Graph-neural-network workflow for coercivity:

```bash
python scripts/train_model_Hc_gnn_uncertainty.py
```

## Data requirements

### Included in this repository

- All `TC` scripts use `data/curie/DS1.csv`.

### Not included in this repository

- `train_model_Ha_bnn.py`
- `train_model_Ha_gpr.py`
- `train_model_Ha_rfr_bagging.py`
- `train_model_Ms_bnn.py`
- `train_model_Ms_gpr.py`
- `train_model_Ms_rfr_bagging.py`

These scripts require the proprietary file `data/tmc/2025-06-05_physical_properties.csv`.

## Output folders

Most scripts create their own output folder in this directory, for example:

- `scripts/out_train_model_TC_bnn/`
- `scripts/out_evaluate_uncertainty_TC_bnn/`

These folders usually contain trained weights, CSV summaries, and figures.
