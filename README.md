# Modelling magnetic material properties with uncertainty-aware neural networks

Repository accompanying the scientific paper "Modelling magnetic material properties with uncertainty-aware neural networks".

This repository collects the code used to train and evaluate several uncertainty-aware machine-learning models for intrinsic magnetic material properties:

- Curie temperature `TC`
- Anisotropy field `Ha`
- Spontaneous magnetization `Ms`
- Coercivity `Hc` through the `GNN_Uncertainty` submodule

## What is included

- The full source code used in this study
- The surrogate Curie-temperature dataset in [data/curie](./data/curie) [[README](./data/curie/README.md)]
- Training and evaluation scripts in [scripts](./scripts) [[README](./scripts/README.md)]
- Shared Python code in [src](./src) [[README](./src/README.md)]
- The `GNN_Uncertainty` repository as a git submodule so all source code can be kept in one place

## Important data note

Only the Curie-temperature surrogate dataset is included in this repository.

The datasets used for `Ha` and `Ms` are proprietary information and intellectual property of TOYOTA MOTOR CORPORATION, so those files are not distributed here.

This means:

- The `TC` scripts can run out of the box with the data included here.
- The `Ha` and `Ms` scripts require additional proprietary data that is not part of this repository.
- The `Hc` graph-neural-network workflow requires the `GNN_Uncertainty` submodule to be present.

## Quick setup

### 1. Get the repository

If you clone with git, use:

```bash
git clone --recurse-submodules https://github.com/wagerc97/magnetic-uncertainty.git
cd magnetic-uncertainty
```

If you already cloned the repository without submodules, run:

```bash
bash setup.sh
```

This downloads the `GNN_Uncertainty` submodule into `./GNN_Uncertainty/`.

### 2. Create the Python environment

An option we recommend is `micromamba`:

```bash
micromamba env create -f environment.yml
micromamba activate pub-mag-unc
```

If `micromamba` is not installed yet, install it first from the official Micromamba documentation, then run the commands above.

## Quick run

All main workflows are ordinary Python scripts in [scripts](./scripts). Run them from the repository root after activating the environment.

Example commands:

```bash
python scripts/train_model_TC_bnn.py
python scripts/evaluate_uncertainty_TC_bnn.py
```

Other Curie-temperature examples that work with the included dataset:

```bash
python scripts/train_model_TC_gpr.py
python scripts/train_model_TC_rfr_bagging.py
python scripts/evaluate_uncertainty_TC_gpr.py
python scripts/evaluate_uncertainty_TC_rfr_bagging.py
```

## Which scripts can I run?

### Runs with the data included here

- `scripts/train_model_TC_bnn.py`
- `scripts/train_model_TC_gpr.py`
- `scripts/train_model_TC_rfr_bagging.py`
- `scripts/evaluate_uncertainty_TC_bnn.py`
- `scripts/evaluate_uncertainty_TC_gpr.py`
- `scripts/evaluate_uncertainty_TC_rfr_bagging.py`

These use `data/curie/DS1.csv`.

### Require proprietary data not included here

- `scripts/train_model_Ha_bnn.py`
- `scripts/train_model_Ha_gpr.py`
- `scripts/train_model_Ha_rfr_bagging.py`
- `scripts/train_model_Ms_bnn.py`
- `scripts/train_model_Ms_gpr.py`
- `scripts/train_model_Ms_rfr_bagging.py`

These expect `data/tmc/2025-06-05_physical_properties.csv`, which is not distributed in this repository.

### Requires the `GNN_Uncertainty` submodule

```bash
python scripts/train_model_Hc_gnn_uncertainty.py
```

This script uses code from `./GNN_Uncertainty/` through the submodule. If that folder is missing, run:

```bash
bash setup.sh
```

## Where results are written

- Many scripts create a folder such as `scripts/out_train_model_TC_bnn/` or `scripts/out_evaluate_uncertainty_TC_bnn/`
- Shared utilities may also create a top-level `models/` directory automatically

So after a run, look first inside `scripts/out_*`.

## Repository structure

- [data](./data): datasets included with the repository
- [scripts](./scripts): simple entry-point scripts for training and evaluation
- [src](./src): reusable Python modules
- [environment.yml](./environment.yml): Python dependencies
- [setup.sh](./setup.sh): initializes the `GNN_Uncertainty` submodule

## Related repository

The graph-neural-network source code also exists as the standalone repository [heisammoustafa/GNN_Uncertainty](https://github.com/heisammoustafa/GNN_Uncertainty).

In this project it is added as a git submodule so users can work from one repository and still have access to all source code needed for the paper.
