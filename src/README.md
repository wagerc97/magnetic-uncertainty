# Source Code

This folder contains the core Python modules and training notebooks for the magnetic-uncertainty project. 

## Layout
- `evaluation/`: plotting and evaluation utilities (confidence curves, residual plots, validation helpers).
- `ml/`: machine-learning models, including approximate Bayesian neural network implementations.
- `util.py`: shared paths, preprocessing helpers, and data-loading utilities.
- `train_model_*.ipynb`: notebooks for training models for anisotropy field $\mu_0H_\mathrm{a}$ (T) and spontaneous magnetization $\mu_0M_\mathrm{s}$ (T) targets.

## Notes
- Model artifacts are written under `models/` at the repository root; `util.py` creates the folder and adds a `.gitignore` if needed.
- Datasets are expected under `data/` at the repository root.
