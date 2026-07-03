# Source code

This folder contains the reusable Python modules used by the scripts in [../scripts](../scripts).

Most users do not need to run files in `src/` directly. In normal use, run the scripts from the repository root and let them import these modules automatically.

## Layout

- `ml/`: machine-learning models, including approximate Bayesian neural-network code
- `evaluation/`: plotting and uncertainty-evaluation utilities
- `util.py`: shared data-loading, preprocessing, timestamp, and output-folder helpers
- `notebook2pdf.py`: utility script for notebook export

## Notes

- Some helper code automatically creates a top-level `models/` folder if it does not exist yet.
- The Curie-temperature scripts use data from `data/curie/`.
- The `Ha` and `Ms` workflows expect additional proprietary data under `data/tmc/`, which is not included in this repository.
- The `Hc` workflow imports code from the `GNN_Uncertainty` submodule.
