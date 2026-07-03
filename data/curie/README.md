# Curie-temperature surrogate dataset

This folder contains the only dataset distributed with this repository.

The file `DS1.csv` is used by the Curie-temperature (`TC`) scripts in [../../scripts](../../scripts), for example:

```bash
python scripts/train_model_TC_bnn.py
python scripts/evaluate_uncertainty_TC_bnn.py
```

## Source

Source repository: https://github.com/msg-byu/ML-for-CurieTemp-Predictions

From that source, `DS1.csv` is described as a cleaned version of `DS1-RAW.csv` compiled by James Nelson and Stefano Sanvito. The feature vector contains 85 features describing the elemental composition of each compound.

## Important note

This dataset is a surrogate dataset for Curie temperature. It is not the proprietary Toyota dataset used for the `Ha` and `Ms` workflows.
