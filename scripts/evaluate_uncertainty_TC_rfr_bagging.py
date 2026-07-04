#!/usr/bin/env python
# coding: utf-8

"""Tran-style uncertainty evaluation for the TC RFR + bagging model.

This script mirrors the RFR + bagging setup from
`scripts/train_model_TC_rfr_bagging.py` and adds uncertainty evaluation
following the guidance summarized in `docs/uncertainty_evaluation.pdf`:
- predictive accuracy
- calibration curve and calibration metrics
- sharpness
- proper scoring rules (NLL, CRPS)
- residuals vs predicted standard deviation

For this ensemble, the predictive spread is treated as epistemic uncertainty.
Reported metrics are averaged across multiple random seeds to reduce dependence
on a single train/test split.
"""

import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
print(f"Project root added to python path:\n{_root}")

try:
    import uncertainty_toolbox as uct
except ImportError as exc:
    raise ImportError(
        "This script requires the `uncertainty-toolbox` package. "
        "Install the project environment from environment.yml first."
    ) from exc

print('Imports OK')

warnings.filterwarnings(
    "ignore", message="X has feature names, but.*was fitted without feature names"
)

SCRIPT_NAME = Path(__file__).stem
OUT_PATH = Path(__file__).parent / f"out_{SCRIPT_NAME}"
OUT_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH = (Path(__file__).parents[1] / "data/curie").absolute()
CSV_FILE = "DS1.csv"

LABEL = "TC"
UNIT = "K"
SYMBOL = r"$T_C$"
SEED = 123
N_EVAL_SEEDS = 10
EVAL_SEEDS = list(range(SEED, SEED + N_EVAL_SEEDS))
TEST_SIZE = 0.3
BASE_RF_N_ESTIMATORS = 50
BAGGING_N_ESTIMATORS = 100
DPI = 600
np.random.seed(SEED)
random.seed(SEED)


class CustomBaggingRegressor(BaseEstimator, RegressorMixin):
    """BaggingRegressor wrapper that returns predictive std by default."""

    def __init__(self, estimator, n_estimators, random_state=None, n_jobs=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.model_ = BaggingRegressor(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X, return_std=True, return_cov=False):
        y_pred = self.model_.predict(X)
        if not return_std:
            return y_pred

        all_preds = np.array([est.predict(X) for est in self.model_.estimators_])
        y_std = np.std(all_preds, axis=0)
        return y_pred, y_std



def load_and_preprocess(data_path: Path):
    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    y = df[LABEL]
    X = df.drop(columns=[LABEL]).select_dtypes(include=[np.number])

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    constant_features = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_features:
        print(f"[*] Removing {len(constant_features)} constant features: {constant_features}")
        X = X.drop(columns=constant_features)

    df = pd.concat([X, y], axis=1)
    return df, X, y



def build_regressor(seed: int) -> CustomBaggingRegressor:
    base_estimator = RandomForestRegressor(
        n_estimators=BASE_RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=8,
    )
    return CustomBaggingRegressor(
        estimator=base_estimator,
        n_estimators=BAGGING_N_ESTIMATORS,
        random_state=seed,
        n_jobs=8,
    )



def split_and_scale_data(X, y, df, seed: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, shuffle=True
    )

    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)

    df_scaled = df.copy()
    df_scaled[X_train.columns] = scaler.transform(df_scaled[X_train.columns])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "df_scaled": df_scaled,
    }



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def get_scoring_rule_metrics(y_pred: np.ndarray, y_std: np.ndarray, y_true: np.ndarray) -> dict:
    if hasattr(uct.metrics, 'get_all_scoring_rule_metrics'):
        return uct.metrics.get_all_scoring_rule_metrics(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            resolution=99,
            scaled=True,
            verbose=False,
        )
    return uct.metrics.get_all_scoring_rule_metric(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        resolution=99,
        scaled=True,
        verbose=False,
    )



def flatten_uncertainty_metrics(split_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    accuracy = uct.metrics.get_all_accuracy_metrics(
        y_pred=y_pred,
        y_true=y_true,
        verbose=False,
    )
    calibration = uct.metrics.get_all_average_calibration(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        num_bins=100,
        verbose=False,
    )
    sharpness = uct.metrics.get_all_sharpness_metrics(
        y_std=y_std,
        verbose=False,
    )
    scoring = get_scoring_rule_metrics(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
    )

    return {
        'split': split_name,
        'n_samples': len(y_true),
        'MAE': float(accuracy.get('mae', mean_absolute_error(y_true, y_pred))),
        'RMSE': float(accuracy.get('rmse', rmse(y_true, y_pred))),
        'MDAE': float(accuracy.get('mdae', np.median(np.abs(y_true - y_pred)))),
        'R2': float(accuracy.get('r2', r2_score(y_true, y_pred))),
        'MACE': float(calibration.get('ma_cal')),
        'RMSCE': float(calibration.get('rms_cal')),
        'miscalibration_area': float(calibration.get('miscal_area')),
        'sharpness': float(sharpness.get('sharp')),
        'NLL': float(scoring.get('nll')),
        'CRPS': float(scoring.get('crps')),
    }



def save_plot(fig, output_path: Path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {output_path}")



def make_uncertainty_figures(split_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray):
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    uct.viz.plot_calibration(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        curve_label=f"RFR + bagging ({split_name})",
        ax=ax1,
    )
    ax1.set_title(f"Calibration curve: RFR + bagging {split_name} ({SYMBOL})")
    save_plot(fig1, OUT_PATH / f'{LABEL}_{split_name}_calibration_curve.png')

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    uct.viz.plot_sharpness(y_std=y_std, ax=ax2)
    ax2.set_title(f"Sharpness: RFR + bagging {split_name} ({SYMBOL})")
    save_plot(fig2, OUT_PATH / f'{LABEL}_{split_name}_sharpness.png')

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    uct.viz.plot_residuals_vs_stds(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        ax=ax3,
    )
    ax3.set_title(f"Residuals vs predicted std: RFR + bagging {split_name} ({SYMBOL})")
    save_plot(fig3, OUT_PATH / f'{LABEL}_{split_name}_residuals_vs_std.png')



def evaluate_seed(X, y, df, seed: int, keep_artifacts: bool = False):
    np.random.seed(seed)
    random.seed(seed)

    split_data = split_and_scale_data(X=X, y=y, df=df, seed=seed)
    regressor = build_regressor(seed)
    regressor.fit(split_data['X_train_processed'], split_data['y_train'])

    y_pred_test, y_std_test = regressor.predict(split_data['X_test_processed'])

    metrics_row = flatten_uncertainty_metrics(
        split_name='test',
        y_true=np.asarray(split_data['y_test'].to_numpy()).flatten(),
        y_pred=np.asarray(y_pred_test).flatten(),
        y_std=np.asarray(y_std_test).flatten(),
    )
    metrics_row['seed'] = seed
    metrics_rows = [metrics_row]

    artifacts = None
    if keep_artifacts:
        artifacts = {
            **split_data,
            'regressor': regressor,
            'y_pred_test': np.asarray(y_pred_test).flatten(),
            'y_std_test': np.asarray(y_std_test).flatten(),
        }

    return metrics_rows, artifacts



def summarize_metrics(metrics_by_seed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_by_seed_df
        .agg(
            n_samples_mean=('n_samples', 'mean'),
            MAE_mean=('MAE', 'mean'),
            MAE_std=('MAE', 'std'),
            RMSE_mean=('RMSE', 'mean'),
            RMSE_std=('RMSE', 'std'),
            MDAE_mean=('MDAE', 'mean'),
            MDAE_std=('MDAE', 'std'),
            R2_mean=('R2', 'mean'),
            R2_std=('R2', 'std'),
            MACE_mean=('MACE', 'mean'),
            MACE_std=('MACE', 'std'),
            RMSCE_mean=('RMSCE', 'mean'),
            RMSCE_std=('RMSCE', 'std'),
            miscalibration_area_mean=('miscalibration_area', 'mean'),
            miscalibration_area_std=('miscalibration_area', 'std'),
            sharpness_mean=('sharpness', 'mean'),
            sharpness_std=('sharpness', 'std'),
            NLL_mean=('NLL', 'mean'),
            NLL_std=('NLL', 'std'),
            CRPS_mean=('CRPS', 'mean'),
            CRPS_std=('CRPS', 'std'),
        )
        .T
    )



def write_summary(metrics_by_seed_df: pd.DataFrame, metrics_summary_df: pd.DataFrame, regressor: CustomBaggingRegressor):
    csv_by_seed_path = OUT_PATH / 'uncertainty_metrics_by_seed.csv'
    csv_summary_path = OUT_PATH / 'uncertainty_metrics_summary.csv'
    txt_path = OUT_PATH / 'uncertainty_metrics_summary.txt'
    metrics_by_seed_df.to_csv(csv_by_seed_path, index=False)
    metrics_summary_df.to_csv(csv_summary_path, index=False)

    bagging_model = regressor.model_
    base_estimator = bagging_model.estimator
    lines = [
        'Tran-style uncertainty evaluation for RFR + bagging',
        '=' * 80,
        f'Data file: {DATA_PATH / CSV_FILE}',
        f'Label: {LABEL}',
        f'Representative seed for plots: {SEED}',
        f'Evaluation seeds for reported metrics: {EVAL_SEEDS}',
        f'Base estimator: {base_estimator}',
        f'Bagging estimators: {bagging_model.n_estimators}',
        f'Stored estimators: {len(bagging_model.estimators_)}',
        f'Uncertainty type: epistemic',
        '',
        'Per-seed metrics:',
        metrics_by_seed_df.to_string(index=False),
        '',
        'Mean and standard deviation across seeds:',
        metrics_summary_df.to_string(index=False),
        '',
        'Metrics follow docs/uncertainty_evaluation.pdf:',
        '- accuracy: MAE, RMSE, MDAE, R2',
        '- calibration: MACE, RMSCE, miscalibration area',
        '- sharpness: expected predictive std',
        '- scoring rules: NLL, CRPS',
    ]
    txt_path.write_text('\n'.join(lines))
    print(f"Saved metrics table: {csv_by_seed_path}")
    print(f"Saved metrics summary table: {csv_summary_path}")
    print(f"Saved metrics summary: {txt_path}")



def main():
    df, X, y = load_and_preprocess(DATA_PATH / CSV_FILE)

    metrics_rows = []
    representative_artifacts = None

    print(f"[*] evaluating RFR + bagging uncertainty across {len(EVAL_SEEDS)} random seeds ...")
    for seed in EVAL_SEEDS:
        seed_rows, artifacts = evaluate_seed(
            X=X,
            y=y,
            df=df,
            seed=seed,
            keep_artifacts=(seed == SEED),
        )
        metrics_rows.extend(seed_rows)
        if artifacts is not None:
            representative_artifacts = artifacts

    metrics_by_seed_df = pd.DataFrame(metrics_rows)
    metrics_summary_df = summarize_metrics(metrics_by_seed_df)

    if representative_artifacts is None:
        raise RuntimeError(f"Representative artifacts for seed {SEED} were not captured.")

    write_summary(
        metrics_by_seed_df=metrics_by_seed_df,
        metrics_summary_df=metrics_summary_df,
        regressor=representative_artifacts['regressor'],
    )

    make_uncertainty_figures(
        split_name='test',
        y_true=representative_artifacts['y_test'].to_numpy(),
        y_pred=representative_artifacts['y_pred_test'],
        y_std=representative_artifacts['y_std_test'],
    )

    print(f"\nFINISHED UNCERTAINTY EVALUATION. All results saved to {OUT_PATH}\n")


if __name__ == '__main__':
    main()
