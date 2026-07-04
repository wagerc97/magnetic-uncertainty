#!/usr/bin/env python
# coding: utf-8

"""Tran-style uncertainty evaluation for the TC BNN model.

This script mirrors the fixed BNN setup from `scripts/train_model_TC_bnn.py`
and runs separate uncertainty evaluations for:
- total uncertainty
- aleatoric uncertainty
- epistemic uncertainty

The evaluation follows the guidance summarized in
`docs/uncertainty_evaluation.pdf`:
- predictive accuracy
- calibration curve and calibration metrics
- sharpness
- proper scoring rules (NLL, CRPS)
- residuals vs predicted standard deviation

Reported metrics are averaged across multiple random seeds to reduce dependence
on a single train/test split.
"""

import os
import random
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential, optimizers
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
print(f"Project root added to python path:\n{_root}")

from src.ml.bnn_base import BnnBase, gaussian_nll

try:
    import uncertainty_toolbox as uct
except ImportError as exc:
    raise ImportError(
        "This script requires the `uncertainty-toolbox` package. "
        "Install the project environment from environment.yml first."
    ) from exc

print('Imports OK')

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
BNN_N_SAMPLES = 50
BNN_EPOCHS = 1000
LEARNING_RATE = 0.005
TEST_SIZE = 0.30
DPI = 600
np.random.seed(SEED)
random.seed(SEED)
keras.utils.set_random_seed(SEED)


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    


class BnnTC(BnnBase):
    """BNN model for Curie temperature prediction."""

    epochs: int = BNN_EPOCHS
    name: str = "Bnn_CurieTemperature"

    def __init__(self, n_features: int, n_samples: int = BNN_N_SAMPLES, random_seed: int = None):
        super().__init__(
            n_features=n_features,
            epochs=self.epochs,
            n_samples=n_samples,
            model_name=self.name,
            random_seed=random_seed,
        )

    @classmethod
    def _build_model(cls, n_features: int, random_seed: int = None) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(n_features,)))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.2, seed=random_seed))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.2, seed=random_seed))

        model.add(Dense(2, activation=None))

        optim = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optim, loss=gaussian_nll)
        return model



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



def build_regressor(n_features: int, seed: int) -> BnnBase:
    return BnnTC(
        n_features=n_features,
        n_samples=BNN_N_SAMPLES,
        random_seed=seed,
    )



def split_and_scale_data(X, y, df, seed: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df_scaled = df.copy()
    df_scaled[X_train.columns] = scaler.transform(df_scaled[X_train.columns])

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'df_scaled': df_scaled,
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



def flatten_uncertainty_metrics(
    uncertainty_type: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict:
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
        'uncertainty_type': uncertainty_type,
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



def make_uncertainty_figures(
    uncertainty_type: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
):
    label = f"BNN {uncertainty_type} ({split_name})"

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    uct.viz.plot_calibration(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        curve_label=label,
        ax=ax1,
    )
    ax1.set_title(f"Calibration curve: {label} ({SYMBOL})")
    save_plot(fig1, OUT_PATH / f'{LABEL}_{split_name}_{uncertainty_type}_calibration_curve.png')

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    uct.viz.plot_sharpness(y_std=y_std, ax=ax2)
    ax2.set_title(f"Sharpness: {label} ({SYMBOL})")
    save_plot(fig2, OUT_PATH / f'{LABEL}_{split_name}_{uncertainty_type}_sharpness.png')

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    uct.viz.plot_residuals_vs_stds(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        ax=ax3,
    )
    ax3.set_title(f"Residuals vs predicted std: {label} ({SYMBOL})")
    save_plot(fig3, OUT_PATH / f'{LABEL}_{split_name}_{uncertainty_type}_residuals_vs_std.png')



def evaluate_seed(X, y, df, seed: int, keep_artifacts: bool = False):
    set_all_seeds(seed)

    split_data = split_and_scale_data(X=X, y=y, df=df, seed=seed)
    regressor = build_regressor(split_data['X_train_scaled'].shape[1], seed)
    regressor.fit(split_data['X_train_scaled'], split_data['y_train'])

    y_pred_test, y_std_test, y_std_al_test, y_std_ep_test = regressor.predict(split_data['X_test_scaled'])

    uncertainty_payloads = {
        'total': {
            'test_std': np.asarray(y_std_test).flatten(),
        },
        'aleatoric': {
            'test_std': np.asarray(y_std_al_test).flatten(),
        },
        'epistemic': {
            'test_std': np.asarray(y_std_ep_test).flatten(),
        },
    }

    y_pred_test = np.asarray(y_pred_test).flatten()
    y_test_arr = split_data['y_test'].to_numpy()

    metrics_rows = []
    for uncertainty_type, payload in uncertainty_payloads.items():
        metrics_row_test = flatten_uncertainty_metrics(
            uncertainty_type=uncertainty_type,
            split_name='test',
            y_true=y_test_arr,
            y_pred=y_pred_test,
            y_std=payload['test_std'],
        )
        metrics_row_test['seed'] = seed
        metrics_rows.append(metrics_row_test)

    artifacts = {
        'y_true_test': y_test_arr,
        'y_pred_test': y_pred_test,
        'uncertainty_payloads': uncertainty_payloads,
    }
    if keep_artifacts:
        artifacts['regressor'] = regressor

    return metrics_rows, artifacts



def summarize_metrics(metrics_by_seed_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for uncertainty_type, group in metrics_by_seed_df.groupby('uncertainty_type', sort=False):
        summary_rows.append({
            'uncertainty_type': uncertainty_type,
            'n_samples_mean': group['n_samples'].mean(),
            'MAE_mean': group['MAE'].mean(),
            'MAE_std': group['MAE'].std(),
            'RMSE_mean': group['RMSE'].mean(),
            'RMSE_std': group['RMSE'].std(),
            'MDAE_mean': group['MDAE'].mean(),
            'MDAE_std': group['MDAE'].std(),
            'R2_mean': group['R2'].mean(),
            'R2_std': group['R2'].std(),
            'MACE_mean': group['MACE'].mean(),
            'MACE_std': group['MACE'].std(),
            'RMSCE_mean': group['RMSCE'].mean(),
            'RMSCE_std': group['RMSCE'].std(),
            'miscalibration_area_mean': group['miscalibration_area'].mean(),
            'miscalibration_area_std': group['miscalibration_area'].std(),
            'sharpness_mean': group['sharpness'].mean(),
            'sharpness_std': group['sharpness'].std(),
            'NLL_mean': group['NLL'].mean(),
            'NLL_std': group['NLL'].std(),
            'CRPS_mean': group['CRPS'].mean(),
            'CRPS_std': group['CRPS'].std(),
        })
    return pd.DataFrame(summary_rows)



def write_summary(metrics_by_seed_df: pd.DataFrame, metrics_summary_df: pd.DataFrame, regressor: BnnBase):
    csv_by_seed_path = OUT_PATH / 'uncertainty_metrics_by_seed.csv'
    csv_summary_path = OUT_PATH / 'uncertainty_metrics_summary.csv'
    txt_path = OUT_PATH / 'uncertainty_metrics_summary.txt'
    metrics_by_seed_df.to_csv(csv_by_seed_path, index=False)
    metrics_summary_df.to_csv(csv_summary_path, index=False)

    lines = [
        'Tran-style uncertainty evaluation for fixed-architecture BNN',
        '=' * 80,
        f'Data file: {DATA_PATH / CSV_FILE}',
        f'Label: {LABEL}',
        f'Evaluation seeds for reported metrics: {EVAL_SEEDS}',
        'Plots pool test predictions from all evaluation seeds.',
        f'Epochs: {regressor.epochs}',
        f'MC samples: {regressor.n_samples}',
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
        '- uncertainty types are evaluated separately: total, aleatoric, epistemic',
    ]
    txt_path.write_text('\n'.join(lines))
    print(f"Saved metrics table: {csv_by_seed_path}")
    print(f"Saved metrics summary table: {csv_summary_path}")
    print(f"Saved metrics summary: {txt_path}")



def main():
    df, X, y = load_and_preprocess(DATA_PATH / CSV_FILE)

    metrics_rows = []
    representative_artifacts = None
    pooled_y_true = []
    pooled_y_pred = []
    pooled_std_by_type = {
        'total': [],
        'aleatoric': [],
        'epistemic': [],
    }

    print(f"[*] evaluating BNN uncertainty across {len(EVAL_SEEDS)} random seeds ...")
    for seed in EVAL_SEEDS:
        print(f"[*] evaluating seed {seed} ...")
        seed_rows, artifacts = evaluate_seed(
            X=X,
            y=y,
            df=df,
            seed=seed,
            keep_artifacts=(seed == SEED),
        )
        metrics_rows.extend(seed_rows)
        pooled_y_true.append(artifacts['y_true_test'])
        pooled_y_pred.append(artifacts['y_pred_test'])
        for uncertainty_type, payload in artifacts['uncertainty_payloads'].items():
            pooled_std_by_type[uncertainty_type].append(payload['test_std'])
        if seed == SEED:
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

    pooled_y_true_arr = np.concatenate(pooled_y_true)
    pooled_y_pred_arr = np.concatenate(pooled_y_pred)

    for uncertainty_type, std_chunks in pooled_std_by_type.items():
        make_uncertainty_figures(
            uncertainty_type=uncertainty_type,
            split_name='test',
            y_true=pooled_y_true_arr,
            y_pred=pooled_y_pred_arr,
            y_std=np.concatenate(std_chunks),
        )

    print(f"\nFINISHED UNCERTAINTY EVALUATION. All results saved to {OUT_PATH}\n")


if __name__ == '__main__':
    main()
