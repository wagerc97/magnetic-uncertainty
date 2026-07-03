#!/usr/bin/env python
# coding: utf-8

# # Train new Anisotropy model with GPR
#
# Requires environment defined in /environment.yml

import os
import random
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
print(f"Project root added to python path:\n{_root}")
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
TEST_SIZE = 0.3
np.random.seed(SEED)
random.seed(SEED)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_and_preprocess(data_path):
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


def plot_histogram(df, filename=None):
    from src.evaluation.styles.pyplot_style import style

    with style(size="single"):
        if filename is None:
            filename = f"{LABEL}_distribution.png"
        plt.hist(df[LABEL], bins=15, edgecolor='black', zorder=50)
        plt.title(f"Distribution of {SYMBOL} in the dataset (N={len(df)})")
        plt.xlabel(f"{SYMBOL} ({UNIT})")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75, zorder=0)
        plt.savefig(OUT_PATH / filename)
        plt.close()


class CustomGPR(GaussianProcessRegressor):
    """Custom GaussianProcessRegressor that always predicts with std."""

    def predict(self, X, return_std=True, return_cov=False):
        return super().predict(X, return_std=return_std, return_cov=return_cov)



def build_regressor(seed):
    return CustomGPR(
        kernel=(
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e2))
        ),
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=seed,
    )



def split_and_scale_data(X, y, df, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, shuffle=True
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train.copy(),
        "X_test_processed": X_test.copy(),
        "df_scaled": df.copy(),
    }



def evaluate_seed(X, y, df, seed, keep_artifacts=False):
    np.random.seed(seed)
    random.seed(seed)

    split_data = split_and_scale_data(X=X, y=y, df=df, seed=seed)
    regressor = build_regressor(seed)
    regressor.fit(split_data["X_train_processed"], split_data["y_train"])

    y_pred_train, y_std_train = regressor.predict(split_data["X_train_processed"])
    y_pred_test, y_std_test = regressor.predict(split_data["X_test_processed"])

    metrics_rows = []
    for split_name, y_true, y_pred in (
        ("train", split_data["y_train"], y_pred_train),
        ("test", split_data["y_test"], y_pred_test),
    ):
        metrics_rows.append({
            "seed": seed,
            "split": split_name,
            "n_samples": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
        })

    artifacts = None
    if keep_artifacts:
        artifacts = {
            **split_data,
            "regressor": regressor,
            "y_pred_train": np.asarray(y_pred_train).flatten(),
            "y_std_train": np.asarray(y_std_train).flatten(),
            "y_pred_test": np.asarray(y_pred_test).flatten(),
            "y_std_test": np.asarray(y_std_test).flatten(),
        }

    return metrics_rows, artifacts



def write_model_summary_to_file(regressor, model_summary_path, metrics_by_seed_df, metrics_summary_df):
    gpr_step = regressor.named_steps["gpr"] if hasattr(regressor, "named_steps") else regressor

    lines = [
        "Model Summary",
        "=" * 80,
        "GaussianProcessRegressor Summary",
        "*" * 80,
        f"Representative seed for plots/model export: {SEED}",
        f"Evaluation seeds for reported metrics: {EVAL_SEEDS}",
        f"Initial Kernel: {gpr_step.kernel}",
    ]

    if hasattr(gpr_step, "kernel_"):
        lines.append(f"Optimized Kernel: {gpr_step.kernel_}")
        lines.append(f"Log-Marginal Likelihood: {gpr_step.log_marginal_likelihood_value_:.3f}")

    lines.extend([
        f"Optimizer: {gpr_step.optimizer}",
        f"Alpha: {gpr_step.alpha}",
        f"Normalize_y: {gpr_step.normalize_y}",
        f"Number of restarts: {gpr_step.n_restarts_optimizer}",
        "",
        "Reported metrics are averaged over repeated train/test splits and refits.",
        "",
        "Per-seed metrics:",
        metrics_by_seed_df.to_string(index=False),
        "",
        "Mean and standard deviation across seeds:",
        metrics_summary_df.to_string(index=False),
    ])

    summary = "\n".join(lines)
    print(summary)
    model_summary_path.write_text(summary)



def main():
    df, X, y = load_and_preprocess(DATA_PATH / CSV_FILE)
    plot_histogram(df, filename=f"{LABEL}_distribution.png")

    metrics_rows = []
    representative_artifacts = None

    print(f"[*] evaluating GPR across {len(EVAL_SEEDS)} random seeds ...")
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
    metrics_by_seed_path = OUT_PATH / "metrics_by_seed.csv"
    metrics_by_seed_df.to_csv(metrics_by_seed_path, index=False)

    metrics_summary_df = (
        metrics_by_seed_df
        .groupby("split", as_index=False)
        .agg(
            n_samples_mean=("n_samples", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    metrics_summary_path = OUT_PATH / "metrics_summary_across_seeds.csv"
    metrics_summary_df.to_csv(metrics_summary_path, index=False)

    if representative_artifacts is None:
        raise RuntimeError(f"Representative artifacts for seed {SEED} were not captured.")

    regressor = representative_artifacts["regressor"]
    X_train = representative_artifacts["X_train"]
    y_train = representative_artifacts["y_train"]
    y_test = representative_artifacts["y_test"]
    y_pred_train = representative_artifacts["y_pred_train"]
    y_std_train = representative_artifacts["y_std_train"]
    y_pred_test = representative_artifacts["y_pred_test"]
    y_std_test = representative_artifacts["y_std_test"]
    df_scaled = representative_artifacts["df_scaled"]

    summary_path = OUT_PATH / "model_summary.txt"
    write_model_summary_to_file(
        regressor=regressor,
        model_summary_path=summary_path,
        metrics_by_seed_df=metrics_by_seed_df,
        metrics_summary_df=metrics_summary_df,
    )

    train_summary = metrics_summary_df.loc[metrics_summary_df["split"] == "train"].iloc[0]
    test_summary = metrics_summary_df.loc[metrics_summary_df["split"] == "test"].iloc[0]
    print("\nTrain set performance across seeds:")
    print(
        f" MAE: {train_summary['MAE_mean']:.4f} ± {train_summary['MAE_std']:.4f},"
        f" MSE: {train_summary['MSE_mean']:.4f} ± {train_summary['MSE_std']:.4f},"
        f" R2: {train_summary['R2_mean']:.4f} ± {train_summary['R2_std']:.4f}"
    )
    print("Test set performance across seeds:")
    print(
        f" MAE: {test_summary['MAE_mean']:.4f} ± {test_summary['MAE_std']:.4f},"
        f" MSE: {test_summary['MSE_mean']:.4f} ± {test_summary['MSE_std']:.4f},"
        f" R2: {test_summary['R2_mean']:.4f} ± {test_summary['R2_std']:.4f}"
    )
    print(f"\nRepresentative seed for plots/model export: {SEED}")
    print(f"Initial: {regressor.kernel}")
    print(f"Optimum: {regressor.kernel_}")
    print(f"Log-Marginal-Likelihood: {regressor.log_marginal_likelihood(regressor.kernel_.theta)}")
    print()

    joblib.dump(regressor, OUT_PATH / "model.joblib")

    from src.evaluation.residual_styled import plot_cv_predictions_hexbin, plot_predictions_color_std

    title = f"Parity plot train-set ({SYMBOL})"
    plot_predictions_color_std(
        y_pred=y_pred_train,
        y_true=y_train,
        std=y_std_train,
        title=title,
        uncertainty_type="total",
        save_path=OUT_PATH / f'{LABEL}_train_total_unc',
    )
    title = f"Parity plot test-set ({SYMBOL})"
    plot_predictions_color_std(
        y_pred=y_pred_test,
        y_true=y_test,
        std=y_std_test,
        title=title,
        uncertainty_type="total",
        save_path=OUT_PATH / f'{LABEL}_test_total_unc',
    )

    title = f"Parity plot train-set ({SYMBOL})"
    plot_cv_predictions_hexbin(
        y_pred=y_pred_train,
        y_true=y_train,
        std=y_std_train,
        title=title,
        uncertainty_type="total",
        save_path=OUT_PATH / f'{LABEL}_train_total_unc_hexbin',
    )

    title = f"Parity plot test-set ({SYMBOL})"
    plot_cv_predictions_hexbin(
        y_pred=y_pred_test,
        y_true=y_test,
        std=y_std_test,
        title=title,
        uncertainty_type="total",
        save_path=OUT_PATH / f'{LABEL}_test_total_unc_hexbin',
    )

    from src.evaluation.confidence_styled import plot_confidence_curve

    plot_confidence_curve(
        regressor=regressor,
        df=df_scaled,
        features=X_train.columns.tolist(),
        label="TC",
        save_path=OUT_PATH / f'{LABEL}_test_confidence_curve.png',
        title=f"Confidence curve of GPR model on test-set ({SYMBOL})",
        seed=SEED,
        uncertainty_types=["total"],
        step=10,
    )

    print(f"\nFINISHED TRAINING AND EVALUATION. All results saved to {OUT_PATH}\n")


if __name__ == '__main__':
    main()
