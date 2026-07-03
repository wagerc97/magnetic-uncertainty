import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression

from .common import fix_random_seed, validate_dir, validate_metric, validate_step, validate_min_remaining


# =========================================================================================================
# Helper functions
# =========================================================================================================

def _fill_error(i, std, y_test, y_pred, error_list, metric, keep_counts):
    """Fill error list based on selected metric (MAE or MSE)."""
    ranked_confidence_list = np.argsort(std, axis=0).flatten()
    for k, keep_count in enumerate(keep_counts):
        conf = ranked_confidence_list[: keep_count]
        if metric == "mae":
            error = mean_absolute_error(y_test[conf], y_pred[conf])
        elif metric == "mse":
            error = mean_squared_error(y_test[conf], y_pred[conf])
        elif metric == "rmse":
            error = root_mean_squared_error(y_test[conf], y_pred[conf])
        else:
            errMsg = f"Invalid metric given: {metric}"
            raise ValueError(errMsg)
        error_list[i, k] = error


def _add_to_confidence_plot(discard_percentages, y_values_list, label, color, style='-', fill=True):
    """Add Y values to the confidence plot with given style and label."""
    err_mean = np.mean(y_values_list, axis=0)
    err_std = np.std(y_values_list, axis=0)
    lower, upper = err_mean - err_std, err_mean + err_std
    plt.plot(discard_percentages, err_mean, style, label=label, color=color, linewidth=2, zorder=100)
    if fill:
        plt.fill_between(discard_percentages, lower, upper, alpha=0.1, color=color, zorder=90)


def _build_keep_counts(n_samples: int, step: int, min_remaining: int) -> list[int]:
    """Return list of sample counts to keep per iteration (descending)."""
    if min_remaining > n_samples:
        errMsg = f"min_remaining ({min_remaining}) cannot exceed number of test samples ({n_samples})."
        raise ValueError(errMsg)
    keep_counts = list(range(n_samples, min_remaining - 1, -step))
    return keep_counts


def _validate_uncertainty_types(uncertainty_types: list[str]) -> list[str]:
    """Validate uncertainty types while preserving caller-provided order."""
    valid_types = {"total", "aleatoric", "epistemic"}
    invalid_types = [u for u in uncertainty_types if u not in valid_types]
    if invalid_types:
        errMsg = (
            f"Invalid uncertainty type(s): {invalid_types}. "
            f"Valid values are: {sorted(valid_types)}."
        )
        raise ValueError(errMsg)
    if not uncertainty_types:
        errMsg = "At least one uncertainty type must be provided."
        raise ValueError(errMsg)
    if len(set(uncertainty_types)) != len(uncertainty_types):
        errMsg = f"Duplicate uncertainty types are not allowed: {uncertainty_types}."
        raise ValueError(errMsg)
    return uncertainty_types


def _compute_confidences(
        metric: str,
        output_dir: Path,
        plot_name: str,
        title: str,
        y_test_arr: np.ndarray,
        y_pred_arr: np.ndarray,
        uncertainty_arr_by_type: dict[str, np.ndarray],
        plot_linear_fit: bool,
        step: int,
        min_remaining: int,
        test_size: float,
):
    """Compute and plot confidence intervals based on error metric (MAE or MSE)."""
    n_trials, n_samples = y_test_arr.shape
    keep_counts = _build_keep_counts(n_samples, step, min_remaining)
    discard_percentages = 100 * (1 - np.array(keep_counts) / n_samples)
    error_confidence_arr = np.zeros((n_trials, len(keep_counts)))

    if metric == "mae":
        ylabel = "mean absolute error"
    elif metric == "mse":
        ylabel = "mean squared error"
    elif metric == "rmse":
        ylabel = "root mean squared error"
    else:
        errMsg = f"Invalid metric given: {metric}"
        raise ValueError(errMsg)

    # --- PLOTTING ---
    # ideal curve
    for i in range(n_trials):
        _fill_error(i, np.abs(y_test_arr[i] - y_pred_arr[i]), y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric, keep_counts)
    _add_to_confidence_plot(discard_percentages, error_confidence_arr, 'ideal (by true error)', 'red', '--', False)
    y_max = np.max(error_confidence_arr)
    y_mean_total = np.mean(error_confidence_arr, axis=0)

    # baseline
    _add_to_confidence_plot(discard_percentages, np.full((n_trials, len(keep_counts)), y_mean_total[0]), 'baseline', 'gray', ':', False)

    available_types = [u for u in ("epistemic", "aleatoric", "total") if u in uncertainty_arr_by_type]
    plot_types = available_types

    color_by_type = {
        "total": "green",
        "epistemic": "blue",
        "aleatoric": "orange",
    }
    for uncertainty_type in plot_types:
        std_arr = uncertainty_arr_by_type[uncertainty_type]
        for i in range(n_trials):
            _fill_error(i, std_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric, keep_counts)
        _add_to_confidence_plot(discard_percentages, error_confidence_arr, uncertainty_type, color_by_type[uncertainty_type])

    # linear fit
    if plot_linear_fit:
        slope_string = ", slope: "
        slope_list = []
        linear_fit_list = []
        x = discard_percentages
        for i in range(n_trials):
            model = LinearRegression().fit(x.reshape(-1, 1), error_confidence_arr[i])
            slope, intercept = model.coef_[0], model.intercept_
            slope_list.append(slope)
            linear_fit_list.append(slope * x + intercept)

        mean_slope = np.mean(slope_list)
        slope_string += f", {mean_slope:.3f}" if mean_slope > 0 else f"-{mean_slope:.3f}"
        fit_label = plot_types[-1]
        fit_color = color_by_type[fit_label]
        _add_to_confidence_plot(discard_percentages, np.array(linear_fit_list), f'linear fit ({fit_label})', fit_color, fill=False)
    else:
        slope_string = ""

    # --- FINISH ---
    plt.legend(loc="lower left")
    plt.ylabel(ylabel)
    plt.title(title + f"\n(trials: {n_trials}, test_size: {test_size}, step: {step}, min remaining: {min_remaining}{slope_string})", fontsize='small')
    plt.grid(0.3, zorder=0)

    plt.xlabel('% discarded samples')
    plt.xlim([-4, 104])
    plt.ylim([y_max * -0.05, y_max * 1.4])

    plt.tight_layout()
    filepath = output_dir / f'{plot_name}.png'
    plt.savefig(filepath, dpi=600)
    plt.show()
    print(f"Saved to file: {filepath}")


# =========================================================================================================
# Main function
# =========================================================================================================

def plot_confidence_curve(
        regressor: BaseEstimator,
        df: pd.DataFrame,
        features: list,
        label: str,
        uncertainty_types: list[str],
        n_trials: int = 10,
        test_size: float = 0.3,
        plot_linear_fit: bool = False,
        output_dir: Path | str = ".",
        plot_name: str = "confidence_plot",
        title: str = "Confidence plot",
        seed: int = None,
        metric: str = "mae",
        step: int = 5,
        min_remaining: int = 10,
):
    """
    Plot the confidence curves of an sklearn model.
    Can average over `n_trials` repetitions for more robust results.

    About the regression model:
        The machine learning model is expected to predict `y_pred` followed by
        one or more uncertainty arrays.
        Use `uncertainty_types` to declare what those additional arrays represent.
        Example:
            [y_pred, std_total] with uncertainty_types=["total"]
            [y_pred, std_total, std_aleatoric, std_epistemic]
                with uncertainty_types=["total", "aleatoric", "epistemic"]

    :param regressor: an untrained sklearn model
    :param df: a dataframe holding features and labels data
    :param features: column headers for features
    :param label: column header for the label (one single label only)
    :param uncertainty_types: names of uncertainty arrays returned after `y_pred`
    :param n_trials: number of repetitions
    :param plot_linear_fit: if True, plot a linear fit line on top
    :param output_dir: directory to save plot to. Default = "."
    :param plot_name: name of the plot. Default = "confidence_plot"
    :param title: title for plot. Default = "Confidence plot"
    :param seed: random seed. Default = None
    :param metric: error metric for prediction error. Default = "mae" (mean absolute error)
    :param step: number of samples to discard per iteration when building the curve. Default = 5
    :param min_remaining: minimum number of samples that will always be kept. Default = 10
    :return:
    """

    metric = validate_metric(metric)
    output_dir = validate_dir(output_dir)
    n_test_samples = int(df.shape[0] * test_size)
    step = validate_step(step, n_test_samples)
    min_remaining = validate_min_remaining(min_remaining, n_test_samples)

    y_test_list, y_pred_list = [], []
    std_lists_by_type = {
        "total": [],
        "aleatoric": [],
        "epistemic": [],
    }
    active_uncertainty_types = _validate_uncertainty_types(uncertainty_types)

    for i in range(n_trials):
        seed_i = seed + i if seed is not None else None
        fix_random_seed(seed_i)
        reg_clone = clone(regressor)
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df[label], test_size=test_size, random_state=seed_i
        )
        print(f"Trial {i + 1}/{n_trials}: fitting ...")
        reg_clone.fit(X_train, y_train.values)
        result = reg_clone.predict(X_test)

        expected_result_len = 1 + len(active_uncertainty_types)
        if len(result) != expected_result_len:
            errMsg = (
                f"Prediction result length mismatch: expected {expected_result_len} "
                f"(y_pred + {active_uncertainty_types}), got {len(result)}."
            )
            raise ValueError(errMsg)

        y_pred_test, *uncertainty_results = result
        y_pred_list.append(y_pred_test.flatten())
        y_test_list.append(y_test.values.flatten())
        for uncertainty_type, uncertainty_result in zip(active_uncertainty_types, uncertainty_results):
            std_lists_by_type[uncertainty_type].append(uncertainty_result.flatten())

    # Cast to ndarray
    y_test_arr = np.array(y_test_list)
    y_pred_arr = np.array(y_pred_list)
    uncertainty_arr_by_type = {
        uncertainty_type: np.array(std_lists)
        for uncertainty_type, std_lists in std_lists_by_type.items()
        if len(std_lists) > 0
    }

    _compute_confidences(
        metric=metric,
        output_dir=output_dir,
        y_test_arr=y_test_arr, y_pred_arr=y_pred_arr,
        uncertainty_arr_by_type=uncertainty_arr_by_type,
        plot_name=plot_name,
        title=title,
        plot_linear_fit=plot_linear_fit,
        step=step,
        min_remaining=min_remaining,
        test_size=test_size,
    )
