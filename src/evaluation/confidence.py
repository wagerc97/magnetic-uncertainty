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


def _compute_confidences(
        metric: str,
        output_dir: Path,
        plot_name: str,
        title: str,
        y_test_arr: np.ndarray,
        y_pred_arr: np.ndarray,
        std_total_arr: np.ndarray,
        std_al_arr: np.ndarray,
        std_ep_arr: np.ndarray,
        plot_std_total_only: bool,
        plot_linear_fit: bool,
        step: int,
        min_remaining: int,
        test_size: float,
        epistemic_only: bool
):
    """Compute and plot confidence intervals based on error metric (MAE or MSE)."""
    n_trials, n_samples = y_test_arr.shape
    keep_counts = _build_keep_counts(n_samples, step, min_remaining)
    discard_percentages = 100 * (1 - np.array(keep_counts) / n_samples)  #> shape: (n_keep_counts,)
    error_confidence_arr = np.zeros((n_trials, len(keep_counts)))  #> shape: (n_trials, n_keep_counts)

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
 
    if not plot_std_total_only and len(std_al_arr) > 0:
        # epistemic uncertainty 
        for i in range(n_trials):
            _fill_error(i, std_ep_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric, keep_counts)
        _add_to_confidence_plot(discard_percentages, error_confidence_arr, 'epistemic', 'blue')

        # aleatoric uncertainty
        for i in range(n_trials):
            _fill_error(i, std_al_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric, keep_counts)
        _add_to_confidence_plot(discard_percentages, error_confidence_arr, 'aleatoric', 'orange')

    # total uncertainty 
    for i in range(n_trials):
        _fill_error(i, std_total_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric, keep_counts)
    tmp_name = "epistemic" if epistemic_only else "total"
    tmp_color = "blue" if epistemic_only else "green"
    _add_to_confidence_plot(discard_percentages, error_confidence_arr, tmp_name, tmp_color)


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
        _add_to_confidence_plot(discard_percentages, np.array(linear_fit_list), 'linear fit (total)', 'green', fill=False, lw=1.5)
    else:
        slope_string = ""


    # --- FINISH ---
    plt.legend(loc="lower left")
    plt.ylabel(ylabel)
    plt.title(title + f"\n(trials: {n_trials}, test_size: {test_size}, step: {step}, min remaining: {min_remaining}{slope_string})", fontsize='small')
    plt.grid(0.3, zorder=0)

    plt.xlabel('% discarded samples')
    plt.xlim([-4, 104])

    plt.tight_layout()
    filepath = output_dir / f'{plot_name}.png'
    plt.savefig(filepath, dpi=600)
    plt.show()
    print(f"Saved to file: {filepath}")


# =========================================================================================================
# Main function
# =========================================================================================================

def plot_confidence(
        regressor: BaseEstimator,
        df: pd.DataFrame,
        features: list,
        label: str,
        n_trials: int = 5,
        test_size: float = 0.5,
        plot_std_total_only: bool = False,
        plot_linear_fit: bool = False,
        output_dir: Path | str = ".",
        plot_name: str = "confidence_plot",
        title: str = "Confidence plot",
        seed: int = None,
        metric: str = "mae",
        step: int = 1,
        min_remaining: int = 1,
):
    """
    Plot the confidence curves of an sklearn model.
    Can average over `n_trials` repetitions for more robust results.

    About the regression model: 
        The machine learning model is expected to predict 2 or 4 values.
        2 values: [y_pred, std_total]
        2 values: [y_pred, std_total, std_aleatoric, std_epistemic]

    :param regressor: an untrained sklearn model
    :param df: a dataframe holding features and labels data
    :param features: column headers for features
    :param label: column header for the label (one single label only)
    :param n_trials: number of repetitions
    :param plot_std_total_only: if True, plot only the total standard deviation and not aleatoric & epistemic
    :param plot_linear_fit: if True, plot a linear fit line on top
    :param output_dir: directory to save plot to. Default = "."
    :param plot_name: name of the plot. Default = "confidence_plot"
    :param title: title for plot. Default = "Confidence plot"
    :param seed: random seed. Default = None
    :param metric: error metric for prediction error. Default = "mae" (mean absolute error)
    :param step: number of samples to discard per iteration when building the curve. Default = 1
    :param min_remaining: minimum number of samples that will always be kept. Default = 1
    :return:
    """

    metric = validate_metric(metric)
    output_dir = validate_dir(output_dir)
    n_test_samples = int(df.shape[0] * test_size)
    step = validate_step(step, n_test_samples)
    min_remaining = validate_min_remaining(min_remaining, n_test_samples)


    y_test_list, y_pred_list, std_list, std_al_list, std_ep_list = [], [], [], [], []
    epistemic_only = True

    for i in range(n_trials):
        fix_random_seed(seed + i if seed is not None else None)
        reg_clone = clone(regressor)
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=test_size, random_state=seed)
        print(f"Trial {i + 1}/{n_trials}: fitting ...")
        reg_clone.fit(X_train, y_train.values)
        result = reg_clone.predict(X_test)

        if len(result) == 2:
            y_pred_test, std_total = result
            y_pred_list.append(y_pred_test.flatten())
            y_test_list.append(y_test.values.flatten())
            std_list.append(std_total.flatten())
            epistemic_only=True

        elif len(result) == 4:
            y_pred_test, std_total, std_al, std_ep = result
            y_pred_list.append(y_pred_test.flatten())
            y_test_list.append(y_test.values.flatten())
            std_list.append(std_total.flatten())
            std_al_list.append(std_al.flatten())
            std_ep_list.append(std_ep.flatten())
            epistemic_only=False
        else:
            errMsg = f"Found prediction result of len={len(result)}! Unknown length! Valid length is 2 or 4 ({len(result)}).\nresult={result}"
            raise ValueError(errMsg)

        # Append values

    # Cast to ndarray
    y_test_arr = np.array(y_test_list)
    y_pred_arr = np.array(y_pred_list)
    std_total_arr = np.array(std_list)
    std_al_arr = np.array(std_al_list)
    std_ep_arr = np.array(std_ep_list)

    _compute_confidences(
        metric=metric,
        output_dir=output_dir,
        y_test_arr=y_test_arr, y_pred_arr=y_pred_arr,
        std_total_arr=std_total_arr, std_al_arr=std_al_arr, std_ep_arr=std_ep_arr,
        plot_name=plot_name,
        title=title,
        plot_std_total_only=plot_std_total_only,
        plot_linear_fit=plot_linear_fit,
        step=step,
        min_remaining=min_remaining,
        test_size=test_size,
        epistemic_only=epistemic_only
    )
