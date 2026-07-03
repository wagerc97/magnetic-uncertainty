import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from .styles import pyplot_style
from .common import validate_metric, validate_step, validate_min_remaining, fix_random_seed


# ── helpers ──────────────────────────────────────────────────────────────────

def _fill_error(i, std, y_test, y_pred, error_list, metric, keep_counts):
    """Fill error list based on selected metric (MAE, MSE, or RMSE)."""
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
            raise ValueError(f"Invalid metric: {metric}")
        error_list[i, k] = error


def _add_to_confidence_plot(ax, discard_percentages, y_values_list, label, color, style='-', fill=True):
    """Add Y values to the confidence plot with given style and label."""
    err_mean = np.mean(y_values_list, axis=0)
    err_std = np.std(y_values_list, axis=0)
    lower, upper = err_mean - err_std, err_mean + err_std

    ax.plot(discard_percentages, err_mean, style, label=label, color=color, linewidth=2, zorder=100)
    if fill:
        ax.fill_between(discard_percentages, lower, upper, alpha=0.1, color=color, zorder=90)


def _build_keep_counts(n_samples: int, step: int, min_remaining: int) -> list[int]:
    """Return list of sample counts to keep per iteration (descending)."""
    if min_remaining > n_samples:
        errMsg = f"min_remaining ({min_remaining}) cannot exceed number of test samples ({n_samples})."
        raise ValueError(errMsg)
    keep_counts = list(range(n_samples, min_remaining - 1, -step))
    return keep_counts


def _save(fig, save_path, show):
    """Save the figure to the specified path and optionally display it."""
    fig.savefig(Path(save_path))
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to: {save_path}")


def get_metric_label(metric):
    """Return a formatted label for the given error metric."""
    if metric == "mae":
        return "Mean Absolute Error (T)"
    elif metric == "mse":
        return r"Mean Squared Error ($T^2$)"
    elif metric == "rmse":
        return "Root Mean Squared Error (T)"
    return metric


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


# ── plot_confidence ──────────────────────────────────────────────────────────


def _plot_confidence(
        y_test_arr: np.ndarray,
        y_pred_arr: np.ndarray,
        uncertainty_arr_by_type: dict[str, np.ndarray],
        save_path: str | Path | None = None,
        title: str = "Confidence Plot",
        metric: str = "mae",
        step: int = 1,
        min_remaining: int = 1,
        plot_linear_fit: bool = False,
        show: bool = False
):
    """
    Compute and plot confidence curves based on error metric.

    This function expects arrays of shape (n_trials, n_samples).
    """
    metric = validate_metric(metric)

    if save_path is None:
        raise ValueError("save_path is required.")
    n_trials, n_samples = y_test_arr.shape
    step = validate_step(step, n_samples)
    min_remaining = validate_min_remaining(min_remaining, n_samples)

    keep_counts = _build_keep_counts(n_samples, step, min_remaining)
    discard_percentages = 100 * (1 - np.array(keep_counts) / n_samples)
    error_confidence_arr = np.zeros((n_trials, len(keep_counts)))

    ylabel = get_metric_label(metric)

    with pyplot_style.style("single"):
        fig, ax = plt.subplots()

        # 1. Ideal curve (by true error)
        for i in range(n_trials):
            _fill_error(i, np.abs(y_test_arr[i] - y_pred_arr[i]),
                        y_test_arr[i], y_pred_arr[i],
                        error_confidence_arr, metric, keep_counts)
        _add_to_confidence_plot(ax, discard_percentages, error_confidence_arr,
                                'ideal (by true error)', 'red', '--', fill=False)

        y_max = np.max(error_confidence_arr)
        y_mean_total = np.mean(error_confidence_arr, axis=0)

        # 2. Baseline
        baseline_vals = np.full((n_trials, len(keep_counts)), y_mean_total[0])
        _add_to_confidence_plot(ax, discard_percentages, baseline_vals,
                                'baseline', 'gray', ':', fill=False)

        # 3. Model-provided uncertainty estimates
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
                _fill_error(i, std_arr[i], y_test_arr[i], y_pred_arr[i],
                            error_confidence_arr, metric, keep_counts)
            _add_to_confidence_plot(
                ax,
                discard_percentages,
                error_confidence_arr,
                uncertainty_type,
                color_by_type[uncertainty_type],
            )

        # 4. Linear fit
        slope_string = ""
        if plot_linear_fit:
            slope_list = []
            linear_fit_list = []
            x = discard_percentages
            for i in range(n_trials):
                model = LinearRegression().fit(x.reshape(-1, 1), error_confidence_arr[i])
                slope, intercept = model.coef_[0], model.intercept_
                slope_list.append(slope)
                linear_fit_list.append(slope * x + intercept)

            mean_slope = np.mean(slope_list)
            slope_string = f", slope: {mean_slope:.3f}"
            fit_label = plot_types[-1]
            fit_color = color_by_type[fit_label]
            _add_to_confidence_plot(ax, discard_percentages, np.array(linear_fit_list),
                                    f'linear fit ({fit_label})', fit_color, style='-', fill=False)

        # --- Finalize ---
        ax.set_title(f"{title}\n"
                     f"(trials: {n_trials}, step: {step}, min rem: {min_remaining}{slope_string})",
                     fontsize='small')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('% discarded samples')
        ax.set_xlim([-4, 104])
        ax.set_ylim([y_max * -0.05, y_max * 1.4])
        ax.legend(loc="lower left", fontsize='x-small', frameon=True)
        ax.grid(True, alpha=0.3, zorder=0)

        fig.tight_layout()
        _save(fig, save_path, show)


# ── main function ───────────────────────────────────────────────────────────

def plot_confidence_curve(
        regressor: BaseEstimator | None = None,
        df: pd.DataFrame | None = None,
        features: list | None = None,
        label: str | None = None,
        save_path: str | Path | None = None,
        uncertainty_types: list[str] | None = None,
        plot_linear_fit: bool = False,
        title: str = "Confidence plot",
        seed: int = None,
        metric: str = "mae",
        n_trials: int = 10,
        test_size: float = 0.3,
        step: int = 5,
        min_remaining: int = 10,
        show: bool = False,
        y_test_arr: np.ndarray | None = None,
        y_pred_arr: np.ndarray | None = None,
        uncertainty_arr_by_type: dict[str, np.ndarray] | None = None,
):
    """
    Compute the confidence curves of an sklearn model.
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
    :param save_path: path to save the plot. Default = "."
    :param title: title for plot. Default = "Confidence plot"
    :param seed: random seed. Default = None
    :param metric: error metric for prediction error. Default = "mae" (mean absolute error)
    :param step: number of samples to discard per iteration when building the curve. Default = 5
    :param min_remaining: minimum number of samples that will always be kept. Default = 10
    :return:
    """

    metric = validate_metric(metric)

    if y_test_arr is not None or y_pred_arr is not None or uncertainty_arr_by_type is not None:
        if y_test_arr is None or y_pred_arr is None or uncertainty_arr_by_type is None:
            raise ValueError(
                "y_test_arr, y_pred_arr, and uncertainty_arr_by_type must all be provided together."
            )
        y_test_arr = np.asarray(y_test_arr)
        y_pred_arr = np.asarray(y_pred_arr)
        if uncertainty_types is None:
            uncertainty_types = list(uncertainty_arr_by_type.keys())
        active_uncertainty_types = _validate_uncertainty_types(uncertainty_types)
        filtered_uncertainty = {
            uncertainty_type: np.asarray(uncertainty_arr_by_type[uncertainty_type])
            for uncertainty_type in active_uncertainty_types
            if uncertainty_type in uncertainty_arr_by_type
        }
        _plot_confidence(
            y_test_arr=y_test_arr,
            y_pred_arr=y_pred_arr,
            uncertainty_arr_by_type=filtered_uncertainty,
            save_path=save_path,
            title=title,
            metric=metric,
            step=step,
            min_remaining=min_remaining,
            plot_linear_fit=plot_linear_fit,
            show=show,
        )
        return

    if regressor is None or df is None or features is None or label is None or uncertainty_types is None:
        raise ValueError(
            "regressor, df, features, label, and uncertainty_types are required when plotting from model fits."
        )

    n_test_samples = int(df.shape[0] * test_size)
    step = validate_step(step, n_test_samples)
    min_remaining = validate_min_remaining(min_remaining, n_test_samples)

    # Initialize lists to hold results across trials
    y_test_list, y_pred_list = [], []
    std_lists_by_type = {
        "total": [],
        "aleatoric": [],
        "epistemic": [],
    }
    active_uncertainty_types = _validate_uncertainty_types(uncertainty_types)

    # Run trials
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

    # Cast to numpy arrays for plotting
    y_test_arr = np.array(y_test_list)
    y_pred_arr = np.array(y_pred_list)
    uncertainty_arr_by_type = {
        uncertainty_type: np.array(std_lists)
        for uncertainty_type, std_lists in std_lists_by_type.items()
        if len(std_lists) > 0
    }

    # Call plot function
    _plot_confidence(
        y_test_arr=y_test_arr,
        y_pred_arr=y_pred_arr,
        uncertainty_arr_by_type=uncertainty_arr_by_type,
        save_path=save_path,
        title=title,
        metric=metric,
        step=step,
        min_remaining=min_remaining,
        plot_linear_fit=plot_linear_fit,
        show=show
    )
