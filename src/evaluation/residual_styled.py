import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .styles import pyplot_style   # ← your style module


# ── helpers ──────────────────────────────────────────────────────────────────

def _r2_label(y_true, y_pred, tag="") -> str:
    """Generate a formatted R^2 label for plots."""
    if tag == "test": tag = r"$R^2_{test}$"
    elif tag == "train": tag = r"$R^2_{train}$"
    else: tag = r"$R^2$"
    return f"{tag} = {r2_score(y_true, y_pred)*100:.1f}%"

def _scatter_kw(color, marker, alpha=0.6) -> dict:
    """Return shared scatter plot keyword arguments for consistent styling."""
    return dict(s=25, color=color, marker=marker, facecolors="none",
                linewidths=0.8, alpha=alpha, zorder=20, clip_on=False)

def _axis_limits_with_padding(values, pad_ratio=0.03):
    """Return axis limits with a small padding so full markers stay visible."""
    values = np.asarray(values)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span <= 0:
        pad = max(abs(vmin) * pad_ratio, 1.0)
    else:
        pad = span * pad_ratio
    return vmin - pad, vmax + pad

def _save(fig, save_path, show):
    """Save the figure to the specified path and optionally display it."""
    fig.savefig(Path(save_path))   # dpi + bbox from rc
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to: {save_path}")


def get_uncertainty_label(uncertainty_type):
    """Return a formatted label for the given uncertainty type."""
    allowed_types = ["total", "epistemic", "aleatoric"]
    if uncertainty_type not in allowed_types: 
        errMsg = f"Invalid uncertainty type: {uncertainty_type}. " \
                 f"Allowed types: {allowed_types}"
        raise ValueError(errMsg)

    if uncertainty_type == "total":
        return r"Total uncertainty ($\sigma_{total}$)"
    elif uncertainty_type == "epistemic":
        #return r"Epistemic uncertainty ($\sigma_{epistemic}$)"
        return r"Epistemic uncertainty ($\sigma_{e}^2$)"
    elif uncertainty_type == "aleatoric":
        #return r"Aleatoric uncertainty ($\sigma_{aleatoric}$)"
        return r"Aleatoric uncertainty ($\sigma_{a}^2$)"
    


# ── plot_cv_predictions_color_std ─────────────────────────────────────────────

def plot_predictions_color_std(
        y_pred, 
        y_true, 
        std,
        save_path,
        title, 
        uncertainty_type,
        show=False
    ) -> None:
    """
    Plot cross-validation predictions where residuals are colored by uncertainty (std).

    Upper subplot shows residuals colored by the provided standard deviation.
    Lower subplot shows the parity plot (measured vs predicted).
    """

    with pyplot_style.style("square"):
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f"{title} {_r2_label(y_true, y_pred)} | N = {len(y_true)}")

        # upper — residuals coloured by std
        sc = axs[0].scatter(
            y_pred, y_true - y_pred,
            c=std, 
            cmap="viridis",
            edgecolors="#555", linewidths=0.4,
            s=25, alpha=0.6, marker="h", zorder=20
        )
        cbar = fig.colorbar(sc, ax=axs[0], orientation="horizontal",
                            location="top", pad=0.02, aspect=40)
        cbar.set_label(get_uncertainty_label(uncertainty_type), fontsize=8)
        axs[0].axhline(0, color="red", ls="--", alpha=0.6)
        axs[0].set_ylabel("Residual (T)")

        # lower — parity
        lim = max(y_true.max(), y_pred.max()) * 1.2
        axs[1].plot([-lim, lim], [-lim, lim], "r--", alpha=0.6, label="ideal")
        axs[1].scatter(y_pred, y_true, **_scatter_kw(color="k", marker="h", alpha=0.4))
        axs[1].set_xlim(*_axis_limits_with_padding(y_pred))
        axs[1].set_ylim(*_axis_limits_with_padding(y_true))
        axs[1].legend(loc="upper left")
        axs[1].set_ylabel("Measured (T)")
        axs[1].set_xlabel("Predicted (T)")

        fig.align_ylabels(axs)
        fig.tight_layout()
        _save(fig, save_path, show)


# ── plot_cv_predictions_hexbin ────────────────────────────────────────────────

def plot_cv_predictions_hexbin(
        y_pred, 
        y_true, 
        std,
        save_path,
        title, 
        uncertainty_type,
        gridsize=25,
        show=False
    ) -> None:
    """
    Plot cross-validation predictions where the upper subplot is a hexbin of residuals.

    Upper subplot shows a hexbin of residuals vs predicted values.
    Lower subplot shows the parity plot (measured vs predicted).
    """

    with pyplot_style.style("square"):
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f"{title} {_r2_label(y_true, y_pred)} | N = {len(y_true)}")

        # upper — hexbin of residuals
        hb = axs[0].hexbin(
            x=y_pred,         # x-axis: predicted values
            y=y_true-y_pred,  # y-axis: residuals
            C=std,            # color by uncertainty (std)
            gridsize=gridsize, cmap="viridis", mincnt=1, zorder=20
        )
        cbar = fig.colorbar(hb, ax=axs[0], orientation="horizontal",
                            location="top", pad=0.02, aspect=40)
        cbar.set_label(get_uncertainty_label(uncertainty_type), fontsize=8)
        axs[0].axhline(0, color="red", ls="--", alpha=0.6)
        axs[0].set_ylabel("Residual (T)")

        # lower — parity
        lim = max(y_true.max(), y_pred.max()) * 1.2
        axs[1].plot([-lim, lim], [-lim, lim], "r--", alpha=0.6, label="ideal")
        axs[1].scatter(y_pred, y_true, **_scatter_kw(color="k", marker="h", alpha=0.4))
        axs[1].set_xlim(*_axis_limits_with_padding(y_pred))
        axs[1].set_ylim(*_axis_limits_with_padding(y_true))
        axs[1].legend(loc="upper left")
        axs[1].set_ylabel("Measured (T)")
        axs[1].set_xlabel("Predicted (T)")

        fig.align_ylabels(axs)
        fig.tight_layout()
        _save(fig, save_path, show)

