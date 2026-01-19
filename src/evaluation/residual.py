import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



def plot_cv_predictions(y_pred, y_true, std, title, filename, dirname, figsize: tuple=None, std_color="C0"):
    """
    Plots residuals and measured values vs predicted values in two stacked subplots.
    The std parameter is optional and will be plotted as errorbars if provided. 
    """

    # Figure setup
    FIGSIZE = (10, 8)
    if figsize is None:
        figsize = FIGSIZE
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title + f"\n$R^2$ = {r2_score(y_true, y_pred):.2f} | N = {len(y_true)}", fontsize="medium")

    # -------------------- Upper plot --------------------

    residuals = y_true - y_pred
    axs[0].scatter(y_pred, residuals, alpha=0.4, edgecolor='k', facecolor='none', s=20, zorder=20, marker="h")
    axs[0].axhline(0.0, color='red', linestyle='--', lw=1.3, alpha=0.6, label='0.0', zorder=10)
    # looks 
    axs[0].grid(alpha=0.5, zorder=0)
    axs[0].legend(loc='lower left', fontsize=10)
    axs[0].set_ylabel("Residual (T)", fontsize=20)

    # -------------------- Lower plot --------------------

    max_y = max(y_true.max(), y_pred.max()) * 1.1
    axs[1].plot([-max_y, max_y], [-max_y, max_y], 'r--', lw=1.3, alpha=0.6, label="ideal fit", zorder=0)

    axs[1].scatter(y_pred, y_true, alpha=0.4, edgecolor='k', facecolor='none', s=20, marker="h", label="mean", zorder=20)
    if std is not None: 
        axs[1].errorbar(y_pred, y_true, xerr=std, fmt='s', markersize=0, elinewidth=1.6, color=std_color, alpha=0.3, label="$\sigma$", zorder=10)

    axs[1].grid(alpha=0.5, zorder=0)
    axs[1].legend(loc='upper left', fontsize=10)
    axs[1].set_ylabel("Measured (T)", fontsize=20)
    axs[1].set_xlabel("Predicted (T)", fontsize=20)

    # limit axes to min and max of y_true and y_pred
    axs[1].set_xlim(y_pred.min()*0.9, y_pred.max()*1.1)
    axs[1].set_ylim(y_true.min()*0.9, y_true.max()*1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    filepath = Path(dirname) / filename
    plt.savefig(filepath, dpi=600, bbox_inches='tight')  # bbox_inches='tight' to avoid clipping
    plt.show()
    print(f"Saved to dir: {dirname}")



def plot_cv_predictions_color_std(y_pred, y_true, std, title, filename, dirname, details=""):
    """
    Plots residuals and measured values vs predicted values using 5-fold CV.
    The upper plot uses colors to represent std values and includes a colorbar.

    Parameters:
    - y_pred: predicted values (array-like)
    - y_true: true values (array-like)
    - std: standard deviation values for predictions (array-like)
    - title: plot title
    - filename: file name to save the plot
    - dirname: directory name to save the plot
    - figsize: tuple, optional, figure size
    - cmap: colormap for std values in upper plot
    """
    # Figure setup
    #FIGfigsizeSIZE = (10, 8)
    figsize = (8, 7)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    TICKLABELSIZE = 14  
    LABELSIZE = 20 
    TITLESIZE = 20 

    # TITLES
    # small detail line
    details = details + f" | $R^2$ = {r2_score(y_true, y_pred)*100:.1f}% | N = {len(y_true)}\n"
    fig.text(
        0.5, 1, 
        details,
        #transform=axs[0].transAxes,
        fontsize=TITLESIZE/2,
        ha="center", va="bottom"
    )

    # big bold main title below it
    fig.text(
        0.5, .97, 
        title,
        fontsize=TITLESIZE, 
        ha="center", va="bottom"
    )


    # -------------------- Upper plot --------------------

    residuals = y_true - y_pred

    sc = axs[0].scatter(
        y_pred, residuals,
        c=std,
        cmap="viridis_r",
        alpha=0.6,
        edgecolor="#555555",  # dark grey
        linewidth=0.4,   # controls thickness of edge
        s=70,  
        marker="h",
        zorder=20
    )

    # Create a dedicated colorbar axes ABOVE the upper plot
    bbox = axs[0].get_position()
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([bbox.x0    -0.02,       # left
                            bbox.y1    +0.03,       # bottom
                            bbox.width +0.09,       # width
                                        0.05])      # height

    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    TOP = False
    BOTTOM = True
    cbar.ax.tick_params(labeltop=TOP, labelbottom=BOTTOM, top=TOP, bottom=BOTTOM, 
                        labelsize=TICKLABELSIZE,
                        direction="inout", size=5) # hide ticks with size=0
    cbar.set_label(loc="center", label="Uncertainty $\sigma$", 
                   labelpad=-40, fontsize=LABELSIZE*0.75)

    axs[0].axhline(0.0, color='red', linestyle='--', lw=1.3, alpha=0.6, zorder=10)
    axs[0].grid(alpha=0.5, zorder=0)
    axs[0].set_ylabel("Residual (T)", fontsize=20)

    # -------------------- Lower plot --------------------

    max_y = max(y_true.max(), y_pred.max()) * 1.1
    axs[1].plot([-max_y, max_y], [-max_y, max_y], 'r--', lw=1.3, alpha=0.6, label="ideal fit", zorder=0)

    axs[1].scatter(
        y_pred, 
        y_true, 
        alpha=0.4, 
        edgecolor="black",
        linewidth=1.3,   # controls thickness of edge
        facecolor='none', 
        s=50, 
        marker="h", 
        zorder=20
    )
    axs[1].grid(alpha=0.5, zorder=0)
    axs[1].legend(loc='upper left', fontsize=15)
    axs[1].set_ylabel("Measured (T)", fontsize=LABELSIZE)
    axs[1].set_xlabel("Predicted (T)", fontsize=LABELSIZE)

    # -------------------- Final adjustments -------------------

    # limit axes to min and max of y_true and y_pred
    axs[1].set_xlim(y_pred.min()*0.7, y_pred.max()*1.1)
    axs[1].set_ylim(y_true.min()*0.7, y_true.max()*1.1)

    fig.align_ylabels(axs)
    for ax in axs: 
        ax.tick_params(labelsize=TICKLABELSIZE)

    # Adjust for suptitle
    plt.tight_layout(rect=[0,       # left
                           0,       # bottom
                           1,       # right
                           0.88])    # top
    filepath = dirname / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to dir {dirname}")



def residual_plot(
        y_test: np.ndarray,  
        y_pred_test: np.ndarray, 
        output_dir: Path | str = ".",
        plot_name: str = "residual_plot",
        title: str = "residual plot",
        x_label: str = "predicted",
        y_label_upper: str = "residual",
        y_label_lower: str = "measured",
        y_train: np.ndarray=None,
        y_pred_train: np.ndarray=None,
        y_std_test: np.ndarray=None,
        y_std_train: np.ndarray=None
):
    """
    Create two subplots in one figure.
    The upper plot plots the residuals against the predictions.
    The lower plot plots the true values against the predictions.

    :param y_test: true y values of unseen data
    :param y_pred_test: predicted y values on unseen data
    :param output_dir: directory where the plot will be saved
    :param plot_name: the name of the plot
    :param title: The title on the plot
    :param x_label: label for x-axis
    :param y_label_upper: label for upper y-axis
    :param y_label_lower: label for lower y_axis
    :param y_train: true y values of train data set
    :param y_pred_train: predicted y values of train data set
    :param y_std_test: (if you model predicted uncertainty estimates) std on unseen data
    :param y_std_train: (if you model predicted uncertainty estimates) std on train data
    :return:
    """
    # -------------------------------------------------------------------
    # Data analysis
    # -------------------------------------------------------------------
    has_train = (y_train is not None) and (y_pred_train is not None)

    # Determine the min and max values across all points
    y_pred_min = min(y_pred_test.min(), y_pred_train.min()) if has_train else y_pred_test.min()
    y_pred_max = max(y_pred_test.max(), y_pred_train.max()) if has_train else y_pred_test.max()
    y_true_min = min(y_test.min(), y_train.min()) if has_train else y_test.min()
    y_true_max = max(y_test.max(), y_train.max()) if has_train else y_test.max()

    max_val = max(y_true_max, y_pred_max)

    # -------------------------------------------------------------------
    # Figure pre-sets
    # -------------------------------------------------------------------
    orig_size = (8, 6)
    fig, axs = plt.subplots(2, 1, figsize=orig_size, sharex=True)
    try: 
        from sklearn.metrics import r2_score
        r2_test = r2_score(y_test, y_pred_test)
        suptitle = r"$R^2_{test}$" + f" = {r2_test*100:.1f}%"
        if y_train is not None:
            r2_train = r2_score(y_train, y_pred_train)
            suptitle += f", " + r"$R^2_{train}$" + f" = {r2_train*100:.1f}%"
        fig.suptitle(suptitle, fontsize="medium")
    except ImportError:
        print("sklearn not installed, cannot compute R2 score.")
        pass

    s = 16; lw = 2; alpha = 0.6

    # -------------------------------------------------------------------
    # Upper plot
    # -------------------------------------------------------------------
    axs[0].axhline(0.0, color='red', linestyle='--', lw=1.3, alpha=0.6,zorder=10)
    # TEST DATA
    res_test = y_pred_test - y_test
    axs[0].scatter(y_pred_test, res_test, s=s, label=f"test (N={len(res_test)})",
                    color="C0", marker="o", facecolors='none', alpha=alpha, zorder=55)
    # TRAIN DATA
    if y_train is not None:
        res_train = y_pred_train - y_train
        axs[0].scatter(y_pred_train, res_train, s=s, label=f"train (N={len(res_train)})",
                        color="orange", marker="s", facecolors='none', alpha=alpha, zorder=50)
    axs[0].grid(0.3, zorder=0)

    # -------------------------------------------------------------------
    # Lower plot
    # -------------------------------------------------------------------
    axs[1].plot([0, max_val*100], [0, max_val*100], '--', color='red', label="ideal fit", zorder=10)
    # TEST DATA
    axs[1].scatter(y_pred_test, y_test, s=s, label=f"test (N={len(y_test)})", 
                   color="C0", marker="o", facecolors='none', alpha=alpha, zorder=45)
    if y_std_test is not None:
        axs[1].errorbar(y_pred_test, y_test, xerr=y_std_test, fmt='o', markersize=0, 
                        elinewidth=lw, label=f"test std (N={len(y_test)})", color="C0", 
                        alpha=alpha, zorder=40)
    # TRAIN DATA
    if y_train is not None:
        axs[1].scatter(y_pred_train, y_train, s=s, label=f"train (N={len(y_train)})", 
                       color="orange", marker="s", facecolors='none', alpha=alpha, zorder=40)
        if y_std_train is not None:
            axs[1].errorbar(y_pred_train, y_train, xerr=y_std_train, fmt='s', markersize=0, 
                            elinewidth=lw, label=f"train std (N={len(y_train)})", color="orange", 
                            alpha=alpha, zorder=40)
    axs[1].grid(0.3, zorder=0)
    axs[1].legend(loc="upper left", fontsize=12)

    # -------------------------------------------------------------------
    # Axis descriptions
    # -------------------------------------------------------------------
    axs[0].set_title(title, fontsize=14)
    axs[1].set_xlabel(x_label, fontsize=20)
    axs[1].set_ylabel(y_label_lower, fontsize=20)
    axs[0].set_ylabel(y_label_upper, fontsize=20)

    # limit axes to min and max of y_true and y_pred
    axs[1].set_xlim(y_pred_min*0.5, y_pred_max*1.1)
    axs[1].set_ylim(y_true_min*0.5, y_true_max*1.1)

    fig.align_ylabels(axs)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig(output_dir / plot_name, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Saved to dir: {output_dir}")
