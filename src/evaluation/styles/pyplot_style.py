"""
Print-ready matplotlib defaults for journal figures, presentations, etc.

USAGE: 

import plot_style

plot_style.apply("single")   # global default at top of script

# or per-figure:
with plot_style.style("double"):
    fig, ax = plt.subplots()
    ...
    fig.savefig("fig1.pdf")

"""

# plot_style.py — publish-ready matplotlib defaults
import matplotlib as mpl
import matplotlib.pyplot as plt
from contextlib import contextmanager
import copy

# ── Physical sizes (inches) ──────────────────────────────────────────────────
SIZES = {
    "single":  (3.5,   2.625),   # 1-column journal figure
    "double":  (7.0,   2.625),   # 2-column spanning
    "square":  (3.5,   3.5),     # phase diagram etc.
    "wide":    (7.0,   3.5),     # timeseries / spectra
    "slide":   (6.0,   4.0),     # presentation
}

# ── Base rc params ───────────────────────────────────────────────────────────
BASE_RC = {
    # Font
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    # Fontsize for titles, labels, ticks, legend
    "figure.titlesize":   8,  # suptitle
    # Axes
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.title_fontsize": 7,
    # Ticks
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.top":          True,   # mirror ticks
    "ytick.right":        True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Axes / spines
    "axes.linewidth":     0.6,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.grid":          False,
    # Lines / markers
    "lines.linewidth":    1.0,
    "lines.markersize":   3.5,
    "errorbar.capsize":   2.0,
    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "legend.handlelength": 1.5,
    # Figure / saving
    "figure.dpi":         300,   # screen preview
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,    # embed fonts (Type 1 → TrueType)
    "ps.fonttype":        42,
    "svg.fonttype":       "none", # keep text editable in Inkscape
}

# ── Tick-off override dicts ──────────────────────────────────────────────────
TICK_OFF_X = {
    "xtick.bottom": False, "xtick.top": False,
    "xtick.minor.visible": False,
}
TICK_OFF_Y = {
    "ytick.left": False, "ytick.right": False,
    "ytick.minor.visible": False,
}

def apply(size: str = "single") -> None:
    """
    Apply BASE_RC globally + set figure size. Call once at module level.
    
    Size: "single", "double", "square", "wide", or "slide".
    """
    mpl.rcParams.update(BASE_RC)
    w, h = SIZES[size]
    mpl.rcParams["figure.figsize"] = (w, h)


@contextmanager
def style(size: str = "single", no_ticks: bool = False,
          no_xticks: bool = False, no_yticks: bool = False, **overrides):
    """
    Context manager for a single figure with a given size + optional overrides.
    Restores original rcParams on exit.

    Size: "single", "double", "square", "wide", or "slide".

    Usage:
        with plot_style.style("double", **{"lines.linewidth": 1.5}):
            fig, ax = plt.subplots()
            ...
            fig.savefig("fig1.pdf")

        with plot_style.style("square", no_ticks=True):   # hide all ticks
            ...
        with plot_style.style("single", no_xticks=True):  # hide x ticks only
            ...
    """
    assert size in SIZES, f"Invalid size '{size}'! Must be one of: {list(SIZES.keys())}"
    if no_ticks or no_xticks:
        overrides = {**TICK_OFF_X, **overrides}
    if no_ticks or no_yticks:
        overrides = {**TICK_OFF_Y, **overrides}
    saved = copy.deepcopy(mpl.rcParams)
    try:
        mpl.rcParams.update(BASE_RC)
        w, h = SIZES[size]
        mpl.rcParams["figure.figsize"] = (w, h)
        if overrides:
            mpl.rcParams.update(overrides)
        yield
    finally:
        mpl.rcParams.update(saved)
