"""Figures for the report/slides, with one consistent (colour-blind safe) style."""
import sys

import matplotlib

if "ipykernel" not in sys.modules:  # scripts write files only; notebooks keep the inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .data import CLASSES  # noqa: E402

# One fixed colour per class (validated categorical palette, never cycled)
CLASS_COLORS = {"Present": "#2a78d6", "Unknown": "#eb6834", "Absent": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _finish(fig, path):
    """Save and close the figure when a path is given (scripts), else return it (notebooks)."""
    if path is None:
        return fig
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(m, title, path=None, ax=None):
    """Confusion matrix in the Challenge convention: rows = predicted, columns = true label."""
    setup_style()
    m = np.asarray(m)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(m, cmap=BLUES, vmin=0, vmax=max(m.max(), 1))
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            dark = m[i, j] > 0.55 * m.max()
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if dark else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    if own and path:
        fig.savefig(path)
        plt.close(fig)
    return ax


def plot_learning_curves(histories, title, path=None):
    """Losses (train/validation) and validation WA per epoch, one thin line per fold."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, h in enumerate(histories):
        lab = k == 0
        ax1.plot(h["epoch"], h["train_loss"], color=SERIES[0], lw=1.5, alpha=0.8, label="Train" if lab else None)
        ax1.plot(h["epoch"], h["val_loss"], color=SERIES[1], lw=1.5, alpha=0.8, label="Validation" if lab else None)
        ax2.plot(h["epoch"], h["val_wa"], color=SERIES[0], lw=1.5, alpha=0.8)
        best = h["val_wa"].idxmax()
        ax2.plot(h.loc[best, "epoch"], h.loc[best, "val_wa"], "o", ms=6, color=SERIES[0], mec=SURFACE, mew=2)
    ax1.set_title("Weighted cross-entropy")
    ax1.set_xlabel("Epoch")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper right")
    ax2.set_title("Validation weighted accuracy (dot = selected epoch)")
    ax2.set_xlabel("Epoch")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path)


def plot_cv_comparison(names, means, stds, path=None, baselines=None):
    """Horizontal bars of the CV weighted accuracy (mean +/- std over folds) of several experiments."""
    setup_style()
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, np.asarray(means)[order], height=0.55, color=SERIES[0])
    ax.errorbar(np.asarray(means)[order], y, xerr=np.asarray(stds)[order], fmt="none", ecolor=INK_2, lw=1, capsize=3)
    for yi, (m, s) in enumerate(zip(np.asarray(means)[order], np.asarray(stds)[order])):
        ax.text(m + s + 0.01, yi, f"{m:.3f}", va="center", color=INK_2, fontsize=9)
    for label, value in (baselines or {}).items():
        ax.axvline(value, color=MUTED, lw=1)
        ax.text(value, len(names) - 0.4, f" {label}", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(y, np.asarray(names)[order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Weighted accuracy (patient level, CV mean ± std)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _finish(fig, path)
