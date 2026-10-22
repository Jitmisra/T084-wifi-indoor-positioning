"""All figures of the project in one shared style, saved as PNG (slides) and PDF (report)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from .config import FILL_VALUE, FLOORS_PER_BUILDING, MODEL_NAMES, WAP_COLS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # easy to tell apart, also for color blindness
INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#b5b3ab"
# One-hue scales from light (few scans, small error) to dark (many scans, large error)
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
ERROR_COLORS = LinearSegmentedColormap.from_list("errors", ["#86b6ef", "#2a78d6", "#0d366b"])
COLUMN, PAGE = 3.5, 7.16  # width of one column and of the whole page in the report (inches)
STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed fonts as TrueType, as PDF checks for papers require
}


def _figure(width: float, height: float, ncols: int = 1):
    plt.rcParams.update(STYLE)
    return plt.subplots(1, ncols, figsize=(width, height))


def save(fig, name: str, folder) -> None:
    """Save a figure as <folder>/<name>.png and <name>.pdf, then close it."""
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None} if ext == "pdf" else None  # same file on every re-run
        fig.savefig(Path(folder) / f"{name}.{ext}", metadata=metadata)
    plt.close(fig)


def _write_values(ax, x, heights, values, digits: int) -> None:
    """Write each value just above the given height (the top of a bar or of its error bar)."""
    for xi, height, value in zip(x, heights, values, strict=True):
        text = f"{value:.{digits}f}"
        ax.annotate(
            text,
            (xi, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=MUTED,
        )
    ax.margins(y=0.12)  # room for the values above the highest bar


def campus_map(train: pd.DataFrame, test: pd.DataFrame):
    """Where the scans were taken: training positions by building, test positions as crosses."""
    fig, ax = _figure(COLUMN, 2.8)
    places = train.drop_duplicates(["BUILDINGID", "LONGITUDE", "LATITUDE"])
    for building, color in zip(FLOORS_PER_BUILDING, (BLUE, ORANGE, AQUA), strict=True):
        rows = places[places["BUILDINGID"] == building]
        label = f"Building {building} (training)"
        ax.scatter(rows["LONGITUDE"], rows["LATITUDE"], s=6, color=color, lw=0, label=label)
        top = (rows["LONGITUDE"].mean(), rows["LATITUDE"].max())
        ax.annotate(f"B{building}", top, xytext=(0, 4), textcoords="offset points", ha="center")
    xy = test["LONGITUDE"], test["LATITUDE"]
    ax.scatter(*xy, s=3, marker="x", color=INK, lw=0.3, alpha=0.7, label="Test scans")
    ax.set(xlabel="LONGITUDE (m)", ylabel="LATITUDE (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="lower left", markerscale=2)
    return fig


def waps_per_scan(train: pd.DataFrame, test: pd.DataFrame):
    """How many WAPs one scan hears, for the training and the test scans."""
    fig, ax = _figure(COLUMN, 2.2)
    n_heard = [(df[WAP_COLS] > FILL_VALUE).sum(axis=1) for df in (train, test)]
    bins = np.arange(0, max(n.max() for n in n_heard) + 3, 2)
    for name, n, color in zip(("Training", "Test"), n_heard, (BLUE, ORANGE), strict=True):
        share = np.full(len(n), 100 / len(n))  # each scan counts as its share of all scans (%)
        label = f"{name} (mean {n.mean():.1f})"
        ax.hist(n, bins=bins, weights=share, histtype="step", lw=1.4, color=color, label=label)
    ax.set(xlabel="WAPs heard in one scan", ylabel="Share of scans (%)")
    ax.legend()
    return fig


def scans_per_floor(train: pd.DataFrame, test: pd.DataFrame):
    """Share of the training and the test scans on each floor of each building."""
    fig, ax = _figure(COLUMN, 2.3)
    classes = [(b, f) for b, n_floors in FLOORS_PER_BUILDING.items() for f in range(n_floors)]
    x = np.arange(len(classes))
    for shift, name, df, color in ((-0.19, "Training", train, BLUE), (0.19, "Test", test, ORANGE)):
        counts = df.groupby(["BUILDINGID", "FLOOR"]).size().reindex(classes, fill_value=0)
        ax.bar(x + shift, 100 * counts / len(df), width=0.36, color=color, label=name)
    ax.set_xticks(x, [f"B{b}-F{f}" for b, f in classes], rotation=90)
    ax.set(xlabel="Building and floor", ylabel="Share of scans (%)")
    ax.legend()
    return fig


def feature_check(table: pd.DataFrame):
    """CV floor accuracy and mean 2D error of the Baseline RF settings on each feature set."""
    fig, axes = _figure(COLUMN, 2.4, ncols=2)
    x = np.arange(len(table))
    labels = [name.replace("+", "\n+") for name in table["feature_set"]]
    panels = (("floor_acc", "Floor accuracy (CV)", 3), ("mean_2d", "Mean 2D error (m, CV)", 1))
    for ax, (metric, ylabel, digits) in zip(axes, panels, strict=True):
        ax.bar(x, table[metric], width=0.6, color=BLUE)
        spread = table[f"{metric}_std"]  # standard deviation over the folds
        ax.errorbar(x, table[metric], yerr=spread, fmt="none", ecolor=MUTED, lw=0.6, capsize=2)
        _write_values(ax, x, table[metric] + spread, table[metric], digits)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
    return fig


def model_comparison(cv_results: pd.DataFrame, test_results: pd.DataFrame):
    """Floor accuracy and mean 2D error of every model: user-group CV next to the test set."""
    fig, axes = _figure(PAGE, 2.3, ncols=2)
    x = np.arange(len(cv_results))
    names = [MODEL_NAMES[key] for key in cv_results["model"]]
    panels = (("floor_acc", "Floor accuracy", 3), ("mean_2d", "Mean 2D error (m)", 2))
    for ax, (metric, ylabel, digits) in zip(axes, panels, strict=True):
        ax.bar(x - 0.19, cv_results[metric], width=0.36, color=BLUE, label="User-group CV")
        ax.bar(x + 0.19, test_results[metric], width=0.36, color=ORANGE, label="Test set")
        _write_values(ax, x + 0.19, test_results[metric], test_results[metric], digits)
        ax.set_xticks(x, names)
        ax.set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    return fig


def floor_confusion(table: pd.DataFrame):
    """Floor confusion matrices of the final model, one per building; colors show row shares."""
    fig, axes = _figure(PAGE, 2.3, ncols=len(FLOORS_PER_BUILDING))
    for ax, (building, rows) in zip(axes, table.groupby("building"), strict=True):
        counts = rows.filter(like="pred_floor_").to_numpy()
        shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(shares, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for (i, j), count in np.ndenumerate(counts):  # every cell: row i, column j
            if count:
                color = "white" if shares[i, j] > 0.6 else INK
                ax.text(j, i, count, ha="center", va="center", fontsize=6.5, color=color)
        ax.set_xticks(range(counts.shape[1]))
        ax.set_yticks(range(len(rows)), rows["true_floor"])
        ax.set(xlabel="Predicted floor", title=f"Building {building} ({counts.sum()} test scans)")
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("True floor")
    fig.colorbar(image, ax=list(axes), label="Share of the true floor's scans", shrink=0.8)
    return fig


def error_map(predictions: pd.DataFrame):
    """True test positions (gray) and predicted positions colored by their 2D error, per building.

    Each panel shows meters east and north of the building's south-west corner.
    """
    fig, axes = _figure(PAGE, 2.7, ncols=len(FLOORS_PER_BUILDING))
    vmax = np.percentile(predictions["err_2d"], 95)  # Why: a few huge errors would hide the rest
    for ax, building in zip(axes, FLOORS_PER_BUILDING, strict=True):
        rows = predictions[predictions["true_BUILDINGID"] == building]
        x0, y0 = rows["true_LONGITUDE"].min(), rows["true_LATITUDE"].min()
        true_x, true_y = rows["true_LONGITUDE"] - x0, rows["true_LATITUDE"] - y0
        ax.scatter(true_x, true_y, s=4, color=GRAY, lw=0, label="True position")
        points = ax.scatter(
            rows["pred_LONGITUDE"] - x0,
            rows["pred_LATITUDE"] - y0,
            s=4,
            c=rows["err_2d"],
            cmap=ERROR_COLORS,
            vmin=0,
            vmax=vmax,
            lw=0,
            label="Prediction (color = 2D error)",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="Meters east", title=f"Building {building}")
    axes[0].set_ylabel("Meters north")
    axes[0].legend(loc="upper left", markerscale=2)
    fig.colorbar(points, ax=list(axes), label="2D error (m)", extend="max", shrink=0.8)
    return fig
