import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Arial itself isn't installed on Linux (proprietary); Liberation Sans is
# metric-compatible with it (same glyph widths/spacing, so layouts built
# against one match the other) and is what's actually available here.
matplotlib.rcParams["font.family"] = "Liberation Sans"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "pruning")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# One chart showing every pruning result at once: accuracy vs. percent of
# tiles kept, one panel per dataset (MNIST and CIFAR-10 need separate y-axes
# - MNIST lives in the 99%+ range and CIFAR-10 in the 55-85% range, so a
# shared axis would flatten MNIST's real-but-small degradation to an
# invisible sliver; see dataviz's "two measures of different scale -> small
# multiples" rule). Within each panel, "zero-pruned" and "noise-pruned" are
# lines across percent_kept = 100 (the un-pruned baseline, at which both
# conditions are identical by construction - pruning 0% is a no-op
# regardless of mode) -> 75 -> 50, so the two conditions' lines visibly
# diverge as more tiles get pruned - the actual "how far can we crank it"
# question this sweep was run to answer.
#
# The "original" (100%-kept) value plotted isn't from a single run: the
# original condition doesn't depend on percent_kept at all (no pixels are
# touched), but it was still retrained independently once per percent_kept
# invocation (train_pruned_classifiers.py always trains all 3 conditions
# fresh), so there are two slightly different measured values from random
# init/data-order noise alone. Averaging them for the shared x=100 anchor
# is more honest than arbitrarily picking one.

DATASETS = ["mnist", "cifar10"]
DATASET_LABELS = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
PERCENTS = [0.75, 0.5]
CONDITION_COLORS = {"original": "#2a78d6", "zero": "#1baf7a", "noise": "#eda100"}
CONDITION_LABELS = {"zero": "Zero-pruned", "noise": "Noise-pruned"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load_metrics(image_type, condition, percent_kept):
    tag = f"{image_type}_{condition}_p{int(round(percent_kept * 100))}"
    path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    with open(path) as handle:
        return json.load(handle)


def main():
    (fig, axes_row) = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 3.8))

    for (axes, image_type) in zip(axes_row, DATASETS):
        original_accs = [load_metrics(image_type, "original", p)["test_accuracy"] for p in PERCENTS]
        original_avg = sum(original_accs) / len(original_accs)

        x_values = [100] + [int(round(p * 100)) for p in PERCENTS]
        axes.plot(100, original_avg * 100, marker="o", markersize=7, color=CONDITION_COLORS["original"],
                   linestyle="none", zorder=3)
        axes.annotate(f"{original_avg * 100:.2f}%", (100, original_avg * 100),
                       xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=TEXT_PRIMARY)

        series_y = {}
        for condition in ("zero", "noise"):
            accs = [original_avg] + [load_metrics(image_type, condition, p)["test_accuracy"] for p in PERCENTS]
            y_values = [a * 100 for a in accs]
            series_y[condition] = y_values
            axes.plot(x_values, y_values, marker="o", markersize=6, linewidth=2,
                       color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition], zorder=2)

        # Label placement is direction-aware (whichever series is higher at
        # a given x gets its label above the point, the lower one below)
        # rather than a fixed offset for both - a fixed "always below"
        # offset stacks the two labels on top of each other when the two
        # conditions' accuracies land close together (e.g. CIFAR-10 at 75%
        # kept, 73.52% vs 72.61%).
        for i in range(1, len(x_values)):
            x = x_values[i]
            (zero_y, noise_y) = (series_y["zero"][i], series_y["noise"][i])
            (higher, lower) = ("zero", "noise") if zero_y >= noise_y else ("noise", "zero")
            axes.annotate(f"{series_y[higher][i]:.2f}%", (x, series_y[higher][i]), xytext=(0, 8),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY)
            axes.annotate(f"{series_y[lower][i]:.2f}%", (x, series_y[lower][i]), xytext=(0, -10),
                           textcoords="offset points", ha="center", va="top", fontsize=8.5, color=TEXT_SECONDARY)

        axes.set_xlim(105, 45)  # descending: 100 -> 75 -> 50, reading left-to-right as "more pruning"
        axes.set_xticks(x_values)
        axes.set_xlabel("Percent of tiles kept", fontsize=10)
        axes.set_ylabel("Test accuracy (%)", fontsize=10)
        axes.set_title(DATASET_LABELS[image_type], fontsize=12)
        axes.tick_params(labelsize=9)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.legend(loc="lower left", fontsize=8.5, frameon=False)

    fig.suptitle("Accuracy vs. how aggressively the lowest-MI tiles are pruned", fontsize=13)
    plt.tight_layout(w_pad=1.2)
    fig.subplots_adjust(wspace=0.25)

    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"pruning_trend_all_results.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
