import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "pruning")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# A poster-ready table of test accuracy across the pruning experiment's 6
# runs (2 datasets x {original, zero-pruned, noise-pruned}), read straight
# from train_pruned_classifiers.py's saved *_metrics.json files - not
# hand-copied numbers, so rerunning this after a training run finishes
# picks up the real result automatically. A condition whose metrics file
# doesn't exist yet (e.g. still training) is rendered as a "training..."
# placeholder rather than erroring out, so this can be regenerated
# mid-sweep and again once everything finishes.
#
# Colors are the first three slots of this project's validated categorical
# palette (see the dataviz skill's palette.md), assigned in that fixed
# order to Original/Zero-pruned/Noise-pruned and used only as small accent
# chips/tints - all numbers stay in dark text, per "text wears text tokens,
# never the series color".

DATASETS = ["mnist", "cifar10"]
DATASET_LABELS = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
CONDITIONS = ["original", "zero", "noise"]
CONDITION_LABELS = {"original": "Original", "zero": "Zero-pruned", "noise": "Noise-pruned"}
CONDITION_COLORS = {"original": "#2a78d6", "zero": "#1baf7a", "noise": "#eda100"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#9a9890"
RULE_COLOR = "#d8d6cf"


def load_metrics(percent_kept=0.75):
    metrics = {}
    for image_type in DATASETS:
        for condition in CONDITIONS:
            tag = f"{image_type}_{condition}_p{int(round(percent_kept * 100))}"
            path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
            if os.path.exists(path):
                with open(path) as handle:
                    metrics[(image_type, condition)] = json.load(handle)
            else:
                metrics[(image_type, condition)] = None
    return metrics


def make_table(metrics, output_tag="p75"):
    n_rows = len(DATASETS)
    n_cols = len(CONDITIONS)
    row_height = 1.0
    col_width = 2.1
    label_col_width = 1.5
    header_height = 0.6

    fig_width = label_col_width + col_width * n_cols
    fig_height = header_height + row_height * n_rows
    (fig, axes) = plt.subplots(figsize=(fig_width, fig_height))
    axes.set_xlim(0, fig_width)
    axes.set_ylim(0, fig_height)
    axes.invert_yaxis()
    axes.axis("off")

    # Column headers, each centered on the same x_center the body cells
    # below use (so header and values line up), identified by a thin
    # colored accent bar spanning the full column width rather than a
    # small chip - a chip positioned relative to the label's rendered
    # text width would risk colliding with the next column for longer
    # labels ("Zero-pruned"/"Noise-pruned"), since matplotlib text width
    # isn't known until render time.
    bar_height = 0.07
    for (col, condition) in enumerate(CONDITIONS):
        x_start = label_col_width + col * col_width
        x_center = x_start + col_width / 2
        axes.text(x_center, header_height / 2 - 0.05, CONDITION_LABELS[condition],
                   ha="center", va="center", fontsize=13, color=TEXT_PRIMARY, fontweight="bold")
        axes.add_patch(Rectangle((x_start + 0.15, header_height - bar_height - 0.08),
                                  col_width - 0.3, bar_height,
                                  facecolor=CONDITION_COLORS[condition], edgecolor="none"))

    # Header/body rule.
    axes.plot([0, fig_width], [header_height, header_height], color=TEXT_PRIMARY, linewidth=1.5)

    for (row, image_type) in enumerate(DATASETS):
        y_center = header_height + row * row_height + row_height / 2
        axes.text(0.15, y_center, DATASET_LABELS[image_type], ha="left", va="center",
                   fontsize=14, color=TEXT_PRIMARY, fontweight="bold")

        original = metrics[(image_type, "original")]
        original_acc = original["test_accuracy"] if original else None

        for (col, condition) in enumerate(CONDITIONS):
            x_center = label_col_width + col * col_width + col_width / 2
            entry = metrics[(image_type, condition)]
            if entry is None:
                axes.text(x_center, y_center, "training…", ha="center", va="center",
                           fontsize=13, color=TEXT_MUTED, style="italic")
                continue
            accuracy = entry["test_accuracy"]
            axes.text(x_center, y_center - 0.14, f"{accuracy * 100:.2f}%", ha="center", va="center",
                       fontsize=17, color=TEXT_PRIMARY, fontweight="bold")
            if condition != "original" and original_acc is not None:
                delta = (accuracy - original_acc) * 100
                axes.text(x_center, y_center + 0.22, f"{delta:+.2f} pp vs original", ha="center", va="center",
                           fontsize=10.5, color=TEXT_SECONDARY)

        # Row separator (skip after the last row).
        if row < n_rows - 1:
            y_rule = header_height + (row + 1) * row_height
            axes.plot([0, fig_width], [y_rule, y_rule], color=RULE_COLOR, linewidth=1)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"accuracy_table_{output_tag}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {path}")
    plt.close()


def main():
    metrics = load_metrics(0.75)
    make_table(metrics, "p75")


if __name__ == "__main__":
    main()
