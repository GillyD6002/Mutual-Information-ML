import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "pruning")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# A plain gridded table (matplotlib's axes.table, not pixel-positioned text
# like make_accuracy_table.py) of the same 12 pruning results - a simple
# black-on-white grid rather than a styled/colored poster graphic, for
# whichever context wants the plainer look instead.

DATASETS = ["mnist", "cifar10"]
DATASET_LABELS = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
CONDITIONS = ["original", "zero", "noise"]
CONDITION_LABELS = {"original": "Original", "zero": "Zero-pruned", "noise": "Noise-pruned"}
PERCENTS = [0.75, 0.5]


def load_metrics(image_type, condition, percent_kept):
    tag = f"{image_type}_{condition}_p{int(round(percent_kept * 100))}"
    path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    with open(path) as handle:
        return json.load(handle)


def main():
    header = ["Dataset", "Percent kept", "Original", "Zero-pruned", "Noise-pruned"]
    rows = []
    for image_type in DATASETS:
        for percent_kept in PERCENTS:
            original_acc = load_metrics(image_type, "original", percent_kept)["test_accuracy"]
            row = [DATASET_LABELS[image_type], f"{int(round(percent_kept * 100))}%",
                   f"{original_acc * 100:.2f}%"]
            for condition in ("zero", "noise"):
                accuracy = load_metrics(image_type, condition, percent_kept)["test_accuracy"]
                delta = (accuracy - original_acc) * 100
                row.append(f"{accuracy * 100:.2f}% ({delta:+.2f}pp)")
            rows.append(row)

    (fig, axes) = plt.subplots(figsize=(8, 1.7))
    axes.axis("off")
    # axis("off") hides ticks/spines but not the Axes' own background
    # patch, which still spans the full figsize above - left visible, it
    # would make savefig's bbox_inches="tight" crop to that full canvas
    # instead of the now content-sized table.
    axes.patch.set_visible(False)
    table = axes.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    # Columns aren't all the same width by default - "Dataset"/"Percent
    # kept" only ever hold short strings ("CIFAR-10", "50%") while
    # "Noise-pruned" holds much longer ones ("58.07% (-22.96pp)"), so
    # sizing every column equally leaves the short columns mostly empty.
    # auto_set_column_width sizes each to its own longest cell instead,
    # which also lets the figure itself shrink to the table's real width.
    table.auto_set_column_width(col=list(range(len(header))))
    # auto_set_column_width only registers columns to be auto-sized at
    # draw time (table._autoColumns) rather than setting widths right
    # away, so a draw has to happen before its widths can be read - and
    # that same list has to be cleared afterwards, or the next draw (at
    # savefig time) recomputes them and silently discards our widening
    # below, which exists because auto_set_column_width's own padding is
    # barely wider than the text itself (e.g. "CIFAR-10" ends up flush
    # against the cell border).
    fig.canvas.draw()
    col_widths = {col: cell.get_width() for ((row, col), cell) in table.get_celld().items()}
    table._autoColumns = []
    for ((row, col), cell) in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.PAD = 0.04
        cell.set_width(col_widths[col] * 1.2)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")
        else:
            cell.set_facecolor("#ffffff")

    # No plt.tight_layout() here: it recomputes the Axes' subplot
    # position to fill the full figsize, which undoes the point of
    # auto_set_column_width and makes savefig's bbox_inches="tight" crop
    # back to the original full canvas instead of the smaller table.
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"simple_table.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
