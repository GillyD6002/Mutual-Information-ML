import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One standalone bar chart per dataset (unlike train_pruned_classifiers.py's
# plot_comparison, which puts every dataset in a single figure as subplots,
# and unlike that figure's per-percent_kept files, which put every dataset's
# 75%-kept run in one file and every 50%-kept run in another). Each dataset's
# chart instead groups both percent_kept sweeps side by side: baseline/
# noise/zero at 75% kept, then the same three at 50% kept. Reads the
# summary_p75.json/summary_p50.json that train_pruned_classifiers.py already
# produced, so this doesn't need TensorFlow and runs anywhere. Run as:
#     python -m MI_scaling_non_middle.plot_pruning_by_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "pruning")

# Order requested: baseline first, then the two pruned conditions - kept
# distinct from CONDITIONS in train_pruned_classifiers.py ("original", "zero",
# "noise"), which is ordered for that module's own dict/loop bookkeeping
# rather than for how the bars should read left to right.
CONDITION_ORDER = ["original", "noise", "zero"]
CONDITION_LABELS = {"original": "baseline", "noise": "noise-pruned", "zero": "zero-pruned"}
COLORS = {"original": "#4C72B0", "zero": "#C44E52", "noise": "#DD8452"}
PERCENTS = [0.75, 0.50]
GROUP_GAP = 1.0


def load_summary(percent_kept):
    tag = f"p{int(round(percent_kept * 100))}"
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    with open(path) as handle:
        metrics = json.load(handle)
    return {(m["image_type"], m["condition"]): m for m in metrics}


def plot_dataset(dataset, by_percent):
    n_cond = len(CONDITION_ORDER)
    group_starts = [i * (n_cond + GROUP_GAP) for i in range(len(PERCENTS))]

    (fig, axes) = plt.subplots(figsize=(2.4 * n_cond * len(PERCENTS), 5))
    for (group_start, percent_kept) in zip(group_starts, PERCENTS):
        by_condition = by_percent[percent_kept]
        xs = group_start + np.arange(n_cond)
        accuracies = [by_condition[cond]["test_accuracy"] for cond in CONDITION_ORDER]
        bars = axes.bar(xs, accuracies, color=[COLORS[cond] for cond in CONDITION_ORDER])
        for (x, bar, accuracy) in zip(xs, bars, accuracies):
            axes.text(bar.get_x() + bar.get_width() / 2, accuracy, f"{accuracy:.3f}",
                      ha="center", va="bottom", fontsize=10)
        axes.text(xs.mean(), -0.18, f"Keep {int(round(percent_kept * 100))}%",
                  transform=axes.get_xaxis_transform(), ha="center", va="top",
                  fontsize=11, fontweight="bold")

    if len(PERCENTS) > 1:
        separator_x = (group_starts[0] + n_cond - 1 + group_starts[1]) / 2
        axes.axvline(separator_x, color="0.7", linestyle="--", linewidth=0.8)

    all_xs = np.concatenate([group_start + np.arange(n_cond) for group_start in group_starts])
    axes.set_xticks(all_xs)
    axes.set_xticklabels(CONDITION_LABELS[cond] for _ in PERCENTS for cond in CONDITION_ORDER)
    axes.set_ylim(0, 1.08)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.set_ylabel("Test accuracy")
    axes.set_title(dataset)
    # Extra bottom margin to fit the "Keep NN%" group labels sitting below
    # the per-bar condition tick labels.
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    plt.savefig(os.path.join(RESULTS_DIR, f"pruning_by_dataset_{dataset}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"pruning_by_dataset_{dataset}.png"), dpi=150)
    plt.close()
    print(f"[{dataset}] saved pruning_by_dataset_{dataset}.png", flush=True)


def main():
    summaries = {percent_kept: load_summary(percent_kept) for percent_kept in PERCENTS}
    datasets = sorted(set(image_type for summary in summaries.values() for (image_type, _) in summary))
    for dataset in datasets:
        by_percent = {
            percent_kept: {
                cond: summaries[percent_kept][(dataset, cond)]
                for cond in CONDITION_ORDER
            }
            for percent_kept in PERCENTS
        }
        plot_dataset(dataset, by_percent)


if __name__ == "__main__":
    main()
