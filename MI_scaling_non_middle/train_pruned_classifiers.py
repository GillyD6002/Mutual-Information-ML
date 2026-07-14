import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as ks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import image as img
from MI_scaling_non_middle.pixel_pruning import (
    build_tile_mask, build_voronoi_assignment, voronoi_pixel_mask, load_corner_grid_mi, apply_pruning,
)

# Trains the same CNN architecture, for the same training budget, on three
# versions of each dataset:
#   - "original": untouched images.
#   - "zero": pixels in the bottom (1 - percent_kept) fraction of tiles (by
#     MI, reusing corner_mi_scaling.py's already-computed 3x3 grid sweep -
#     see pixel_pruning.load_corner_grid_mi) forced to 0.
#   - "noise": those same pruned pixels replaced with fresh i.i.d. noise
#     drawn uniformly between the dataset's own min/max pixel value.
# The comparison this produces (does classification accuracy survive
# pruning the *lowest*-MI tiles?) is the actual pruning experiment; this
# script only trains and records metrics - pixel_pruning.py supplies the MI
# ranking (reused from corner_mi_scaling.py, no fresh MI measurement is run)
# and builds the mask/does the corruption.
#
# Architecture is deliberately NOT train_cnn.py's 3-conv/no-pooling network
# - that one was built so its activations stay 28x28 for re-mining MI, not
# for classification accuracy. This uses a conventional conv+pool+dropout
# CNN sized for MNIST/CIFAR-10-scale classification instead.
#
# "Do the same amount of training to keep it fair" (the user's framing) is
# implemented as: identical architecture, optimizer, batch size, max epochs,
# and early-stopping patience/monitor across all three conditions of a given
# dataset - the same training *protocol* and *budget*, not a hand-picked
# fixed epoch count. Early stopping is kept (rather than disabled) because
# every other training script in this project already uses it this way
# (train_cnn.py), and forcing a fixed epoch count regardless of overfitting
# would bias the comparison in the other direction (rewarding whichever
# condition happens to overfit slowest).
#
# Run as (needs corner_mi_scaling.py's `--grid` sweep to have been run for
# each dataset first - already true for mnist/cifar10 as of this writing):
#     python -m MI_scaling_non_middle.train_pruned_classifiers
#     python -m MI_scaling_non_middle.train_pruned_classifiers --percent-kept 0.5
#     python -m MI_scaling_non_middle.train_pruned_classifiers --datasets mnist

MI_RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
RESULTS_DIR = os.path.join(MI_RESULTS_DIR, "pruning")
IMAGE_SIZE = img.DEFAULT_IMAGE_SIZE
NUM_CLASSES = 10

CONDITIONS = ["original", "zero", "noise"]
DATASETS = ["mnist", "cifar10"]

GRID_SIZE = 3
# Matches this project's ~9px tile granularity (28 // 3) - see
# pixel_pruning.load_corner_grid_mi's docstring for why this particular
# length slice of the existing MI-vs-length curves is what's used here.
TILE_LENGTH = 9

BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 6

# Fixed so that the *same* noise draw is used for a given (dataset,
# percent_kept) pair across reruns - a killed/resumed run doesn't silently
# get a different noise-pruned dataset than a prior partial run of the same
# condition.
NOISE_SEED = 20260714


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_mnist_labeled():
    ((x_train, y_train), (x_test, y_test)) = ks.datasets.mnist.load_data()
    x_train = img.conform_size(x_train / 255, IMAGE_SIZE, mode="crop").astype(np.float32)
    x_test = img.conform_size(x_test / 255, IMAGE_SIZE, mode="crop").astype(np.float32)
    return (x_train, y_train, x_test, y_test)


def load_cifar10_labeled():

    # Same grayscale + center-crop-to-28x28 pipeline as train_cnn.py's
    # load_cifar10_grayscale, kept here as a duplicate (rather than an
    # import) so this module has no dependency on train_cnn.py's
    # CNN-activation-extraction-specific constants (CONV_FILTERS etc).

    ((x_train, y_train), (x_test, y_test)) = ks.datasets.cifar10.load_data()
    x_train = img.conform_size(img.convert_to_grayscale(x_train / 255), IMAGE_SIZE, mode="crop").astype(np.float32)
    x_test = img.conform_size(img.convert_to_grayscale(x_test / 255), IMAGE_SIZE, mode="crop").astype(np.float32)
    y_train = y_train.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)
    return (x_train, y_train, x_test, y_test)


LOADERS = {"mnist": load_mnist_labeled, "cifar10": load_cifar10_labeled}


def load_tile_mask(image_type, percent_kept):

    # Sources the per-tile MI ranking from corner_mi_scaling.py's existing
    # `--grid` sweep (load_corner_grid_mi) rather than running a fresh MI
    # measurement - see pixel_pruning.py's module docstring for why this is
    # both free (already computed) and methodologically cleaner (every cell
    # measured at the same fixed size, unlike a literal non-overlapping
    # tiling of a 28px image into 3 equal parts). The keep/prune decision
    # per cell is expanded to a full pixel mask via a Voronoi partition of
    # the image around the same 9 cell centers, not the (slightly
    # non-tiling) box each cell's MI was actually measured on.

    heatmap = load_corner_grid_mi(image_type, MI_RESULTS_DIR, grid_size=GRID_SIZE, tile_length=TILE_LENGTH)
    tile_mask = build_tile_mask(heatmap, percent_kept)
    assignment = build_voronoi_assignment(IMAGE_SIZE, GRID_SIZE)
    pixel_mask = voronoi_pixel_mask(tile_mask, assignment)
    return (pixel_mask, tile_mask, heatmap)


def build_classifier():

    # Two conv+conv+pool+dropout blocks (32 then 64 filters) followed by a
    # dense head - a conventional, reasonably strong CNN for 28x28
    # single-channel classification (unlike train_cnn.py's flat,
    # pooling-free network, which trades accuracy for keeping every conv
    # layer's activation map at input resolution).

    inputs = ks.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = inputs
    for filters in (32, 64):
        x = ks.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.MaxPooling2D(2)(x)
        x = ks.layers.Dropout(0.25)(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(256, activation="relu")(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Dropout(0.5)(x)
    outputs = ks.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def condition_tag(image_type, condition, percent_kept):
    return f"{image_type}_{condition}_p{int(round(percent_kept * 100))}"


def condition_paths(image_type, condition, percent_kept):
    tag = condition_tag(image_type, condition, percent_kept)
    return {
        "history": os.path.join(RESULTS_DIR, f"{tag}_history.json"),
        "metrics": os.path.join(RESULTS_DIR, f"{tag}_metrics.json"),
        "model": os.path.join(RESULTS_DIR, f"{tag}.keras"),
    }


def run_condition(image_type, condition, x_train, y_train, x_test, y_test, pixel_mask, percent_kept, batch_size):
    ensure_results_dir()
    paths = condition_paths(image_type, condition, percent_kept)

    # Resumable like every other sweep in this project: a condition whose
    # metrics file already exists is loaded from disk rather than retrained.
    if os.path.exists(paths["metrics"]):
        print(f"[{image_type}/{condition}] metrics already exist, skipping", flush=True)
        with open(paths["metrics"]) as handle:
            return json.load(handle)

    if condition == "original":
        (xt, xv) = (x_train, x_test)
    else:
        rng = np.random.RandomState(NOISE_SEED)
        xt = apply_pruning(x_train, pixel_mask, condition, rng)
        xv = apply_pruning(x_test, pixel_mask, condition, rng)

    model = build_classifier()
    history = model.fit(
        np.expand_dims(xt, axis=3), y_train,
        validation_data=(np.expand_dims(xv, axis=3), y_test),
        batch_size=batch_size, epochs=EPOCHS,
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        verbose=2,
    )
    (test_loss, test_accuracy) = model.evaluate(np.expand_dims(xv, axis=3), y_test, verbose=0)

    metrics = {
        "image_type": image_type,
        "condition": condition,
        "percent_kept": percent_kept,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_trained": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
    }
    with open(paths["history"], "w") as handle:
        json.dump({key: [float(v) for v in values] for (key, values) in history.history.items()}, handle)
    with open(paths["metrics"], "w") as handle:
        json.dump(metrics, handle, indent=2)
    model.save(paths["model"])
    print(f"[{image_type}/{condition}] test_accuracy={test_accuracy:.4f} "
          f"(epochs_trained={metrics['epochs_trained']})", flush=True)
    return metrics


def plot_comparison(all_metrics, percent_kept):
    ensure_results_dir()
    datasets = sorted(set(m["image_type"] for m in all_metrics))
    by_dataset = {d: {m["condition"]: m for m in all_metrics if m["image_type"] == d} for d in datasets}

    (fig, axes_row) = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    axes_row = axes_row[0]
    colors = {"original": "#4C72B0", "zero": "#C44E52", "noise": "#DD8452"}
    for (axes, dataset) in zip(axes_row, datasets):
        accuracies = [by_dataset[dataset][c]["test_accuracy"] for c in CONDITIONS]
        bars = axes.bar(CONDITIONS, accuracies, color=[colors[c] for c in CONDITIONS])
        for (bar, accuracy) in zip(bars, accuracies):
            axes.text(bar.get_x() + bar.get_width() / 2, accuracy, f"{accuracy:.3f}",
                       ha="center", va="bottom", fontsize=10)
        axes.set_ylim(0, 1.0)
        axes.set_ylabel("Test accuracy")
        axes.set_title(dataset)
    fig.suptitle(f"Effect of pruning the lowest-MI {int(round((1 - percent_kept) * 100))}% of tiles "
                 f"(keep {int(round(percent_kept * 100))}%)", fontsize=13)
    plt.tight_layout()

    tag = f"p{int(round(percent_kept * 100))}"
    plt.savefig(os.path.join(RESULTS_DIR, f"pruning_comparison_{tag}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"pruning_comparison_{tag}.png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent-kept", type=float, default=0.75,
                         help="Fraction of (brightest, by MI) tiles to keep unmodified (default: 0.75)")
    parser.add_argument("--datasets", type=str, default=None,
                         help="Comma-separated subset of mnist,cifar10 to run (default: both)")
    parser.add_argument("--conditions", type=str, default=None,
                         help="Comma-separated subset of original,zero,noise to run (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    datasets = DATASETS
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    conditions = CONDITIONS
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]

    strategy = tf.distribute.MirroredStrategy()
    print(f"MirroredStrategy running on {strategy.num_replicas_in_sync} device(s)")
    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    all_metrics = []
    for image_type in datasets:
        (x_train, y_train, x_test, y_test) = LOADERS[image_type]()
        (pixel_mask, tile_mask, heatmap) = load_tile_mask(image_type, args.percent_kept)
        print(f"[{image_type}] tile mask (True=kept):\n{tile_mask}", flush=True)
        for condition in conditions:
            with strategy.scope():
                metrics = run_condition(
                    image_type, condition, x_train, y_train, x_test, y_test,
                    pixel_mask, args.percent_kept, global_batch_size,
                )
            all_metrics.append(metrics)

    summary_path = os.path.join(RESULTS_DIR, f"summary_p{int(round(args.percent_kept * 100))}.json")
    with open(summary_path, "w") as handle:
        json.dump(all_metrics, handle, indent=2)

    print("\n=== Summary ===")
    for metrics in all_metrics:
        print(f"{metrics['image_type']:>8} | {metrics['condition']:>8} | "
              f"test_acc={metrics['test_accuracy']:.4f} | epochs={metrics['epochs_trained']}")

    if set(conditions) == set(CONDITIONS):
        # Only a full run of all three conditions makes the side-by-side
        # comparison plot meaningful, same reasoning as
        # sliding_window_mi.py's plot_combined guard on a full window-size
        # sweep.
        plot_comparison(all_metrics, args.percent_kept)


if __name__ == "__main__":
    main()
