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
from MI_scaling_non_middle.pixel_pruning import build_tile_mask, pixel_mask_from_edges, load_sliding_window_mi, apply_pruning

# Trains a matched CNN, for the same training budget across its own three
# conditions, on three versions of each dataset:
#   - "original": untouched images.
#   - "zero": pixels in the bottom (1 - percent_kept) fraction of tiles (by
#     MI, reusing sliding_window_mi.py's already-computed window=3/stride=3
#     sweep - see pixel_pruning.load_sliding_window_mi) forced to 0.
#   - "noise": those same pruned pixels replaced with fresh i.i.d. noise
#     drawn uniformly between the dataset's own min/max pixel value.
# The comparison this produces (does classification accuracy survive
# pruning the *lowest*-MI tiles?) is the actual pruning experiment; this
# script only trains and records metrics - pixel_pruning.py supplies the MI
# ranking (reused from sliding_window_mi.py, no fresh MI measurement is run)
# and builds the mask/does the corruption.
#
# Architecture is deliberately NOT train_cnn.py's 3-conv/no-pooling network
# - that one was built so its activations stay 28x28 for re-mining MI, not
# for classification accuracy. MNIST uses a conventional 2-block
# conv+pool+dropout CNN; CIFAR-10, lfw_faces, and fer2013_hf are all
# meaningfully harder tasks at this same grayscale, 28x28 resolution (for
# CIFAR-10, well below what color, full-resolution CIFAR-10 benchmarks
# reach), so they share a deeper, wider, augmented network of their own
# (build_augmented_classifier), on a larger epoch budget, since a bigger
# network needs more room to converge. Datasets were never meant to share
# one architecture's capacity across each other; only each dataset's *own*
# three conditions need to train identically for the pruning comparison to
# be fair (see below), and they still do.
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
# lfw_faces (identity classification) and fer2013_hf (7-class emotion
# classification) plug into this same pipeline via img.get_images's
# existing support for both source names (see src/image.py) - the only
# thing that had to be added for them was a *labeled* loader (get_images
# only returns raw pixels; MI measurement doesn't need labels, but training
# a classifier does) and their own model builder, since neither is a
# built-in keras.datasets set. Both still need to go through
# NUM_CLASSES-generic model builders (unlike mnist/cifar10's fixed 10),
# since fer2013_hf's class count depends on the emotion taxonomy and
# lfw_faces's depends on LFW_MIN_FACES_PER_PERSON.
#
# Run as (needs sliding_window_mi.py's window=3 sweep to have been run for
# each dataset first - already true for mnist/cifar10 as of this writing;
# lfw_faces/fer2013_hf need `python -m MI_scaling_non_middle.sliding_window_mi
# --datasets lfw_faces,fer2013_hf --window-sizes 3` run once beforehand):
#     python -m MI_scaling_non_middle.train_pruned_classifiers
#     python -m MI_scaling_non_middle.train_pruned_classifiers --percent-kept 0.5
#     python -m MI_scaling_non_middle.train_pruned_classifiers --datasets mnist

MI_RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
RESULTS_DIR = os.path.join(MI_RESULTS_DIR, "pruning")
IMAGE_SIZE = img.DEFAULT_IMAGE_SIZE

CONDITIONS = ["original", "zero", "noise"]
DATASETS = ["mnist", "cifar10", "lfw_faces", "fer2013_hf"]

# Only people with at least this many images are kept, so that every
# identity class has enough examples for both a training set and a
# held-out test split - unlike src/image.py's own lfw_faces loader
# (min_faces_per_person=1), which never needs a train/test split or per
# class minimum since it's only used for unsupervised MI measurement, not
# classification.
LFW_MIN_FACES_PER_PERSON = 20

# window_size == stride is what makes this sweep a non-overlapping tiling -
# see pixel_pruning.load_sliding_window_mi's docstring.
TILE_WINDOW_SIZE = 3
TILE_STRIDE = 3

BATCH_SIZE = 128
# Per-dataset training budget: CIFAR-10's deeper/augmented network needs
# more epochs to converge than MNIST's smaller one, so it gets a larger
# budget - this is still "the same amount of training" in the sense that
# matters (see module docstring): identical across a given dataset's own
# three conditions, just not identical *between* datasets, which was never
# a fairness requirement. lfw_faces and fer2013_hf both get cifar10's
# larger budget - like cifar10, both use its augmented 3-block
# architecture (see build_lfw_classifier/build_fer2013_classifier).
TRAINING_CONFIG = {
    "mnist": {"epochs": 40, "patience": 6},
    "cifar10": {"epochs": 60, "patience": 8},
    "lfw_faces": {"epochs": 60, "patience": 8},
    "fer2013_hf": {"epochs": 60, "patience": 8},
}

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


def load_lfw_labeled():

    # Unlike src/image.py's own lfw_faces loader (identity labels aren't
    # needed for unsupervised MI measurement), this is an actual
    # train/test split for identity classification, so people with too few
    # images to both train and test on are excluded via
    # LFW_MIN_FACES_PER_PERSON. LFW ships with no canonical train/test
    # split for this task, so one is carved out here (stratified, so every
    # kept person appears in both splits regardless of how few images they
    # have). Native images are resized (not cropped) to IMAGE_SIZE like
    # src/image.py's lfw_faces branch - they're already a non-square
    # 125x94 crop, so "crop to 28x28" would cut away most of the face
    # rather than shrink it.

    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    data = fetch_lfw_people(min_faces_per_person=LFW_MIN_FACES_PER_PERSON, resize=1.0)
    (x_train, x_test, y_train, y_test) = train_test_split(
        data.images, data.target, test_size=0.2, random_state=NOISE_SEED, stratify=data.target)
    x_train = img.conform_size(x_train, IMAGE_SIZE, mode="resize").astype(np.float32)
    x_test = img.conform_size(x_test, IMAGE_SIZE, mode="resize").astype(np.float32)
    return (x_train, y_train, x_test, y_test)


def load_fer2013_labeled():

    # FER-2013 (7-class facial emotion) via the same clip-benchmark/wds_fer2013
    # Hugging Face mirror src/image.py's fer2013_hf loader uses, but with
    # labels ("cls", an int 0-6) kept rather than discarded, and using the
    # dataset's own train (28,709) / test (7,178) split as-is rather than
    # carving out a new one - unlike lfw_faces, FER-2013 already ships with
    # a canonical split for its intended (emotion) task. Native images are
    # 48x48; resized (not cropped) down to IMAGE_SIZE, since a 48->28 center
    # crop would trim 10px off every edge and risk losing the
    # mouth/eyebrows that carry most of the expression (see the matching
    # note on src/image.py's get_images fer2013_hf branch).

    from datasets import load_dataset

    dataset = load_dataset("clip-benchmark/wds_fer2013")

    def to_arrays(split):
        images = np.stack([np.asarray(example["jpg"], dtype=np.float32) / 255 for example in dataset[split]])
        labels = np.array([example["cls"] for example in dataset[split]])
        return (images, labels)

    (x_train, y_train) = to_arrays("train")
    (x_test, y_test) = to_arrays("test")
    x_train = img.conform_size(x_train, IMAGE_SIZE, mode="resize").astype(np.float32)
    x_test = img.conform_size(x_test, IMAGE_SIZE, mode="resize").astype(np.float32)
    return (x_train, y_train, x_test, y_test)


LOADERS = {
    "mnist": load_mnist_labeled,
    "cifar10": load_cifar10_labeled,
    "lfw_faces": load_lfw_labeled,
    "fer2013_hf": load_fer2013_labeled,
}


def load_tile_mask(image_type, percent_kept):

    # Sources the per-tile MI ranking from sliding_window_mi.py's existing
    # window=3/stride=3 sweep (load_sliding_window_mi) rather than running a
    # fresh MI measurement - see pixel_pruning.py's module docstring for why
    # this is both free (already computed) and finer-grained/cleaner than
    # the coarser alternatives (a 3x3 grid of 9 unequal-size tiles, or a
    # Voronoi partition of independently-measured boxes). The 81 identically
    # -sized 3x3 tiles this sweep measured already tile the image almost
    # exactly (edges from load_sliding_window_mi close the 1px gap its last
    # position leaves short of the far edge), so pixel_mask_from_edges is a
    # plain box expansion - no Voronoi step needed.

    (heatmap, edges) = load_sliding_window_mi(image_type, MI_RESULTS_DIR, window_size=TILE_WINDOW_SIZE, stride=TILE_STRIDE)
    tile_mask = build_tile_mask(heatmap, percent_kept)
    pixel_mask = pixel_mask_from_edges(tile_mask, edges)
    return (pixel_mask, tile_mask, heatmap)


def build_mnist_classifier(num_classes):

    # Two conv+conv+pool+dropout blocks (32 then 64 filters) followed by a
    # dense head - a conventional, reasonably strong CNN for 28x28
    # single-channel classification (unlike train_cnn.py's flat,
    # pooling-free network, which trades accuracy for keeping every conv
    # layer's activation map at input resolution). MNIST is easy enough at
    # this resolution that this size is already sufficient to reach ~99.6%.

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
    outputs = ks.layers.Dense(num_classes, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_augmented_classifier(num_classes):

    # A deeper, wider version of build_mnist_classifier: three
    # conv+conv+pool+dropout blocks (32/64/128 filters, vs MNIST's two at
    # 32/64) and a bigger dense head (512 vs 256), plus light train-time-only
    # data augmentation (RandomFlip/RandomTranslation/RandomZoom - Keras
    # preprocessing layers that are no-ops during model.evaluate/predict, so
    # the held-out test set this is scored on is never itself augmented).
    # Used for every dataset harder than MNIST at this same grayscale/28x28
    # resolution - CIFAR-10 (object classification), lfw_faces (identity,
    # far fewer images per class than CIFAR-10 has per object class) and
    # fer2013_hf (emotion) - all of which saw train accuracy pull well
    # ahead of val accuracy at MNIST's plain, unaugmented 2-block
    # architecture. Only a horizontal flip and small shifts/zooms are used
    # - nothing that would plausibly flip a class label for any of the
    # three.

    inputs = ks.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = inputs
    x = ks.layers.RandomFlip("horizontal")(x)
    x = ks.layers.RandomTranslation(0.1, 0.1)(x)
    x = ks.layers.RandomZoom(0.1)(x)
    for filters in (32, 64, 128):
        x = ks.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.MaxPooling2D(2)(x)
        x = ks.layers.Dropout(0.3)(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(512, activation="relu")(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Dropout(0.5)(x)
    outputs = ks.layers.Dense(num_classes, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {
    "mnist": build_mnist_classifier,
    "cifar10": build_augmented_classifier,
    "lfw_faces": build_augmented_classifier,
    "fer2013_hf": build_augmented_classifier,
}


def condition_tag(image_type, condition, percent_kept):
    return f"{image_type}_{condition}_p{int(round(percent_kept * 100))}"


def condition_paths(image_type, condition, percent_kept):
    tag = condition_tag(image_type, condition, percent_kept)
    return {
        "history": os.path.join(RESULTS_DIR, f"{tag}_history.json"),
        "metrics": os.path.join(RESULTS_DIR, f"{tag}_metrics.json"),
        "model": os.path.join(RESULTS_DIR, f"{tag}.keras"),
    }


def run_condition(image_type, condition, x_train, y_train, x_test, y_test, pixel_mask, percent_kept, batch_size, num_classes):
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

    config = TRAINING_CONFIG[image_type]
    model = MODEL_BUILDERS[image_type](num_classes)
    history = model.fit(
        np.expand_dims(xt, axis=3), y_train,
        validation_data=(np.expand_dims(xv, axis=3), y_test),
        batch_size=batch_size, epochs=config["epochs"],
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True)],
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
        # Headroom above 1.0 (rather than capping the axis at the data's own
        # max of 1.0) so a near-ceiling accuracy's value label - drawn just
        # above its bar - has room to clear the subplot title instead of
        # overlapping it, which happened for MNIST's ~0.99+ bars.
        axes.set_ylim(0, 1.08)
        axes.set_yticks(np.arange(0, 1.01, 0.2))
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
                         help="Comma-separated subset of mnist,cifar10,lfw_faces,fer2013_hf to run (default: all)")
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
        # Computed from the loaded labels rather than a fixed constant,
        # since only mnist/cifar10 have a known-in-advance class count (10)
        # - lfw_faces's depends on LFW_MIN_FACES_PER_PERSON and fer2013_hf's
        # is fixed at 7 but not worth hardcoding separately.
        num_classes = int(max(y_train.max(), y_test.max())) + 1
        print(f"[{image_type}] {x_train.shape[0]} train / {x_test.shape[0]} test images, {num_classes} classes", flush=True)
        (pixel_mask, tile_mask, heatmap) = load_tile_mask(image_type, args.percent_kept)
        print(f"[{image_type}] tile mask (True=kept):\n{tile_mask}", flush=True)
        for condition in conditions:
            with strategy.scope():
                metrics = run_condition(
                    image_type, condition, x_train, y_train, x_test, y_test,
                    pixel_mask, args.percent_kept, global_batch_size, num_classes,
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
