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
# conv+pool+dropout CNN; CIFAR-10 is a meaningfully harder task at this same
# grayscale, 28x28 resolution (well below what color, full-resolution
# CIFAR-10 benchmarks reach), so it gets a deeper, wider, augmented network
# of its own (build_augmented_classifier), on a larger epoch budget, since a
# bigger network needs more room to converge.
#
# lfw_faces and fer2013_hf are different from all three of the above in two
# ways, not just one:
#   - They run at their own, much larger native (or near-native) resolution
#     - 96x96 and 48x48 respectively - instead of being squished down to
#     28x28. A first pass at this pipeline resized both down to 28x28 to
#     match mnist/cifar10, which tanked accuracy (identity/emotion detail
#     doesn't survive that much downsampling) - see IMAGE_SIZES below and
#     each loader's docstring for how the new sizes are reached.
#   - They use transfer learning (build_transfer_classifier: a pretrained
#     MobileNetV2 ImageNet backbone, fine-tuned) rather than a from-scratch
#     CNN, since both are fundamentally data-scarce problems a bigger
#     from-scratch network doesn't fix - lfw_faces has only on the order of
#     tens of images per identity across dozens of identities, and even
#     fer2013_hf's ~29k training images is small next to what a CNN
#     typically needs to learn good features from scratch. A pretrained
#     backbone starts with strong generic visual features already learned
#     from millions of images, which a small/specialized dataset then only
#     has to adapt rather than learn from zero. See
#     TRANSFER_LEARNING_DATASETS/run_condition for the resulting two-phase
#     (frozen head, then fine-tune) training loop this requires - a plain
#     single model.fit() call, as mnist/cifar10 use, doesn't fit a
#     pretrained backbone's standard training recipe.
#
# Datasets were never meant to share one architecture's capacity/resolution
# across each other; only each dataset's *own* three conditions need to
# train identically for the pruning comparison to be fair (see below), and
# they still do - "the same amount of training to keep it fair" (the user's
# original framing for mnist/cifar10) is implemented as identical
# architecture, optimizer, batch size, max epochs, and early-stopping
# patience/monitor across all three conditions of a given dataset, not
# across datasets.
#
# Each dataset's own tile grid for the MI-guided pruning mask is also
# dataset-specific now (TILE_WINDOW_SIZES below), chosen so that
# window_size tiles its own IMAGE_SIZES entry with zero leftover edge - see
# pixel_pruning.load_sliding_window_mi's docstring on why window_size must
# equal stride, and why mnist/cifar10's 28px images still close a 1px gap
# rather than tiling perfectly (28 has no reasonable divisor near 3).
#
# Run as (needs sliding_window_mi.py's sweep to have been run for each
# dataset first, at that dataset's own window size - see
# run_pruning_lfw_fer2013.sh for the lfw_faces/fer2013_hf invocations):
#     python -m MI_scaling_non_middle.train_pruned_classifiers
#     python -m MI_scaling_non_middle.train_pruned_classifiers --percent-kept 0.5
#     python -m MI_scaling_non_middle.train_pruned_classifiers --datasets mnist

MI_RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
RESULTS_DIR = os.path.join(MI_RESULTS_DIR, "pruning")

# Per-dataset native (or near-native) resolution - must match
# sliding_window_mi.py's DATASETS dict exactly, since the MI heatmap/tile
# grid this pipeline reuses was measured at that size.
IMAGE_SIZES = {
    "mnist": img.DEFAULT_IMAGE_SIZE,
    "cifar10": img.DEFAULT_IMAGE_SIZE,
    "lfw_faces": 96,
    "fer2013_hf": 48,
}

CONDITIONS = ["original", "zero", "noise"]
DATASETS = ["mnist", "cifar10", "lfw_faces", "fer2013_hf"]

# Only people with at least this many images are kept, so that every
# identity class has enough examples for both a training set and a
# held-out test split - unlike src/image.py's own lfw_faces loader
# (min_faces_per_person=1), which never needs a train/test split or per
# class minimum since it's only used for unsupervised MI measurement, not
# classification.
LFW_MIN_FACES_PER_PERSON = 20

# window_size == stride for every dataset - what makes each sweep a
# non-overlapping tiling (see pixel_pruning.load_sliding_window_mi's
# docstring). mnist/cifar10 keep the original 3 (their sweep already
# exists at that value); lfw_faces (96px) and fer2013_hf (48px) use larger
# windows chosen to divide their own IMAGE_SIZES entry evenly with no
# leftover edge, at a similar tiles-per-axis granularity to the original
# (96/8 = 48/4 = 12 tiles per axis, vs mnist/cifar10's 28/3 ~= 9).
TILE_WINDOW_SIZES = {
    "mnist": 3,
    "cifar10": 3,
    "lfw_faces": 8,
    "fer2013_hf": 4,
}

# lfw_faces/fer2013_hf use transfer learning (see build_transfer_classifier
# and run_condition) rather than a plain single model.fit() call - this set
# is what run_condition checks to decide which training path to use.
TRANSFER_LEARNING_DATASETS = {"lfw_faces", "fer2013_hf"}

BATCH_SIZE = 128
# Per-dataset training budget. mnist/cifar10 use a single training phase
# ({"epochs", "patience"}); lfw_faces/fer2013_hf use a different shape of
# config since transfer learning needs two phases - a frozen-backbone
# "head" phase (train just the new classification head) followed by a
# "finetune" phase (unfreeze the backbone, continue training end-to-end at
# a much lower learning rate, the standard transfer-learning recipe). See
# run_condition's TRANSFER_LEARNING_DATASETS branch.
TRAINING_CONFIG = {
    "mnist": {"epochs": 40, "patience": 6},
    "cifar10": {"epochs": 60, "patience": 8},
    "lfw_faces": {"head_epochs": 15, "head_patience": 5,
                  "finetune_epochs": 30, "finetune_patience": 8, "finetune_lr": 1e-5},
    "fer2013_hf": {"head_epochs": 15, "head_patience": 5,
                   "finetune_epochs": 30, "finetune_patience": 8, "finetune_lr": 1e-5},
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
    x_train = img.conform_size(x_train / 255, IMAGE_SIZES["mnist"], mode="crop").astype(np.float32)
    x_test = img.conform_size(x_test / 255, IMAGE_SIZES["mnist"], mode="crop").astype(np.float32)
    return (x_train, y_train, x_test, y_test)


def load_cifar10_labeled():

    # Same grayscale + center-crop-to-28x28 pipeline as train_cnn.py's
    # load_cifar10_grayscale, kept here as a duplicate (rather than an
    # import) so this module has no dependency on train_cnn.py's
    # CNN-activation-extraction-specific constants (CONV_FILTERS etc).

    ((x_train, y_train), (x_test, y_test)) = ks.datasets.cifar10.load_data()
    x_train = img.conform_size(img.convert_to_grayscale(x_train / 255), IMAGE_SIZES["cifar10"], mode="crop").astype(np.float32)
    x_test = img.conform_size(img.convert_to_grayscale(x_test / 255), IMAGE_SIZES["cifar10"], mode="crop").astype(np.float32)
    y_train = y_train.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)
    return (x_train, y_train, x_test, y_test)


def _lfw_to_square(images, target_size):

    # LFW's native crop (fetch_lfw_people(resize=1.0)'s default slice) is
    # (125, 94) - non-square, and target_size (96) sits *between* those two
    # values, so neither cropping alone nor padding alone reaches it on
    # both axes: height (125) is center-cropped down to target_size, width
    # (94) is edge-padded (replicating the border pixel, not a stark black
    # band) up to target_size. Either way, every kept pixel is a real,
    # unaltered original value - unlike img.conform_size's "resize" mode
    # (interpolation), used here for exactly zero pixels, since the whole
    # point of this rewrite was to stop synthesizing/squishing pixel
    # values and train on the real native resolution instead.

    (_, height, width) = images.shape
    top = (height - target_size) // 2
    cropped = images[:, top:top + target_size, :]
    pad_total = target_size - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(cropped, ((0, 0), (0, 0), (pad_left, pad_right)), mode="edge")


def load_lfw_labeled():

    # Unlike src/image.py's own lfw_faces loader (identity labels aren't
    # needed for unsupervised MI measurement), this is an actual
    # train/test split for identity classification, so people with too few
    # images to both train and test on are excluded via
    # LFW_MIN_FACES_PER_PERSON. LFW ships with no canonical train/test
    # split for this task, so one is carved out here (stratified, so every
    # kept person appears in both splits regardless of how few images they
    # have).

    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    data = fetch_lfw_people(min_faces_per_person=LFW_MIN_FACES_PER_PERSON, resize=1.0)
    (x_train, x_test, y_train, y_test) = train_test_split(
        data.images, data.target, test_size=0.2, random_state=NOISE_SEED, stratify=data.target)
    target_size = IMAGE_SIZES["lfw_faces"]
    x_train = _lfw_to_square(x_train, target_size).astype(np.float32)
    x_test = _lfw_to_square(x_test, target_size).astype(np.float32)
    return (x_train, y_train, x_test, y_test)


def load_fer2013_labeled():

    # FER-2013 (7-class facial emotion) via the same clip-benchmark/wds_fer2013
    # Hugging Face mirror src/image.py's fer2013_hf loader uses, but with
    # labels ("cls", an int 0-6) kept rather than discarded, and using the
    # dataset's own train (28,709) / test (7,178) split as-is rather than
    # carving out a new one - unlike lfw_faces, FER-2013 already ships with
    # a canonical split for its intended (emotion) task. Native images are
    # already 48x48 (IMAGE_SIZES["fer2013_hf"]), so - unlike the previous
    # version of this loader, which squished them down to 28x28 - no
    # resize/crop is needed here at all.

    from datasets import load_dataset

    dataset = load_dataset("clip-benchmark/wds_fer2013")

    def to_arrays(split):
        images = np.stack([np.asarray(example["jpg"], dtype=np.float32) / 255 for example in dataset[split]])
        labels = np.array([example["cls"] for example in dataset[split]])
        return (images, labels)

    (x_train, y_train) = to_arrays("train")
    (x_test, y_test) = to_arrays("test")
    return (x_train, y_train, x_test, y_test)


LOADERS = {
    "mnist": load_mnist_labeled,
    "cifar10": load_cifar10_labeled,
    "lfw_faces": load_lfw_labeled,
    "fer2013_hf": load_fer2013_labeled,
}


def load_tile_mask(image_type, percent_kept):

    # Sources the per-tile MI ranking from sliding_window_mi.py's existing
    # sweep at this dataset's own window size (load_sliding_window_mi)
    # rather than running a fresh MI measurement - see pixel_pruning.py's
    # module docstring for why this is both free (already computed) and
    # finer-grained/cleaner than the coarser alternatives (a 3x3 grid of 9
    # unequal-size tiles, or a Voronoi partition of independently-measured
    # boxes). For lfw_faces/fer2013_hf the tiles measured tile the image
    # exactly (window_size divides IMAGE_SIZES[image_type] with zero
    # remainder, by construction - see TILE_WINDOW_SIZES); for mnist/
    # cifar10, edges from load_sliding_window_mi close the 1px gap the
    # sweep's last position leaves short of the far edge. Either way
    # pixel_mask_from_edges is a plain box expansion - no Voronoi step
    # needed.

    window_size = TILE_WINDOW_SIZES[image_type]
    (heatmap, edges) = load_sliding_window_mi(
        image_type, MI_RESULTS_DIR, IMAGE_SIZES[image_type], window_size=window_size, stride=window_size)
    tile_mask = build_tile_mask(heatmap, percent_kept)
    pixel_mask = pixel_mask_from_edges(tile_mask, edges)
    return (pixel_mask, tile_mask, heatmap)


def build_mnist_classifier(num_classes, image_size):

    # Two conv+conv+pool+dropout blocks (32 then 64 filters) followed by a
    # dense head - a conventional, reasonably strong CNN for 28x28
    # single-channel classification (unlike train_cnn.py's flat,
    # pooling-free network, which trades accuracy for keeping every conv
    # layer's activation map at input resolution). MNIST is easy enough at
    # this resolution that this size is already sufficient to reach ~99.6%.

    inputs = ks.Input(shape=(image_size, image_size, 1))
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


def build_augmented_classifier(num_classes, image_size):

    # A deeper, wider version of build_mnist_classifier: three
    # conv+conv+pool+dropout blocks (32/64/128 filters, vs MNIST's two at
    # 32/64) and a bigger dense head (512 vs 256), plus light train-time-only
    # data augmentation (RandomFlip/RandomTranslation/RandomZoom - Keras
    # preprocessing layers that are no-ops during model.evaluate/predict, so
    # the held-out test set this is scored on is never itself augmented).
    # Used only for CIFAR-10 - a meaningfully harder task than MNIST at this
    # same grayscale/28x28 resolution, which saw train accuracy pull well
    # ahead of val accuracy at MNIST's plain, unaugmented 2-block
    # architecture. (lfw_faces/fer2013_hf use build_transfer_classifier
    # instead - see its docstring for why a bigger from-scratch network
    # like this one doesn't fix what's wrong with those two.) Only a
    # horizontal flip and small shifts/zooms are used - nothing that would
    # plausibly flip a CIFAR-10 class label.

    inputs = ks.Input(shape=(image_size, image_size, 1))
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


# MobileNetV2's own pretrained-weights resolution we standardize both
# transfer-learning datasets on - one of the handful of sizes (96/128/160/
# 192/224) Keras actually ships ImageNet weights for, and conveniently
# equal to lfw_faces's own IMAGE_SIZES entry already (chosen partly for
# this reason), so only fer2013_hf (48px) needs the internal upsize below.
TRANSFER_BACKBONE_SIZE = 96


def build_transfer_classifier(num_classes, image_size):

    # Transfer learning instead of a from-scratch CNN: both lfw_faces and
    # fer2013_hf are fundamentally data-scarce problems (lfw_faces has only
    # on the order of tens of images per identity across dozens of
    # identities; fer2013_hf's ~29k training images is still small next to
    # what a CNN typically needs to learn good features from zero) - a
    # bigger build_augmented_classifier-style network doesn't fix a
    # data-scarcity problem, it just overfits a bigger model to the same
    # small dataset. A pretrained ImageNet backbone (MobileNetV2) starts
    # with strong generic visual features already learned from millions of
    # images, and only has to adapt them to this dataset rather than learn
    # from scratch - the standard approach for both face-identity and
    # emotion recognition on limited data in practice.
    #
    # Grayscale (1-channel) input is replicated to 3 channels (Concatenate)
    # since MobileNetV2 was trained on RGB, and rescaled from this
    # pipeline's [0, 1] range to the [-1, 1] range MobileNetV2 expects
    # (equivalent to the official mobilenet_v2.preprocess_input for
    # originally-[0,255] input, just applied directly to [0,1] instead of
    # scaling up through 255 first). fer2013_hf (48px) is upsized to
    # TRANSFER_BACKBONE_SIZE via an internal Resizing layer *after* the
    # input - i.e. after pruning has already been applied to the true
    # 48x48 array - so the backbone gets a size it's actually pretrained
    # for without changing what resolution the MI-guided pruning mask
    # itself operates on; lfw_faces (already 96px) passes straight through.
    #
    # The backbone starts frozen (trainable=False) here; run_condition's
    # TRANSFER_LEARNING_DATASETS branch unfreezes it for a second,
    # low-learning-rate fine-tuning phase after this initial head-only
    # phase - the standard two-phase transfer-learning recipe.

    inputs = ks.Input(shape=(image_size, image_size, 1))
    x = inputs
    if image_size != TRANSFER_BACKBONE_SIZE:
        x = ks.layers.Resizing(TRANSFER_BACKBONE_SIZE, TRANSFER_BACKBONE_SIZE)(x)
    x = ks.layers.Concatenate(axis=-1)([x, x, x])
    x = ks.layers.Rescaling(2.0, offset=-1.0)(x)
    base_model = ks.applications.MobileNetV2(
        input_shape=(TRANSFER_BACKBONE_SIZE, TRANSFER_BACKBONE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model(x, training=False)
    x = ks.layers.GlobalAveragePooling2D()(x)
    x = ks.layers.Dropout(0.3)(x)
    outputs = ks.layers.Dense(num_classes, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Stashed so run_condition's fine-tuning phase can unfreeze this
    # specific layer later - a plain attribute on the Model object, not a
    # Keras API of its own.
    model.transfer_base = base_model
    return model


MODEL_BUILDERS = {
    "mnist": build_mnist_classifier,
    "cifar10": build_augmented_classifier,
    "lfw_faces": build_transfer_classifier,
    "fer2013_hf": build_transfer_classifier,
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


def train_transfer_learning(model, xt, y_train, xv, y_test, batch_size, config):

    # Two-phase fine-tuning, the standard recipe for a pretrained backbone:
    #   1. "head" phase - backbone frozen (as build_transfer_classifier left
    #      it), train only the new classification head. The backbone's
    #      pretrained weights would otherwise get scrambled by large,
    #      randomly-initialized gradients flowing back from the brand-new
    #      head during its first few, least-informed updates.
    #   2. "finetune" phase - unfreeze the backbone and continue training
    #      end-to-end at a much lower learning rate (config["finetune_lr"],
    #      typically ~100x smaller than the head phase's), so the
    #      pretrained features get gently adapted to this dataset rather
    #      than overwritten.
    # Both phases' per-epoch histories are concatenated into one dict
    # (rather than returning two separate History objects) so the rest of
    # run_condition - epoch counts, best-val-accuracy, the saved history
    # JSON - doesn't need to know two model.fit() calls happened.

    xt = np.expand_dims(xt, axis=3)
    xv = np.expand_dims(xv, axis=3)

    head_history = model.fit(
        xt, y_train, validation_data=(xv, y_test),
        batch_size=batch_size, epochs=config["head_epochs"],
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=config["head_patience"], restore_best_weights=True)],
        verbose=2,
    )

    model.transfer_base.trainable = True
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=config["finetune_lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    finetune_history = model.fit(
        xt, y_train, validation_data=(xv, y_test),
        batch_size=batch_size, epochs=config["finetune_epochs"],
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=config["finetune_patience"], restore_best_weights=True)],
        verbose=2,
    )

    combined = {key: head_history.history[key] + finetune_history.history[key] for key in head_history.history}
    return combined


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
    model = MODEL_BUILDERS[image_type](num_classes, IMAGE_SIZES[image_type])
    if image_type in TRANSFER_LEARNING_DATASETS:
        history = train_transfer_learning(model, xt, y_train, xv, y_test, batch_size, config)
    else:
        history = model.fit(
            np.expand_dims(xt, axis=3), y_train,
            validation_data=(np.expand_dims(xv, axis=3), y_test),
            batch_size=batch_size, epochs=config["epochs"],
            callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True)],
            verbose=2,
        ).history
    (test_loss, test_accuracy) = model.evaluate(np.expand_dims(xv, axis=3), y_test, verbose=0)

    metrics = {
        "image_type": image_type,
        "condition": condition,
        "percent_kept": percent_kept,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_trained": len(history["loss"]),
        "best_val_accuracy": float(max(history["val_accuracy"])),
    }
    with open(paths["history"], "w") as handle:
        json.dump({key: [float(v) for v in values] for (key, values) in history.items()}, handle)
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

    # Merged into (rather than overwriting) any existing summary file, so a
    # partial run (--datasets lfw_faces,fer2013_hf, say) updates just those
    # datasets' entries instead of silently dropping every other dataset's
    # results that a previous, differently-scoped run had already recorded
    # here - this file previously lost its mnist/cifar10 entries exactly
    # this way when a lfw_faces/fer2013_hf-only run overwrote it wholesale.
    summary_path = os.path.join(RESULTS_DIR, f"summary_p{int(round(args.percent_kept * 100))}.json")
    existing_metrics = []
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            existing_metrics = json.load(handle)
    new_keys = {(m["image_type"], m["condition"]) for m in all_metrics}
    merged_metrics = [m for m in existing_metrics if (m["image_type"], m["condition"]) not in new_keys] + all_metrics
    with open(summary_path, "w") as handle:
        json.dump(merged_metrics, handle, indent=2)

    print("\n=== Summary ===")
    for metrics in all_metrics:
        print(f"{metrics['image_type']:>8} | {metrics['condition']:>8} | "
              f"test_acc={metrics['test_accuracy']:.4f} | epochs={metrics['epochs_trained']}")

    if set(conditions) == set(CONDITIONS):
        # Only a full run of all three conditions makes the side-by-side
        # comparison plot meaningful, same reasoning as
        # sliding_window_mi.py's plot_combined guard on a full window-size
        # sweep. Plotted from merged_metrics (every dataset ever recorded
        # in this summary file), not just all_metrics (this run's subset),
        # for the same reason the summary merge above exists - a
        # lfw_faces/fer2013_hf-only run shouldn't make the comparison chart
        # forget mnist/cifar10's panels.
        plot_comparison(merged_metrics, args.percent_kept)


if __name__ == "__main__":
    main()
