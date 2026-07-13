import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import image as img

# Trains a small CNN classifier on CIFAR-10 and saves the trained model plus
# the intermediate activations of its 3 conv layers, so that the sliding-
# window MI technique in sliding_window_mi.py can be re-applied to those
# activation maps as if they were "images" - see
# cnn_activation_sliding_mi.py, which consumes the .npy files this script
# produces.
#
# All 3 conv layers use stride 1 and "same" padding (no pooling anywhere in
# between), so every layer's activation map stays exactly 28 x 28 - matching
# img.DEFAULT_IMAGE_SIZE, the size every pixel-space MI experiment in this
# project already uses. That's what makes an activation map directly
# comparable to a pixel image under the sliding-window technique: the
# window/stride geometry (positions, grid resolution) is identical at every
# layer and at the input, only the number of channels per position differs.
#
# CIFAR-10 is converted to grayscale and center-cropped to 28 x 28 exactly
# the way img.get_images("cifar10", ...) already does, rather than trained
# on natively-sized color images, so that the "first NUM_IMAGES images" used
# here for activation extraction are pixel-identical to the ones
# sliding_window_mi.py already computed direct pixel-space MI heatmaps for.

ROOT_RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
MODEL_DIR = os.path.join(ROOT_RESULTS_DIR, "cnn")

IMAGE_SIZE = img.DEFAULT_IMAGE_SIZE

# Channel counts are kept small (rather than a more typical 32/64/128) purely
# for disk/RAM footprint: each layer's saved activation array is
# NUM_IMAGES x 28 x 28 x channels floats, and the sliding-window MI sweep
# needs the whole array resident in memory. At channels=64 and
# NUM_IMAGES=10000 the largest layer is already ~2.4GB as float32; go bigger
# than this only if that's an acceptable resident-memory cost.
CONV_FILTERS = [16, 32, 64]
DENSE_UNITS = [256, 128]
NUM_CLASSES = 10

# Deliberately smaller than sliding_window_mi.py's NUM_IMAGES=30000: that
# constant sizes a single-channel (1x) pixel array, while this sizes a
# multi-channel (up to 64x) activation array, so keeping this pool smaller
# keeps per-layer activation files at a comparable order of magnitude.
NUM_IMAGES = 10000

BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 6

CONV_LAYER_NAMES = ["conv1", "conv2", "conv3"]


def load_cifar10_grayscale():

    # Loads CIFAR-10's native train/test split (labels included, unlike
    # img.get_images which discards them), then applies the same
    # grayscale-conversion + center-crop-to-28x28 pipeline img.get_images
    # uses for source="cifar10". Slicing to NUM_IMAGES happens on x_train
    # only (see module docstring): since img._load_keras_dataset builds its
    # pool as concatenate([train, test])[:num_images] and NUM_IMAGES here is
    # smaller than the 50000-image train split, "first NUM_IMAGES of the
    # concatenated pool" and "first NUM_IMAGES of x_train" are the same
    # images in the same order.

    ((x_train, y_train), (x_test, y_test)) = ks.datasets.cifar10.load_data()
    x_train = img.conform_size(img.convert_to_grayscale(x_train / 255), IMAGE_SIZE, mode="crop")
    x_test = img.conform_size(img.convert_to_grayscale(x_test / 255), IMAGE_SIZE, mode="crop")
    y_train = y_train.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)
    return (x_train, y_train, x_test, y_test)


def build_classifier():

    # 3 same-padding, stride-1 conv layers (named conv1/conv2/conv3, so
    # extract_activations can look them up by name) followed by 2 dense
    # ("deep") layers before the final 10-way softmax classification head.
    # No pooling/strided conv anywhere, so every conv layer's output is
    # still 28 x 28 x channels - see module docstring for why that matters.

    inputs = ks.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = inputs
    for (index, filters) in enumerate(CONV_FILTERS):
        x = ks.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu",
                              name=CONV_LAYER_NAMES[index])(x)
    x = ks.layers.Flatten()(x)
    for units in DENSE_UNITS:
        x = ks.layers.Dense(units, activation="relu")(x)
    outputs = ks.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_classifier(model, x_train, y_train, x_test, y_test):
    model.fit(
        np.expand_dims(x_train, axis=3), y_train,
        validation_data=(np.expand_dims(x_test, axis=3), y_test),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        verbose=2,
    )
    return model


def extract_activations(model, images):

    # Builds a side model sharing conv1/conv2/conv3's weights with the
    # trained classifier (same trick as mine.py's Model.build_model calling
    # one shared model_core twice) and runs it once over `images` to collect
    # every conv layer's activations in a single forward pass, rather than
    # slicing the classifier apart three separate times.

    conv_outputs = [model.get_layer(name).output for name in CONV_LAYER_NAMES]
    activation_model = ks.Model(inputs=model.input, outputs=conv_outputs)
    activations = activation_model.predict(np.expand_dims(images, axis=3), batch_size=BATCH_SIZE, verbose=1)
    return dict(zip(CONV_LAYER_NAMES, activations))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    (x_train, y_train, x_test, y_test) = load_cifar10_grayscale()

    model = build_classifier()
    train_classifier(model, x_train, y_train, x_test, y_test)
    (_, test_accuracy) = model.evaluate(np.expand_dims(x_test, axis=3), y_test, verbose=0)
    print(f"Final held-out test accuracy: {test_accuracy:.4f}")

    model.save(os.path.join(MODEL_DIR, "cifar10_cnn.keras"))

    activation_images = x_train[:NUM_IMAGES]
    activations = extract_activations(model, activation_images)
    for (name, layer_activations) in activations.items():
        path = os.path.join(MODEL_DIR, f"cifar10_{name}_activations.npy")
        np.save(path, layer_activations.astype(np.float32))
        print(f"Saved {name} activations {layer_activations.shape} -> {path}")


if __name__ == "__main__":
    main()
