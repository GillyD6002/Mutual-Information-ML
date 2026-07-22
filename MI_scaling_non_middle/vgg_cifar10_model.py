import argparse
import os
import sys

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Builds the "pretrained model" the VGG conv-layer MI sweep/pruning experiment (see
# vgg_conv_sliding_mi.py, vgg_conv_pruning.py) operates on: keras.applications.VGG16's
# official ImageNet-pretrained weights, fine-tuned into a CIFAR-10 classifier via the
# same two-phase transfer-learning recipe train_pruned_classifiers.py already uses for
# lfw_faces/fer2013_hf (build_transfer_classifier/train_transfer_learning) - frozen-
# backbone head training, then a low-LR full fine-tune. Unlike this project's other
# CIFAR-10 handling (src/image.py, train_cnn.py, train_pruned_classifiers.py all convert
# CIFAR-10 to grayscale at 28x28), this keeps it in full color at native 32x32 - VGG16's
# ImageNet weights are for 3-channel input, and grayscale-then-replicated-to-3-channels
# would throw away exactly the color information those weights were trained to use.
#
# CIFAR-10's native 32x32 is far below VGG16's 224x224 ImageNet training resolution -
# resized up via an internal Resizing layer (the same "resize inside the model, after the
# input" approach build_transfer_classifier uses for fer2013_hf's 48px images) rather than
# resizing the raw dataset once up front, so the true 32x32 array stays what every later
# script actually loads/prunes at (irrelevant here since this script prunes nothing, but
# kept for consistency with that established pattern).
#
# CONV_LAYER_NAMES/CONV_LAYER_SPATIAL_SIZE below are VGG16-at-224-input's own architecture
# facts (confirmed by direct inspection: 13 conv layers, spatial sizes 224 down to 14,
# every one of which divides evenly by GRID_SIZE=7) - vgg_conv_sliding_mi.py and
# vgg_conv_pruning.py both import these rather than re-deriving them, the same way
# cnn_activation_sliding_mi.py imports CONV_LAYER_NAMES from train_cnn.py.

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "vgg")
MODEL_PATH = os.path.join(RESULTS_DIR, "vgg16_cifar10.keras")
BASELINE_OUTPUTS_PATH = os.path.join(RESULTS_DIR, "vgg16_cifar10_baseline_outputs.npy")
EVAL_IMAGES_PATH = os.path.join(RESULTS_DIR, "vgg16_cifar10_eval_images.npy")

BACKBONE_SIZE = 224  # VGG16's native ImageNet training resolution.
BACKBONE_NAME = "vgg16"  # keras.applications.VGG16's default Model name, used to find the
# nested backbone submodel inside the full classifier (see get_backbone below).
NUM_CLASSES = 10
NUM_EVAL_IMAGES = 2000  # Fixed held-out sample whose softmax outputs become the pruning
# script's baseline-to-diff-against (vgg_conv_pruning.py), not a training hyperparameter.

CONV_LAYER_NAMES = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
]
CONV_LAYER_SPATIAL_SIZE = {
    "block1_conv1": 224, "block1_conv2": 224,
    "block2_conv1": 112, "block2_conv2": 112,
    "block3_conv1": 56, "block3_conv2": 56, "block3_conv3": 56,
    "block4_conv1": 28, "block4_conv2": 28, "block4_conv3": 28,
    "block5_conv1": 14, "block5_conv2": 14, "block5_conv3": 14,
}
GRID_SIZE = 7  # Every conv layer tiled into the same GRID_SIZE x GRID_SIZE non-overlapping
# grid (window == stride == spatial_size // GRID_SIZE), so every layer's heatmap/mask is
# directly comparable at a glance - the same "one shared grid across every layer"
# convention cnn_activation_sliding_mi.py already established for its 3-conv-layer CNN.

BATCH_SIZE = 64  # Conservative default: VGG16 at 224x224 input is memory-heavy (early
# conv layers' activation maps are 224x224x64), well below train_pruned_classifiers.py's
# BATCH_SIZE=128 used for its much smaller/lower-resolution models.
HEAD_EPOCHS = 15
HEAD_PATIENCE = 5
FINETUNE_EPOCHS = 30
FINETUNE_PATIENCE = 8
FINETUNE_LR = 1e-5


def grid_window_size(layer_name):
    return CONV_LAYER_SPATIAL_SIZE[layer_name] // GRID_SIZE


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_cifar10_color():

    # Native color CIFAR-10 at its native 32x32 resolution, rescaled to [0, 1] - no
    # grayscale conversion (see module docstring) and no cropping/resizing here (the
    # model's internal Resizing layer handles the up-size to BACKBONE_SIZE).

    ((x_train, y_train), (x_test, y_test)) = ks.datasets.cifar10.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    x_test = (x_test / 255.0).astype(np.float32)
    y_train = y_train.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)
    return (x_train, y_train, x_test, y_test)


@keras.saving.register_keras_serializable(package="vgg_cifar10")
class VGGPreprocess(ks.layers.Layer):

    # keras.applications.vgg16.preprocess_input expects [0, 255]-range RGB and converts
    # to the BGR, ImageNet-mean-centered input VGG16's pretrained weights were actually
    # trained on - skipping this (e.g. just feeding raw [0, 1] RGB) would silently
    # mismatch the pretrained weights' expected input distribution and hurt transfer
    # performance. load_cifar10_color already rescaled to [0, 1], so that's undone (* 255)
    # before calling it. A custom Layer subclass (serialized by class reference) is used
    # instead of a Lambda layer (which pickles/inspects its function and is refused by
    # Keras 3's safe-mode deserialization, even when the function is a registered,
    # non-lambda callable) - this reloads cleanly with no safe_mode override needed in
    # vgg_conv_sliding_mi.py/vgg_conv_pruning.py.

    def call(self, x):
        return ks.applications.vgg16.preprocess_input(x * 255.0)


def build_model():

    # Same two-phase-transfer-learning shape as train_pruned_classifiers.py's
    # build_transfer_classifier: a frozen pretrained backbone (unfrozen later by train()'s
    # fine-tune phase) feeding a small new classification head. base_model is stashed as
    # a plain attribute (model.vgg_base) so train() can flip .trainable on it later,
    # exactly like that function stashes model.transfer_base.

    inputs = ks.Input(shape=(32, 32, 3))
    x = ks.layers.Resizing(BACKBONE_SIZE, BACKBONE_SIZE)(inputs)
    x = VGGPreprocess(name="vgg_preprocess")(x)
    base_model = ks.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=(BACKBONE_SIZE, BACKBONE_SIZE, 3))
    base_model.trainable = False
    x = base_model(x, training=False)
    x = ks.layers.GlobalAveragePooling2D()(x)
    x = ks.layers.Dropout(0.3)(x)
    outputs = ks.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.vgg_base = base_model
    return model


def get_backbone(model):

    # Recovers the nested VGG16 submodel from a full classifier - either one just built
    # by build_model (which stashed it at model.vgg_base) or one reloaded from disk via
    # load_trained_model (which has no such attribute, since Keras doesn't restore plain
    # Python attributes on load, but does preserve the nested submodel itself under its
    # own name). vgg_conv_sliding_mi.py/vgg_conv_pruning.py both need this to look up
    # individual conv layers (get_backbone(model).get_layer(layer_name)) or to rebuild the
    # backbone's forward pass layer-by-layer for mask insertion.

    if hasattr(model, "vgg_base"):
        return model.vgg_base
    return model.get_layer(BACKBONE_NAME)


def train(model, x_train, y_train, x_test, y_test, batch_size):

    # Phase 1: backbone frozen (as build_model left it), train only the new head - avoids
    # the brand-new head's large, randomly-initialized early gradients scrambling the
    # pretrained backbone's weights. Phase 2: unfreeze the backbone, continue end-to-end
    # at a much lower learning rate. Identical structure to train_pruned_classifiers.py's
    # train_transfer_learning, just without that function's extra np.expand_dims (CIFAR-10
    # is already channel-last color, not single-channel grayscale).

    head_history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test),
        batch_size=batch_size, epochs=HEAD_EPOCHS,
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=HEAD_PATIENCE, restore_best_weights=True)],
        verbose=2,
    )

    backbone = get_backbone(model)
    backbone.trainable = True
    model.compile(optimizer=ks.optimizers.Adam(learning_rate=FINETUNE_LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    finetune_history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test),
        batch_size=batch_size, epochs=FINETUNE_EPOCHS,
        callbacks=[ks.callbacks.EarlyStopping(monitor="val_loss", patience=FINETUNE_PATIENCE, restore_best_weights=True)],
        verbose=2,
    )

    combined = {key: head_history.history[key] + finetune_history.history[key] for key in head_history.history}
    return combined


def load_trained_model():
    return ks.models.load_model(MODEL_PATH, custom_objects={"VGGPreprocess": VGGPreprocess})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help=f"Per-replica batch size (default: {BATCH_SIZE})")
    parser.add_argument("--num-eval-images", type=int, default=NUM_EVAL_IMAGES,
                         help=f"Held-out test images to save baseline outputs for (default: {NUM_EVAL_IMAGES})")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    (x_train, y_train, x_test, y_test) = load_cifar10_color()

    # Same MirroredStrategy pattern as train_cnn.py/train_pruned_classifiers.py: splits
    # each batch across every visible GPU, transparently reduces to single-GPU with only
    # one device visible. train()'s fine-tune phase re-compiles the model with a fresh
    # optimizer partway through - that optimizer's slot variables are built lazily on
    # its first apply_gradients call, so the *entire* train() call (not just
    # build_model()) has to run inside strategy.scope(), exactly like
    # train_pruned_classifiers.py wraps its whole run_condition call (build + both
    # compile/fit phases) in one `with strategy.scope():`. Splitting it (build_model()
    # in scope, train() outside) builds the fine-tune optimizer's variables under the
    # default (non-distributed) strategy instead, which raises "Mixing different
    # tf.distribute.Strategy objects" as soon as the fine-tune phase's first
    # apply_gradients call runs.
    strategy = tf.distribute.MirroredStrategy()
    print(f"MirroredStrategy running on {strategy.num_replicas_in_sync} device(s)")
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        model = build_model()
        train(model, x_train, y_train, x_test, y_test, global_batch_size)

    (test_loss, test_accuracy) = model.evaluate(x_test, y_test, batch_size=global_batch_size, verbose=0)
    print(f"Final held-out test accuracy: {test_accuracy:.4f} (test_loss={test_loss:.4f})")

    model.save(MODEL_PATH)
    print(f"Saved fine-tuned model -> {MODEL_PATH}")

    # A fixed evaluation sample (first num_eval_images of the test split) whose baseline
    # (unpruned) softmax outputs the pruning script's KL-divergence/agreement-rate
    # criterion diffs every pruned condition against - saved alongside the images
    # themselves so vgg_conv_pruning.py doesn't need to re-derive "which images" from
    # scratch (and would otherwise risk silently diffing against a different sample if
    # e.g. NUM_EVAL_IMAGES's default ever changes).
    num_eval = min(args.num_eval_images, x_test.shape[0])
    x_eval = x_test[:num_eval]
    np.save(EVAL_IMAGES_PATH, x_eval)
    baseline_outputs = model.predict(x_eval, batch_size=global_batch_size, verbose=1)
    np.save(BASELINE_OUTPUTS_PATH, baseline_outputs.astype(np.float32))
    print(f"Saved {num_eval} eval images -> {EVAL_IMAGES_PATH}")
    print(f"Saved baseline outputs {baseline_outputs.shape} -> {BASELINE_OUTPUTS_PATH}")


if __name__ == "__main__":
    main()
