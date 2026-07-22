import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import renyi_mi
from MI_scaling_non_middle import vgg_cifar10_model as vm

# Re-applies cnn_activation_sliding_mi.py's technique (slide a fixed-size window across a
# CNN layer's activation map and measure I(window; rest of map) at every position) to the
# fine-tuned VGG16 built by vgg_cifar10_model.py, across all 13 conv layers. Unlike
# cnn_activation_sliding_mi.py, which calls src.mine.run_bipartition (a fresh neural
# classifier trained per measurement), this calls src.renyi_mi.renyi_mutual_information -
# a closed-form, training-free estimator - since training 13 layers x 49 positions worth
# of classifiers would be far too slow at VGG16's channel counts. See src/renyi_mi.py's
# module docstring for why that estimator was chosen instead.
#
# Every layer uses the same GRID_SIZE x GRID_SIZE non-overlapping grid (window == stride
# == that layer's own spatial_size // GRID_SIZE - see vgg_cifar10_model.grid_window_size),
# so heatmaps are directly comparable across layers exactly like
# cnn_activation_sliding_mi.py's shared-grid convention - and, unlike that script's 28px
# images (which leave a 1px gap needing pixel_pruning.py's edge-extension trick), every
# one of VGG16-at-224's 13 layer sizes divides evenly by 7, so no gap-closing is needed
# here: get_positions below is a clean, exact tiling.
#
# Run as: python -m MI_scaling_non_middle.vgg_conv_sliding_mi
# (needs vgg_cifar10_model.py to have been run first, to produce the trained model at
# vgg_cifar10_model.MODEL_PATH)

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "vgg")

# Only N drives this estimator's statistical quality (see src/renyi_mi.py) - not the
# dimensionality of each layer's activations - so this is far smaller than the
# classifier-based sweeps' NUM_IMAGES (e.g. sliding_window_mi.py's 30000). Chosen so the
# largest layer (block1_conv1/2, 224x224x64) stays a manageable ~6GB resident array
# (500 * 224 * 224 * 64 * 4 bytes) - lower via --num-images on a more memory-constrained
# box, or raise it on a bigger one.
NUM_IMAGES = 500
EXTRACTION_BATCH_SIZE = 32


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_positions(layer_name):
    window = vm.grid_window_size(layer_name)
    spatial_size = vm.CONV_LAYER_SPATIAL_SIZE[layer_name]
    return list(range(0, spatial_size, window))


def result_paths(layer_name):
    tag = f"cifar10_vgg_{layer_name}_mi_w{vm.grid_window_size(layer_name)}"
    return {
        "heatmap": os.path.join(RESULTS_DIR, f"{tag}.npy"),
        "positions": os.path.join(RESULTS_DIR, f"{tag}_positions.npy"),
        "heatmap_pdf": os.path.join(RESULTS_DIR, f"{tag}_heatmap.pdf"),
        "heatmap_png": os.path.join(RESULTS_DIR, f"{tag}_heatmap.png"),
    }


def split_inner_outer(activations, top, left, window):

    # Splits one layer's (N, H, W, C) activation array into the inner window's
    # per-sample flattened vectors and the outer (rest-of-map) per-sample flattened
    # vectors - the same "patch vs. everything else" split mine.py's get_finite_dataset
    # builds joint/marginal samples around, but consumed directly (no cross-image
    # splicing needed - see src/renyi_mi.py's module docstring on why this estimator
    # doesn't need marginal samples at all).

    inner = activations[:, top:top + window, left:left + window, :]
    inner_flat = inner.reshape(inner.shape[0], -1)
    spatial_mask = np.ones(activations.shape[1:3], dtype=bool)
    spatial_mask[top:top + window, left:left + window] = False
    outer = activations[:, spatial_mask, :]
    outer_flat = outer.reshape(outer.shape[0], -1)
    return (inner_flat, outer_flat)


def build_preprocess_model(model):

    # The Resizing + VGGPreprocess layers sit directly in the outer model's own graph
    # (unlike the VGG16 backbone, which is nested as a single opaque call - see
    # extract_layer_activations below), so this submodel can be built straight from
    # model.input with no special handling.

    return ks.Model(inputs=model.input, outputs=model.get_layer("vgg_preprocess").output)


def extract_layer_activations(backbone, preprocessed_images, layer_name, batch_size=EXTRACTION_BATCH_SIZE):

    # Building `ks.Model(inputs=model.input, outputs=backbone.get_layer(name).output)`
    # directly would fail (a disconnected-graph error): the backbone was called as one
    # opaque node inside the outer model (`base_model(x, training=False)` in
    # vgg_cifar10_model.build_model), so its *internal* layer tensors aren't reachable
    # from the outer model's own graph. Building the activation model from the backbone's
    # own standalone input/graph instead (backbone.input, a plain Input(224,224,3) tensor
    # from when VGG16(...) was first constructed) sidesteps that entirely, and still
    # shares the exact same trained weights (verified against a real load/predict
    # roundtrip during development). preprocessed_images must already be the (224,224,3)
    # output of build_preprocess_model - i.e. what the backbone actually sees at inference.

    layer_output = backbone.get_layer(layer_name).output
    activation_model = ks.Model(inputs=backbone.input, outputs=layer_output)
    return activation_model.predict(preprocessed_images, batch_size=batch_size, verbose=1)


def run_sliding_sweep(layer_name, activations):
    ensure_results_dir()
    positions = get_positions(layer_name)
    paths = result_paths(layer_name)

    heatmap = np.full((len(positions), len(positions)), np.nan)
    if os.path.exists(paths["heatmap"]):
        existing = np.load(paths["heatmap"])
        if existing.shape == heatmap.shape:
            heatmap = existing

    window = vm.grid_window_size(layer_name)
    for (row_index, top) in enumerate(positions):
        for (col_index, left) in enumerate(positions):
            if not np.isnan(heatmap[row_index, col_index]):
                print(f"[{layer_name}] window at (top={top}, left={left}) already complete, skipping", flush=True)
                continue
            print(f"[{layer_name}] window at (top={top}, left={left})", flush=True)
            (inner, outer) = split_inner_outer(activations, top, left, window)
            mi = renyi_mi.renyi_mutual_information(inner, outer)
            heatmap[row_index, col_index] = mi
            print(f"  renyi_mi={mi:.4f}", flush=True)
            np.save(paths["heatmap"], heatmap)

    np.save(paths["positions"], np.asarray(positions))
    return (heatmap, positions)


def plot_heatmap(layer_name, heatmap, positions):
    ensure_results_dir()
    paths = result_paths(layer_name)

    plt.figure(figsize=(7, 6))
    plt.imshow(np.clip(heatmap, 0, None), cmap="viridis", origin="upper")
    plt.colorbar(label="Matrix-based Renyi MI (bits)")
    tick_labels = [str(p) for p in positions]
    plt.xticks(range(len(positions)), tick_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(positions)), tick_labels, fontsize=8)
    plt.xlabel("Window left edge (activation-map pixels)")
    plt.ylabel("Window top edge (activation-map pixels)")
    window = vm.grid_window_size(layer_name)
    plt.title(f"Sliding {window}x{window} window MI map (VGG16 {layer_name}, cifar10)", fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["heatmap_pdf"])
    plt.savefig(paths["heatmap_png"], dpi=150)
    plt.close()


def plot_combined(heatmaps_by_layer):
    ensure_results_dir()
    vmax = max(np.nanmax(np.clip(heatmap, 0, None)) for (heatmap, _) in heatmaps_by_layer.values())

    (fig, axes_row) = plt.subplots(1, len(vm.CONV_LAYER_NAMES), figsize=(3.2 * len(vm.CONV_LAYER_NAMES), 4.5))
    for (axes, layer_name) in zip(axes_row, vm.CONV_LAYER_NAMES):
        (heatmap, _) = heatmaps_by_layer[layer_name]
        image = axes.imshow(np.clip(heatmap, 0, None), cmap="viridis", origin="upper", vmin=0, vmax=vmax)
        axes.set_title(layer_name, fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])
    fig.colorbar(image, ax=axes_row, fraction=0.02, pad=0.02, label="Matrix-based Renyi MI (bits)")
    fig.suptitle(f"Sliding-window Renyi MI maps across VGG16 conv layers (cifar10, {vm.GRID_SIZE}x{vm.GRID_SIZE} grid)", fontsize=15)

    pdf_path = os.path.join(RESULTS_DIR, "cifar10_vgg_conv_sliding_combined_heatmap.pdf")
    png_path = os.path.join(RESULTS_DIR, "cifar10_vgg_conv_sliding_combined_heatmap.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default=None,
                         help="Comma-separated subset of vgg_cifar10_model.CONV_LAYER_NAMES to run (default: all 13)")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                         help=f"Images used per layer's MI measurement (default: {NUM_IMAGES})")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    layer_names = vm.CONV_LAYER_NAMES
    if args.layers:
        layer_names = [name.strip() for name in args.layers.split(",")]

    model = vm.load_trained_model()
    backbone = vm.get_backbone(model)
    preprocess_model = build_preprocess_model(model)

    (x_train, _, _, _) = vm.load_cifar10_color()
    images = x_train[:args.num_images]
    preprocessed_images = preprocess_model.predict(images, batch_size=EXTRACTION_BATCH_SIZE, verbose=1)

    heatmaps_by_layer = {}
    for layer_name in layer_names:
        print(f"=== {layer_name} ===", flush=True)
        activations = extract_layer_activations(backbone, preprocessed_images, layer_name)
        (heatmap, positions) = run_sliding_sweep(layer_name, activations)
        plot_heatmap(layer_name, heatmap, positions)
        heatmaps_by_layer[layer_name] = (heatmap, positions)
        # Freed explicitly (rather than left to the next loop iteration's reassignment)
        # so peak memory is one layer's activation array at a time, not two - matters
        # most for the largest layers (block1_conv1/2 at 224x224x64).
        del activations

    if len(heatmaps_by_layer) == len(vm.CONV_LAYER_NAMES):
        # Only the full 13-layer sweep makes the shared-color-scale combined panel
        # meaningful - same guard sliding_window_mi.py/cnn_activation_sliding_mi.py use
        # for their own combined plots.
        plot_combined(heatmaps_by_layer)


if __name__ == "__main__":
    main()
