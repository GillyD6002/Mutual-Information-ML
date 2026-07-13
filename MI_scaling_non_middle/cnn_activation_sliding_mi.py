import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mine
from MI_scaling_non_middle.sliding_window_mi import build_sliding_region_fn
from MI_scaling_non_middle.train_cnn import MODEL_DIR, IMAGE_SIZE, CONV_LAYER_NAMES

# Re-applies sliding_window_mi.py's technique - slide a fixed-size window
# across the image and measure I(window; rest of image) at every position -
# to the CNN activation maps train_cnn.py saves, instead of to raw pixel
# images. Each conv layer's activations are just another (N, 28, 28,
# channels) array to mine.run_bipartition, which now accepts a pre-loaded
# `images` array directly (see mine.py's `images` parameter) rather than
# only one of img.get_images's named pixel datasets - the multi-channel
# splicing/flattening machinery in mine.py already generalizes to that
# without modification (get_finite_dataset splices [top:bottom, left:right]
# regions, which numpy indexing preserves any trailing channel dimension
# through unchanged; Model.build_model flattens whatever channel depth its
# input has before its first Dense layer).
#
# Unlike sliding_window_mi.py, which sweeps WINDOW_SIZES 3-8 at STRIDE=3,
# this uses a single fixed window size/stride pair chosen to tile the image
# into exactly a 7x7 grid of positions (0, 4, 8, ..., 24 along each axis) -
# one heatmap per conv layer, all directly comparable at a glance since
# they share the same window geometry, rather than a whole family of sizes
# per layer.
#
# Run as: python -m MI_scaling_non_middle.cnn_activation_sliding_mi

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "cnn")
WINDOW_SIZE = 4
STRIDE = 4  # (28 - 4) / 6 = 4 exactly -> 7 positions per axis

PARAM_SETTINGS = dict(
    drop=0,
    learn=1e-4,
    layers="[256, 256]",
    patience=12,
    optm="rms",
    val=1 / 7,
    batch=64,
    epoch=60,
)
EVAL_STEPS = 400


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_positions():
    return list(range(0, IMAGE_SIZE - WINDOW_SIZE + 1, STRIDE))


def result_paths(layer_name):
    tag = f"cifar10_{layer_name}_sliding_w{WINDOW_SIZE}_s{STRIDE}"
    return {
        "direct": os.path.join(RESULTS_DIR, f"{tag}_mi_direct.npy"),
        "indirect": os.path.join(RESULTS_DIR, f"{tag}_mi_indirect.npy"),
        "positions": os.path.join(RESULTS_DIR, f"{tag}_positions.npy"),
        "heatmap_pdf": os.path.join(RESULTS_DIR, f"{tag}_heatmap.pdf"),
        "heatmap_png": os.path.join(RESULTS_DIR, f"{tag}_heatmap.png"),
    }


def load_activations(layer_name):
    path = os.path.join(MODEL_DIR, f"cifar10_{layer_name}_activations.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `python -m MI_scaling_non_middle.train_cnn` first "
            "to train the CNN and save its layer activations."
        )
    return np.load(path)


def run_sliding_sweep(layer_name, activations):
    ensure_results_dir()
    positions = get_positions()
    paths = result_paths(layer_name)

    # image_type/num_images/strength are unused here (activations is passed
    # directly to run_bipartition via `images`), but algorithm is still
    # read from this dict, so the same alg_settings shape as
    # sliding_window_mi.py is kept for consistency.
    alg_settings = dict(
        image_type="cifar10",
        num_images=activations.shape[0],
        strength="small",
        algorithm="logistic",
    )

    heatmap_direct = np.full((len(positions), len(positions)), np.nan)
    heatmap_indirect = np.full((len(positions), len(positions)), np.nan)

    if os.path.exists(paths["direct"]) and os.path.exists(paths["indirect"]):
        existing_direct = np.load(paths["direct"])
        existing_indirect = np.load(paths["indirect"])
        if existing_direct.shape == heatmap_direct.shape:
            heatmap_direct = existing_direct
            heatmap_indirect = existing_indirect

    for (row_index, top) in enumerate(positions):
        for (col_index, left) in enumerate(positions):
            if not np.isnan(heatmap_direct[row_index, col_index]):
                print(f"[{layer_name}] window at (top={top}, left={left}) already complete, skipping", flush=True)
                continue
            print(f"[{layer_name}] window at (top={top}, left={left})", flush=True)
            region_fn = build_sliding_region_fn(top, left, WINDOW_SIZE)
            (indirect_mi, direct_mi) = mine.run_bipartition(
                WINDOW_SIZE,
                alg_settings,
                PARAM_SETTINGS,
                eval_steps=EVAL_STEPS,
                target_size=IMAGE_SIZE,
                region_fn=region_fn,
                images=activations,
            )
            heatmap_direct[row_index, col_index] = direct_mi
            heatmap_indirect[row_index, col_index] = indirect_mi
            print(f"  indirect={indirect_mi:.4f}  direct={direct_mi:.4f}", flush=True)
            np.save(paths["direct"], heatmap_direct)
            np.save(paths["indirect"], heatmap_indirect)

    np.save(paths["positions"], np.asarray(positions))
    return (heatmap_direct, positions)


def plot_heatmap(layer_name, heatmap_direct, positions):
    ensure_results_dir()
    paths = result_paths(layer_name)

    plt.figure(figsize=(7, 6))
    plt.imshow(np.clip(heatmap_direct, 0, None), cmap="viridis", origin="upper")
    plt.colorbar(label="Direct MI (nats)")
    tick_labels = [str(p) for p in positions]
    plt.xticks(range(len(positions)), tick_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(positions)), tick_labels, fontsize=8)
    plt.xlabel("Window left edge (pixels)")
    plt.ylabel("Window top edge (pixels)")
    plt.title(f"Sliding {WINDOW_SIZE}x{WINDOW_SIZE} window MI map (cifar10 {layer_name}, stride={STRIDE})", fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["heatmap_pdf"])
    plt.savefig(paths["heatmap_png"], dpi=150)
    plt.close()


def plot_combined(heatmaps_by_layer):
    ensure_results_dir()
    vmax = max(np.nanmax(np.clip(heatmap, 0, None)) for (heatmap, _) in heatmaps_by_layer.values())

    (fig, axes_row) = plt.subplots(1, len(CONV_LAYER_NAMES), figsize=(4 * len(CONV_LAYER_NAMES), 4.5))
    for (axes, layer_name) in zip(axes_row, CONV_LAYER_NAMES):
        (heatmap_direct, positions) = heatmaps_by_layer[layer_name]
        image = axes.imshow(np.clip(heatmap_direct, 0, None), cmap="viridis", origin="upper", vmin=0, vmax=vmax)
        axes.set_title(layer_name, fontsize=12)
        axes.set_xticks([])
        axes.set_yticks([])
    fig.colorbar(image, ax=axes_row, fraction=0.02, pad=0.02, label="Direct MI (nats)")
    fig.suptitle(f"Sliding-window MI maps across CNN conv layers (cifar10, {WINDOW_SIZE}x{WINDOW_SIZE} window, 7x7 grid)", fontsize=15)

    pdf_path = os.path.join(RESULTS_DIR, "cifar10_cnn_sliding_combined_heatmap.pdf")
    png_path = os.path.join(RESULTS_DIR, "cifar10_cnn_sliding_combined_heatmap.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    heatmaps_by_layer = {}
    for layer_name in CONV_LAYER_NAMES:
        activations = load_activations(layer_name)
        (heatmap_direct, positions) = run_sliding_sweep(layer_name, activations)
        plot_heatmap(layer_name, heatmap_direct, positions)
        heatmaps_by_layer[layer_name] = (heatmap_direct, positions)
    plot_combined(heatmaps_by_layer)


if __name__ == "__main__":
    main()
