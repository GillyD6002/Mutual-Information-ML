import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mine, image as img

# Every other experiment in this folder grows a patch from some fixed anchor
# (a corner, the middle, a 3x3 grid of cell centers) and sweeps the *length*
# of that patch. This script instead holds the patch size fixed at each of
# several window sizes and slides its *position* across the entire image,
# producing a genuine spatial heatmap of I(window; rest of image) rather
# than a handful of growth curves - the fine-grained version of the 9-cell
# grid experiment. Run once per size in WINDOW_SIZES (3x3 through 8x8), it
# also shows how that spatial map's *shape* changes as the receptive field
# grows, not just how it looks at one arbitrary size.
#
# This sidesteps the H(inner)-normalization problem discussed for the
# region-importance/pruning question: since every window at a given size is
# identical in size to every other position at that size, comparing raw MI
# across positions *within one size's heatmap* is already an apples-to-apples
# comparison (no position gets credit just for having a bigger patch to work
# with) - unlike comparing MI across different partition lengths. The
# resulting maps should reveal actual spatial structure - e.g. the real
# footprint of where MNIST digit strokes tend to fall - rather than just the
# coarse "closer to center is better" story the 9-cell grid could show.
#
# Positions are placed on a fixed stride across each axis (0, stride,
# 2*stride, ... up to img_size - window_size), not swept per pixel: a
# stride-1 sweep of a 3x3 window over a 28x28 image would be 26x26 = 676
# positions for that size alone. STRIDE=3 (roughly tiling the image with
# only minor overlap) keeps each size's heatmap to a 7x7-9x9 resolution
# instead - a genuinely useful middle ground between the 9-cell grid and
# full pixel resolution, repeated at each of six window sizes. Total scope
# across WINDOW_SIZES for one dataset is on the order of 400 trainings
# (9^2 + 9^2 + 8^2 + 8^2 + 8^2 + 7^2 = 403 for sizes 3-8 at stride 3); built
# with GPU throughput in mind rather than the CPU-scale sweeps run so far in
# this project.
#
# Resumable like corner_mi_scaling.py's grid sweep: each (dataset,
# window_size) pair's heatmap is saved to disk after every single position,
# and a position already present in a saved heatmap (not NaN) is skipped
# rather than retrained - a killed process only costs whatever position was
# in flight, not the whole sweep, and a whole window size already fully
# completed costs nothing to "redo" on a rerun. Run as:
#     python -m MI_scaling_non_middle.sliding_window_mi

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
DATASETS = {
    "mnist": img.DEFAULT_IMAGE_SIZE,
    "cifar10": img.DEFAULT_IMAGE_SIZE,
}
WINDOW_SIZES = list(range(3, 9))  # 3x3 through 8x8 inclusive
STRIDE = 3

NUM_IMAGES = 30000
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


def get_positions(img_size, window_size):
    return list(range(0, img_size - window_size + 1, STRIDE))


def build_sliding_region_fn(top, left, window_size):

    # Ignores the inner_length/img_height/img_width run_bipartition would
    # otherwise pass in - position and size are already fully determined by
    # the closure, exactly like build_region_fn("middle") ignoring its
    # arguments to always return the same thing.

    def region_fn(inner_length, img_height, img_width):
        return (top, top + window_size, left, left + window_size)
    return region_fn


def result_paths(image_type, window_size):
    tag = f"{image_type}_sliding_w{window_size}_s{STRIDE}"
    return {
        "direct": os.path.join(RESULTS_DIR, f"{tag}_mi_direct.npy"),
        "indirect": os.path.join(RESULTS_DIR, f"{tag}_mi_indirect.npy"),
        "positions": os.path.join(RESULTS_DIR, f"{tag}_positions.npy"),
        "heatmap_pdf": os.path.join(RESULTS_DIR, f"{tag}_heatmap.pdf"),
        "heatmap_png": os.path.join(RESULTS_DIR, f"{tag}_heatmap.png"),
    }


def run_sliding_sweep(image_type, target_size, window_size):
    ensure_results_dir()
    positions = get_positions(target_size, window_size)
    paths = result_paths(image_type, window_size)
    alg_settings = dict(
        image_type=image_type,
        num_images=NUM_IMAGES,
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
                print(f"[{image_type}] w={window_size} window at (top={top}, left={left}) already complete, skipping", flush=True)
                continue
            print(f"[{image_type}] w={window_size} window at (top={top}, left={left})", flush=True)
            region_fn = build_sliding_region_fn(top, left, window_size)
            (indirect_mi, direct_mi) = mine.run_bipartition(
                window_size,
                alg_settings,
                PARAM_SETTINGS,
                eval_steps=EVAL_STEPS,
                target_size=target_size,
                region_fn=region_fn,
            )
            heatmap_direct[row_index, col_index] = direct_mi
            heatmap_indirect[row_index, col_index] = indirect_mi
            print(f"  indirect={indirect_mi:.4f}  direct={direct_mi:.4f}", flush=True)
            np.save(paths["direct"], heatmap_direct)
            np.save(paths["indirect"], heatmap_indirect)

    np.save(paths["positions"], np.asarray(positions))
    return (heatmap_direct, positions)


def plot_heatmap(image_type, window_size, heatmap_direct, positions):
    ensure_results_dir()
    paths = result_paths(image_type, window_size)

    plt.figure(figsize=(7, 6))
    plt.imshow(np.clip(heatmap_direct, 0, None), cmap="viridis", origin="upper")
    plt.colorbar(label="Direct MI (nats)")
    tick_labels = [str(p) for p in positions]
    plt.xticks(range(len(positions)), tick_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(positions)), tick_labels, fontsize=8)
    plt.xlabel("Window left edge (pixels)")
    plt.ylabel("Window top edge (pixels)")
    plt.title(f"Sliding {window_size}x{window_size} window MI map ({image_type}, stride={STRIDE})", fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["heatmap_pdf"])
    plt.savefig(paths["heatmap_png"], dpi=150)
    plt.close()


def plot_combined(image_type, heatmaps_by_size):

    # One figure per dataset, one subplot per window size, sharing a single
    # color scale (the max direct-MI value across every size's heatmap) so
    # the panels are directly comparable at a glance - does the map just get
    # uniformly brighter as the window grows, or does its actual shape
    # change? A shared scale is what makes that distinction visible; letting
    # each subplot auto-scale its own colors would hide it.

    ensure_results_dir()
    vmax = max(np.nanmax(np.clip(heatmap, 0, None)) for (heatmap, _) in heatmaps_by_size.values())

    (fig, axes_row) = plt.subplots(1, len(WINDOW_SIZES), figsize=(4 * len(WINDOW_SIZES), 4.5))
    for (axes, window_size) in zip(axes_row, WINDOW_SIZES):
        (heatmap_direct, positions) = heatmaps_by_size[window_size]
        image = axes.imshow(np.clip(heatmap_direct, 0, None), cmap="viridis", origin="upper", vmin=0, vmax=vmax)
        axes.set_title(f"{window_size}x{window_size}", fontsize=12)
        axes.set_xticks([])
        axes.set_yticks([])
    fig.colorbar(image, ax=axes_row, fraction=0.02, pad=0.02, label="Direct MI (nats)")
    fig.suptitle(f"Sliding-window MI maps across sizes 3x3-8x8 (stride={STRIDE}, {image_type})", fontsize=15)

    pdf_path = os.path.join(RESULTS_DIR, f"{image_type}_sliding_combined_heatmap.pdf")
    png_path = os.path.join(RESULTS_DIR, f"{image_type}_sliding_combined_heatmap.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def parse_args():

    # Lets a single (dataset, window-size) slice of the sweep be run per
    # process - e.g. one container per GPU - without touching the defaults
    # that make plain `python -m ...sliding_window_mi` still run everything,
    # same as before this flag existed.

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None,
                         help="Comma-separated subset of DATASETS keys to run (default: all)")
    parser.add_argument("--window-sizes", type=str, default=None,
                         help="Comma-separated subset of WINDOW_SIZES to run (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = DATASETS
    if args.datasets:
        keys = [key.strip() for key in args.datasets.split(",")]
        datasets = {key: DATASETS[key] for key in keys}
    window_sizes = WINDOW_SIZES
    if args.window_sizes:
        window_sizes = [int(w.strip()) for w in args.window_sizes.split(",")]

    for (image_type, target_size) in datasets.items():
        heatmaps_by_size = {}
        for window_size in window_sizes:
            (heatmap_direct, positions) = run_sliding_sweep(image_type, target_size, window_size)
            plot_heatmap(image_type, window_size, heatmap_direct, positions)
            heatmaps_by_size[window_size] = (heatmap_direct, positions)
        if len(heatmaps_by_size) == len(WINDOW_SIZES):
            # Only the full, unsliced set of window sizes makes plot_combined's
            # shared-color-scale panel meaningful - a partial slice run in one
            # of several parallel containers would otherwise produce a
            # misleadingly incomplete combined figure.
            plot_combined(image_type, heatmaps_by_size)


if __name__ == "__main__":
    main()
