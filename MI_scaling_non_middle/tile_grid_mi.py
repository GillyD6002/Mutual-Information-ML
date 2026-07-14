import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mine, image as img
from MI_scaling_non_middle.pixel_pruning import get_tile_regions, build_tile_region_fn

# Computes I(tile; rest of image) for every tile of a non-overlapping
# GRID_SIZE x GRID_SIZE tiling of the image (image.py's get_tile_regions),
# for both mnist and cifar10 - the input the pixel-pruning experiment
# (pixel_pruning.py, train_pruned_classifiers.py) ranks tiles by. This is a
# deliberately coarser, non-overlapping alternative to
# sliding_window_mi.py's stride=3 sweep - see pixel_pruning.py's module
# docstring for why non-overlap matters for turning a heatmap into an
# unambiguous keep/prune pixel mask.
#
# Mechanically this is corner_mi_scaling.py's --grid sweep (run_grid_sweep)
# with one crucial difference: that sweep anchors a *growing* patch on each
# of the 9 cell centers (image.get_grid_region), producing an MI-vs-length
# curve per cell, whereas this holds a *fixed*, non-overlapping tile per
# cell and reports one MI value per tile - directly analogous to
# sliding_window_mi.py's single-window-size heatmaps, just tiled instead of
# strided/overlapping. Saved under a "tile{GRID_SIZE}x{GRID_SIZE}" tag
# (rather than corner_mi_scaling's "grid_{row}_{col}") specifically so the
# two don't collide in results/.
#
# Resumable exactly like sliding_window_mi.py: each tile's MI value is saved
# to disk as soon as it's computed, and a tile already present (not NaN) in
# a saved heatmap is skipped on rerun. Run as:
#     python -m MI_scaling_non_middle.tile_grid_mi

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
DATASETS = {
    "mnist": img.DEFAULT_IMAGE_SIZE,
    "cifar10": img.DEFAULT_IMAGE_SIZE,
}
GRID_SIZE = 3

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


def result_paths(image_type):
    tag = f"{image_type}_tile{GRID_SIZE}x{GRID_SIZE}"
    return {
        "direct": os.path.join(RESULTS_DIR, f"{tag}_mi_direct.npy"),
        "indirect": os.path.join(RESULTS_DIR, f"{tag}_mi_indirect.npy"),
        "heatmap_pdf": os.path.join(RESULTS_DIR, f"{tag}_heatmap.pdf"),
        "heatmap_png": os.path.join(RESULTS_DIR, f"{tag}_heatmap.png"),
    }


def run_tile_sweep(image_type, target_size):
    ensure_results_dir()
    paths = result_paths(image_type)
    regions = get_tile_regions(target_size, GRID_SIZE)

    heatmap_direct = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    heatmap_indirect = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    if os.path.exists(paths["direct"]) and os.path.exists(paths["indirect"]):
        existing_direct = np.load(paths["direct"])
        existing_indirect = np.load(paths["indirect"])
        if existing_direct.shape == heatmap_direct.shape:
            heatmap_direct = existing_direct
            heatmap_indirect = existing_indirect

    alg_settings = dict(
        image_type=image_type,
        num_images=NUM_IMAGES,
        strength="small",
        algorithm="logistic",
    )

    for (index, region) in enumerate(regions):
        (row, col) = divmod(index, GRID_SIZE)
        if not np.isnan(heatmap_direct[row, col]):
            print(f"[{image_type}] tile ({row},{col}) already complete, skipping", flush=True)
            continue
        print(f"[{image_type}] tile ({row},{col}) region={region}", flush=True)
        region_fn = build_tile_region_fn(region)
        # inner_length is ignored by region_fn (see build_tile_region_fn)
        # but run_bipartition still requires one; the tile's own row extent
        # is passed just for a plausible value, not because it's used.
        nominal_length = int(region[1] - region[0])
        (indirect_mi, direct_mi) = mine.run_bipartition(
            nominal_length,
            alg_settings,
            PARAM_SETTINGS,
            eval_steps=EVAL_STEPS,
            target_size=target_size,
            region_fn=region_fn,
        )
        heatmap_direct[row, col] = direct_mi
        heatmap_indirect[row, col] = indirect_mi
        print(f"  indirect={indirect_mi:.4f}  direct={direct_mi:.4f}", flush=True)
        np.save(paths["direct"], heatmap_direct)
        np.save(paths["indirect"], heatmap_indirect)

    return heatmap_direct


def plot_heatmap(image_type, heatmap_direct):
    ensure_results_dir()
    paths = result_paths(image_type)

    plt.figure(figsize=(5, 4.5))
    clipped = np.clip(heatmap_direct, 0, None)
    plt.imshow(clipped, cmap="viridis", origin="upper")
    plt.colorbar(label="Direct MI (nats)")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            plt.text(col, row, f"{heatmap_direct[row, col]:.3f}", ha="center", va="center",
                      color="white", fontsize=11)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.xlabel("Tile column")
    plt.ylabel("Tile row")
    plt.title(f"{GRID_SIZE}x{GRID_SIZE} non-overlapping tile MI map ({image_type})", fontsize=12)
    plt.tight_layout()
    plt.savefig(paths["heatmap_pdf"])
    plt.savefig(paths["heatmap_png"], dpi=150)
    plt.close()


def main():
    for (image_type, target_size) in DATASETS.items():
        heatmap_direct = run_tile_sweep(image_type, target_size)
        plot_heatmap(image_type, heatmap_direct)


if __name__ == "__main__":
    main()
