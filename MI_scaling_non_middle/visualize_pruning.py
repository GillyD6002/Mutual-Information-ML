import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import image as img
from MI_scaling_non_middle.pixel_pruning import build_tile_mask, pixel_mask_from_edges, load_sliding_window_mi, apply_pruning

# A quick, GPU-free sanity check of the pruning pipeline: for each dataset,
# shows the tile MI heatmap (reused from sliding_window_mi.py's existing
# window=3/stride=3 sweep, see pixel_pruning.load_sliding_window_mi), the
# resulting keep/prune mask, and a handful of real example images before/
# after zero- and noise-pruning. Doesn't train anything - just lets the
# masking logic (pixel_pruning.py) be checked by eye before spending GPU
# time on the actual training sweep (train_pruned_classifiers.py). Run as:
#     python -m MI_scaling_non_middle.visualize_pruning
#     python -m MI_scaling_non_middle.visualize_pruning --percent-kept 0.5

MI_RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
RESULTS_DIR = os.path.join(MI_RESULTS_DIR, "pruning")
TILE_WINDOW_SIZE = 3
TILE_STRIDE = 3
NUM_EXAMPLES = 4
NOISE_SEED = 20260714


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def visualize(image_type, percent_kept):
    (heatmap, edges) = load_sliding_window_mi(image_type, MI_RESULTS_DIR, window_size=TILE_WINDOW_SIZE, stride=TILE_STRIDE)
    tile_mask = build_tile_mask(heatmap, percent_kept)
    pixel_mask = pixel_mask_from_edges(tile_mask, edges)

    (images, _, _) = img.get_images(image_type, NUM_EXAMPLES, target_size=img.DEFAULT_IMAGE_SIZE)
    rng = np.random.RandomState(NOISE_SEED)
    zero_images = apply_pruning(images, pixel_mask, "zero", rng)
    rng = np.random.RandomState(NOISE_SEED)
    noise_images = apply_pruning(images, pixel_mask, "noise", rng)

    grid_size = tile_mask.shape[0]
    keep_n = int(tile_mask.sum())
    n_cols = 3 + NUM_EXAMPLES
    (fig, axes) = plt.subplots(3, n_cols, figsize=(2.2 * n_cols, 6.6))

    axes[0, 0].imshow(np.clip(heatmap, 0, None), cmap="viridis")
    axes[0, 0].set_title("Tile MI")
    axes[1, 0].imshow(tile_mask, cmap="Greys_r", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Kept tiles ({keep_n}/{grid_size * grid_size})")
    axes[2, 0].imshow(pixel_mask, cmap="Greys_r", vmin=0, vmax=1)
    axes[2, 0].set_title("Pixel mask")
    for row in range(3):
        for col in (1, 2):
            axes[row, col].axis("off")

    for col in range(NUM_EXAMPLES):
        axes[0, 3 + col].imshow(images[col], cmap="gray", vmin=0, vmax=1)
        axes[1, 3 + col].imshow(zero_images[col], cmap="gray", vmin=0, vmax=1)
        axes[2, 3 + col].imshow(noise_images[col], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_ylabel("Original", fontsize=11)
    axes[1, 3].set_ylabel("Zero-pruned", fontsize=11)
    axes[2, 3].set_ylabel("Noise-pruned", fontsize=11)
    for row in range(3):
        for col in range(NUM_EXAMPLES):
            axes[row, 3 + col].set_xticks([])
            axes[row, 3 + col].set_yticks([])

    fig.suptitle(f"Pruning preview: {image_type}, keep {int(round(percent_kept * 100))}% of tiles", fontsize=14)
    plt.tight_layout()

    ensure_results_dir()
    tag = f"{image_type}_p{int(round(percent_kept * 100))}"
    plt.savefig(os.path.join(RESULTS_DIR, f"prune_preview_{tag}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"prune_preview_{tag}.png"), dpi=150)
    plt.close()
    print(f"[{image_type}] saved prune_preview_{tag}.png", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent-kept", type=float, default=0.75)
    parser.add_argument("--datasets", type=str, default=None)
    args = parser.parse_args()

    datasets = ["mnist", "cifar10"]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    for image_type in datasets:
        visualize(image_type, args.percent_kept)


if __name__ == "__main__":
    main()
