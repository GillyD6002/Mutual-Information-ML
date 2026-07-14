import os

import numpy as np

# Shared utilities for the tile-based MI pruning experiment: turns a
# non-overlapping grid of per-tile MI values into a pixel-level keep/prune
# mask, and applies that mask to a real image dataset by either zeroing or
# noising the pruned pixels. Used by train_pruned_classifiers.py (which
# trains classifiers on the resulting pruned datasets) and
# visualize_pruning.py (which previews the mask before spending GPU time).
#
# A *non-overlapping* tiling is used deliberately, rather than reusing the
# existing stride=3 sliding-window MI heatmaps (sliding_window_mi.py):
# stride=3 windows at window_size=7 overlap by 4 pixels, so a majority of
# pixels are covered by both a "kept" and a "pruned" window and there's no
# non-arbitrary way to decide those pixels' fate. Tiling the image into a
# grid where every pixel belongs to exactly one tile removes that ambiguity
# entirely.
#
# The actual per-tile MI values come from corner_mi_scaling.py's existing
# `--grid` sweep (already computed and committed for both mnist and
# cifar10 - see load_corner_grid_mi below), not a fresh sweep: that sweep
# already measured I(patch; rest of image) for a patch of every length 1-28
# centered on each of a 3x3 grid of 9 anchor points
# (image.get_grid_region), so the length=9 slice of that existing data is
# already exactly what's needed here, at zero additional GPU cost. It's
# also methodologically *cleaner* than a fresh non-overlapping tiling would
# be: every one of the 9 measurements uses an identically-sized 9x9 patch
# (image.get_grid_region always requests a square `length x length` patch,
# regardless of cell), whereas a literal non-overlapping tiling of a
# 28x28 image into a 3x3 grid necessarily mixes 9x9 corner tiles with a
# 10x10 center tile (28 isn't divisible by 3) - comparing raw MI across
# differently-sized regions confounds "this region is more informative"
# with "this region just has more pixels to work with" (see
# sliding_window_mi.py's docstring on this same normalization issue).
#
# get_grid_region's 9 boxes at length=9 don't quite tile the image exactly,
# though: they're independently centered and clamped per cell, so a thin
# ~1-pixel seam between adjacent cells (55 of 784 pixels for a 28x28 image)
# isn't covered by any of them. build_voronoi_assignment resolves this by
# assigning every pixel to its nearest of the 9 cell centers instead of to
# get_grid_region's literal box - a complete, gap-free, overlap-free
# partition of the whole image by construction, while still deciding each
# region's keep/prune status from the actual measured box MI.

PRUNE_MODES = ("zero", "noise")


def get_tile_edges(img_size, grid_size):

    # Splits [0, img_size] into grid_size roughly-equal integer segments
    # (e.g. img_size=28, grid_size=3 -> edges [0, 9, 19, 28], tile sizes
    # 9/10/9) via evenly-spaced float boundaries rounded to the nearest
    # pixel. This covers the whole image with no gaps or overlaps even when
    # img_size isn't evenly divisible by grid_size, unlike a fixed
    # window/stride pair (e.g. window=7/stride=7 only tiles cleanly because
    # 28 happens to be divisible by 7).

    edges = np.linspace(0, img_size, grid_size + 1)
    return np.round(edges).astype(int)


def get_tile_regions(img_size, grid_size):

    # Returns the grid_size*grid_size (top, bottom, left, right) boxes of a
    # non-overlapping grid_size x grid_size tiling, in row-major order
    # (index = row * grid_size + col) - the same flattening convention
    # np.ndarray.flatten() uses, which build_tile_mask relies on.

    edges = get_tile_edges(img_size, grid_size)
    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            regions.append((edges[row], edges[row + 1], edges[col], edges[col + 1]))
    return regions


def build_tile_region_fn(region):

    # Ignores the inner_length/img_height/img_width mine.run_bipartition
    # would otherwise pass in - the region is already fully determined by
    # the closure, exactly like sliding_window_mi.py's
    # build_sliding_region_fn.

    (top, bottom, left, right) = region

    def region_fn(inner_length, img_height, img_width):
        return (top, bottom, left, right)
    return region_fn


def keep_count_for(grid_size, percent_kept):

    # Number of tiles to keep out of grid_size**2, rounded to the nearest
    # whole tile since percent_kept*grid_size**2 generally isn't an integer
    # (e.g. 0.75 * 9 = 6.75 -> 7 tiles kept, 2 pruned - the closest a 3x3
    # grid can get to a 75/25 split).

    total = grid_size * grid_size
    return int(round(percent_kept * total))


def build_tile_mask(heatmap, percent_kept):

    # Ranks every tile by its MI value (descending) and marks the top
    # `percent_kept` fraction as kept (True) - the "brightest squares" from
    # the user's request - with ties broken by flat (row-major) index order
    # via np.argsort's stable sort, so the result is deterministic. Returns
    # a boolean (grid_size, grid_size) mask, True = keep.

    grid_size = heatmap.shape[0]
    flat = heatmap.flatten()
    order = np.argsort(-flat, kind="stable")
    keep_n = keep_count_for(grid_size, percent_kept)
    mask_flat = np.zeros(flat.shape, dtype=bool)
    mask_flat[order[:keep_n]] = True
    return mask_flat.reshape(grid_size, grid_size)


def tile_mask_to_pixel_mask(tile_mask, img_size):

    # Expands a (grid_size, grid_size) tile mask into a full (img_size,
    # img_size) pixel mask, True = keep - every pixel in a kept tile is
    # kept, every pixel in a pruned tile is pruned.

    grid_size = tile_mask.shape[0]
    edges = get_tile_edges(img_size, grid_size)
    pixel_mask = np.zeros((img_size, img_size), dtype=bool)
    for row in range(grid_size):
        for col in range(grid_size):
            if tile_mask[row, col]:
                pixel_mask[edges[row]:edges[row + 1], edges[col]:edges[col + 1]] = True
    return pixel_mask


def get_grid_cell_centers(img_size, grid_size):

    # Reproduces image.get_grid_region's own row_center/col_center formula
    # exactly (int((row + 0.5) * img_size / grid_size)), in the same
    # row-major order as get_tile_regions/heatmap.flatten() - so a Voronoi
    # partition built from these centers (build_voronoi_assignment) assigns
    # each pixel to the *same* cell that corner_mi_scaling.py's saved MI
    # curve for that cell was actually centered on.

    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            row_center = int((row + 0.5) * img_size / grid_size)
            col_center = int((col + 0.5) * img_size / grid_size)
            centers.append((row_center, col_center))
    return np.asarray(centers)


def build_voronoi_assignment(img_size, grid_size):

    # Assigns every pixel in an img_size x img_size image to the flat
    # (row-major) index of its nearest of the grid_size**2 cell centers -
    # a complete, gap-free, overlap-free partition of the whole image by
    # construction (unlike get_grid_region's independently-clamped boxes,
    # see module docstring), used to turn per-cell MI values measured on a
    # small centered patch into a full-image pixel mask.

    centers = get_grid_cell_centers(img_size, grid_size)
    (rows, cols) = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing="ij")
    pixel_coords = np.stack([rows, cols], axis=-1).reshape(-1, 2)
    dist2 = ((pixel_coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(dist2, axis=1).reshape(img_size, img_size)


def voronoi_pixel_mask(tile_mask, assignment):

    # Expands a (grid_size, grid_size) tile mask into a full pixel mask via
    # a precomputed build_voronoi_assignment array, the Voronoi-partition
    # analog of tile_mask_to_pixel_mask (which instead uses an exact
    # non-overlapping box tiling).

    return tile_mask.flatten()[assignment]


def load_corner_grid_mi(image_type, results_dir, grid_size=3, tile_length=9):

    # Reuses corner_mi_scaling.py's already-computed `--grid` sweep output
    # (results/{image_type}_grid_{row}_{col}_mi_direct.npy, one MI-vs-length
    # curve per cell, lengths 1-28) instead of running a fresh MI
    # measurement: slices out just the `tile_length` entry of each of the
    # grid_size**2 cells' curves and returns them as a (grid_size,
    # grid_size) heatmap, in the same row-major layout build_tile_mask
    # expects. See module docstring for why length=9 (matching this
    # project's ~9px tile granularity) is a meaningful, ready-made slice of
    # that data rather than an arbitrary one.

    lengths_path = os.path.join(results_dir, f"{image_type}_mi_lengths.npy")
    lengths = np.load(lengths_path)
    matches = np.nonzero(lengths == tile_length)[0]
    if matches.size == 0:
        raise ValueError(f"tile_length={tile_length} not found in {lengths_path} (lengths {lengths.min()}-{lengths.max()})")
    length_index = int(matches[0])

    heatmap = np.full((grid_size, grid_size), np.nan)
    for row in range(grid_size):
        for col in range(grid_size):
            cell_path = os.path.join(results_dir, f"{image_type}_grid_{row}_{col}_mi_direct.npy")
            if not os.path.exists(cell_path):
                raise FileNotFoundError(
                    f"{cell_path} not found - run `python -m MI_scaling_non_middle.corner_mi_scaling --grid` first."
                )
            curve = np.load(cell_path)
            heatmap[row, col] = curve[length_index]
    return heatmap


def apply_pruning(images, pixel_mask, mode, rng):

    # Returns a copy of `images` (N, H, W) with every pixel outside
    # pixel_mask (the pruned 25%) replaced according to `mode`:
    #   - "zero": set to 0.
    #   - "noise": replaced with fresh i.i.d. Uniform(min, max) noise, where
    #     min/max are the dataset's own actual pixel value range (not a
    #     hardcoded [0, 1]) - see the module docstring on why this matters
    #     for the specific dataset being pruned.
    # Kept pixels (inside pixel_mask) are left completely untouched, and
    # `images` itself is never modified in place.

    if mode not in PRUNE_MODES:
        raise ValueError(f"Unknown pruning mode {mode!r}, expected one of {PRUNE_MODES}")
    pruned = images.copy()
    prune_area = ~pixel_mask
    if mode == "zero":
        pruned[:, prune_area] = 0.0
    elif mode == "noise":
        low = images.min()
        high = images.max()
        noise_shape = (images.shape[0], int(prune_area.sum()))
        pruned[:, prune_area] = rng.uniform(low, high, size=noise_shape).astype(images.dtype)
    return pruned
