import os

import numpy as np

# Shared utilities for the tile-based MI pruning experiment: turns a
# non-overlapping grid of per-tile MI values into a pixel-level keep/prune
# mask, and applies that mask to a real image dataset by either zeroing or
# noising the pruned pixels. Used by train_pruned_classifiers.py (which
# trains classifiers on the resulting pruned datasets) and
# visualize_pruning.py (which previews the mask before spending GPU time).
#
# The per-tile MI values are reused from sliding_window_mi.py's existing
# window_size=3, stride=3 sweep (results/{image_type}_sliding_w3_s3_*.npy,
# already computed and committed for both mnist and cifar10) rather than a
# fresh measurement - window_size == stride is what makes that particular
# sweep a (near-)exact non-overlapping tiling rather than the overlapping
# windows every other (window_size, stride) pair in that sweep produces
# (e.g. window_size=7/stride=3 overlaps by 4px, which is exactly why an
# earlier version of this module built a coarser 3x3 grid instead - see
# git history). At window=3/stride=3 the 9 positions 0,3,...,24 tile pixels
# 0-26 with zero overlap, leaving only the single last row/column (pixel 27)
# uncovered - closed by extending the last tile in each axis by that one
# extra pixel (see load_sliding_window_mi/pixel_mask_from_edges), not by a
# Voronoi partition. This is both finer-grained (81 tiles instead of 9,
# letting individual small regions be pruned rather than whole quadrants)
# and methodologically cleaner (every one of the 81 measurements is an
# identically-sized 3x3 patch - image.get_center_region-equivalent boxes,
# not centered+clamped per-cell boxes of varying effective size).

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


def pixel_mask_from_edges(tile_mask, edges):

    # Expands a (grid_size, grid_size) tile mask into a full pixel mask
    # using explicit per-axis boundary positions (edges[i]:edges[i+1] is
    # tile i's extent along either axis, edges[-1] is the image size) -
    # the general form tile_mask_to_pixel_mask's equal-division edges are
    # just one particular case of.

    grid_size = tile_mask.shape[0]
    img_size = edges[-1]
    pixel_mask = np.zeros((img_size, img_size), dtype=bool)
    for row in range(grid_size):
        for col in range(grid_size):
            if tile_mask[row, col]:
                pixel_mask[edges[row]:edges[row + 1], edges[col]:edges[col + 1]] = True
    return pixel_mask


def tile_mask_to_pixel_mask(tile_mask, img_size):

    # Expands a (grid_size, grid_size) tile mask into a full (img_size,
    # img_size) pixel mask, True = keep - every pixel in a kept tile is
    # kept, every pixel in a pruned tile is pruned. Equal-division case of
    # pixel_mask_from_edges; kept as a convenience for any caller that
    # wants a plain equal-size tiling rather than sourcing edges from an
    # existing sliding-window sweep (see load_sliding_window_mi).

    grid_size = tile_mask.shape[0]
    edges = get_tile_edges(img_size, grid_size)
    return pixel_mask_from_edges(tile_mask, edges)


def load_sliding_window_mi(image_type, results_dir, img_size, window_size=3, stride=3):

    # Reuses sliding_window_mi.py's already-computed sweep output
    # (results/{image_type}_sliding_w{window_size}_s{stride}_mi_direct.npy
    # + _positions.npy) instead of running a fresh MI measurement. Only
    # meaningful here when window_size == stride (see module docstring) -
    # that's what makes the swept positions a non-overlapping tiling rather
    # than the overlapping windows every other (window_size, stride) pair
    # in that sweep produces.
    #
    # Returns (heatmap, edges): heatmap is the (grid_size, grid_size) MI
    # array as saved; edges is positions with the image size appended, so
    # edges[i]:edges[i+1] is exactly tile i's extent along either axis
    # (pixel_mask_from_edges expects this) - the appended final boundary is
    # what closes the 1-pixel gap the last position leaves short of the
    # image edge (e.g. positions [0,3,...,24] with window=3 cover pixels
    # 0-26 exactly; appending img_size=28 extends the last tile from 3px to
    # 4px rather than leaving pixel 27 unassigned to any tile).

    if window_size != stride:
        raise ValueError(
            f"window_size ({window_size}) must equal stride ({stride}) for a non-overlapping tiling - "
            "any other pair produces overlapping windows (see module docstring)."
        )
    tag = f"{image_type}_sliding_w{window_size}_s{stride}"
    heatmap_path = os.path.join(results_dir, f"{tag}_mi_direct.npy")
    positions_path = os.path.join(results_dir, f"{tag}_positions.npy")
    if not os.path.exists(heatmap_path) or not os.path.exists(positions_path):
        raise FileNotFoundError(
            f"{heatmap_path} not found - run `python -m MI_scaling_non_middle.sliding_window_mi "
            f"--window-sizes {window_size}` first."
        )
    heatmap = np.load(heatmap_path)
    if np.isnan(heatmap).any():
        raise ValueError(f"{heatmap_path} has unfinished (NaN) tiles - that sweep hasn't completed yet.")
    positions = np.load(positions_path)
    # The true image size isn't stored in positions.npy itself, and can't
    # be safely inferred from positions[-1] + window_size either: that
    # only equals the real image size when img_size divides evenly by
    # window_size (true for lfw_faces/fer2013_hf's grids, not for
    # mnist/cifar10's 28px images at window=3, where the last position
    # falls 1px short of the edge - see the closing-the-gap note above).
    # So the caller has to pass the actual size sliding_window_mi.py ran
    # this sweep against (each dataset's entry in that script's DATASETS
    # dict) rather than a single project-wide constant.
    edges = np.concatenate([positions, [img_size]]).astype(int)
    return (heatmap, edges)


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
