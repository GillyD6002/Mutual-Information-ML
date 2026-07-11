import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mine, image as img

# The paper's original MI-scaling sweep always grows the "inner" patch
# outward from the image's center (image.get_center_region). This script
# redoes that same growing-partition sweep, but anchors the inner patch at
# each of the four image corners instead (image.get_corner_region), via
# mine.run_bipartition's region_fn hook, and includes the original centered
# ("middle") sweep alongside them as the baseline they're compared against.
# Mechanically the corner/middle choice is a drop-in swap - get_finite_dataset's
# splice-based marginal construction operates on an (top, bottom, left,
# right) box regardless of where that box sits, so no change was needed
# there. If the four corners' MI-vs-length curves look similar to each
# other (and to the middle's), that's evidence MI scaling here is a generic
# property of "how much of the image is revealed," not an artifact of the
# center-out ordering the paper used; if they diverge from each other (or
# from the middle), that's a sign the choice of starting region is doing
# real work - a centered patch borders "outer" pixels on all four sides for
# any length short of the full image, while a corner patch only ever
# borders "outer" pixels on two sides (the other two are already at the
# image's own edge), so a smaller inner/outer boundary at a given length is
# expected to translate into slower-rising, lower-peaking MI. Run as:
#     python -m MI_scaling_non_middle.corner_mi_scaling

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
DATASETS = {
    "mnist": img.DEFAULT_IMAGE_SIZE,
    "cifar10": img.DEFAULT_IMAGE_SIZE,
}
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
REGIONS = ["middle"] + CORNERS

# 3x3 "moving slideshow" grid: the same growing-patch sweep, but anchored at
# all nine grid-cell centers (image.get_grid_region) instead of just the
# four corners plus the true middle - see plot_grid_results below for the
# resulting 3x3 arrangement of subplots.
GRID_SIZE = 3
GRID_CELLS = [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)]

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


def save_results(image_type, lengths, region_results):
    ensure_results_dir()
    np.save(os.path.join(RESULTS_DIR, f"{image_type}_mi_lengths.npy"), np.asarray(lengths))
    for region, values in region_results.items():
        np.save(os.path.join(RESULTS_DIR, f"{image_type}_{region}_mi_direct.npy"), np.asarray(values["direct"]))
        np.save(os.path.join(RESULTS_DIR, f"{image_type}_{region}_mi_indirect.npy"), np.asarray(values["indirect"]))


def plot_corner_results(image_type, lengths, region_results):
    ensure_results_dir()

    # "middle" is drawn separately (thicker, dashed, black) so the baseline
    # it's being compared against doesn't get lost among the four corners'
    # colors.
    series = {}
    for region in CORNERS:
        if region in region_results:
            series[region.replace("_", " ").title()] = region_results[region]["direct"]

    axes = img.plot_mi_scaling(series, lengths=lengths, clip_negative=True, save_path=None)
    # plot_mi_scaling's own lines aren't individually .set_label()'d (it
    # builds its legend via axes.legend(labels, ...) instead), so a bare
    # axes.legend() call here would only pick up the "middle" line added
    # below - handles/labels are grabbed and passed explicitly instead.
    handles = list(axes.get_lines())
    labels = list(series.keys())
    if "middle" in region_results:
        middle_values = np.clip(region_results["middle"]["direct"], 0, None)
        (middle_line,) = axes.plot(lengths, middle_values, linewidth = 3, linestyle = "--", color = "black")
        handles.append(middle_line)
        labels.append("Middle")
    axes.legend(handles, labels, fontsize = 16)
    axes.set_title(f"MI scaling from image corners vs. middle ({image_type})", fontsize=16)
    plt.tight_layout()

    pdf_path = os.path.join(RESULTS_DIR, f"{image_type}_corner_mi_scaling.pdf")
    png_path = os.path.join(RESULTS_DIR, f"{image_type}_corner_mi_scaling.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def build_region_fn(region):

    # "middle" maps to None, which is run_bipartition's own signal to fall
    # back to its default image.get_center_region behavior - not a special
    # case bolted on here, just reusing the hook exactly as every other
    # existing (region_fn-less) caller in this project already does.

    if region == "middle":
        return None
    def region_fn(inner_length, img_height, img_width):
        return img.get_corner_region(inner_length, img_height, img_width, region)
    return region_fn


def run_corner_sweep(image_type, target_size, lengths):
    alg_settings = dict(
        image_type=image_type,
        num_images=NUM_IMAGES,
        strength="small",
        algorithm="logistic",
    )

    region_results = {}
    for region in REGIONS:
        region_results[region] = {
            "direct": [],
            "indirect": [],
        }
        region_fn = build_region_fn(region)
        for length in lengths:
            print(f"[{image_type}] {region} partition length: {length}", flush=True)
            indirect_mi, direct_mi = mine.run_bipartition(
                length,
                alg_settings,
                PARAM_SETTINGS,
                eval_steps=EVAL_STEPS,
                target_size=target_size,
                region_fn=region_fn,
            )
            region_results[region]["direct"].append(direct_mi)
            region_results[region]["indirect"].append(indirect_mi)

    return region_results


def build_grid_region_fn(row, col):
    def region_fn(inner_length, img_height, img_width):
        return img.get_grid_region(inner_length, img_height, img_width, row, col, grid_size=GRID_SIZE)
    return region_fn


def run_grid_sweep(image_type, target_size, lengths):
    alg_settings = dict(
        image_type=image_type,
        num_images=NUM_IMAGES,
        strength="small",
        algorithm="logistic",
    )

    grid_results = {}
    for (row, col) in GRID_CELLS:
        cell = f"grid_{row}_{col}"

        # Resumable: a process-memory leak in the underlying TF/Keras
        # training loop (not fixed by clear_session()/gc.collect() - see
        # src/mine.py's run_bipartition) means a long enough sweep can run
        # the process out of memory partway through. Since each cell is
        # saved to disk as soon as it finishes, a cell whose output already
        # exists is loaded from disk instead of retrained, so a killed and
        # restarted run only redoes whatever cell was actually in progress.
        direct_path = os.path.join(RESULTS_DIR, f"{image_type}_{cell}_mi_direct.npy")
        indirect_path = os.path.join(RESULTS_DIR, f"{image_type}_{cell}_mi_indirect.npy")
        if os.path.exists(direct_path) and os.path.exists(indirect_path):
            print(f"[{image_type}] {cell} already complete, loading from disk", flush=True)
            grid_results[cell] = {
                "direct": np.load(direct_path).tolist(),
                "indirect": np.load(indirect_path).tolist(),
            }
            continue

        grid_results[cell] = {
            "direct": [],
            "indirect": [],
        }
        region_fn = build_grid_region_fn(row, col)
        for length in lengths:
            print(f"[{image_type}] {cell} partition length: {length}", flush=True)
            indirect_mi, direct_mi = mine.run_bipartition(
                length,
                alg_settings,
                PARAM_SETTINGS,
                eval_steps=EVAL_STEPS,
                target_size=target_size,
                region_fn=region_fn,
            )
            grid_results[cell]["direct"].append(direct_mi)
            grid_results[cell]["indirect"].append(indirect_mi)
        # Saved after each cell completes (not just once at the very end of
        # all nine) so a crash partway through a dataset - e.g. the
        # out-of-memory failure this was added after - only costs the
        # in-progress cell's work, not every cell already finished.
        save_results(image_type, lengths, {cell: grid_results[cell]})

    return grid_results


def plot_grid_results(image_type, lengths, grid_results):

    # Arranges the nine cells' MI-vs-length curves into a 3x3 grid of
    # subplots that mirrors their actual spatial position in the image
    # (row 0 = top, row 2 = bottom), sharing y-axis limits across all nine
    # so heights are directly comparable at a glance, rather than overlaying
    # nine lines on one axes (unreadable at this count).

    ensure_results_dir()

    max_mi = max(np.clip(values["direct"], 0, None).max() for values in grid_results.values())

    (fig, axes_grid) = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(12, 10), sharex=True, sharey=True)
    for (row, col) in GRID_CELLS:
        cell = f"grid_{row}_{col}"
        values = np.clip(grid_results[cell]["direct"], 0, None)
        axes = axes_grid[row][col]
        axes.plot(lengths, values, linewidth=2, color="#4C72B0")
        axes.set_title(f"({row}, {col})", fontsize=12)
        axes.set_ylim(0, max_mi * 1.05)
    for col in range(GRID_SIZE):
        axes_grid[GRID_SIZE - 1][col].set_xlabel("Partition Length (pixels)", fontsize=11)
    for row in range(GRID_SIZE):
        axes_grid[row][0].set_ylabel("MI (nats)", fontsize=11)

    fig.suptitle(f"MI scaling across a {GRID_SIZE}x{GRID_SIZE} grid of anchor points ({image_type})", fontsize=16)
    plt.tight_layout()

    pdf_path = os.path.join(RESULTS_DIR, f"{image_type}_grid_mi_scaling.pdf")
    png_path = os.path.join(RESULTS_DIR, f"{image_type}_grid_mi_scaling.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    lengths = list(range(1, img.DEFAULT_IMAGE_SIZE + 1))
    for image_type, target_size in DATASETS.items():
        region_results = run_corner_sweep(image_type, target_size, lengths)
        save_results(image_type, lengths, region_results)
        plot_corner_results(image_type, lengths, region_results)


def main_grid():
    lengths = list(range(1, img.DEFAULT_IMAGE_SIZE + 1))
    for image_type, target_size in DATASETS.items():
        grid_results = run_grid_sweep(image_type, target_size, lengths)
        save_results(image_type, lengths, grid_results)
        plot_grid_results(image_type, lengths, grid_results)


if __name__ == "__main__":
    if "--grid" in sys.argv:
        main_grid()
    else:
        main()
