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
# mine.run_bipartition's region_fn hook. Mechanically this is a drop-in
# swap - get_finite_dataset's splice-based marginal construction operates
# on an (top, bottom, left, right) box regardless of where that box sits,
# so no change was needed there. If the four corners' MI-vs-length curves
# look similar to each other (and to the center's), that's evidence MI
# scaling here is a generic property of "how much of the image is
# revealed," not an artifact of the center-out ordering the paper used; if
# they diverge from each other, that's a sign the choice of starting corner
# (or its distance from an edge/other corner) is doing real work. Run as:
#     python -m MI_scaling_non_middle.corner_mi_scaling

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results")
DATASETS = {
    "mnist": img.DEFAULT_IMAGE_SIZE,
    "cifar10": img.DEFAULT_IMAGE_SIZE,
}
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]

NUM_IMAGES = 10000
PARAM_SETTINGS = dict(
    drop=0,
    learn=1e-4,
    layers="[256, 256]",
    patience=8,
    optm="rms",
    val=1 / 7,
    batch=64,
    epoch=30,
)
EVAL_STEPS = 400


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(image_type, lengths, corner_results):
    ensure_results_dir()
    np.save(os.path.join(RESULTS_DIR, f"{image_type}_mi_lengths.npy"), np.asarray(lengths))
    for corner, values in corner_results.items():
        np.save(os.path.join(RESULTS_DIR, f"{image_type}_{corner}_mi_direct.npy"), np.asarray(values["direct"]))
        np.save(os.path.join(RESULTS_DIR, f"{image_type}_{corner}_mi_indirect.npy"), np.asarray(values["indirect"]))


def plot_corner_results(image_type, lengths, corner_results):
    ensure_results_dir()

    series = {}
    for corner, values in corner_results.items():
        label = corner.replace("_", " ").title()
        series[label] = values["direct"]

    axes = img.plot_mi_scaling(series, lengths=lengths, clip_negative=True, save_path=None)
    axes.set_title(f"MI scaling from image corners ({image_type})", fontsize=16)
    plt.tight_layout()

    pdf_path = os.path.join(RESULTS_DIR, f"{image_type}_corner_mi_scaling.pdf")
    png_path = os.path.join(RESULTS_DIR, f"{image_type}_corner_mi_scaling.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close()


def build_region_fn(corner):
    def region_fn(inner_length, img_height, img_width):
        return img.get_corner_region(inner_length, img_height, img_width, corner)
    return region_fn


def run_corner_sweep(image_type, target_size, lengths):
    alg_settings = dict(
        image_type=image_type,
        num_images=NUM_IMAGES,
        strength="small",
        algorithm="logistic",
    )

    corner_results = {}
    for corner in CORNERS:
        corner_results[corner] = {
            "direct": [],
            "indirect": [],
        }
        region_fn = build_region_fn(corner)
        for length in lengths:
            print(f"[{image_type}] {corner} partition length: {length}", flush=True)
            indirect_mi, direct_mi = mine.run_bipartition(
                length,
                alg_settings,
                PARAM_SETTINGS,
                eval_steps=EVAL_STEPS,
                target_size=target_size,
                region_fn=region_fn,
            )
            corner_results[corner]["direct"].append(direct_mi)
            corner_results[corner]["indirect"].append(indirect_mi)

    return corner_results


def main():
    lengths = list(range(1, img.DEFAULT_IMAGE_SIZE + 1))
    for image_type, target_size in DATASETS.items():
        corner_results = run_corner_sweep(image_type, target_size, lengths)
        save_results(image_type, lengths, corner_results)
        plot_corner_results(image_type, lengths, corner_results)


if __name__ == "__main__":
    main()
