import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import image as img

# Builds a poster-ready figure illustrating the "splicing" trick
# src/mine.py's get_finite_dataset uses to construct marginal (fake) samples
# for MI estimation: a "real" (joint) sample is an untouched image, whose
# inner 8x8 patch and outer surrounding ring are genuinely correlated
# (both come from the same digit); a "fake" (marginal) sample splices the
# inner patch from a *different, unrelated* image onto the outer ring,
# destroying that correlation - the two distributions a classifier is
# trained to tell apart, per mine.py's module docstring. This script is a
# standalone illustration, not part of the research pipeline - it doesn't
# touch mine.py/run_bipartition, it just reproduces get_finite_dataset's
# actual splice operation for two chosen example digits so it can be shown
# on a poster.

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_SIZE = 10
IMAGE_SIZE = img.DEFAULT_IMAGE_SIZE


def load_two_digits(index_a, index_b):
    from tensorflow import keras as ks
    ((x_train, y_train), (_, _)) = ks.datasets.mnist.load_data()
    image_a = x_train[index_a].astype(np.float64) / 255
    image_b = x_train[index_b].astype(np.float64) / 255
    print(f"digit A: index={index_a} label={y_train[index_a]}")
    print(f"digit B: index={index_b} label={y_train[index_b]}")
    return (image_a, image_b)


def splice(outer_source, inner_source, region):
    (top, bottom, left, right) = region
    spliced = outer_source.copy()
    spliced[top:bottom, left:right] = inner_source[top:bottom, left:right]
    return spliced


def draw_patch_box(axes, region, color="#DD2222", linewidth=2.5):
    (top, bottom, left, right) = region
    axes.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top,
                              fill=False, edgecolor=color, linewidth=linewidth))


def style_axes(axes):
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def make_figure(image_a, image_b, region, output_tag):
    frankenstein = splice(image_b, image_a, region)  # B's outer ring + A's inner patch

    (fig, axes_row) = plt.subplots(1, 5, figsize=(16, 3.6),
                                    gridspec_kw={"width_ratios": [1, 0.25, 1, 0.25, 1]})
    (ax_a, ax_plus, ax_b, ax_equals, ax_frank) = axes_row

    ax_a.imshow(image_a, cmap="gray", vmin=0, vmax=1)
    draw_patch_box(ax_a, region)
    style_axes(ax_a)

    ax_b.imshow(image_b, cmap="gray", vmin=0, vmax=1)
    draw_patch_box(ax_b, region)
    style_axes(ax_b)

    ax_frank.imshow(frankenstein, cmap="gray", vmin=0, vmax=1)
    draw_patch_box(ax_frank, region, color="#2266DD")
    style_axes(ax_frank)

    ax_plus.axis("off")
    ax_equals.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"frankenstein_{output_tag}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, facecolor="white")
        print(f"saved {path}")
    plt.close()
    return frankenstein


def make_real_vs_fake_figure(image_a, image_b, region, frankenstein, output_tag):

    # A second, more MI-focused panel: the actual real/joint vs. fake/
    # marginal pair a classifier is trained to distinguish, side by side,
    # with no splicing arithmetic shown - just the two things a poster
    # viewer needs to see the classifier is telling apart.

    (fig, axes_row) = plt.subplots(1, 2, figsize=(7, 3.8))
    (ax_real, ax_fake) = axes_row

    ax_real.imshow(image_b, cmap="gray", vmin=0, vmax=1)
    draw_patch_box(ax_real, region, color="#22AA44")
    style_axes(ax_real)

    ax_fake.imshow(frankenstein, cmap="gray", vmin=0, vmax=1)
    draw_patch_box(ax_fake, region, color="#DD2222")
    style_axes(ax_fake)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"real_vs_fake_{output_tag}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, facecolor="white")
        print(f"saved {path}")
    plt.close()


def main():
    # Indices 129 (a "6") and 41 (an "8") were picked over the first two
    # training images (a "5" and a "0") because both have real, dense ink
    # in the center 8x8 patch - a "0"'s center is naturally blank, which
    # made the spliced-vs-real comparison too subtle to read on a poster.
    (image_a, image_b) = load_two_digits(129, 41)
    region = img.get_center_region(PATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    print("inner patch region (top, bottom, left, right):", region)

    frankenstein = make_figure(image_a, image_b, region, "6_8")
    make_real_vs_fake_figure(image_a, image_b, region, frankenstein, "6_8")


if __name__ == "__main__":
    main()
