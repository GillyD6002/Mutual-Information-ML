import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import mine, image as img

# This script redoes the real-dataset MI-scaling sweep (see examples.ipynb's
# "Visualizing the scaling for a real dataset" section, and real_dataset_scaling.pdf)
# for the datasets that get most aggressively downsized by the default 28x28
# pipeline - CIFAR-10 (32x32 native), LFW faces (125x94 native), and FER-2013
# (48x48 native, via the new fer2013_hf source) - but at each dataset's own
# native (or best achievable square) resolution instead, to check how much
# MI the 28x28 conforming step was throwing away. MNIST/Fashion-MNIST are
# excluded: they're already natively 28x28, so conform_size is a no-op for
# them and there's nothing to undo.
#
# Each dataset is swept all the way from length 1 to its own full target_size
# (the entire uncropped image, not just the first 27 pixels), so the curve
# shows the whole "inner patch grows to swallow the image" story rather than
# an arbitrary truncation. At length = target_size the inner patch *is* the
# whole image and the outer patch is empty, so MI(inner; outer) necessarily
# collapses back toward 0 there - the classifier has nothing left to
# discriminate, since both the "joint" and "marginal" batches are just draws
# of a whole image at that point. That's an expected, meaningful boundary
# condition (mirroring how bipartite entanglement entropy is symmetric and
# vanishes at both ends of a pure-state bipartition), not a bug.
#
# This is a NEW, standalone script: src/image.py and src/mine.py keep their
# existing default behavior unchanged (mine.run_bipartition's target_size
# argument defaults to the original 28, so every existing caller -
# alg.ini/mine.ini-driven runs, examples.ipynb, language_experiment.py - is
# unaffected). Run as:
#     python -m src.image_noncrop_experiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image_noncrop_results")

# Each dataset's target_size is its own native resolution (cifar10,
# fer2013_hf), or, for LFW's non-square 125x94 images, the larger of its two
# dimensions' minimum (94) - the biggest square conform_size can produce
# without upsampling in either dimension. At these sizes conform_size either
# performs no resize at all (cifar10, fer2013_hf are already exactly square
# at these values) or the mildest resize possible (lfw_faces).
DATASETS = {
    "cifar10": 32,
    "lfw_faces": 94,
    "fer2013_hf": 48,
}

MIN_LENGTH = 1

# Settings match examples.ipynb's existing reduced real-dataset sweep
# (10,000 images, up to 30 epochs, patience 8, 400 eval steps), rather than
# the paper's full scale in mine.ini (70,000 images, up to 3,000 epochs).
# Sweeping all the way to each dataset's full target_size (94 for lfw_faces,
# vs. the 27-length truncated sweep this replaces) roughly triples lfw_faces'
# runtime; based on the truncated sweep's measured per-length cost, the full
# extended sweep is estimated at around an hour total on this CPU-only box.
# Raise NUM_IMAGES/epoch for higher-fidelity curves if more time is available.
NUM_IMAGES = 10000
PARAM_SETTINGS = dict(
    drop = 0,
    learn = 1e-4,
    layers = "[256, 256]",
    patience = 8,
    optm = "rms",
    val = 1 / 7,
    batch = 64,
    epoch = 30)
EVAL_STEPS = 400

def run_sweep(image_type, target_size, lengths = None):

    # Sweeps mine.run_bipartition over the given partition lengths (default:
    # MIN_LENGTH..target_size inclusive, i.e. the entire uncropped image)
    # for one dataset at its non-cropped target_size, returning the
    # (indirect, direct) MI estimate at every length.

    if lengths is None:
        lengths = list(range(MIN_LENGTH, target_size + 1))
    alg_settings = dict(
        image_type = image_type,
        num_images = NUM_IMAGES,
        strength = "small",
        algorithm = "logistic")

    indirect_values = []
    direct_values = []
    for length in lengths:
        print("[{}] partition length: {}".format(image_type, length), flush = True)
        (indirect_mi, direct_mi) = mine.run_bipartition(
            length, alg_settings, PARAM_SETTINGS, eval_steps = EVAL_STEPS, target_size = target_size)
        indirect_values.append(indirect_mi)
        direct_values.append(direct_mi)
        print("  indirect={:.4f}  direct={:.4f}".format(indirect_mi, direct_mi), flush = True)
    return (lengths, indirect_values, direct_values)

def save_results(image_type, lengths, indirect_values, direct_values):
    os.makedirs(RESULTS_DIR, exist_ok = True)
    np.save(os.path.join(RESULTS_DIR, "{}_mi_lengths.npy".format(image_type)), np.asarray(lengths))
    np.save(os.path.join(RESULTS_DIR, "{}_mi_indirect.npy".format(image_type)), np.asarray(indirect_values))
    np.save(os.path.join(RESULTS_DIR, "{}_mi_direct.npy".format(image_type)), np.asarray(direct_values))

def plot_combined(results):

    # Plots every dataset's direct-MI scaling curve together, reusing
    # image.plot_mi_scaling's exact figure style unmodified. The *direct*
    # estimate is plotted rather than the indirect (Donsker-Varadhan-style)
    # one, matching examples.ipynb's own real-dataset sweep - see README.md's
    # note on why direct is more stable for real (non-Gaussian) data.
    #
    # Each dataset now sweeps a different length range (1..target_size, and
    # target_size differs per dataset), so a single shared `lengths` array
    # can't be passed to plot_mi_scaling for all three series - it plots
    # every series against whatever `lengths` is given, but here each
    # series's x-axis differs from the others'. Instead, lengths is left as
    # None so plot_mi_scaling falls back to its own per-series default of
    # np.arange(1, len(values) + 1), which is exactly each dataset's actual
    # 1..target_size range since MIN_LENGTH is always 1.

    series = {}
    for (image_type, (dataset_lengths, direct_values)) in results.items():
        size = DATASETS[image_type]
        series["{} ({}x{}, non-cropped)".format(image_type, size, size)] = direct_values
    axes = img.plot_mi_scaling(series, clip_negative = True, save_path = None)
    plt.savefig(os.path.join(RESULTS_DIR, "non_cropped_mi_scaling.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "non_cropped_mi_scaling.png"), dpi = 150)
    plt.close()

if __name__ == "__main__":

    all_results = {}
    for (image_type, target_size) in DATASETS.items():
        (lengths, indirect_values, direct_values) = run_sweep(image_type, target_size)
        save_results(image_type, lengths, indirect_values, direct_values)
        all_results[image_type] = (lengths, direct_values)
    plot_combined(all_results)
