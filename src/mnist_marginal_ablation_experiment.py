import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import mine, image as img

# This script tests a hypothesis about get_finite_dataset's "marginal" batch
# construction: that method builds a marginal sample by splicing an inner
# patch from one real image onto the outer patch of a different, unrelated
# real image (via get_mixed_indices) - a genuine approximation of a draw
# from the product of marginals p(inner)*p(outer). That machinery exists to
# guarantee the two patches are truly unrelated, but it's also the most
# complex part of the data pipeline. This script checks whether it's
# actually necessary, by comparing the resulting MI-scaling curve against
# two much cheaper "marginal" constructions (mine.get_corrupted_dataset),
# each of which keeps a single real image's own real inner patch and just
# corrupts everything outside it instead of splicing in a second image:
#
#   - blackout: outer patch forced to 0.
#   - randomize_real: each image's own outer pixel values reshuffled among
#     themselves (preserves that image's real brightness/contrast, destroys
#     spatial structure).
#
# (A third mode, randomize_uniform, was tried in an earlier run at lower
# training budget and dropped here - see mnist_{mode}_mi_*.npy for that
# three-mode, 10,000 image / 30 epoch sweep.)
#
# If any of these produce a similar MI-vs-partition-length curve to the
# original splice method, that's evidence the cross-image splicing is
# redundant. If they instead produce a wildly different (typically inflated)
# curve, that's evidence the classifier is exploiting how easy corrupted
# outer patches are to spot - a good example of a "marginal" construction
# that's *too* easy to distinguish from the real thing, rather than a good
# approximation of drawing from p(inner)*p(outer).
#
# This run uses a substantially larger training budget (NUM_IMAGES/epoch,
# see RUN_LABEL below) than the original 10,000 image / 30 epoch sweep,
# specifically to test whether that sweep's dramatically inflated MI values
# were a training-budget artifact (values should shrink toward the splice
# baseline here if so) or a structural property of corrupted marginals
# (values should stay inflated, or grow further, regardless of budget) - see
# plot_hightrain_comparison, which plots both training budgets together.
# Run as:
#     python -m src.mnist_marginal_ablation_experiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mnist_marginal_ablation_results")
BASELINE_LENGTHS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image_noncrop_results", "mnist_mi_lengths.npy")
BASELINE_DIRECT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image_noncrop_results", "mnist_mi_direct.npy")

NUM_IMAGES = 30000
PARAM_SETTINGS = dict(
    drop = 0,
    learn = 1e-4,
    layers = "[256, 256]",
    patience = 12,
    optm = "rms",
    val = 1 / 7,
    batch = 64,
    epoch = 60)
EVAL_STEPS = 400

MODES = ["blackout", "randomize_real"]

# Distinguishes this run's saved files from the earlier (10,000 image, 30
# epoch) sweep's, so both stay on disk and can be compared side by side -
# this run exists specifically to test whether that earlier sweep's
# ballooned MI values were an under-training artifact (see the comparison
# plot built in plot_hightrain_comparison below).
RUN_LABEL = "n{}_e{}".format(NUM_IMAGES, PARAM_SETTINGS["epoch"])

def run_sweep(mode, lengths):

    # Sweeps mine.run_bipartition over `lengths` on MNIST with the given
    # corruption mode, returning the (indirect, direct) MI estimate at every
    # length.

    alg_settings = dict(
        image_type = "mnist",
        num_images = NUM_IMAGES,
        strength = "small",
        algorithm = "logistic")

    indirect_values = []
    direct_values = []
    for length in lengths:
        print("[{}] partition length: {}".format(mode, length), flush = True)
        (indirect_mi, direct_mi) = mine.run_bipartition(
            length, alg_settings, PARAM_SETTINGS, eval_steps = EVAL_STEPS, marginal_mode = mode)
        indirect_values.append(indirect_mi)
        direct_values.append(direct_mi)
        print("  indirect={:.4f}  direct={:.4f}".format(indirect_mi, direct_mi), flush = True)
    return (indirect_values, direct_values)

def save_results(mode, lengths, indirect_values, direct_values):
    os.makedirs(RESULTS_DIR, exist_ok = True)
    np.save(os.path.join(RESULTS_DIR, "mnist_{}_{}_mi_lengths.npy".format(mode, RUN_LABEL)), np.asarray(lengths))
    np.save(os.path.join(RESULTS_DIR, "mnist_{}_{}_mi_indirect.npy".format(mode, RUN_LABEL)), np.asarray(indirect_values))
    np.save(os.path.join(RESULTS_DIR, "mnist_{}_{}_mi_direct.npy".format(mode, RUN_LABEL)), np.asarray(direct_values))

def plot_hightrain_comparison(lengths, baseline_direct, results):

    # Plots the splice baseline alongside both the *original* (10,000 image,
    # 30 epoch) ablation curves and this run's higher-training (RUN_LABEL)
    # curves for the same two modes, so the two training budgets can be
    # compared directly - the whole point of this run is to check whether
    # the original sweep's ballooned values were a training-budget artifact
    # (in which case they'd shrink toward the baseline here) or a structural
    # property of corrupted marginals (in which case they'd stay inflated,
    # or grow further).

    series = {"mnist (splice, original)": baseline_direct}
    for mode in MODES:
        original_path = os.path.join(RESULTS_DIR, "mnist_{}_mi_direct.npy".format(mode))
        if os.path.exists(original_path):
            series["mnist ({}, 10k img/30 epoch)".format(mode)] = np.load(original_path).tolist()
    for (mode, direct_values) in results.items():
        series["mnist ({}, {})".format(mode, RUN_LABEL)] = direct_values
    img.plot_mi_scaling(series, lengths = lengths, clip_negative = True, save_path = None)
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_marginal_ablation_hightrain_comparison.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_marginal_ablation_hightrain_comparison.png"), dpi = 150)
    plt.close()

if __name__ == "__main__":

    lengths = np.load(BASELINE_LENGTHS_PATH).tolist()
    baseline_direct = np.load(BASELINE_DIRECT_PATH).tolist()

    all_results = {}
    for mode in MODES:
        (indirect_values, direct_values) = run_sweep(mode, lengths)
        save_results(mode, lengths, indirect_values, direct_values)
        all_results[mode] = direct_values
    plot_hightrain_comparison(lengths, baseline_direct, all_results)
    print("Done.", flush = True)
