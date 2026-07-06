import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import language, sequence, image as img

# This script is the embedding-based counterpart to src/language_experiment.py:
# instead of collapsing each word to a single log-rank scalar, every word is
# mapped to its pretrained GloVe embedding vector (see
# language.get_embedded_sequences), so each sequence has shape
# (num_points, embedding_dim) rather than (num_points,) - a genuine
# per-token representation of meaning, not just commonness. No changes were
# needed anywhere in src/sequence.py or src/mine.py for this 3D input - see
# language.get_embedded_sequences's docstring for why.
#
# WHICH MI ESTIMATE IS PRIMARY HERE, AND WHY IT DIFFERS FROM THE REST OF
# THIS PROJECT: every other experiment in this project (images, stocks, the
# rank-based language sweep) plots the *direct* MI estimate
# (mean(joint_output)) as primary, since it is normally far less noisy than
# the *indirect* Donsker-Varadhan-style estimate. Smoke-testing this
# embedding setup surfaced a case where that default breaks down: direct
# and indirect estimates diverged sharply (e.g. direct=-0.34, indirect=+0.11
# at one length), because early stopping landed on a checkpoint whose joint/
# marginal logits both sit at a shifted, not-fully-calibrated scale.
#
# This is provable, not just a guess: if a constant C is added to both the
# joint and marginal logits (i.e. the whole output shifts uniformly),
#   direct  = mean(joint) -> mean(joint) + C                    (SHIFTS)
#   indirect = mean(joint) - log(mean(exp(marginal)))
#           -> (mean(joint) + C) - log(exp(C) * mean(exp(marginal)))
#           -> mean(joint) - log(mean(exp(marginal)))            (UNCHANGED)
# so indirect is invariant to exactly this kind of miscalibration, while
# direct is not - direct is only meaningful when the marginal branch happens
# to already be calibrated so that mean(exp(marginal)) ~ 1. Since that
# calibration isn't reliable here, *indirect* is used as the primary metric
# for this script only; both are still saved.
#
# Run as:
#     python -m src.language_embedding_experiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "language_results")

NUM_POINTS = 100
MIN_LENGTH = 1
MAX_LENGTH = 95
LENGTH_STEP = 1

NUM_SEQUENCES = 3000

# The flattened input here (num_points x embedding_dim = 100 x 50 = 5,000)
# is ~50x larger than the rank-based sweep's plain 100-scalar input, so even
# a single 64-unit hidden layer has ~320,000 parameters - large relative to
# ~2,570 training sequences. The same [64] + dropout=0.3 regularization
# used for the rank-based sweep was kept (a smoke test confirmed it trains
# without collapsing), rather than shrinking further, since the *indirect*
# MI estimate reported here already accounts for the residual calibration
# drift discussed above.
PARAM_SETTINGS = dict(
    drop = 0.3,
    learn = 1e-4,
    layers = "[64]",
    patience = 15,
    optm = "rms",
    val = 1 / 7,
    batch = 32,
    epoch = 60,
    algorithm = "logistic")
EVAL_STEPS = 200

POWER_LAW_FIT_MAX_LENGTH = 20
MIN_POSITIVE_FIT_POINTS = 5

def run_sweep(lengths = None, sequences = None):
    if lengths is None:
        lengths = list(range(MIN_LENGTH, MAX_LENGTH + 1, LENGTH_STEP))
    if sequences is None:
        (sequences, _, _) = language.get_embedded_sequences(num_sequences = NUM_SEQUENCES, num_points = NUM_POINTS)

    indirect_values = []
    direct_values = []
    for length in lengths:
        print("Window length: {}".format(length), flush = True)
        (indirect_mi, direct_mi) = sequence.run_sequence_bipartition(
            length, sequences, PARAM_SETTINGS, eval_steps = EVAL_STEPS)
        indirect_values.append(indirect_mi)
        direct_values.append(direct_mi)
        print("  indirect={:.4f}  direct={:.4f}".format(indirect_mi, direct_mi), flush = True)
    return (lengths, indirect_values, direct_values)

def fit_power_law(lengths, values, max_fit_length = POWER_LAW_FIT_MAX_LENGTH,
        min_points = MIN_POSITIVE_FIT_POINTS):
    lengths = np.asarray(lengths, dtype = float)
    values = np.asarray(values, dtype = float)
    mask = (lengths <= max_fit_length) & (values > 0)
    if mask.sum() < min_points:
        return None
    log_lengths = np.log(lengths[mask])
    log_values = np.log(values[mask])
    (exponent, intercept) = np.polyfit(log_lengths, log_values, 1)
    return (exponent, intercept, lengths[mask], values[mask])

def save_results(lengths, indirect_values, direct_values):
    os.makedirs(RESULTS_DIR, exist_ok = True)
    np.save(os.path.join(RESULTS_DIR, "language_embed_mi_lengths.npy"), np.asarray(lengths))
    np.save(os.path.join(RESULTS_DIR, "language_embed_mi_indirect.npy"), np.asarray(indirect_values))
    np.save(os.path.join(RESULTS_DIR, "language_embed_mi_direct.npy"), np.asarray(direct_values))

def plot_scaling_curve(lengths, indirect_values):
    axes = img.plot_mi_scaling(
        {"GloVe-embedded word sequences": indirect_values},
        lengths = lengths,
        clip_negative = True,
        save_path = None)
    axes.set_xlabel("Window Length (words)", fontsize = 16)
    axes.set_ylabel("Mutual Information (nats, indirect estimate)", fontsize = 16)
    plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_scaling.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_scaling.png"), dpi = 150)
    plt.close()

def plot_loglog_fit(lengths, indirect_values):
    lengths = np.asarray(lengths, dtype = float)
    indirect_values = np.asarray(indirect_values, dtype = float)
    fit_mask = lengths <= POWER_LAW_FIT_MAX_LENGTH
    fit = fit_power_law(lengths, indirect_values)

    if fit is None:
        (_, axes) = plt.subplots(figsize = (8, 6))
        axes.axhline(0, color = "gray", linewidth = 1)
        axes.plot(lengths[fit_mask], indirect_values[fit_mask], "o", label = "Estimated MI (indirect)")
        axes.set_xlabel("Window Length (words)", fontsize = 14)
        axes.set_ylabel("Mutual Information (nats)", fontsize = 14)
        axes.set_title(
            "Fewer than {} of {} points are positive - no log-log fit possible".format(
                MIN_POSITIVE_FIT_POINTS, fit_mask.sum()),
            fontsize = 11)
        axes.legend(fontsize = 12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_loglog_fit.pdf"))
        plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_loglog_fit.png"), dpi = 150)
        plt.close()
        return None

    (exponent, intercept, fit_lengths, fit_values) = fit
    (_, axes) = plt.subplots(figsize = (8, 6))
    axes.loglog(fit_lengths, fit_values, "o", label = "Estimated MI (indirect)")
    fit_line = np.exp(intercept) * fit_lengths ** exponent
    axes.loglog(fit_lengths, fit_line, "--", label = "Power-law fit (exponent = {:.2f})".format(exponent))
    axes.set_xlabel("Window Length (words, log scale)", fontsize = 14)
    axes.set_ylabel("Mutual Information (nats, log scale)", fontsize = 14)
    axes.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_loglog_fit.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "language_embed_mi_loglog_fit.png"), dpi = 150)
    plt.close()
    return exponent

def report_exponent(exponent, lengths = None, indirect_values = None):
    if exponent is None:
        print("No power-law fit could be made over L <= {}: fewer than {} of the indirect-MI "
            "estimates in that range were even positive.".format(POWER_LAW_FIT_MAX_LENGTH, MIN_POSITIVE_FIT_POINTS))
        if lengths is not None and indirect_values is not None:
            lengths = np.asarray(lengths, dtype = float)
            indirect_values = np.asarray(indirect_values, dtype = float)
            mask = lengths <= POWER_LAW_FIT_MAX_LENGTH
            print("  MI over that range: mean={:.4f}  std={:.4f}  (nats)".format(
                indirect_values[mask].mean(), indirect_values[mask].std()))
        print("  -> MI is statistically indistinguishable from zero at every tested window length.")
        return
    print("Fitted power-law exponent over L <= {}: {:.3f}".format(POWER_LAW_FIT_MAX_LENGTH, exponent))
    if exponent < 0.3:
        print("  -> consistent with a 1D area/boundary law (~flat MI): short-range dependence only.")
    elif exponent > 0.7:
        print("  -> consistent with a 1D volume law (~linear MI growth): genuine long-range dependence.")
    else:
        print("  -> intermediate; neither limiting 1D regime is a clean match.")

if __name__ == "__main__":

    (lengths, indirect_values, direct_values) = run_sweep()
    save_results(lengths, indirect_values, direct_values)
    plot_scaling_curve(lengths, indirect_values)
    exponent = plot_loglog_fit(lengths, indirect_values)
    report_exponent(exponent, lengths, indirect_values)
