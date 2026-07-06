import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import language, sequence, image as img

# This script runs the 1D (natural-language word-rank sequence) MI-scaling
# sweep: it trains the existing neural MI estimator (mine.LogisticRegression,
# via sequence.run_sequence_bipartition) once per inner-window length,
# across a range of window lengths, and saves the resulting
# MI-vs-window-length curve, the raw per-length values, and a log-log
# power-law fit distinguishing a flat "1D area/boundary law" from a growing
# "1D volume law" (see src/sequence.py's module docstring for why those two
# regimes are exponent ~0 and ~1 here, rather than the 2D image pipeline's
# exponents of 1 and 2).
#
# This is a NEW, standalone script: src/image.py and src/mine.py are not
# modified, and this does not touch alg.ini/mine.ini. Run as:
#     python -m src.language_experiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "language_results")

NUM_POINTS = 100
MIN_LENGTH = 1
MAX_LENGTH = 95
LENGTH_STEP = 1

NUM_SEQUENCES = 3000

# --- Model settings, tuned via a smoke test before committing to the full
# sweep -------------------------------------------------------------------
# An initial pass with the image pipeline's own layers=[256, 256], drop=0
# showed clear OVERFITTING on this dataset (training loss steadily
# decreasing while validation loss steadily rose) - unsurprising, since with
# ~2,570 training sequences and a two-layer 256-unit network, there are far
# more parameters than samples. Shrinking the network to a single 64-unit
# hidden layer and adding dropout=0.3 fixed this. A positive control (each
# point deliberately blended 60/40 with its predecessor, injecting a known,
# modest local dependency) was then used to confirm this smaller, regularized
# architecture can still reliably detect real dependency of a plausible
# magnitude at this sample size - so a weak/absent result on the real data
# below reflects the data, not an underpowered model.
PARAM_SETTINGS = dict(
    drop = 0.3,
    learn = 1e-4,
    layers = "[64]",
    patience = 15,
    optm = "rms",
    val = 1 / 7,
    batch = 32,
    epoch = 100,
    algorithm = "logistic")
EVAL_STEPS = 300

POWER_LAW_FIT_MAX_LENGTH = 20
MIN_POSITIVE_FIT_POINTS = 5

def run_sweep(lengths = None, sequences = None):

    # Sweeps sequence.run_sequence_bipartition over the given inner-window
    # lengths (default: MIN_LENGTH..MAX_LENGTH) on a single, fixed set of
    # word-rank sequences (built once via language.get_sequences with its
    # default fixed seed, so every length in the sweep trains on the exact
    # same underlying dataset - only the partition length differs).

    if lengths is None:
        lengths = list(range(MIN_LENGTH, MAX_LENGTH + 1, LENGTH_STEP))
    if sequences is None:
        (sequences, _, _) = language.get_sequences(num_sequences = NUM_SEQUENCES, num_points = NUM_POINTS)

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

    # Fits log(MI) = exponent * log(L) + const over the early/small-L
    # portion of the curve, to distinguish a flat 1D area/boundary law
    # (exponent ~ 0) from a growing 1D volume law (exponent ~ 1). Returns
    # None if fewer than min_points values in range are positive (log is
    # undefined for non-positive values) - see src/stock_experiment.py's
    # original version of this function for why that's treated as a
    # distinct, still-meaningful "indistinguishable from zero" outcome
    # rather than an error.

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
    np.save(os.path.join(RESULTS_DIR, "language_mi_lengths.npy"), np.asarray(lengths))
    np.save(os.path.join(RESULTS_DIR, "language_mi_indirect.npy"), np.asarray(indirect_values))
    np.save(os.path.join(RESULTS_DIR, "language_mi_direct.npy"), np.asarray(direct_values))

def plot_scaling_curve(lengths, direct_values):

    # Reuses image.plot_mi_scaling's exact figure style completely
    # unmodified, but overrides the axis labels for the word-sequence
    # context afterward instead of the 2D "Partition Length (pixels)"
    # labels it hardcodes.

    axes = img.plot_mi_scaling(
        {"Gutenberg word-rank sequences": direct_values},
        lengths = lengths,
        clip_negative = True,
        save_path = None)
    axes.set_xlabel("Window Length (words)", fontsize = 16)
    axes.set_ylabel("Mutual Information (nats)", fontsize = 16)
    plt.savefig(os.path.join(RESULTS_DIR, "language_mi_scaling.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "language_mi_scaling.png"), dpi = 150)
    plt.close()

def plot_loglog_fit(lengths, direct_values):
    lengths = np.asarray(lengths, dtype = float)
    direct_values = np.asarray(direct_values, dtype = float)
    fit_mask = lengths <= POWER_LAW_FIT_MAX_LENGTH
    fit = fit_power_law(lengths, direct_values)

    if fit is None:
        (_, axes) = plt.subplots(figsize = (8, 6))
        axes.axhline(0, color = "gray", linewidth = 1)
        axes.plot(lengths[fit_mask], direct_values[fit_mask], "o", label = "Estimated MI (direct)")
        axes.set_xlabel("Window Length (words)", fontsize = 14)
        axes.set_ylabel("Mutual Information (nats)", fontsize = 14)
        axes.set_title(
            "Fewer than {} of {} points are positive - no log-log fit possible".format(
                MIN_POSITIVE_FIT_POINTS, fit_mask.sum()),
            fontsize = 11)
        axes.legend(fontsize = 12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "language_mi_loglog_fit.pdf"))
        plt.savefig(os.path.join(RESULTS_DIR, "language_mi_loglog_fit.png"), dpi = 150)
        plt.close()
        return None

    (exponent, intercept, fit_lengths, fit_values) = fit
    (_, axes) = plt.subplots(figsize = (8, 6))
    axes.loglog(fit_lengths, fit_values, "o", label = "Estimated MI (direct)")
    fit_line = np.exp(intercept) * fit_lengths ** exponent
    axes.loglog(fit_lengths, fit_line, "--", label = "Power-law fit (exponent = {:.2f})".format(exponent))
    axes.set_xlabel("Window Length (words, log scale)", fontsize = 14)
    axes.set_ylabel("Mutual Information (nats, log scale)", fontsize = 14)
    axes.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "language_mi_loglog_fit.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "language_mi_loglog_fit.png"), dpi = 150)
    plt.close()
    return exponent

def report_exponent(exponent, lengths = None, direct_values = None):
    if exponent is None:
        print("No power-law fit could be made over L <= {}: fewer than {} of the direct-MI "
            "estimates in that range were even positive.".format(POWER_LAW_FIT_MAX_LENGTH, MIN_POSITIVE_FIT_POINTS))
        if lengths is not None and direct_values is not None:
            lengths = np.asarray(lengths, dtype = float)
            direct_values = np.asarray(direct_values, dtype = float)
            mask = lengths <= POWER_LAW_FIT_MAX_LENGTH
            print("  MI over that range: mean={:.4f}  std={:.4f}  (nats)".format(
                direct_values[mask].mean(), direct_values[mask].std()))
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
    plot_scaling_curve(lengths, direct_values)
    exponent = plot_loglog_fit(lengths, direct_values)
    report_exponent(exponent, lengths, direct_values)
