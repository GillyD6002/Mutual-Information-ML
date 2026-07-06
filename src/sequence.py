import math
import numpy as np

from src import mine

# This module is a 1D generalization of the 2D partition/splice logic in
# src/image.py (get_center_region) and src/mine.py (get_finite_dataset,
# run_bipartition), for testing MI scaling along a single axis - e.g. word
# sequences from natural-language text (src/language.py) - rather than
# across a 2D pixel grid. This is a NEW module: src/image.py and
# src/mine.py are not modified by anything here.
#
# ------------------------------------------------------------------------
# IMPORTANT CONCEPTUAL POINT - 1D scaling exponents differ from the 2D ones:
#
# In the existing 2D image code, a centered square inner patch of side
# length L sits inside a 2D grid, and its BOUNDARY (perimeter) grows
# linearly with L (perimeter ~ 4L), while its AREA grows quadratically with
# L (area ~ L^2). That is why, over there, a "boundary/area law" (MI set by
# the boundary) is LINEAR in L and a "volume law" (MI set by the enclosed
# area) is QUADRATIC in L.
#
# In 1D, a contiguous inner WINDOW of length L sitting inside a 1D sequence
# has a boundary of just 2 POINTS - its left and right edges - no matter how
# large L gets. There is no perimeter that grows with L. So the two 1D
# scaling regimes are instead:
#
#   - "1D area/boundary law": MI is roughly CONSTANT in L, independent of
#     window length. This is the regime where only short-range dependence
#     exists (e.g. a word only depends on its immediate neighbors), and is
#     the regime that a boundary-respecting ansatz like a matrix product
#     state (MPS) / tensor train handles efficiently.
#   - "1D volume law": MI GROWS with L (roughly linearly, in the cleanest
#     case). This indicates genuine long-range dependence spread throughout
#     the window, not concentrated at its two edges - plausible for text,
#     where topical/thematic coherence can span an entire passage rather
#     than just adjacent words.
#
# Do NOT reuse the 2D module's "linear = boundary law, quadratic = volume
# law" language when describing results computed here - the exponents that
# distinguish the two regimes are different (0 vs. 1 here, instead of 1 vs.
# 2 for images).
#
# ------------------------------------------------------------------------
# WHY mine.py NEEDS NO CHANGES:
#
# mine.py's Model.build_model constructs the network as
# ks.Input(shape = image_shape) -> Flatten() -> Dense layers, with no
# assumption anywhere about the rank of image_shape. A shape of (100,) (a
# length-100 sequence) flows through this exactly as a (28, 28, 1) image
# would: Input(shape=(100,)) accepts a (batch, 100) tensor, and Flatten()
# applied to an already-rank-2 (batch, features) tensor is simply a no-op.
# So Model/LogisticRegression/MINE are reused from mine.py completely
# unmodified below - only mine.run_bipartition itself is 2D-specific (it
# calls image.get_center_region and hardcodes a channel-dimension expansion
# via np.expand_dims(images, axis=3), neither of which apply to a plain
# (num_samples, length) sequence array), which is why it gets a 1D analog
# (run_sequence_bipartition) here rather than being reused directly.

def get_center_interval(length, total_length):

    # 1D analog of image.get_center_region: returns the (left, right)
    # bounds of a centered window of the given length within a 1D sequence
    # of total_length points, using the same "shorter side rounds down"
    # convention as get_center_region's (top, left) calculation.

    left = total_length // 2 - length // 2
    return (left, left + length)

def get_finite_sequence_dataset(sequences, inner_interval, batch_size, loop = True):

    # 1D analog of mine.get_finite_dataset: a generator over (joint,
    # marginal) sequence-pair batches. The "marginal" sample splices the
    # inner_interval window of one sequence into the outer (before + after)
    # wings of another, exactly mirroring get_finite_dataset's 2D
    # inner-patch swap but along a single axis instead of two. Reuses
    # mine.Index and mine.get_mixed_indices unmodified, since neither makes
    # any assumption about the dimensionality of the data being indexed.

    num_batches = math.ceil(sequences.shape[0] / batch_size)
    (left, right) = inner_interval
    if loop:
        itr = iter(int, 1) # Infinite iterator
    else:
        itr = range(1)
    for _ in itr:
        sequence_indices = mine.Index(sequences.shape[0])
        (mixed_inner_indices, mixed_outer_indices) = mine.get_mixed_indices(sequences.shape[0])
        for _ in range(num_batches):
            sequence_choice = sequence_indices.draw(batch_size)
            mixed_inner_choice = mixed_inner_indices.draw(batch_size)
            mixed_outer_choice = mixed_outer_indices.draw(batch_size)
            mixed_sequences = sequences[mixed_outer_choice]
            mixed_sequences[:, left:right] = sequences[mixed_inner_choice][:, left:right]
            joint_sequences = sequences[sequence_choice]
            yield ((joint_sequences, mixed_sequences), (np.zeros(joint_sequences.shape[0]), np.zeros(mixed_sequences.shape[0])))

def run_sequence_bipartition(inner_length, sequences, param_settings, eval_steps = 5000):

    # 1D analog of mine.run_bipartition: trains a fresh MI-estimation model
    # for one inner-window length on the given (num_samples, length) array
    # of already-loaded sequences, and returns its (indirect, direct) MI
    # estimate. Unlike mine.run_bipartition (which re-loads its dataset from
    # src/image.py on every call), this takes `sequences` directly, so the
    # (possibly expensive to build) dataset is constructed once by the
    # caller and reused across an entire length sweep.

    total_length = sequences.shape[1]
    inner_interval = get_center_interval(inner_length, total_length)

    algorithm = param_settings.get("algorithm", "logistic")
    if algorithm == "mine":
        net = mine.MINE(sequences.shape[1:], param_settings)
    elif algorithm == "logistic":
        net = mine.LogisticRegression(sequences.shape[1:], param_settings)
    else:
        raise ValueError("Algorithm {} not recognized.".format(algorithm))

    val_start = int(sequences.shape[0] * float(param_settings["val"]))
    train_sequences = sequences[val_start:]
    val_sequences = sequences[:val_start]
    batch_size = int(param_settings["batch"])

    train_steps = int(np.ceil(train_sequences.shape[0] / batch_size))
    val_steps = int(np.ceil(val_sequences.shape[0] / batch_size))

    train_itr = get_finite_sequence_dataset(train_sequences, inner_interval, batch_size, loop = True)
    val_itr = mine.cycle_generator(get_finite_sequence_dataset, val_sequences, inner_interval, batch_size)

    net.train(train_itr, val_itr, train_steps, val_steps, int(param_settings["epoch"]))
    (est_mi, direct_mi) = net.evaluate_MI(val_itr, eval_steps)
    return [est_mi, direct_mi]
