import configparser
import ast
import math

import tensorflow as tf
from tensorflow import keras as ks
import numpy as np

from src import image as img

# This module implements the actual MI estimation algorithm: a neural
# network is trained as a binary classifier to distinguish real (a, b) pairs
# drawn from the joint distribution p(a, b) - here, a whole image, with `a`
# the inner patch and `b` the surrounding outer patch - from "fake" pairs
# built by splicing an inner patch from one image onto the outer patch of an
# unrelated image, which approximates a draw from the product-of-marginals
# distribution p(a)p(b). If the classifier is trained well, its output on a
# real sample approximates the log-density-ratio log[p(a,b) / p(a)p(b)],
# whose expectation over p(a,b) *is* the mutual information (a standard
# result sometimes called the "density-ratio trick" or, in this classifier
# formulation, related to the Donsker-Varadhan/NWJ family of MI lower
# bounds). See evaluate_MI below for exactly how the two MI estimates
# (`direct`/`indirect`) are built from the classifier's output on held-out
# data, and get_finite_dataset for how the "real" vs "spliced" batches are
# actually constructed.

class Model():

    # This class holds the fully-connected neural network that is used for
    # MI estimation. It has two identically-weighted "towers" (see
    # build_model) that are fed different inputs each training step: one
    # gets real, untouched images (samples from the joint p(a,b)); the other
    # gets the spliced "fake" images described above (samples from the
    # product of marginals p(a)p(b)). Model itself is agnostic to what loss
    # function turns those two outputs into a training signal - that's
    # supplied by a subclass (LogisticRegression or MINE, below), which
    # determines what quantity the network's scalar output actually
    # approximates once trained.

    def __init__(self, image_shape, settings):

        # The model can be configured based on the numuber of
        # layers, learning rate, optimizers, and dropout.

        self.drop = float(settings['drop'])
        self.learn_rate = float(settings['learn'])
        self.layers = ast.literal_eval(settings['layers'])
        self.patience = int(settings['patience'])
        self.optimizer = settings['optm']
        self.build_model(image_shape)

    def build_model(self, image_shape):

        # The model is constructed such that it contains two inputs, one
        # corresponding to the real ("joint") images and another to the
        # spliced ("marginal") images. Both inputs are run through the same
        # `model_core` sub-network (weight sharing - it's the same Sequential
        # instance called twice, not two separate copies), so the model
        # learns a single scoring function T(x) and just evaluates it on two
        # different batches per step; the two resulting scalar outputs are
        # what the loss functions below (e.g. logistic_loss_joint/_marginal)
        # compare against a target label to know whether a given image
        # looked more "real" (joint) or "fake" (marginal).

        joint_input = ks.Input(shape = image_shape)
        marginal_input = ks.Input(shape = image_shape)
        model_core = ks.models.Sequential()
        model_core.add(ks.Input(shape = image_shape))
        model_core.add(ks.layers.Flatten())
        for layer_size in self.layers:
            model_core.add(ks.layers.Dense(layer_size, activation = 'relu'))
            model_core.add(ks.layers.Dropout(self.drop))
        model_core.add(ks.layers.Dense(1, activation = None))
        joint_output = model_core(joint_input)
        marginal_output = model_core(marginal_input)
        self.model = ks.Model(inputs = [joint_input, marginal_input], outputs = [joint_output, marginal_output])

    def compile_model(self, loss_functions):

        # The model is compiled based on the specified opimizer
        # and learning rate.

        if self.optimizer == 'adam':
            optimizer = ks.optimizers.Adam(learning_rate = self.learn_rate)
        elif self.optimizer == 'rms':
            optimizer = ks.optimizers.RMSprop(learning_rate = self.learn_rate)
        elif self.optimizer == 'sgd':
            optimizer = ks.optimizers.SGD(learning_rate = self.learn_rate)
        else:
            raise ValueError("Optimizer not recognized.")
        self.model.compile(
            optimizer = optimizer,
            loss = loss_functions)

    def train(self, train_itr, val_itr, train_steps, val_steps, epochs):

        # The model is trained using early stoppage according to the
        # patience setting: training halts once `val_loss` fails to improve
        # for `self.patience` consecutive epochs, and the best-seen weights
        # (not necessarily the final epoch's) are restored before returning.

        self.model.fit(
            train_itr,
            steps_per_epoch = train_steps,
            epochs = epochs,
            validation_data = val_itr,
            validation_steps = val_steps,
            callbacks =  [tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss', min_delta = 0, patience = self.patience, restore_best_weights = True)],
            verbose = 1
        )

    def evaluate_MI(self, image_iterator, num_steps):

        # This function uses the trained model to estimate the MI of a
        # given image set, by averaging its classifier output over
        # `num_steps` held-out batches and combining the joint/marginal
        # averages into two different MI estimates:
        #
        #   direct_mi   = E_joint[T(x)]
        #   indirect_mi = E_joint[T(x)] - log(E_marginal[exp(T(x))])
        #
        # `direct_mi` is just the mean classifier score on real (joint)
        # samples. At the Bayes-optimal classifier, T(x) equals the true
        # log-density-ratio, so this mean is exactly the MI by definition -
        # but it carries no correction if the classifier hasn't fully
        # converged, so it can be biased in either direction.
        # `indirect_mi` additionally subtracts log(mean(exp(T))) over
        # marginal samples, a Donsker-Varadhan-style correction term that is
        # mathematically forced to be ~0 if T truly is the log-density-ratio
        # (since E_marginal[exp(log-density-ratio)] = 1). Subtracting it
        # gives a bound that's more robust to imperfect training, at the
        # cost of being sensitive to the `exp()` term blowing up on rare
        # high-scoring marginal batches - which of the two estimates is more
        # trustworthy is dataset-dependent (see README.md's note on this).

        cum_joint = 0
        cum_marginal = 0
        for (count, (image_batch, _)) in enumerate(image_iterator):
            print('\rCount: {}'.format(count), end = '')
            [joint_outputs, marginal_outputs] = self.model.predict_on_batch(image_batch)
            cum_joint += np.mean(joint_outputs)
            cum_marginal += np.mean(np.exp(marginal_outputs))
            if count >= num_steps:
                break
        print('')
        est_mi = cum_joint / num_steps - np.log(cum_marginal / num_steps)
        direct_mi = cum_joint / num_steps
        return (est_mi, direct_mi)

class LogsiticRegression(Model):

    # This model uses the cross-entropy (logistic regression) loss: it
    # trains model_core as an ordinary binary classifier (real vs. spliced),
    # via logistic_loss_joint/_marginal below. At the Bayes-optimal solution
    # this recovers the true log-density-ratio directly (see the module
    # docstring at the top of this file), which is the "direct" MI estimate
    # in evaluate_MI. This is the algorithm used throughout this project's
    # notebook and experiment scripts (`algorithm = "logistic"` in
    # alg.ini-style settings dicts).

    def __init__(self, image_shape, settings):
        Model.__init__(self, image_shape, settings)
        loss_functions = [logistic_loss_joint, logistic_loss_marginal]
        self.compile_model(loss_functions)

# Backwards-compatible alias for the (historically misspelled) class name.
# Existing code and notebooks reference `LogsiticRegression`; new code can use
# the correctly spelled `LogisticRegression`.
LogisticRegression = LogsiticRegression

class MINE(Model):

    # This model instead directly optimizes the Donsker-Varadhan lower bound
    # on MI as its training objective (via biased_MINE_loss_joint/_marginal
    # below, negated since Keras minimizes), rather than a classification
    # loss - the approach introduced by Belghazi et al., "MINE: Mutual
    # Information Neural Estimation" (2018), which this class and module are
    # named after. Selected via `algorithm = "mine"` in alg.ini-style
    # settings dicts; not exercised by this project's own notebook (which
    # uses "logistic" throughout) but fully functional as an alternative.

    def __init__(self, image_shape, settings):
        Model.__init__(self, image_shape, settings)
        loss_functions = [biased_MINE_loss_joint, biased_MINE_loss_marginal]
        self.compile_model(loss_functions)

class Index():

    # A small stateful helper for drawing successive, non-overlapping random
    # batches from a fixed pool of `num_indicies` indices: the constructor
    # shuffles all the indices once, and each call to draw(size) hands out
    # (and permanently removes) the next `size` of them. Used so that a
    # single epoch's worth of batches partitions the dataset exactly once,
    # rather than sampling with replacement.

    def __init__(self, num_indicies):
        self.indices = np.random.permutation(np.arange(num_indicies))

    def draw(self, size):
        choice = self.indices[:size]
        self.indices = self.indices[size:]
        return choice

def get_mixed_indices(num_indices):

    # Generates two independent random permutations of the same index range
    # (index_1, index_2), for use as an (inner-patch source, outer-patch
    # source) pairing when building "marginal" (spliced) samples in
    # get_finite_dataset. Positions where both permutations happen to point
    # at the *same* original index are then repeatedly re-shuffled (the
    # while loop below) until no such collisions remain - without this, some
    # "marginal" samples would end up as an inner patch and outer patch both
    # taken from the *same* real image, which is actually a joint sample in
    # disguise and would bias the MI estimate downward.

    index_1 = Index(num_indices)
    index_2 = Index(num_indices)
    dupl_positions = np.nonzero(np.equal(index_1.indices, index_2.indices))[0]
    while dupl_positions.size != 0:
        dupl_indices = index_1.indices[dupl_positions]
        random_positions_1 = np.random.choice(num_indices, dupl_positions.size, replace = False)
        random_positions_2 = np.random.choice(num_indices, dupl_positions.size, replace = False)
        random_indices_1 = index_1.indices[random_positions_1]
        random_indices_2 = index_2.indices[random_positions_2]
        index_1.indices[dupl_positions] = random_indices_1
        index_1.indices[random_positions_1] = dupl_indices
        index_2.indices[dupl_positions] = random_indices_2
        index_2.indices[random_positions_2] = dupl_indices
        dupl_positions = np.nonzero(np.equal(index_1.indices, index_2.indices))[0]
    return (index_1, index_2)

def logistic_loss_joint(unused_y_true, joint_output):

    # Cross-entropy loss for the "joint" (real, unmodified) batch, labeled 1
    # (i.e. "this batch is real"). `unused_y_true` exists only because
    # Keras's loss-function signature always passes the target array from
    # `get_finite_dataset`'s dummy zero labels; the actual label used here is
    # the constant `labels = tf.ones_like(...)` below, not that argument.

    labels = tf.ones_like(joint_output)
    logits = joint_output
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(losses)
    return loss

def logistic_loss_marginal(unused_y_true, marginal_output):

    # Cross-entropy loss for the "marginal" (spliced/fake) batch, labeled 0
    # ("this batch is fake") - the mirror image of logistic_loss_joint above.

    labels = tf.zeros_like(marginal_output)
    logits = marginal_output
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(losses)
    return loss

def biased_MINE_loss_joint(unused_y_true, joint_output):

    # The joint (E_joint[T(x)]) term of the negated Donsker-Varadhan bound:
    # since Keras minimizes, and we want to *maximize* E_joint[T] -
    # log(E_marginal[exp(T)]), this term is just its negation, added to
    # biased_MINE_loss_marginal below via Keras's per-output loss summing.

    loss = -tf.reduce_mean(joint_output)
    return loss

def biased_MINE_loss_marginal(unused_y_true, marginal_output):

    # The -log(E_marginal[exp(T(x))]) term of the negated Donsker-Varadhan
    # bound (see biased_MINE_loss_joint above) - "biased" because this uses
    # a plain minibatch average to estimate E_marginal[exp(T)] rather than a
    # bias-corrected moving average, which is a known source of gradient
    # bias in the DV-bound training objective (see Belghazi et al. 2018).

    avg_marginal_exp = tf.reduce_mean(tf.exp(marginal_output))
    loss = tf.math.log(avg_marginal_exp)
    return loss

def get_finite_dataset(images, inner_region, batch_size, loop = True):

    # This function returns a generator that yields (joint, marginal) image
    # batches for MI estimation, one full epoch's worth of batches per outer
    # loop iteration (`for _ in itr`) if `loop = True`, or exactly one epoch
    # if `loop = False`:
    #
    #   - "joint" batch: real images, completely untouched - samples from
    #     the true joint distribution p(inner_patch, outer_patch).
    #   - "marginal" batch: real *outer* patches, but with their *inner*
    #     patch (the `[top:bottom, left:right]` region defined by
    #     `inner_region`, see image.get_center_region) replaced by the inner
    #     patch cut from a *different, unrelated* image (chosen by
    #     get_mixed_indices, which also guarantees the "different" image is
    #     never accidentally the same one). Splicing together two unrelated
    #     images' patches like this approximates a draw from the product of
    #     marginals p(inner_patch)·p(outer_patch), since the two patches no
    #     longer have anything to do with each other.
    #
    # The (dummy) label arrays yielded alongside each image pair are all
    # zeros and are never actually used - see logistic_loss_joint's comment
    # on why the loss functions ignore their `y_true` argument.

    num_batches = math.ceil(images.shape[0] / batch_size)
    (top, bottom, left, right) = inner_region
    rand = np.random.RandomState()
    if loop:
        itr = iter(int, 1) # Infinite iterator
    else:
        itr = range(1)
    for _ in itr:
        image_indices = Index(images.shape[0])
        (mixed_inner_indices, mixed_outer_indices) = get_mixed_indices(images.shape[0])
        for _ in range(num_batches):
            image_choice = image_indices.draw(batch_size)
            mixed_inner_choice = mixed_inner_indices.draw(batch_size)
            mixed_outer_choice = mixed_outer_indices.draw(batch_size)
            mixed_images = images[mixed_outer_choice]
            mixed_images[:, top:bottom, left:right] = images[mixed_inner_choice][:, top:bottom, left:right]
            joint_images = images[image_choice]
            yield ((joint_images, mixed_images), (np.zeros(joint_images.shape[0]), np.zeros(mixed_images.shape[0])))

CORRUPTION_MODES = ('blackout', 'randomize_real', 'randomize_uniform')

def get_corrupted_dataset(images, inner_region, batch_size, mode, loop = True):

    # An ablation of get_finite_dataset, built to test whether that
    # function's cross-image splicing is actually necessary, or whether a
    # "marginal" sample can instead be built more cheaply by taking a single
    # real image's own real inner patch and just corrupting everything
    # outside it:
    #
    #   - "joint" batch: identical to get_finite_dataset's - real, untouched
    #     images.
    #   - "marginal" batch: each image's own real inner patch, kept exactly
    #     as-is, with everything outside `inner_region` replaced according
    #     to `mode`:
    #       - "blackout": outer patch forced to 0.
    #       - "randomize_uniform": outer patch replaced with fresh i.i.d.
    #         Uniform(0, 1) noise per pixel, ignoring the dataset's real
    #         pixel statistics entirely.
    #       - "randomize_real": each image's own outer pixel *values* are
    #         randomly reshuffled among themselves (a per-image permutation,
    #         restricted to just the outer positions) - this preserves that
    #         image's own real brightness/contrast statistics in the outer
    #         region (unlike "blackout"/"randomize_uniform", which can hand
    #         the classifier an easy give-away like "the outer patch is much
    #         darker/brighter than any real one"), while still destroying
    #         its real spatial structure.
    #
    # Because the "marginal" sample's inner patch here is a real image's own
    # real inner patch rather than one spliced in from an unrelated image
    # (contrast get_finite_dataset), this is *not* a true product-of-
    # marginals p(inner)*p(outer) sample - it instead tests a narrower
    # question: can the classifier tell "real full image" from "real inner
    # patch + corrupted outer" without the get_mixed_indices machinery at
    # all. If the resulting MI estimate looks similar to get_finite_dataset's,
    # that's evidence the expensive cross-image splicing is redundant; if it
    # looks wildly different (e.g. inflated, because corruption is trivially
    # detectable), that's evidence the splicing is doing real work.

    if mode not in CORRUPTION_MODES:
        raise ValueError('Corruption mode {} not recognized.'.format(mode))
    num_batches = math.ceil(images.shape[0] / batch_size)
    (top, bottom, left, right) = inner_region
    outer_mask = np.ones(images.shape[1:3], dtype = bool)
    outer_mask[top:bottom, left:right] = False
    rand = np.random.RandomState()
    if loop:
        itr = iter(int, 1) # Infinite iterator
    else:
        itr = range(1)
    for _ in itr:
        image_indices = Index(images.shape[0])
        for _ in range(num_batches):
            image_choice = image_indices.draw(batch_size)
            joint_images = images[image_choice]
            if mode == 'blackout':
                corrupted_images = np.zeros_like(joint_images)
            elif mode == 'randomize_uniform':
                corrupted_images = rand.uniform(0, 1, size = joint_images.shape).astype(joint_images.dtype)
            elif mode == 'randomize_real':
                corrupted_images = joint_images.copy()
                for i in range(corrupted_images.shape[0]):
                    corrupted_images[i][outer_mask] = rand.permutation(corrupted_images[i][outer_mask])
            corrupted_images[:, top:bottom, left:right] = joint_images[:, top:bottom, left:right]
            yield ((joint_images, corrupted_images), (np.zeros(joint_images.shape[0]), np.zeros(corrupted_images.shape[0])))

def cycle_generator(generator_fn, *args, **kwargs):

    # Repeatedly re-invokes a finite generator factory (i.e. calls
    # generator_fn(*args, **kwargs, loop=False) again each time it runs dry
    # and yields from the fresh one), so the validation set can be cycled
    # through indefinitely across many epochs. This exists in place of the
    # more obvious `itertools.cycle(get_finite_dataset(...))` because Keras's
    # data adapters require an actual generator *instance* (something with a
    # `__next__` bound to a running generator frame) and reject
    # `itertools.cycle` objects, which are iterators but not generators in
    # that stricter sense.

    while True:
        yield from generator_fn(*args, **kwargs, loop = False)

def run_bipartition(inner_length, alg_settings, param_settings, eval_steps = 5000, target_size = img.DEFAULT_IMAGE_SIZE, marginal_mode = 'splice'):

    # This is the top-level, one-call entry point for a single MI
    # measurement: given a dataset (`alg_settings["image_type"]`) and a
    # partition length (`inner_length`, the side length of the centered
    # square inner patch - see image.get_center_region), it loads the
    # images, builds and trains a fresh classifier from scratch to
    # distinguish that dataset's joint vs. marginal samples (get_finite_dataset),
    # and returns its (indirect, direct) MI estimate (Model.evaluate_MI).
    # Every scaling-curve sweep in this project - the notebook, alg.ini's own
    # CLI loop below, image_noncrop_experiment.py, etc. - is just this
    # function called once per partition length.
    #
    # eval_steps controls how many validation batches evaluate_MI averages
    # over; the default of 5000 matches the paper's full-scale runs, but a
    # smaller value is more appropriate when num_images has been reduced for
    # a quicker demo. target_size is passed straight through to
    # img.get_images and defaults to the same 28x28 every existing caller
    # already relies on, so this is purely additive - existing
    # alg.ini/mine.ini-driven runs and notebook cells are unaffected. Pass a
    # larger value to train on less-aggressively downsized images (e.g. a
    # dataset's native resolution) - see src/image_noncrop_experiment.py.
    #
    # marginal_mode selects how "marginal" batches are built: the default
    # 'splice' uses get_finite_dataset's cross-image splicing (the original
    # method); any of CORRUPTION_MODES instead routes through
    # get_corrupted_dataset, an ablation that corrupts a single real image's
    # own outer patch instead of splicing in an unrelated one - see
    # get_corrupted_dataset's docstring for why this exists.

    num_images = max(1, int(alg_settings["num_images"]))
    (images, _, _) = img.get_images(
        alg_settings["image_type"],
        num_images,
        strength = alg_settings["strength"],
        target_size = target_size)
    (_, height, width) = images.shape
    images = np.expand_dims(images, axis = 3)
    inner_region = img.get_center_region(inner_length, height, width)

    # `algorithm` is read from alg_settings here (rather than relied on as a
    # module/enclosing-scope global) so that this function is fully
    # self-contained and callable on its own - every caller in this project
    # other than the __main__ block below invokes it directly, without ever
    # running the config-file-driven code at the bottom of this file.
    algorithm = alg_settings["algorithm"]
    if algorithm == 'mine':
        net = MINE(images.shape[1:], param_settings)
    elif algorithm == 'logistic':
        net = LogisticRegression(images.shape[1:], param_settings)
    else:
        raise ValueError('Algorithm {} not recognized.'.format(algorithm))

    val_start = int(images.shape[0] * float(param_settings['val']))
    train_images = images[val_start:]
    val_images = images[:val_start]
    batch_size = int(param_settings['batch'])

    train_steps = int(np.ceil(train_images.shape[0] / batch_size))
    val_steps = int(np.ceil(val_images.shape[0] / batch_size))

    if marginal_mode == 'splice':
        train_itr = get_finite_dataset(train_images, inner_region, batch_size, loop = True)
        val_itr = cycle_generator(get_finite_dataset, val_images, inner_region, batch_size)
    else:
        train_itr = get_corrupted_dataset(train_images, inner_region, batch_size, marginal_mode, loop = True)
        val_itr = cycle_generator(get_corrupted_dataset, val_images, inner_region, batch_size, marginal_mode)

    net.train(train_itr, val_itr, train_steps, val_steps, int(param_settings['epoch']))
    (est_mi, direct_mi) = net.evaluate_MI(val_itr, eval_steps)
    return [est_mi, direct_mi]

if __name__ == "__main__":

    # Command-line entry point (`python -m src.mine`): loads settings from
    # the alg.ini and mine.ini configuration files, then either sweeps
    # run_bipartition across every partition length from `abs(start_length)`
    # up to 27 (if `start_length` is negative in alg.ini) or runs just the
    # one length `start_length` (if non-negative). This is the original,
    # config-file-driven way to run an experiment; the notebook and the
    # standalone *_experiment.py scripts elsewhere in src/ call
    # run_bipartition directly instead and don't use this code path at all.

    alg_parser = configparser.ConfigParser()
    alg_parser.read("alg.ini")
    alg_settings = alg_parser["alg"]

    param_parser = configparser.ConfigParser()
    param_parser.read("mine.ini")
    param_settings = param_parser["dense"]

    algorithm = alg_settings["algorithm"]
    image_type = alg_settings["image_type"]
    strength = alg_settings["strength"]
    start_length = int(alg_settings["start_length"])
    num_images = int(alg_settings["num_images"])

    if image_type in img.rho_values.keys():
        rho = img.rho_values[image_type][strength]
    else:
        rho = 0

    max_length = 28
    if start_length < 0:
        MIs = []
        for length in range(abs(start_length), max_length):
            print("Length: {}".format(length))
            mi_pair = run_bipartition(length, alg_settings, param_settings)
            MIs.append(mi_pair)
            print("\nMI Lower: {} | MI Direct: {}".format(*mi_pair))
            print("{} - {} - {} - {}".format(algorithm, image_type, rho, num_images))
    else:
        mi = run_bipartition(start_length, alg_settings, param_settings)
        print(mi)
  