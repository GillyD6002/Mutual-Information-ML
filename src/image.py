import numpy as np
import matplotlib.pyplot as plt

# This module is used to process image data for the MI estimation task.
#
# TensorFlow / scikit-learn are imported lazily inside the loading functions
# so that the Gaussian-field and plotting utilities in this module can be
# used without those installed.

rho_values = { # These are the correlation values for the Gaussiam Markov fields
    "area": {"small": -0.12, "large": -0.227},
    "diffuse": {"small": -0.0012, "large": -0.00127712},
    "sparse": {"small": -0.045, "large": -0.11}
}

# The size (in pixels) that every real image dataset is conformed to. The rest
# of the pipeline (partitioning, 28-pixel covariance reshaping, etc.) assumes
# square images of this side length, matching the original 28 x 28 setup.
DEFAULT_IMAGE_SIZE = 28

def get_pixel_numbers(image_height, image_width, region):

    # Numbers every pixel in an image_height x image_width grid row-by-row
    # (0, 1, 2, ... left to right, top to bottom - matching how a
    # (height, width) array flattens under np.reshape/.flatten()), then
    # returns just the numbers that fall inside `region` (a (top, bottom,
    # left, right) box, e.g. from get_center_region). This is what converts
    # a 2D pixel *region* into the flat integer *indices* that
    # get_marginal_entropy/get_gaussian_mutual_information need, since the
    # covariance matrices in this module are indexed by flattened pixel
    # position, not by (row, col).

    indices = np.reshape(np.arange(image_height*image_width), [image_height, image_width])
    (top, bottom, left, right) = region
    region_indices = indices[top:bottom, left:right]
    return region_indices.flatten()

def get_center_region(length, img_height, img_width):

    # Returns the (top, bottom, left, right) bounding box of a centered
    # length x length square patch within an img_height x img_width image
    # (the "inner" patch throughout this project; everything outside it is
    # the "outer" patch). When length is odd relative to the image
    # dimensions, `top`/`left` round down (floor division), so the patch is
    # not perfectly symmetric by a fraction of a pixel - a minor, consistent
    # convention rather than a bug.

    top = img_height // 2 - length // 2
    left = img_width // 2 - length // 2
    region = (top, top + length, left, left + length)
    return region


def get_corner_region(length, img_height, img_width, corner):

    # Returns a length x length square patch anchored at one of the four
    # image corners. This is used for the non-middle MI scaling experiment
    # where the "inner" region starts in a corner rather than in the center.

    if length > img_height or length > img_width:
        raise ValueError("Requested corner patch length {} exceeds image size {}x{}".format(length, img_height, img_width))

    if corner == 'top_left':
        region = (0, length, 0, length)
    elif corner == 'top_right':
        region = (0, length, img_width - length, img_width)
    elif corner == 'bottom_left':
        region = (img_height - length, img_height, 0, length)
    elif corner == 'bottom_right':
        region = (img_height - length, img_height, img_width - length, img_width)
    else:
        raise ValueError("Corner '{}' not recognized".format(corner))
    return region


def get_grid_region(length, img_height, img_width, row, col, grid_size = 3):

    # Returns a length x length square patch centered on the (row, col) cell
    # of a grid_size x grid_size grid overlaid on the image (row/col each in
    # range(grid_size)) - the generalization of get_center_region (which is
    # exactly the grid_size=3, row=col=1 case) to any number of evenly
    # spaced anchor points, for the 9-cell "moving slideshow" MI-scaling
    # sweep. Small patches are genuinely centered on their cell's center
    # point; once a patch grows too large to fit within the image while
    # staying centered there, it's clamped to the nearest valid position
    # (sliding flush against whichever edge(s) it would otherwise cross) -
    # so cells near a border naturally converge toward corner/edge-anchored
    # behavior for larger lengths, exactly like get_corner_region already
    # does, while every cell still converges to the full image at
    # length = img_height = img_width, same as every other region function
    # in this module.

    if length > img_height or length > img_width:
        raise ValueError("Requested grid patch length {} exceeds image size {}x{}".format(length, img_height, img_width))
    if not (0 <= row < grid_size and 0 <= col < grid_size):
        raise ValueError("Grid cell ({}, {}) out of range for grid_size {}".format(row, col, grid_size))

    row_center = int((row + 0.5) * img_height / grid_size)
    col_center = int((col + 0.5) * img_width / grid_size)
    top = min(max(row_center - length // 2, 0), img_height - length)
    left = min(max(col_center - length // 2, 0), img_width - length)
    region = (top, top + length, left, left + length)
    return region

# --- Gaussian Markov random field (GMRF) constructors ---------------------
#
# Each of the next three functions builds the *precision matrix* Q (inverse
# covariance) of a zero-mean multivariate Gaussian over length**2 "pixels"
# arranged on a length x length grid, then returns its inverse (the
# covariance matrix `cov` that get_gaussian_images/get_analytic_MI actually
# need). Off-diagonal entries of Q set the *conditional* correlation between
# two pixels given every other pixel - a classic Gaussian Markov random
# field construction - which is a more direct way to encode "this pair of
# pixels is/isn't locally coupled" than specifying the covariance matrix
# itself. All three differ only in *which* pairs of pixels get a non-zero Q
# entry (their common value is the `rho` argument), which is what gives
# them exactly-known, structurally different scaling behavior when their
# exact MI is computed (get_analytic_MI) - see README.md's "Area law vs.
# volume law" section for how that scaling behavior was actually measured on
# these fields and on real datasets.

def get_area_law_cov(length, rho):

    # Nearest-neighbor nonzero off-diagonal Q entries only (each pixel
    # conditionally correlated with just its up/down/left/right neighbors on
    # the grid). This is the "area"/boundary-law prototype: since only pairs
    # of pixels adjacent across the inner/outer cut can be correlated at
    # all, the MI a partition picks up scales with the *length of that cut*
    # (~4 x length in 2D, i.e. linearly in the partition side length), not
    # with the enclosed area.

    q = np.eye(length**2)
    for i in range(length):
        for j in range(length):
            q_row = i * length + j
            for (m, l) in [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]:
                if (m < length and m >= 0) and (l < length and l >= 0):
                    q_col = m * length + l
                    q[q_row, q_col] = rho
    cov = np.linalg.inv(q)
    return cov

def get_diffuse_volume_cov(length, rho):

    # Every pixel pair gets the same nonzero Q entry `rho`, regardless of
    # distance on the grid - the opposite extreme from get_area_law_cov's
    # nearest-neighbor-only structure. This is the "volume law" prototype:
    # because correlation doesn't decay with distance, the amount of MI a
    # partition picks up scales with how many pixel *pairs* get split across
    # the inner/outer cut, which (for a small inner patch in a much larger
    # image) grows with the inner patch's *area*, not just its boundary.

    q = np.full([length**2, length**2], rho)
    for i in range(length**2):
        q[i, i] = 1
    cov = np.linalg.inv(q)
    return cov

def get_sparse_volume_cov(length, rho):

    # Starts from exactly the same nearest-neighbor Q as get_area_law_cov,
    # but then randomly permutes which grid position each of the length**2
    # variables actually sits at (`shuffle`, applied to both rows and
    # columns of Q so the permutation is consistent). Every variable is
    # still only conditionally correlated with 4 others - the correlation
    # graph itself is unchanged - but those 4 partners are now scattered to
    # essentially random locations on the grid instead of being physically
    # adjacent. A fixed-size square inner/outer cut through this randomized
    # layout severs a number of correlated pairs that scales with the inner
    # patch's *volume* (same mechanism as get_diffuse_volume_cov, despite
    # the underlying graph being sparse/local rather than dense/global) - a
    # real synthetic test case for "what does it look like when genuinely
    # local correlations are physically scattered", which
    # `mnist_shuffle_shared` below (see get_images) reproduces with real
    # image data. The permutation uses a *fixed* seed (123456789) so this
    # field's structure - and therefore its exact MI curve - is
    # reproducible across calls, not re-randomized every time.

    gen = np.random.RandomState(123456789)
    q = np.eye(length**2)
    for i in range(length):
        for j in range(length):
            q_row = i * length + j
            for (m, l) in [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]:
                if (m < length and m >= 0) and (l < length and l >= 0):
                    q_col = m * length + l
                    q[q_row, q_col] = rho
    shuffle = gen.permutation(length**2)
    q = q[shuffle, :]
    q = q[:, shuffle]
    cov = np.linalg.inv(q)
    return cov

def get_marginal_entropy(cov, remove = [], keep = []):

    # Computes the differential (Shannon) entropy of the marginal Gaussian
    # distribution over a subset of variables from a full covariance matrix
    # `cov` - either the variables in `keep`, or all variables *except*
    # those in `remove` (exactly one of the two should be given; if neither
    # is, the full `cov` is used as-is). For a multivariate Gaussian with
    # covariance C, the differential entropy has the closed form
    # S = (1/2) * log((2*pi*e)^n * det(C)) - computed here as
    # 0.5 * slogdet(2*pi*e * C) for numerical stability (slogdet avoids
    # over/underflow from computing det(C) directly for large C).
    #
    # `good_pixels` filters out any variable whose marginal variance
    # (diagonal entry) is effectively zero - this matters for the real-data
    # Gaussian fit (get_gaussian_fit): pixels that are constant across every
    # sample image (e.g. a fixed black border) have zero empirical variance,
    # which would make the sub-covariance matrix singular (and its log-det
    # -infinity) if left in; excluding them treats a deterministic pixel as
    # contributing zero entropy instead, which is the correct limit.

    remove = np.asarray(remove)
    keep = np.asarray(keep)
    cov = np.asarray(cov)
    if remove.size != 0:
        red_cov = np.delete(np.delete(cov, remove, axis = 0), remove, axis = 1)
    elif keep.size != 0:
        red_cov = cov[keep]
        red_cov = red_cov[:, keep]
    else:
        red_cov = cov
    good_pixels = np.nonzero(np.greater(np.diagonal(red_cov), 10**-6))[0]
    valid_cov = red_cov[good_pixels]
    valid_cov = valid_cov[:, good_pixels]
    entropy = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * valid_cov)[1]
    return entropy

def get_gaussian_mutual_information(cov, variable_indices_a):

    # Exact closed-form MI between a set of variables `variable_indices_a`
    # (e.g. the inner patch) and every other variable in the Gaussian
    # (the outer patch), via the standard identity
    # MI(A; B) = S(A) + S(B) - S(A, B) - the sum of the two marginal
    # entropies minus their joint entropy. This is the "ground truth" that
    # the neural estimator in src/mine.py is checked against for the
    # synthetic GMRF fields (see examples.ipynb's Gaussian-validation
    # section), since for a genuinely Gaussian distribution there's no need
    # to estimate MI from samples at all - it follows directly from `cov`.

    entropy_a = get_marginal_entropy(cov, keep = variable_indices_a)
    entropy_b = get_marginal_entropy(cov, remove = variable_indices_a)
    total_entropy = get_marginal_entropy(cov)
    mutual_information = entropy_a + entropy_b - total_entropy
    return mutual_information

def get_gaussian_fit(sample_images):

    # Fits the maximum-likelihood Gaussian (sample mean + sample covariance)
    # to a set of real (non-Gaussian) images, flattened to one long vector
    # per image first. This is only ever an approximation for real data -
    # see get_marginal_entropy's docstring and README.md's discussion of
    # why a Gaussian fit systematically underestimates the true MI of
    # non-Gaussian data (it can only capture pairwise/linear structure) -
    # but it's a useful, instantly-computable lower bound (via
    # get_gaussian_mutual_information) when no closed form for the true
    # distribution exists.

    flat_images = np.reshape(sample_images, [sample_images.shape[0], -1])
    mean = np.mean(flat_images, axis = 0)
    cov = np.cov(flat_images, rowvar = False)
    return (cov, mean)

def get_analytic_MI(cov, image_shape, max_length):

    # Computes the exact MI (get_gaussian_mutual_information) between a
    # centered square inner patch and its surrounding outer patch, for
    # every partition side length from 1 to max_length - 1 inclusive - i.e.
    # the same "MI vs. partition length" scaling curve that mine.py's neural
    # estimator produces from finite samples, but computed directly and
    # exactly from the known covariance matrix, with zero sampling error and
    # no training required.

    mi = []
    for length in range(1, max_length):
        (height, width) = image_shape
        inner_region = get_center_region(length, height, width)
        inner_indices = get_pixel_numbers(height, width, inner_region)
        known_mi = get_gaussian_mutual_information(cov, inner_indices)
        mi.append(known_mi)
    return mi

def get_gaussian_images(cov, mean, num_images):

    # Draws num_images independent samples from the N(mean, cov) Gaussian
    # distribution and reshapes each flat length**2-vector sample back into
    # a length x length square "image" (cov is assumed to be a perfect
    # square in size, i.e. built by one of the get_*_cov functions above or
    # get_gaussian_fit).

    length = int(cov.shape[0] ** 0.5)
    flat_images = np.random.multivariate_normal(mean = mean, cov = cov, size = [num_images], check_valid = 'raise')
    images = np.reshape(flat_images, [num_images, length, length])
    return images

def convert_to_grayscale(images):

    # This function collapses the color channel of an RGB image set into a
    # single grayscale channel using the standard weighted luminance coding.

    images = np.asarray(images)
    if images.ndim == 4 and images.shape[-1] == 3:
        (r, g, b) = (0.3, 0.59, 0.11)
        images = r * images[..., 0] + g * images[..., 1] + b * images[..., 2]
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    return images

def shuffle_pixels_independent(images, seed = None):

    # This function applies a different random pixel permutation to each
    # image independently. A given pixel *position* therefore no longer
    # corresponds to any consistent original location from one image to the
    # next, which destroys the correlation between the inner and outer
    # patches used elsewhere in this module (see get_center_region):
    # whatever ends up in the "inner" region of one image came from a
    # completely unrelated random subset of pixels than in any other image.

    images = np.asarray(images)
    (num_images, height, width) = images.shape
    rand = np.random.RandomState(seed)
    flat = images.reshape(num_images, height * width)
    shuffled = np.empty_like(flat)
    for i in range(num_images):
        shuffled[i] = flat[i, rand.permutation(height * width)]
    return shuffled.reshape(num_images, height, width)

def shuffle_pixels_shared(images, seed = 123456789):

    # This function applies the *same* random pixel permutation to every
    # image, analogous to how get_sparse_volume_cov permutes a
    # nearest-neighbor precision matrix once and reuses that one permutation
    # everywhere. Real spatial correlations between nearby original pixels
    # are preserved, since the same original pair of positions always maps
    # to the same (now scattered) pair of positions across every image, but
    # the "inner square patch" bipartition used elsewhere in this module now
    # cuts through a randomized, non-local subset of the original pixel
    # grid rather than a spatially contiguous one.

    images = np.asarray(images)
    (_, height, width) = images.shape
    rand = np.random.RandomState(seed)
    permutation = rand.permutation(height * width)
    num_images = images.shape[0]
    flat = images.reshape(num_images, height * width)
    shuffled = flat[:, permutation]
    return shuffled.reshape(num_images, height, width)

def center_crop(images, size):

    # This function extracts a centered square patch of the given side
    # length from every image in the set.

    (_, height, width) = images.shape
    top = height // 2 - size // 2
    left = width // 2 - size // 2
    return images[:, top:top + size, left:left + size]

def resize_stack(images, size):

    # This function resizes every (grayscale) image in the set to a
    # size x size square using Pillow. Input images are assumed to be
    # floating-point in the [0, 1] range and are returned in the same range.

    from PIL import Image
    resized = np.empty((images.shape[0], size, size), dtype = np.float64)
    for (i, image) in enumerate(images):
        pil_image = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))
        pil_image = pil_image.resize((size, size), Image.LANCZOS)
        resized[i] = np.asarray(pil_image, dtype = np.float64) / 255
    return resized

def conform_size(images, target_size, mode = "resize"):

    # This function forces every image to be target_size x target_size. When
    # mode is "crop" a centered crop is taken (used when the source images are
    # larger than the target and cropping preserves the pixel statistics), and
    # otherwise the images are resized.

    (_, height, width) = images.shape
    if height == target_size and width == target_size:
        return images
    if mode == "crop" and height >= target_size and width >= target_size:
        return center_crop(images, target_size)
    return resize_stack(images, target_size)

def _load_keras_dataset(name):

    # This helper loads one of the built-in keras image datasets and returns
    # the combined train/test images (labels are not needed for MI scaling).

    from tensorflow import keras
    loaders = {
        "mnist": keras.datasets.mnist,
        "fashion_mnist": keras.datasets.fashion_mnist,
        "cifar10": keras.datasets.cifar10,
    }
    ((train_images, _), (test_images, _)) = loaders[name].load_data()
    images = np.concatenate([train_images, test_images], axis = 0)
    return images

def _load_lfw_faces():

    # This helper loads the Labeled Faces in the Wild (LFW) dataset (over
    # 13,000 grayscale face images across roughly 5,700 people) via
    # scikit-learn. It downloads automatically from a stable, non-Kaggle
    # mirror with no manual steps.

    from sklearn.datasets import fetch_lfw_people
    data = fetch_lfw_people(min_faces_per_person = 1, resize = 1.0)
    return data.images

def _load_fer2013_hf():

    # This helper loads FER-2013 (48 x 48 grayscale facial-emotion images,
    # 7 classes) via the clip-benchmark/wds_fer2013 mirror on the Hugging
    # Face Hub. The original Kaggle source is gone and FER-2013 was removed
    # from the tensorflow-datasets catalog entirely (see the lfw_faces note
    # above), so this restores actual emotion labels rather than LFW's
    # identity labels. clip-benchmark/wds_fer2013 was chosen over several
    # other community mirrors because it stores plain 48x48 grayscale JPEGs
    # in the standard webdataset format - no custom loading script or
    # trust_remote_code needed - and its train (28,709) + test (7,178)
    # splits sum to exactly 35,887 images, matching the canonical FER-2013
    # dataset size.

    from datasets import load_dataset, concatenate_datasets
    dataset = load_dataset("clip-benchmark/wds_fer2013")
    combined = concatenate_datasets([dataset["train"], dataset["test"]])
    images = np.stack([np.asarray(example["jpg"], dtype = np.float64) / 255 for example in combined])
    return images

def get_images(source, num_images, strength = "small", target_size = DEFAULT_IMAGE_SIZE):

    # This function retrieves images from the specified dataset, as
    # well as the mean and covatiance from the Gaussian image sets.
    #
    # Real image datasets ("mnist", "fashion_mnist", "cifar10", "lfw_faces",
    # "fer2013_hf") are returned as grayscale floats in [0, 1], conformed to
    # target_size, with placeholder identity covariance / zero mean (they
    # are not Gaussian).

    if source == 'mnist':
        images = _load_keras_dataset("mnist")[:num_images] / 255
        images = conform_size(images, target_size, mode = "crop")
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'fashion_mnist':
        images = _load_keras_dataset("fashion_mnist")[:num_images] / 255
        images = conform_size(images, target_size, mode = "crop")
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'cifar10':
        images = _load_keras_dataset("cifar10")[:num_images] / 255
        images = convert_to_grayscale(images)
        # The colour images are 32 x 32, so a centered crop to 28 x 28 mirrors
        # the cropping used for the original 32 x 32 image dataset.
        images = conform_size(images, target_size, mode = "crop")
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'lfw_faces':
        images = _load_lfw_faces()[:num_images]
        # LFW images are already grayscale floats in [0, 1], so they are
        # resized down to the target from their native (125 x 94) size.
        images = conform_size(images, target_size, mode = "resize")
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'fer2013_hf':
        images = _load_fer2013_hf()[:num_images]
        # FER-2013 images are already grayscale floats in [0, 1] at their
        # native 48 x 48 size, so target_size = 48 (the "non-cropped" case)
        # is a no-op crop and returns them unchanged.
        images = conform_size(images, target_size, mode = "crop")
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'mnist_shuffle_independent':
        images = _load_keras_dataset("mnist")[:num_images] / 255
        images = conform_size(images, target_size, mode = "crop")
        # A different random pixel permutation per image destroys any
        # dataset-wide correlation between fixed pixel positions. This
        # shouldn't drive MI all the way to zero, though: a permutation
        # preserves each image's own pixel-value multiset, so a residual
        # "do these two patches share the same overall brightness" signal
        # survives even once positional structure is scrambled away.
        images = shuffle_pixels_independent(images)
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'mnist_shuffle_shared':
        images = _load_keras_dataset("mnist")[:num_images] / 255
        images = conform_size(images, target_size, mode = "crop")
        # The same random pixel permutation for every image preserves real
        # pixel-to-pixel correlations, just scattered to random positions
        # instead of local ones - a real-image analog of the "sparse"
        # (randomized boundary-law) GMRF, which is specifically built from
        # *nearest-neighbor* (very short-range) correlations before positions
        # get scattered. MNIST (rather than another real dataset) is used
        # here because its own *unshuffled* scaling curve peaks at the
        # image's midpoint and decays to ~0 well before the edge (see
        # README.md's MNIST scaling note) - a much shorter correlation range
        # than other real datasets show, even ones that share the same
        # (area-law) growth exponent, making MNIST the closer real-data
        # match to `sparse`'s genuinely short-range starting assumption.
        images = shuffle_pixels_shared(images)
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1] * images.shape[2])
    elif source == 'gauss_mnist':
        mnist_images = _load_keras_dataset("mnist") / 255
        mnist_images = conform_size(mnist_images, target_size, mode = "crop")
        (cov, mean) = get_gaussian_fit(mnist_images)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'area':
        rho = rho_values["area"][strength]
        length = 28
        cov = get_area_law_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'diffuse':
        rho = rho_values["diffuse"][strength]
        length = 28
        cov = get_diffuse_volume_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'sparse':
        rho = rho_values["sparse"][strength]
        length = 28
        cov = get_sparse_volume_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    else:
        raise ValueError("Image source not recognized.")
    return (images, cov, mean)

def plot_mi_scaling(results, lengths = None, labels = None, save_path = None, clip_negative = False):

    # This function plots one or more MI-vs-partition-length curves using
    # the same figure style as the paper's scaling plots, but operates
    # directly on MI values already held in
    # memory rather than loading pre-saved trial .npy files. This makes it
    # suitable for visualizing results computed live in a notebook, e.g. by
    # looping mine.run_bipartition over a range of partition lengths.
    #
    # `results` may be a dict mapping a legend label to a sequence of MI
    # values, or a plain list/array of such sequences (in which case
    # `labels` supplies the legend text). Each sequence is assumed to give
    # the MI estimate for partition lengths starting at 1, unless `lengths`
    # is provided explicitly.
    #
    # MI is analytically non-negative, so any negative values in an
    # estimated (as opposed to exact analytic) curve are finite-sample
    # noise rather than real signal. Set clip_negative = True to floor the
    # plotted curves at zero for empirical estimates.

    fontsize = 14
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 1, figsize = (10, 6))

    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    if isinstance(results, dict):
        (labels, series) = (list(results.keys()), list(results.values()))
    else:
        series = list(results)
        if labels is None:
            labels = ["Series {}".format(i + 1) for i in range(len(series))]

    for values in series:
        values = np.asarray(values)
        if clip_negative:
            values = np.clip(values, 0, None)
        plot_lengths = lengths if lengths is not None else np.arange(1, values.shape[0] + 1)
        axes.plot(plot_lengths, values, linewidth = 2)

    axes.set_xlabel('Partition Length (pixels)', fontsize = fontsize + 2)
    axes.set_ylabel('Mutual Information (nats)', fontsize = fontsize + 2)
    axes.legend(labels, fontsize = fontsize + 2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return axes
