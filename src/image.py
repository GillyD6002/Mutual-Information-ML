import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# This module is used to process image data for the MI estimation task. Note that
# much of the code here is non-functional without the associated data.
#
# TensorFlow / tensorflow-datasets are imported lazily inside the loading
# functions so that the Gaussian-field and plotting utilities in this module
# can be used without a TensorFlow installation.

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

    # This function returns a grid that has been numbered
    # row-by-row from left to right.

    indices = np.reshape(np.arange(image_height*image_width), [image_height, image_width])
    (top, bottom, left, right) = region
    region_indices = indices[top:bottom, left:right]
    return region_indices.flatten()

def get_center_region(length, img_height, img_width):

    # This function extracts the four corners of a patch
    # located in the middle of the img_height x img_wdith
    # # image. 

    top = img_height // 2 - length // 2
    left = img_width // 2 - length // 2
    region = (top, top + length, left, left + length)
    return region

def get_area_law_cov(length, rho):

    # This function constructs a covariance matrix
    # with every variable being correlated with its
    # four nearest neighbors.

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

    # This function constructs a covariance matrix
    # with every variable equally correlated to all
    # other variables.

    q = np.full([length**2, length**2], rho)
    for i in range(length**2):
        q[i, i] = 1
    cov = np.linalg.inv(q)
    return cov

def get_sparse_volume_cov(length, rho):

    # This function constructs a covariance matrix
    # with sparse correlations that obey a volume law
    # in their scaling.

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

    # This function computes the marginal entropy of a 
    # Gaussian Markov random field with respect to a specific
    # set of variables, selected by either keeping or removing
    # some of the variables.

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

    # This function computes the MI for a Gaussian distribution
    # with the given covariance matrix.

    entropy_a = get_marginal_entropy(cov, keep = variable_indices_a)
    entropy_b = get_marginal_entropy(cov, remove = variable_indices_a)
    total_entropy = get_marginal_entropy(cov)
    mutual_information = entropy_a + entropy_b - total_entropy
    return mutual_information

def get_gaussian_fit(sample_images):

    # This function fits a Gaussian to a given sample of 
    # images.

    flat_images = np.reshape(sample_images, [sample_images.shape[0], -1])
    mean = np.mean(flat_images, axis = 0)
    cov = np.cov(flat_images, rowvar = False)
    return (cov, mean)

def get_analytic_MI(cov, image_shape, max_length):

    # This function computes the exact mutual information
    # between an inner and outer patch of variable in a 
    # Gaussian Markov random field, with patch sizes ranging
    # from 1 to max_length.

    mi = []
    for length in range(1, max_length):
        (height, width) = image_shape
        inner_region = get_center_region(length, height, width)
        inner_indices = get_pixel_numbers(height, width, inner_region)
        known_mi = get_gaussian_mutual_information(cov, inner_indices)
        mi.append(known_mi)
    return mi
    
def get_gaussian_images(cov, mean, num_images):

    # This function samples images from a Gaussian distribution
    # with the specified mean and covariance.

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

def _load_fer2013(num_images, data_dir = None, csv_path = None):

    # This helper loads the FER-2013 facial-emotion-recognition images
    # (48 x 48 grayscale). It prefers tensorflow-datasets, and falls back to
    # reading the standard fer2013.csv file if tensorflow-datasets or its
    # download is unavailable. Set csv_path to point at a local fer2013.csv.

    try:
        import tensorflow_datasets as tfds
        splits = ["train", "test"]
        arrays = []
        for split in splits:
            data = tfds.load(
                "fer2013",
                split = split,
                data_dir = data_dir,
                as_supervised = True,
                batch_size = -1)
            (split_images, _) = tfds.as_numpy(data)
            arrays.append(split_images)
        images = np.concatenate(arrays, axis = 0)
        return images[:num_images]
    except Exception as tfds_error:
        if csv_path is None:
            raise RuntimeError(
                "Could not load FER-2013 through tensorflow-datasets "
                f"({tfds_error}). Provide csv_path=<path to fer2013.csv> "
                "to load it from the local CSV instead.")
        return _load_fer2013_csv(csv_path, num_images)

def _load_fer2013_csv(csv_path, num_images):

    # This helper parses the pixel strings in a standard fer2013.csv file into
    # a stack of 48 x 48 grayscale images.

    pixel_rows = []
    with open(csv_path, "r") as handle:
        header = handle.readline().strip().split(",")
        pixel_column = header.index("pixels")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(",")
            pixels = np.fromstring(fields[pixel_column].strip('"'), sep = " ")
            pixel_rows.append(pixels)
            if len(pixel_rows) >= num_images:
                break
    images = np.stack(pixel_rows).reshape([-1, 48, 48])
    return images

def get_images(source, num_images, strength = "small", target_size = DEFAULT_IMAGE_SIZE,
        fer_csv_path = None, fer_data_dir = None):

    # This function retrieves images from the specified dataset, as
    # well as the mean and covatiance from the Gaussian image sets.
    #
    # Real image datasets ("mnist", "fashion_mnist", "cifar10", "fer2013") are
    # returned as grayscale floats in [0, 1], conformed to target_size, with
    # placeholder identity covariance / zero mean (they are not Gaussian).

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
    elif source == 'fer2013':
        images = _load_fer2013(num_images, data_dir = fer_data_dir, csv_path = fer_csv_path) / 255
        # FER-2013 images are 48 x 48, so they are resized down to the target.
        images = conform_size(images, target_size, mode = "resize")
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

def plot_cov():

    # This function plots the covariance matricies from
    # the strongly-correlated Gaussian Markov random fields
    # with respect to the center pixel, which is marked in 
    # red.

    fontsize = 16
    (_, ax_list) = plt.subplots(1, 3, figsize = (13, 4))
    for (i, scaling_type) in enumerate(["area", "diffuse", "sparse"]):
        (_, cov, _) = get_images(scaling_type, 1, strength = "large")
        cov[405, 405] = 0
        corr = cov[405].reshape([28, 28])
        ax_list[i].imshow(corr)
        for tick in ax_list[i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax_list[i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
    ax_list[0].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    ax_list[1].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    ax_list[2].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    plt.tight_layout()
    plt.text(-0.15, 1.02, "a)", fontsize = fontsize + 6, transform = ax_list[0].transAxes)
    plt.text(-0.15, 1.02, "b)", fontsize = fontsize + 6, transform = ax_list[1].transAxes)
    plt.text(-0.15, 1.02, "c)", fontsize = fontsize + 6, transform = ax_list[2].transAxes)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.933, wspace=0.23, hspace=0)
    plt.savefig("scaling_covs.pdf")

def plot_gaussian_cov_image(source = "gauss_mnist"):

    # This function samples an image from a Gaussian distribution fit to a
    # real dataset and plots both the covariance with respect to the center
    # pixel (marked in red) and a sample image.

    (_, ax_list) = plt.subplots(1, 2, figsize = (6, 3))
    (images, cov, _) = get_images(source, 1)
    length = int(cov.shape[0] ** 0.5)
    center = (length // 2) * length + (length // 2)
    cov = cov.copy()
    cov[center, center] = 0
    corr = cov[center].reshape([length, length])
    ax_list[0].imshow(corr)
    ax_list[1].imshow(images[0], cmap = "gray")
    ax_list[0].set_title("Covariance")
    ax_list[1].set_title("Sample")
    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0, hspace = None)
    plt.savefig("image_cov.pdf")

def plot_sampled_images():

    # This function samples an image from each of the six
    # Gaussian Markov random fields and then plots it.

    (_, ax_list) = plt.subplots(2, 3)
    for (i, size) in enumerate(["small", "large"]):
        for (j, scaling_type) in enumerate(["area", "diffuse", "sparse"]):
            (image, _, _) = get_images(scaling_type, 1, size)
            ax_list[i][j].imshow(image[0], cmap = "gray")
    ax_list[0][0].set_title("Nearest-Neighbor")
    ax_list[0][1].set_title("Uniform")
    ax_list[0][2].set_title("Randomized")
    ax_list[0][0].set_ylabel("Weak", fontsize = 12)
    ax_list[1][0].set_ylabel("Strong", fontsize = 12)
    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0)
    plt.savefig("gauss_samples.pdf")
        
def plot_gaussian(sources = ("gauss_mnist",), labels = None):

    # This function computes analytic MI values for Gaussian distributions
    # fit to one or more real datasets.

    fontsize = 14
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 1, figsize = (10, 6))

    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    size = 27
    lengths = list(range(1, size))
    for source in sources:
        (_, cov, _) = get_images(source, 1)
        image_length = int(cov.size**(1/4))
        mi = get_analytic_MI(cov, [image_length, image_length], size + 1)
        axes.plot(lengths, mi[:-1])
    axes.set_xlabel('Partition Length (pixels)', fontsize = fontsize + 2)
    axes.set_ylabel('Mutual Information (nats)', fontsize = fontsize + 2)
    if labels is None:
        labels = ["Gaussian fit to {}".format(source) for source in sources]
    plt.legend(labels, fontsize = fontsize + 2)
    plt.tight_layout()
    plt.savefig("gaussian.pdf")

def plot_averages(result_specs = None):

    # This function plots averaged MI estimates for one or more real datasets.
    # Each entry in result_specs is a dict with keys:
    #   "path"       : path to the saved .npy array of trial results
    #   "label"      : legend label
    #   "num_trials" : number of trials stored in the array
    #   "length"     : number of partition lengths per trial (default 27)
    # The saved arrays are assumed to flatten to [num_trials * length * 2].

    if result_specs is None:
        result_specs = [{
            "path": "averages/logistic_dense_mnist_0_1.0_70000.npy",
            "label": "70,000 MNIST Images",
            "num_trials": 20,
            "length": 27}]

    fontsize = 14
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 1, figsize = (10, 6))

    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    labels = []
    for spec in result_specs:
        length = spec.get("length", 27)
        trials = np.load(spec["path"]).reshape([spec["num_trials"], length, 2])
        mean = np.mean(trials, axis = 0)
        std = np.std(trials, axis = 0)
        lengths = np.arange(1, length)
        axes.plot(lengths, mean[:length - 1, 1])
        axes.fill_between(lengths, (mean + std)[:length - 1, 1], (mean - std)[:length - 1, 1], alpha = 0.3)
        labels.append(spec["label"])
    axes.legend(labels, fontsize = fontsize + 2)
    axes.set_xlabel("Partition Length (pixels)", fontsize = fontsize + 2)
    axes.set_ylabel("Mutual Information (nats)", fontsize = fontsize + 2)
    plt.tight_layout()
    plt.savefig("dataset_averages.pdf")

def plot_mi_scaling(results, lengths = None, labels = None, save_path = None):

    # This function plots one or more MI-vs-partition-length curves using
    # the same figure style as the paper's scaling plots (plot_gaussian,
    # plot_averages), but operates directly on MI values already held in
    # memory rather than loading pre-saved trial .npy files. This makes it
    # suitable for visualizing results computed live in a notebook, e.g. by
    # looping mine.run_bipartition over a range of partition lengths.
    #
    # `results` may be a dict mapping a legend label to a sequence of MI
    # values, or a plain list/array of such sequences (in which case
    # `labels` supplies the legend text). Each sequence is assumed to give
    # the MI estimate for partition lengths starting at 1, unless `lengths`
    # is provided explicitly.

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
        plot_lengths = lengths if lengths is not None else np.arange(1, values.shape[0] + 1)
        axes.plot(plot_lengths, values, linewidth = 2)

    axes.set_xlabel('Partition Length (pixels)', fontsize = fontsize + 2)
    axes.set_ylabel('Mutual Information (nats)', fontsize = fontsize + 2)
    axes.legend(labels, fontsize = fontsize + 2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return axes

def plot_large_small_avg(scaling_type):

    # This function creates plots of the MI for the 
    # Gaussian Markov random fields, using the analytic
    # value and the three estimates with different sample
    # sizes.

    if scaling_type not in ["area", "diffuse", "sparse"]:
        raise ValueError("Scaling type '{}' not recognized.".format(scaling_type))
    size = 27

    # Compute the average MI values and the standard deviation
    # for the small correlaton value.

    small_corr = rho_values[scaling_type]["small"]
    small_0 = np.load('trials/logistic_dense_{}_{}_1.0_70000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:20]
    small_1 = np.load('trials/logistic_dense_{}_{}_1.0_700000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:10]
    small_2 =  np.load('trials/logistic_dense_{}_{}_1.0_7000000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:5]
    small_mean_0 = np.mean(small_0, axis = 0)
    small_mean_1 = np.mean(small_1, axis = 0)
    small_mean_2 = np.mean(small_2, axis = 0)
    small_std_0 = np.std(small_0, axis = 0)
    small_std_1 = np.std(small_1, axis = 0)
    small_std_2 = np.std(small_2, axis = 0)
    (_, small_cov, _) = get_images(scaling_type, 1, strength = "small")
    small_length = int(small_cov.size**(1/4))
    small_exact = get_analytic_MI(small_cov, [small_length, small_length], size + 1)
    
    # Compute the average MI values and the standard deviation
    # for the large correlaton value.

    large_corr = rho_values[scaling_type]["large"]
    large_0 = np.load('trials/logistic_dense_{}_{}_1.0_70000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:20]
    large_1 = np.load('trials/logistic_dense_{}_{}_1.0_700000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:10]
    large_2 =  np.load('trials/logistic_dense_{}_{}_1.0_7000000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:5]
    large_mean_0 = np.mean(large_0, axis = 0)
    large_mean_1 = np.mean(large_1, axis = 0)
    large_mean_2 = np.mean(large_2, axis = 0)
    large_std_0 = np.std(large_0, axis = 0)
    large_std_1 = np.std(large_1, axis = 0)
    large_std_2 = np.std(large_2, axis = 0)
    (_, large_cov, _) = get_images(scaling_type, 1, strength = "large")
    large_length = int(large_cov.size**(1/4))
    large_exact = get_analytic_MI(large_cov, [large_length, large_length], size + 1)

    # Plot the mean values and standard deviation on a pair
    # of axes, and then save the result.

    lengths = list(range(1, size))
    fontsize = 14
    linewidth = 2
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 2, figsize = (12, 6))

    for tick in axes[0].xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)
    for tick in axes[0].yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)

    for tick in axes[1].xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)
    for tick in axes[1].yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)

    handles = [mlines.Line2D([], [], color = "C0"), mlines.Line2D([], [], color = "C1"), 
        mlines.Line2D([], [], color = "C2"), mlines.Line2D([], [], color = "C3")]
    axes[0].legend(handles, [r'$7\times10^4$ samples', r'$7\times10^5$ samples', r"$7\times10^6$ samples", 'Exact'],
    fontsize = fontsize + 2).set_zorder(-1)

    axes[0].plot(lengths, small_mean_0[:26, 0], "C0", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_0 + small_std_0)[:26, 0], (small_mean_0 - small_std_0)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_mean_1[:26, 0], "C1", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_1 + small_std_1)[:26, 0], (small_mean_1 - small_std_1)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_mean_2[:26, 0], "C2", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_2 + small_std_2)[:26, 0], (small_mean_2 - small_std_2)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_exact[:26], "C3", linewidth = linewidth)
    axes[0].set_xlabel('Partition Length (L)', fontsize = fontsize + 3)
    axes[0].set_ylabel('Mutual Information (nats)', fontsize = fontsize + 3)

    axes[1].plot(lengths, large_mean_0[:26, 0], "C0", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_0 + large_std_0)[:26, 0], (large_mean_0 - large_std_0)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_mean_1[:26, 0], "C1", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_1 + large_std_1)[:26, 0], (large_mean_1 - large_std_1)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_mean_2[:26, 0], "C2", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_2 + large_std_2)[:26, 0], (large_mean_2 - large_std_2)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_exact[:26], "C3", linewidth = linewidth)
    axes[1].set_xlabel('Partition Length (L)', fontsize = fontsize + 3)

    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.2, hspace = None)
    plt.savefig(f"{scaling_type}.pdf")
