import numpy as np

# A training-free alternative to mine.py's neural-classifier MI estimator, used where
# that estimator's per-measurement training cost doesn't scale (e.g. sweeping VGG16's 13
# conv layers x many window positions x many pruning ratios). Implements the matrix-based
# Renyi's alpha-order entropy functional (Giraldo, Sanchez-Giraldo & Principe, "Measures
# of entropy from data using infinitely divisible kernels", 2014; see also Yu, Sanchez-
# Giraldo, Jenssen & Principe's multivariate extension): entropy of a variable is read off
# the eigenvalues of a normalized Gaussian-kernel Gram matrix built from N samples of that
# variable, and mutual information between two variables X and Y is the sum of their
# individual entropies minus their joint entropy (itself the entropy of the normalized
# Hadamard product of X's and Y's Gram matrices). Cost scales with N (sample count), not
# with the dimensionality of X/Y - the opposite scaling of a KNN/KSG estimator or of
# training a classifier on flattened high-dimensional inputs - which is what makes it
# tractable on VGG's deep, high-channel-count layers.

DEFAULT_ALPHA = 1.01  # Near the alpha -> 1 limit, where this functional reduces to (an
# analog of) Shannon entropy - the standard default in the matrix-based-entropy
# literature, chosen so results are comparable to the Shannon-based estimates elsewhere
# in this project without actually hitting the alpha=1 singularity in (1-alpha).


def _flatten_samples(x):

    # Collapses every non-sample axis into one feature dimension, so a caller can pass
    # activation slices of any shape (H, W, C) per sample without flattening by hand.
    # x is (N, ...) -> returns (N, D).

    return np.reshape(x, (x.shape[0], -1)).astype(np.float64)


def pairwise_sq_dists(x):

    # Squared Euclidean distance between every pair of the N rows of x (N, D), computed
    # as ||xi||^2 + ||xj||^2 - 2 xi.xj - a single (N, D) @ (D, N) matmul plus a norm
    # vector, rather than an explicit N x N x D loop. Clipped at 0 to erase floating-point
    # negative noise on (near-)duplicate rows.

    sq_norms = np.sum(x * x, axis=1)
    dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    return np.clip(dists, 0.0, None)


def median_heuristic_sigma(x):

    # The "median heuristic" bandwidth for a Gaussian/RBF kernel: sigma is set so that
    # 2*sigma^2 equals the median of the sample's own pairwise squared distances (i.e.
    # the kernel evaluates to exp(-1) at the median distance). Standard practice for
    # RBF-kernel methods (kernel MMD/HSIC, and the matrix-based Renyi-entropy literature)
    # precisely because it derives directly from the data's own distance distribution
    # rather than assuming a regime - unlike a multivariate Silverman/KDE-style rule
    # (h ~ (4/(D+2))^(1/(D+4)) N^(-1/(D+4)) * std), which is built for density estimation
    # and collapses the kernel to near-identity here: that rule's implied bandwidth
    # saturates near a constant as D grows while pairwise distances keep growing like
    # sqrt(D), so sigma/distance -> 0 and every off-diagonal Gram entry vanishes -
    # verified empirically (an earlier version of this function used that rule and
    # produced a constant MI of log2(N) regardless of actual dependence, exactly the
    # degenerate near-identity-Gram-matrix failure mode). The median heuristic instead
    # scales with however large the data's actual pairwise distances are, at any D.

    x = _flatten_samples(x)
    sq_dists = pairwise_sq_dists(x)
    num_samples = sq_dists.shape[0]
    off_diagonal = sq_dists[~np.eye(num_samples, dtype=bool)]
    median_sq_dist = np.median(off_diagonal)
    if median_sq_dist == 0:
        # Every sample identical (e.g. a constant window) - no spread to set a bandwidth
        # from; fall back to a tiny constant rather than dividing by zero.
        return 1e-6
    return max(np.sqrt(median_sq_dist / 2.0), 1e-6)


def gaussian_gram_matrix(x, sigma=None):

    # Builds the (N, N) RBF Gram matrix K_ij = exp(-||xi - xj||^2 / (2 sigma^2)) for the
    # N samples in x (any per-sample shape, flattened internally). Diagonal is always 1
    # (each point compared to itself), matching a valid (symmetric, PSD) kernel matrix.

    x = _flatten_samples(x)
    if sigma is None:
        sigma = median_heuristic_sigma(x)
    sq_dists = pairwise_sq_dists(x)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def normalized_gram(k):

    # Divides by the trace so the matrix's eigenvalues sum to 1 (a valid density-like
    # spectrum for the entropy functional below) - the same normalization Giraldo et al.
    # apply before computing entropy from a Gram matrix's eigenvalues.

    trace = np.trace(k)
    return k / trace


def renyi_entropy(a, alpha=DEFAULT_ALPHA):

    # Matrix-based Renyi's alpha-order entropy of a normalized Gram matrix `a`:
    # S_alpha(A) = log2(sum_i lambda_i(A)^alpha) / (1 - alpha), where lambda_i(A) are A's
    # eigenvalues. `a` is symmetric by construction (a Gram matrix or a Hadamard product
    # of two Gram matrices), so eigvalsh (which assumes/exploits symmetry) is used rather
    # than the general eigvals. Tiny negative eigenvalues from floating-point error are
    # clipped to 0 before raising to a fractional power.

    eigvals = np.linalg.eigvalsh(a)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals[eigvals > 0]  # 0^alpha is 0 anyway; drop them to avoid 0**negative
    # blowing up if alpha < 0 (not used here, but keeps this correct in general).
    return np.log2(np.sum(eigvals ** alpha)) / (1.0 - alpha)


def renyi_joint_entropy(a_x, a_y, alpha=DEFAULT_ALPHA):

    # Matrix-based joint entropy of two variables, from the normalized Hadamard
    # (elementwise) product of their individual normalized Gram matrices - the
    # multivariate extension's definition of a joint density matrix for two variables
    # observed on the same N samples (a_x and a_y must therefore be the same shape, one
    # row/column per sample).

    joint = normalized_gram(a_x * a_y)
    return renyi_entropy(joint, alpha)


def renyi_mutual_information(x, y, alpha=DEFAULT_ALPHA, sigma_x=None, sigma_y=None):

    # Matrix-based Renyi's alpha-order mutual information between two variables observed
    # on the same N samples: I_alpha(X; Y) = S_alpha(X) + S_alpha(Y) - S_alpha(X, Y).
    # x and y are (N, ...) arrays of paired samples (e.g. x = an activation map's inner
    # window, y = the same images' outer/rest-of-map region) - unlike mine.py's
    # run_bipartition, no "marginal" (spliced/shuffled) samples are needed: this
    # estimator reads MI directly off the real paired data's Gram matrices, with no
    # training step and no explicit density estimate.

    a_x = normalized_gram(gaussian_gram_matrix(x, sigma_x))
    a_y = normalized_gram(gaussian_gram_matrix(y, sigma_y))
    entropy_x = renyi_entropy(a_x, alpha)
    entropy_y = renyi_entropy(a_y, alpha)
    joint_entropy = renyi_joint_entropy(a_x, a_y, alpha)
    return entropy_x + entropy_y - joint_entropy
