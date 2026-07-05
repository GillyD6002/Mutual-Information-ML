# Quantifying Mutual Information via Logistic Regression

Implements the mutual information estimation algorithm and scaling analysis carried out in [Mutual Information Scaling for Tensor Network Machine Learning (2022)](https://arxiv.org/abs/2103.00105). The abstract of the paper is reproduced here:

> Tensor networks have emerged as promising tools for machine learning, inspired by their widespread use as variational ansatze in quantum many-body physics. It is well known that the success of a given tensor network ansatz depends in part on how well it can reproduce the underlying entanglement structure of the target state, with different network designs favoring different scaling patterns. We demonstrate here how a related correlation analysis can be applied to tensor network machine learning, and explore whether classical data possess correlation scaling patterns similar to those found in quantum states which might indicate the best network to use for a given dataset. We utilize mutual information as a measure of correlations in classical data, and show that it can serve as a lower-bound on the entanglement needed for a probabilistic tensor network classifier. We then develop a logistic regression algorithm to estimate the mutual information between bipartitions of data features, and verify its accuracy on a set of Gaussian distributions designed to mimic different correlation patterns. Using this algorithm, we characterize the scaling patterns in the MNIST and Tiny Images datasets, and find clear evidence of boundary-law scaling in the latter. This quantum-inspired classical analysis offers insight into the design of tensor networks which are best suited for specific learning tasks.

In the Jupyter notebook `examples.ipynb` a summary of the work is presented, along with a portion of the experiments used to generate the paper's numerical results. The full source code is given in the `src` folder, although it is missing some of the large data files that are needed for all portions of the code to run.

A non-interactive write-up of the project can be found on [my website](https://ianconvy.github.io/projects/phd/mi-scaling/mi-scaling.html).

## Datasets

This version replaces the original **Tiny Images** dataset (the `tiny` and `gauss_tiny` sources) with three datasets that download automatically at runtime:

| `image_type` | Dataset | Native format | Conformed to |
| --- | --- | --- | --- |
| `mnist` | MNIST digits | 28x28 grayscale | 28x28 grayscale |
| `fashion_mnist` | Fashion-MNIST | 28x28 grayscale | 28x28 grayscale |
| `cifar10` | CIFAR-10 | 32x32 RGB | 28x28 grayscale (luminance + centre crop) |
| `lfw_faces` | LFW - Labeled Faces in the Wild (facial recognition) | 125x94 grayscale | 28x28 grayscale (resized) |
| `cifar10_shuffle_independent` | CIFAR-10 with a *different* random pixel permutation per image | 28x28 grayscale | 28x28 grayscale |
| `cifar10_shuffle_shared` | CIFAR-10 with the *same* random pixel permutation for every image | 28x28 grayscale | 28x28 grayscale |

Every real dataset is returned as grayscale floats in `[0, 1]` and conformed to a common 28x28 grid so the rest of the pipeline (partitioning, covariance reshaping, plotting) is unchanged. The target size is configurable via `get_images(..., target_size=N)`.

The Gaussian Markov random field sources (`area`, `diffuse`, `sparse`) and the Gaussian fit to MNIST (`gauss_mnist`) are unchanged.

### A note on LFW (Labeled Faces in the Wild)

MNIST, Fashion-MNIST, and CIFAR-10 ship with Keras and download on first use. `lfw_faces` is loaded through [`scikit-learn`](https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset) (`sklearn.datasets.fetch_lfw_people`), which downloads it automatically from a stable, non-Kaggle mirror with no account or manual steps. It's large enough (13,000+ images across ~5,700 people) that its MI estimates are comparable in scale to the other datasets, unlike the much smaller Olivetti Faces dataset (400 images) it replaced during development, which produced noisy, squashed-flat estimates at the same settings.

Note that `lfw_faces` is a face-*recognition* (identity) dataset, not a facial-*emotion* one — this project originally used FER-2013 for that purpose, but FER-2013 has since been removed from the `tensorflow-datasets` catalog entirely and is no longer supported here.

### A note on `cifar10_shuffle_independent` / `cifar10_shuffle_shared`

Real image datasets almost never show the "sparse, randomized" scaling pattern that the synthetic `sparse` GMRF was built to test (nearest-neighbor correlations scattered to random positions by a one-time permutation - see `get_sparse_volume_cov`). These two sources build a real-data analog directly from CIFAR-10 using `image.shuffle_pixels_independent` and `image.shuffle_pixels_shared`:

- `cifar10_shuffle_independent` permutes each image's pixels with its own independent random permutation, which should destroy most of the *positional* correlation between the inner and outer patches (a given pixel position no longer maps to a consistent original location across images). It doesn't drive the MI all the way to zero, though: a permutation preserves each image's own pixel-value multiset, so a residual "do these two patches share the same overall brightness/contrast" signal survives.
- `cifar10_shuffle_shared` applies the same permutation to every image, directly mirroring the synthetic `sparse` field's construction. Real pixel-to-pixel correlations are preserved, just scattered to non-local positions - which in practice produces *higher* MI than unshuffled CIFAR-10, since a fixed-size inner/outer square cut now severs a much larger fraction of the (now scattered) correlated pixel pairs than it would in the original, spatially-compact image.

## Requirements

The code has been updated for modern Python (tested against Python 3.11-3.13) and TensorFlow 2.x / Keras 3. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies with minimum compatible versions; run `pip-compile requirements.in` (from `pip-tools`) to produce a fully pinned lockfile. Key requirements:

- `tensorflow>=2.17` (needed for NumPy 2.x and Python 3.12 support)
- `scikit-learn` (provides LFW)
- `numpy>=2.0`, `matplotlib`, `pillow`, `notebook`

### A note on Python 3.14

TensorFlow does not yet publish a Windows build for Python 3.14 (not even in the `tf-nightly` prereleases, which only ship Linux/macOS wheels for cp314 as of this writing). Since TensorFlow is a hard dependency, use **Python 3.13** on Windows for now — it's the newest interpreter TensorFlow actually supports. Re-check TensorFlow's PyPI release page periodically; once a Windows cp314 wheel ships, this project should work unmodified under 3.14.

## Running the experiments

Set the parameters in `alg.ini` (algorithm, `image_type`, number of images, etc.) and `mine.ini` (model hyperparameters), then run the estimator as a module from the repository root:

```bash
python -m src.mine
```

## What changed in this update

- Replaced the Tiny Images dataset with CIFAR-10, Fashion-MNIST, and LFW (facial recognition) in `src/image.py`.
- Removed the TensorFlow 1.x calls that no longer exist in TensorFlow 2.x (`tf.log` -> `tf.math.log`, removed `tf.reset_default_graph()`).
- Updated model construction and optimizers for Keras 3 (explicit `Input` layer, `learning_rate=` keyword).
- Made the TensorFlow import lazy so the Gaussian-field and plotting utilities work without TensorFlow installed.
- Refreshed `requirements.in`/`requirements.txt` for modern Python.
- Added a correctly spelled `LogisticRegression` alias (the original `LogsiticRegression` name still works).
- Fixed `src/mine.py` for the current Keras 3 data-adapter API, which is stricter about generator inputs than the version this project was originally written against:
  - `get_finite_dataset` now yields `(inputs, targets)` as tuples instead of lists — Keras's generator adapter now infers a `tf.TypeSpec` per input and rejects plain lists.
  - `train_steps`/`val_steps` are now cast to `int` (`np.ceil` returns a `numpy.float64`, which the newer epoch iterator no longer accepts in `range()`).
  - Added `mine.cycle_generator`, replacing `itertools.cycle(...)` for repeating the validation generator — Keras's adapter now requires an actual generator object and rejects `itertools.cycle` instances. `examples.ipynb` was updated to match.
- `examples.ipynb`'s "Visualizing the scaling for a real dataset" section now sweeps **all** available real datasets (MNIST, Fashion-MNIST, CIFAR-10, LFW) instead of only MNIST, plotting them together with `image.plot_mi_scaling`.
- That section now plots the *direct* MI estimate rather than the *indirect* (Donsker-Varadhan-style) one, matching how the original paper's own real-dataset figures (`plot_averages`) were generated. The indirect estimate's `log(mean(exp(...)))` term is much more sensitive to noisy batches and can dip below zero even though MI is analytically non-negative; the direct estimate (a plain mean of classifier logits) is far more stable. `image.plot_mi_scaling` also gained a `clip_negative` option (used here) to floor any residual noise-driven negative values at zero.
- `mine.run_bipartition` gained an optional `eval_steps` argument (default 5000, unchanged) so the number of validation batches averaged over in `evaluate_MI` can be reduced to match a smaller `num_images`, instead of always evaluating over 5000 batches regardless of dataset size.
- The notebook's real-dataset sweep now uses reduced settings (10,000 images, up to 30 epochs, patience 8, 400 eval steps) instead of the paper's full scale (70,000 images, up to 3,000 epochs, see `mine.ini`), so the 27-length sweep across multiple datasets finishes in minutes rather than hours. Raise these values for higher-fidelity curves closer to the published figures.
- Added `lfw_faces` (via `sklearn.datasets.fetch_lfw_people`) as a working facial dataset in `src/image.py`, and included it in the notebook's real-dataset sweep. Added `scikit-learn` to `requirements.in`/`requirements.txt`. This replaced an earlier attempt using the much smaller Olivetti Faces dataset (400 images), whose MI estimates were noisy and squashed flat on the shared-axis comparison plot; LFW's 13,000+ images give comparable-magnitude, much less noisy estimates. The notebook still also plots it alone on its own axis (`lfw_faces_scaling.pdf`) to check.
- Removed FER-2013 support entirely (`_load_fer2013`/`_load_fer2013_csv` in `src/image.py`, the `fer_csv_path`/`fer_data_dir` arguments to `get_images`, and the `tensorflow-datasets` dependency): FER-2013 has been removed from the `tensorflow-datasets` catalog and can no longer be loaded automatically, and LFW now covers the facial-dataset role instead.
- `examples.ipynb`'s Gaussian Markov random field section gained two new live-computed cells after the static reference figures: (1) the *exact* analytic MI curves for all three field types at both correlation strengths, computed instantly from the closed-form Gaussian formula (`image.get_analytic_MI`) with no training; and (2) a validation plot overlaying the neural estimator's *direct* MI estimate (swept across all 27 partition lengths, same reduced settings as the real-dataset sweep) against that exact curve for the `diffuse`/`large` field, to demonstrate the estimator actually recovers the known value.
- Fixed a notebook-format bug where several cells' `source` field had been written as a single raw string instead of a list of lines (both are valid per the nbformat spec, but the string form rendered as blank/invisible cell input in VS Code's notebook editor even though the cells still executed correctly).
- Added `image.shuffle_pixels_independent` and `image.shuffle_pixels_shared`, plus two new `get_images` sources built on them (`cifar10_shuffle_independent`, `cifar10_shuffle_shared`), as a real-data analog of the synthetic `sparse` (randomized boundary-law) GMRF test. `examples.ipynb` gained a new section sweeping both across all 27 partition lengths and plotting them against unshuffled CIFAR-10 (`cifar10_shuffle_scaling.pdf`).
