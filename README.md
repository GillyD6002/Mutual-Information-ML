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
| `fer2013` | FER-2013 (facial emotion recognition) | 48x48 grayscale | 28x28 grayscale (resized) |

Every real dataset is returned as grayscale floats in `[0, 1]` and conformed to a common 28x28 grid so the rest of the pipeline (partitioning, covariance reshaping, plotting) is unchanged. The target size is configurable via `get_images(..., target_size=N)`.

The Gaussian Markov random field sources (`area`, `diffuse`, `sparse`) and the Gaussian fit to MNIST (`gauss_mnist`) are unchanged.

### A note on FER-2013

MNIST, Fashion-MNIST, and CIFAR-10 ship with Keras and download on first use. FER-2013 is **not** bundled with Keras, so it is loaded through [`tensorflow-datasets`](https://www.tensorflow.org/datasets), which downloads it on first use. If the automatic download is unavailable (it has historically been hosted behind Kaggle auth), pass a path to the standard `fer2013.csv` instead:

```python
from src import image as img
images, _, _ = img.get_images("fer2013", 70000, fer_csv_path="path/to/fer2013.csv")
```

## Requirements

The code has been updated for modern Python (tested against Python 3.11-3.13) and TensorFlow 2.x / Keras 3. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies with minimum compatible versions; run `pip-compile requirements.in` (from `pip-tools`) to produce a fully pinned lockfile. Key requirements:

- `tensorflow>=2.17` (needed for NumPy 2.x and Python 3.12 support)
- `tensorflow-datasets` (provides FER-2013)
- `numpy>=2.0`, `matplotlib`, `pillow`, `notebook`

### A note on Python 3.14

TensorFlow does not yet publish a Windows build for Python 3.14 (not even in the `tf-nightly` prereleases, which only ship Linux/macOS wheels for cp314 as of this writing). Since TensorFlow is a hard dependency, use **Python 3.13** on Windows for now — it's the newest interpreter TensorFlow actually supports. Re-check TensorFlow's PyPI release page periodically; once a Windows cp314 wheel ships, this project should work unmodified under 3.14.

## Running the experiments

Set the parameters in `alg.ini` (algorithm, `image_type`, number of images, etc.) and `mine.ini` (model hyperparameters), then run the estimator as a module from the repository root:

```bash
python -m src.mine
```

## What changed in this update

- Replaced the Tiny Images dataset with CIFAR-10, Fashion-MNIST, and FER-2013 in `src/image.py`.
- Removed the TensorFlow 1.x calls that no longer exist in TensorFlow 2.x (`tf.log` -> `tf.math.log`, removed `tf.reset_default_graph()`).
- Updated model construction and optimizers for Keras 3 (explicit `Input` layer, `learning_rate=` keyword).
- Made the TensorFlow import lazy so the Gaussian-field and plotting utilities work without TensorFlow installed.
- Refreshed `requirements.in`/`requirements.txt` for modern Python.
- Added a correctly spelled `LogisticRegression` alias (the original `LogsiticRegression` name still works).
- Fixed `src/mine.py` for the current Keras 3 data-adapter API, which is stricter about generator inputs than the version this project was originally written against:
  - `get_finite_dataset` now yields `(inputs, targets)` as tuples instead of lists — Keras's generator adapter now infers a `tf.TypeSpec` per input and rejects plain lists.
  - `train_steps`/`val_steps` are now cast to `int` (`np.ceil` returns a `numpy.float64`, which the newer epoch iterator no longer accepts in `range()`).
  - Added `mine.cycle_generator`, replacing `itertools.cycle(...)` for repeating the validation generator — Keras's adapter now requires an actual generator object and rejects `itertools.cycle` instances. `examples.ipynb` was updated to match.
- `examples.ipynb`'s "Visualizing the scaling for a real dataset" section now sweeps **all** available real datasets (MNIST, Fashion-MNIST, CIFAR-10, FER-2013) instead of only MNIST, plotting them together with `image.plot_mi_scaling`. FER-2013 is skipped automatically (with a printed note) if it can't be loaded.
- That section now plots the *direct* MI estimate rather than the *indirect* (Donsker-Varadhan-style) one, matching how the original paper's own real-dataset figures (`plot_averages`) were generated. The indirect estimate's `log(mean(exp(...)))` term is much more sensitive to noisy batches and can dip below zero even though MI is analytically non-negative; the direct estimate (a plain mean of classifier logits) is far more stable. `image.plot_mi_scaling` also gained a `clip_negative` option (used here) to floor any residual noise-driven negative values at zero.
- `mine.run_bipartition` gained an optional `eval_steps` argument (default 5000, unchanged) so the number of validation batches averaged over in `evaluate_MI` can be reduced to match a smaller `num_images`, instead of always evaluating over 5000 batches regardless of dataset size.
- The notebook's real-dataset sweep now uses reduced settings (10,000 images, up to 30 epochs, patience 8, 400 eval steps) instead of the paper's full scale (70,000 images, up to 3,000 epochs, see `mine.ini`), so the 27-length sweep across multiple datasets finishes in minutes rather than hours. Raise these values for higher-fidelity curves closer to the published figures.
