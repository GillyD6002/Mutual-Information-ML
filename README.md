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

The code has been updated for modern Python (tested against Python 3.11-3.12) and TensorFlow 2.x / Keras 3. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies with minimum compatible versions; run `pip-compile requirements.in` (from `pip-tools`) to produce a fully pinned lockfile. Key requirements:

- `tensorflow>=2.17` (needed for NumPy 2.x and Python 3.12 support)
- `tensorflow-datasets` (provides FER-2013)
- `numpy>=2.0`, `matplotlib`, `pillow`, `notebook`

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
