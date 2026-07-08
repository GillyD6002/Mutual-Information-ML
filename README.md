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
| `fer2013_hf` | FER-2013 (facial *emotion* recognition, 7 classes) | 48x48 grayscale | 28x28 grayscale (centre crop) |
| `mnist_shuffle_independent` | MNIST with a *different* random pixel permutation per image | 28x28 grayscale | 28x28 grayscale |
| `mnist_shuffle_shared` | MNIST with the *same* random pixel permutation for every image | 28x28 grayscale | 28x28 grayscale |

Every real dataset is returned as grayscale floats in `[0, 1]` and conformed to a common 28x28 grid so the rest of the pipeline (partitioning, covariance reshaping, plotting) is unchanged. The target size is configurable via `get_images(..., target_size=N)`, and `mine.run_bipartition(..., target_size=N)` passes it straight through (defaulting to 28, so every existing caller is unaffected) - see "Non-cropped MI scaling" below for why this matters.

The Gaussian Markov random field sources (`area`, `diffuse`, `sparse`) and the Gaussian fit to MNIST (`gauss_mnist`) are unchanged.

### A note on LFW (Labeled Faces in the Wild)

MNIST, Fashion-MNIST, and CIFAR-10 ship with Keras and download on first use. `lfw_faces` is loaded through [`scikit-learn`](https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset) (`sklearn.datasets.fetch_lfw_people`), which downloads it automatically from a stable, non-Kaggle mirror with no account or manual steps. It's large enough (13,000+ images across ~5,700 people) that its MI estimates are comparable in scale to the other datasets, unlike the much smaller Olivetti Faces dataset (400 images) it replaced during development, which produced noisy, squashed-flat estimates at the same settings.

### A note on FER-2013 (`fer2013_hf`)

FER-2013 was removed from this project once (see the changelog below) because it had been pulled from the `tensorflow-datasets` catalog and its original Kaggle hosting is unreliable. It's back via a Hugging Face Hub mirror instead: [`clip-benchmark/wds_fer2013`](https://huggingface.co/datasets/clip-benchmark/wds_fer2013), loaded with the `datasets` library (`datasets.load_dataset("clip-benchmark/wds_fer2013")`). This mirror was chosen over several other community re-uploads found on the Hub because it stores plain 48x48 grayscale JPEGs in the standard webdataset format - no custom loading script or `trust_remote_code` needed - and its train (28,709) + test (7,178) splits sum to exactly 35,887 images, matching the canonical FER-2013 dataset size. Unlike `lfw_faces` (identity labels), this restores genuine facial *emotion* labels (7 classes), though the labels themselves aren't used here since only the images matter for MI estimation.

## Non-cropped MI scaling

The default 28x28 pipeline throws away real information for every real dataset that isn't already natively 28x28: CIFAR-10's 32x32 images are centre-cropped, and LFW's 125x94 and FER-2013's 48x48 images are resized down. `src/image_noncrop_experiment.py` re-runs the same MI-vs-partition-length sweep as `examples.ipynb`'s real-dataset section, but at each dataset's own native (or best achievable square) resolution instead of 28x28, and sweeps the *entire* image (length 1 up to and including `target_size`) rather than stopping at 27. At length = `target_size` the inner patch is the whole image and the outer patch is empty, so MI necessarily collapses back to ~0 there (there's nothing left for the classifier to discriminate) - this is an expected, meaningful boundary condition, not a bug, and it's what makes the resulting curves show a full rise-then-fall shape instead of an arbitrary truncation mid-rise.

| `image_type` | non-cropped `target_size` | why |
| --- | --- | --- |
| `cifar10` | 32 | native size - `conform_size` becomes a no-op, zero cropping |
| `lfw_faces` | 94 | the larger square achievable without upsampling either of its 125x94 dimensions |
| `fer2013_hf` | 48 | native size - `conform_size` becomes a no-op, zero resizing |

MNIST/Fashion-MNIST are excluded since they're already natively 28x28 - there's no cropping to undo for them. Results (raw `.npy` arrays and a combined scaling plot in the same `image.plot_mi_scaling` style as the rest of this project) are saved to `image_noncrop_results/`. Run it with:

```bash
python -m src.image_noncrop_experiment
```

It reuses `mine.run_bipartition`/`img.get_images` unmodified aside from the additive `target_size` argument described above - the training procedure itself is identical to the cropped 28x28 sweep, just on larger inputs, so it takes proportionally longer (still on the order of tens of minutes rather than hours on a CPU-only machine, per a timing check before committing to the full sweep - see the script's module docstring for the actual numbers).

Note that `lfw_faces` is a face-*recognition* (identity) dataset, not a facial-*emotion* one — this project originally used FER-2013 for that purpose, but FER-2013 has since been removed from the `tensorflow-datasets` catalog entirely and is no longer supported here.

### Area law vs. volume law, and what the full-length sweep revealed

Area/boundary law and volume law are defined by the *growth rate* of the rising portion of the curve: MI ~ L (linear, since a square patch's boundary/perimeter grows as ~4L in 2D) is a boundary/area law, while MI ~ L² (quadratic, tracking the patch's area) is a volume law. That's a claim about the early rise, before finite-size effects take over - and *every* curve here eventually turns over and declines back toward 0 as the inner patch approaches the full image (the outer patch runs out of pixels to correlate with), regardless of which law governs its growth. That decline is universal, not something that only happens to area-law curves - an easy trap to fall into when eyeballing where a curve peaks.

Fitting the actual exponent (log(MI) vs log(L), matching `language_experiment.py`'s existing power-law fit) over the early rise, at a consistent *relative* fraction of each dataset's own native size (this matters: L=10 is 36% of MNIST's 28px width but only 11% of LFW's 94px width, so comparing raw pixel-length windows across differently-sized datasets isn't a fair comparison) gives, robust across several fraction choices:

| Dataset | Exponent | Verdict |
| --- | --- | --- |
| `mnist` | 0.67-0.76 | area-law |
| `cifar10` | 0.70-0.95 | area-law |
| `fer2013_hf` | 1.02-1.25 | area-law (close to exactly linear) |
| `lfw_faces` | 2.43-2.62 | **volume-law** |

So it's not "CIFAR-10/LFW/FER-2013 are volume-law, MNIST is the exception" - it's LFW specifically that's volume-law, and MNIST/CIFAR-10/FER-2013 all land on area-law. This is a correction of an earlier version of this note, which classified datasets by *where their curve peaked relative to the image size* (`image_noncrop_results/non_cropped_mi_scaling.png`, `mnist_vs_others_mi_scaling.png`) rather than by fitting the growth exponent directly - MNIST's unusually early peak (L=14/28, the image's midpoint) is real and reflects its digits being small, fixed-support strokes on an otherwise blank background (once the inner patch captures the whole digit, there's nothing left in the outer ring to correlate with), but a late peak doesn't by itself imply quadratic growth, since a longer correlation length just delays saturation without changing the underlying (linear) law. LFW's very low, noisy MI at small absolute L (its 94px native size makes "small" patches a much smaller fraction of the image than for the other datasets) also means its exponent fit is the least robust of the four and could partly reflect the estimator struggling more at that much higher input dimensionality (8,836 pixels vs. 28²-48² for the others) on the same fixed 10,000-image training budget, rather than a purely intrinsic property - not fully ruled out without a larger run.

With this correction, our results are actually fairly consistent with the original paper (Sec. 6, read directly rather than just the abstract) rather than a departure from it: the paper found Tiny Images confidently boundary-law and called the MNIST evidence "less definitive" but leaning boundary-law too (a second, independent Gaussian-fit method found MNIST "obeys a clear boundary law"). Three of our four real datasets landing on area-law lines up with that, better than the peak-location-based classification this note previously reported.

### A note on `mnist_shuffle_independent` / `mnist_shuffle_shared`

Real image datasets almost never show the "sparse, randomized" scaling pattern that the synthetic `sparse` GMRF was built to test (nearest-neighbor correlations scattered to random positions by a one-time permutation - see `get_sparse_volume_cov`). `image.shuffle_pixels_independent`/`image.shuffle_pixels_shared` build a real-data analog by permuting real image pixels the same way:

- `mnist_shuffle_independent` permutes each image's pixels with its own independent random permutation, which should destroy most of the *positional* correlation between the inner and outer patches (a given pixel position no longer maps to a consistent original location across images). It doesn't drive the MI all the way to zero, though: a permutation preserves each image's own pixel-value multiset, so a residual "do these two patches share the same overall brightness" signal survives.
- `mnist_shuffle_shared` applies the same permutation to every image, directly mirroring the synthetic `sparse` field's construction. Real pixel-to-pixel correlations are preserved, just scattered to non-local positions - which in practice produces *higher* MI than unshuffled MNIST, since a fixed-size inner/outer square cut now severs a much larger fraction of the (now scattered) correlated pixel pairs than it would in the original, spatially-compact image.

MNIST is the dataset used here (rather than CIFAR-10, tried during development) because the `sparse` GMRF's construction specifically starts from *nearest-neighbor* (very short-range) correlations before positions get scattered - and although the previous section's exponent fit shows both MNIST and CIFAR-10 are area-law, MNIST's correlations are also far shorter-*range* in absolute terms: its curve peaks and its real digit content is exhausted by L=14/28 (the image's midpoint), versus CIFAR-10's L=25/32 (78% of the image) - CIFAR-10 still has an approximately linear growth law, but sustained over a much longer distance. That makes MNIST the closer real-data match to `sparse`'s genuinely nearest-neighbor starting assumption, even though both datasets share the same (area-law) growth exponent.

Swept across the full 1-28 range in `image_noncrop_results/mnist_shuffle_vs_sparse_scaling.png` (`image_results/mnist_shuffle_scaling.png`/`.pdf` is the same plot):

- `mnist_shuffle_shared` peaks at L=21/28, well past unshuffled MNIST's own early (L=14) peak, and rises to a notably higher magnitude (≈6.6 nats vs unshuffled MNIST's ≈5.0 nats) - consistent with shuffling generally inflating estimated MI (see above).
- `mnist_shuffle_independent` stays low throughout (peak ≈0.6 nats) with no clear structure - just the residual per-image brightness signal.

For a check against the synthetic field this is meant to mimic: the *exact* analytic `sparse` GMRF curve (`image.get_analytic_MI` - free, no training needed, saved separately as `image_noncrop_results/sparse_gmrf_large_analytic_mi.npy`) peaks at L=21/28 too - the same location as `mnist_shuffle_shared`, even though it isn't plotted alongside it above. That peak-location match is suggestive, but the growth-exponent fit (same method as the previous section) is the real, rigorous check: unshuffled MNIST fits at 0.66-0.68 (area-law), while `mnist_shuffle_shared` fits at 1.76-2.56, converging toward ~2 as the fit window widens - matching the exact `sparse` GMRF's own exponent of 1.97-2.07 almost exactly. Scattering MNIST's short-range correlations really does convert its growth law from area-law to volume-law, exactly as the `sparse` field's own construction predicts.

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

## Project layout

- `src/` - all source code (`image.py`, `mine.py` are the core MI-estimation pipeline; `sequence.py`/`language.py`/`language_experiment.py`/`language_embedding_experiment.py` are the 1D word-sequence analog; `image_noncrop_experiment.py` is the non-cropped real-dataset sweep).
- `examples.ipynb` - the main walkthrough notebook; start here.
- `image_results/`, `image_noncrop_results/`, `language_results/` - generated output (`.npy` arrays and plots) from the notebook and the standalone experiment scripts, one folder per experiment family.
- `alg.ini`, `mine.ini` - config files for the `python -m src.mine` CLI entry point.
- `CHANGELOG.md` - development history of this modernized fork; not needed to use the project, kept for context on *why* things are the way they are.

## What changed in this update

See `CHANGELOG.md` for the full development history of this modernized fork.
