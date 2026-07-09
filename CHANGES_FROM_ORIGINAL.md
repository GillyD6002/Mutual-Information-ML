# Changes from the original repository

This document explains, precisely and completely, how this repository differs from the original it was forked from: **[IanConvy/mutual-information-scaling](https://github.com/IanConvy/mutual-information-scaling)** (the code accompanying [Convy et al., "Mutual Information Scaling for Tensor Network Machine Learning" (2022)](https://arxiv.org/abs/2103.00105)).

It was written by diffing this repository against a fresh clone of the original, file by file — not from memory — so it can be used as an authoritative reference for explaining exactly what changed and why. `CHANGELOG.md` in this repo is a chronological list of edits as they happened; this document is organized by *topic* instead, for explaining the fork to someone else.

## 1. Why this fork exists

The original repository was written in 2021-2022 against TensorFlow 2.6.5 and NumPy 1.19.5, pinned exactly, on Python 3.9. It also depended on two things this copy doesn't have:

- **The Tiny Images dataset** (`tiny`/`gauss_tiny` sources) — this 80-million-image dataset was permanently taken offline by its creators (MIT/NYU) in 2020 due to documented ethical concerns about offensive/biased content, and is no longer obtainable.
- **Large pre-computed trial files** (`averages/*.npy`, `trials/*.npy`) — the arrays behind the paper's published figures (hundreds of training runs, averaged). The original README itself says "it is missing some of the large data files that are needed for all portions of the code to run."

Getting the notebook to run at all on a current machine required upgrading the TensorFlow/Keras API calls, which surfaced several bugs in the original code that had been silently tolerated by the old API. From there, the project grew into a broader modernization: replacing Tiny Images with several currently-available datasets, replacing the missing pre-computed figures with everything reproduced live, and (in this session) a substantial new experiment (1D word-sequence MI scaling, an analog of the paper's 2D image analysis applied to text) that doesn't exist in the original at all.

## 2. Environment and dependencies

| | Original | This fork |
|---|---|---|
| Python | 3.9 | 3.11-3.13 (3.13 recommended on Windows — see README's "A note on Python 3.14") |
| TensorFlow | `== 2.6.5` (exact pin) | `>= 2.17` |
| NumPy | `== 1.19.5` (exact pin) | `>= 2.0` |
| `requirements.txt` | 383-line `pip-compile` lockfile, pinning every transitive dependency (old Jupyter stack, `tensorboard==2.6.0`, etc.) | ~15-line direct-dependency list with minimum versions, each annotated with *why* it's needed |

New direct dependencies added, none of which existed in the original: `scikit-learn` (LFW dataset), `datasets` (Hugging Face, FER-2013), `pillow` (image resizing), `nltk` and `gensim` (the new language/text experiment).

## 3. `src/mine.py` — bug fixes required by the TensorFlow/Keras version bump

These are concrete, mechanical fixes: code that called TF 1.x/early-TF-2.x APIs that no longer exist, or that Keras 3's stricter data-adapter validation now rejects outright.

- **`tf.log` → `tf.math.log`** in `biased_MINE_loss_marginal`. `tf.log` was removed from the TF 2.x namespace entirely; this loss function (used by the `MINE` algorithm option) would raise `AttributeError` on any modern TensorFlow.
- **`tf.reset_default_graph()` removed** from the `__main__` block. This was a TF 1.x graph-mode call; TF 2.x runs eagerly by default and has no default graph to reset. Calling it now raises `AttributeError`.
- **`Model.build_model`**: `model_core.add(ks.layers.Flatten(input_shape = image_shape))` → `model_core.add(ks.Input(shape = image_shape)); model_core.add(ks.layers.Flatten())`. Keras 3 removed the `input_shape=` kwarg on layers other than `Input` itself; a model must now start with an explicit `Input` layer.
- **Optimizer construction**: `ks.optimizers.Adam(self.learn_rate)` (positional) → `ks.optimizers.Adam(learning_rate = self.learn_rate)` (keyword), same for `RMSprop`/`SGD`. Keras 3's optimizer constructors made `learning_rate` keyword-only.
- **`Model.train`**: `validation_data = (data for data in val_itr)` → `validation_data = val_itr`. The original wrapped the validation iterator in a generator expression for reasons that aren't clear from the code; Keras 3's data-adapter inference is stricter about what it accepts as a validation source, and passing the iterator directly is both correct and simpler.
- **`get_finite_dataset`**: `yield ([...], [...])` (lists) → `yield ((...), (...))` (tuples). Keras 3's generator data adapter infers a `tf.TypeSpec` per model input and rejects plain Python lists as a batch container — it requires tuples.
- **`itertools.cycle(...)` replaced with a new `cycle_generator` function**. The original wrapped the validation dataset in `itertools.cycle(get_finite_dataset(...))` to repeat it indefinitely. Keras 3's data adapter now requires an actual generator *instance* (something with `__next__`) and rejects `itertools.cycle` objects, which are iterators but not generators in the technical sense. `cycle_generator(generator_fn, *args, **kwargs)` re-invokes the generator *factory function* in a loop (`yield from generator_fn(*args, **kwargs, loop=False)`) instead, producing a real generator.
- **`train_steps`/`val_steps` cast to `int`**. `np.ceil(...)` returns a `numpy.float64`; the newer Keras epoch iterator no longer silently accepts a float for `steps_per_epoch`/`validation_steps` — it needs a Python `int`.

## 4. `src/mine.py` — a real bug, unrelated to version compatibility

**`run_bipartition` read a bare `algorithm` variable that it never defined or received as a parameter.** In the original:

```python
def run_bipartition(inner_length, alg_settings, param_settings):
    ...
    if algorithm == 'mine':          # <- `algorithm` is not a parameter and not assigned above
        net = MINE(...)
    elif algorithm == 'logistic':
        net = LogisticRegression(...)
```

This only worked because `__main__` happened to set a **module-level global** `algorithm = alg_settings["algorithm"]` before calling `run_bipartition` (Python functions can read module globals they never declared). That makes `run_bipartition` silently dependent on `__main__` having already run — it is not actually a self-contained, reusable function. Since this project now calls `run_bipartition` directly from the notebook, from `src/image_noncrop_experiment.py`, and indirectly from `src/sequence.py`'s 1D analog — none of which ever execute `src/mine.py`'s `__main__` block — the original code would raise `NameError: name 'algorithm' is not defined` in every one of those call sites. Fixed by reading it from the parameter that was already right there: `algorithm = alg_settings["algorithm"]` as the first line of the function body.

## 5. `src/mine.py` — additive changes (new capability, old behavior unchanged)

- **`LogisticRegression = LogsiticRegression`**: a correctly-spelled alias for the original class (the misspelling is preserved for backward compatibility; both names work).
- **`run_bipartition` gained two new optional parameters**, both defaulted to match prior behavior exactly, so `alg.ini`/`mine.ini`-driven runs are unaffected:
  - `eval_steps = 5000` — how many validation batches `evaluate_MI` averages over. The original hardcoded `5000`; that's appropriate at the paper's full scale (tens of thousands of images) but wasteful/slow at the smaller image counts this fork's notebook uses for quick demonstration.
  - `target_size = img.DEFAULT_IMAGE_SIZE` (28) — passed straight through to `img.get_images`. This is what lets `src/image_noncrop_experiment.py` train on a dataset's native resolution instead of the fixed 28×28 crop (see §7).

## 6. `src/image.py` — dataset pipeline: Tiny Images replaced with five real, auto-downloading datasets

The original's `get_images` had exactly four real-data branches: `'tiny'` and `'gauss_tiny'` (both loading `.npy` files — `tiny_images_100.npy`, `tiny_images_cov.npy` — that don't ship with the repo and can't be regenerated, since the source dataset is gone), plus `'mnist'` and `'gauss_mnist'`.

This fork's `get_images` has: `mnist`, `fashion_mnist`, `cifar10`, `lfw_faces`, `fer2013_hf`, `mnist_shuffle_independent`, `mnist_shuffle_shared`, `gauss_mnist`, `area`, `diffuse`, `sparse`. The `tiny`/`gauss_tiny` sources are gone entirely — replaced conceptually by whichever of the new real datasets serves the same "rich, natural-image correlation structure" role in a given experiment.

- **`mnist`, `fashion_mnist`, `cifar10`** load via `tensorflow.keras.datasets` (built into Keras, download automatically on first use — same mechanism the original used for MNIST specifically). CIFAR-10 is new; it's converted to grayscale (`convert_to_grayscale`, new function — standard 0.3R+0.59G+0.11B luminance weighting) and center-cropped from its native 32×32 to 28×28.
- **`lfw_faces`** (Labeled Faces in the Wild) is new, loaded via `sklearn.datasets.fetch_lfw_people` — downloads automatically from a stable non-Kaggle mirror. Added as the "facial dataset" role after an initial attempt with the much smaller Olivetti Faces dataset (400 images) produced noisy, squashed-flat MI estimates; LFW's 13,000+ images fixed that.
- **`fer2013_hf`** is new, loaded via the Hugging Face `datasets` library from the `clip-benchmark/wds_fer2013` mirror. This restores a genuine facial-*emotion* dataset (7 classes) — the original FER-2013's usual Kaggle/`tensorflow-datasets` hosting is gone, so this uses a community mirror chosen specifically because it needs no custom loading script (`trust_remote_code`) and its train+test splits sum to exactly 35,887 images, matching canonical FER-2013.
- **`mnist_shuffle_independent`/`mnist_shuffle_shared`** are new: they apply a random pixel-position permutation to MNIST (a different permutation per image, or one shared permutation for every image, respectively) via two new helper functions, `shuffle_pixels_independent`/`shuffle_pixels_shared`. These build a *real-data* analog of the synthetic `sparse` Gaussian field (nearest-neighbor correlations, positions then randomly scattered) — something the original repository never attempted on real data at all. (An earlier version of this built the same thing from CIFAR-10; it was replaced with MNIST after determining CIFAR-10 doesn't actually satisfy the "genuinely local correlations" assumption the test depends on — see `README.md`'s "A note on `mnist_shuffle_independent`/`mnist_shuffle_shared`" for the full reasoning.)

New generic helpers backing all of the above, none of which existed in the original:
- **`conform_size(images, target_size, mode)`** — forces any image set to `target_size × target_size`, either by centered crop (`mode="crop"`, for sources at least as large as the target) or by resize (`mode="resize"`, via a new `resize_stack` using Pillow's Lanczos filter). This is the generalization that makes `target_size` configurable per dataset at all — the original hardcoded 28×28 crop logic inline, specific to MNIST's shape.
- **`center_crop`**, **`resize_stack`** — the two `conform_size` primitives.
- **`_load_keras_dataset`**, **`_load_lfw_faces`**, **`_load_fer2013_hf`** — one loader per data source, all imported lazily (TensorFlow/scikit-learn/`datasets` are imported *inside* these functions, not at module load time) so the rest of `image.py` — the Gaussian-field math and plotting utilities — can be used in an environment without those installed. The original imported `from tensorflow.keras import datasets` unconditionally at the top of the file.
- **`DEFAULT_IMAGE_SIZE = 28`** module constant, replacing several hardcoded `28`s scattered through the original's `get_images`.

## 7. `src/image.py` — plotting: six dead functions removed, one generic function added

The original file had six plotting functions, every one of which loaded pre-computed data this repository never had (`averages/*.npy`, `trials/*.npy`) or referenced the `tiny`/`gauss_tiny` sources removed in §6: `plot_cov`, `plot_tiny_mnist_cov_images`, `plot_sampled_images`, `plot_gaussian` (hardcoded to MNIST vs. Tiny Images), `plot_averages` (hardcoded to `averages/logistic_dense_mnist_0_1.0_70000.npy` / `..._tiny_...npy`), `plot_large_small_avg` (hardcoded to `trials/logistic_dense_{type}_{rho}_1.0_{n}.npy` for three sample sizes). None of these could run without files that don't exist in this repo and can no longer be regenerated (Tiny Images is gone).

All six were removed and replaced by a single general-purpose function, **`plot_mi_scaling(results, lengths=None, labels=None, save_path=None, clip_negative=False)`**, which plots one or more MI-vs-partition-length curves directly from in-memory arrays (a dict of `label: values`, or a plain list of value-sequences) in the same visual style the original's functions used (font sizes, figure proportions, axis labels). This is what every scaling plot in the modernized notebook and every experiment script in this repo uses — it works on data computed live in the current session rather than data pre-computed offline and saved to disk, which is the only way to produce these figures now that the offline trial data doesn't exist.

(`get_analytic_MI` and the exact-covariance Gaussian-field math — `get_area_law_cov`, `get_diffuse_volume_cov`, `get_sparse_volume_cov`, `get_marginal_entropy`, `get_gaussian_mutual_information` — are **logically unchanged** from the original; they don't depend on any external data and needed no fixes. Their comments were substantially expanded, though — see §13.)

## 8. Entirely new modules (nothing in the original corresponds to these)

- **`src/image_noncrop_experiment.py`** — reruns the real-dataset MI-scaling sweep at each dataset's *native* resolution (CIFAR-10 at 32×32, LFW at 94×94, FER-2013 at 48×48) instead of the fixed 28×28 every dataset gets conformed to elsewhere, to measure how much information that downsizing throws away. Sweeps the entire image (length 1 through the dataset's full size), not just a truncated range, so the resulting curves show the complete rise, peak, and forced decline back to ~0 once the outer patch runs out of pixels. Results land in `image_noncrop_results/`.
- **`src/sequence.py`**, **`src/language.py`**, **`src/language_experiment.py`**, **`src/language_embedding_experiment.py`** — a full 1D analog of the paper's 2D image analysis, applied to natural-language text (the NLTK Gutenberg corpus) instead of images: MI between a centered window of consecutive words and the surrounding text, as a function of window length. `sequence.py` provides the 1D version of `image.py`'s partitioning/splicing logic and explicitly reuses `mine.py`'s `Model`/`LogisticRegression`/`MINE` classes completely unmodified (they make no assumption about input rank — a `(num_samples, length)` sequence flows through the same `Input → Flatten → Dense` architecture a `(28, 28, 1)` image does). `language.py` builds the word sequences two ways: a crude per-word `log(frequency rank)` scalar encoding, or (in the `_embedding` variant) full pretrained GloVe embedding vectors via `gensim`. This experiment and everything it depends on is new; none of it exists in the original repository in any form. Results land in `language_results/`.

## 9. `examples.ipynb` — from static reference figures to a fully live-computed notebook

The original notebook has **20 cells** and, for anything beyond a single hardcoded training example, shows a **static pre-rendered image** loaded from the `figures/` folder (`figures/area.png`, `diffuse.png`, `sparse.png`, `mnist_tiny.png` — five PNGs, generated offline from the missing trial data, never computed in the notebook itself). This fork's notebook has **32 cells**, all of it live-computed:

- The Gaussian Markov random field section gained two new cells computing the *exact* analytic MI curves for all three field types at both correlation strengths (closed-form, no training — `get_analytic_MI`), plus a validation cell that sweeps the neural estimator across all 27 partition lengths for one field and plots it against the exact curve, to demonstrate the estimator actually recovers the known answer.
- The single-dataset MNIST example is followed by a new section that sweeps **every** available real dataset (MNIST, Fashion-MNIST, CIFAR-10, LFW) across all 27 partition lengths and plots them together, replacing the original's single static `figures/mnist_tiny.png`.
- A new section builds the `mnist_shuffle_*` real-data analog of the `sparse` field described in §6, again fully live.
- The `figures/` folder itself (all five PNGs) was dropped — this repository doesn't include or generate the paper's original hero figures, since the trial data behind them is unrecoverable; every scaling plot here is one this fork actually computed.

**A latent bug was also fixed in the notebook.** The original's MNIST training cell calls `img.get_images("mnist", num_images, strength)` — passing the bare name `strength` as a positional argument, which is never defined in that cell. It happens to work *only* if the reader has already run the earlier Gaussian-field cell that sets `strength = "large"` for an unrelated purpose (Jupyter shares variables across all cells in a kernel session) — running the MNIST section on a fresh kernel without first running the Gaussian section raises `NameError`. Every real-dataset cell in this fork's notebook defines its own settings from scratch rather than silently depending on an earlier, conceptually unrelated section having been run first.

Also fixed: a notebook-format bug (unrelated to Python) where several cells' `source` field had been written as one raw string instead of a list of lines — both are valid per the `nbformat` spec, but the raw-string form rendered as blank/invisible input in VS Code's notebook editor even though the cells executed fine.

## 10. `alg.ini`

Only the default `image_type` changed, from `area` (a synthetic Gaussian field) to `mnist` (a real dataset — a more representative smoke-test default now that several real datasets actually work), plus a clarifying comment listing which `image_type` values are valid now that `tiny`/`gauss_tiny` are gone and five new sources exist. `mine.ini` is byte-for-byte unchanged.

## 11. `README.md` / `CHANGELOG.md`

The original's README is 9 lines of prose plus a paper abstract. This fork's README documents: the dataset table (§6), the `target_size`/non-cropped sweep (§8), a section explaining the area-law-vs-volume-law growth-rate analysis and how this fork's own results compare to the original paper's, the `mnist_shuffle_*` reasoning, Python/TensorFlow version guidance, and a "Project layout" map. The original had no changelog; this fork's development history (every dated change, in the order it happened) was split out into `CHANGELOG.md` once the README grew long enough that the change history was crowding out the "how to actually use this" content.

## 12. Repository hygiene (no functional effect)

- Added `.gitignore` (`__pycache__/`, `*.pyc`, `.venv/`, `.ipynb_checkpoints/`); the original had none, and had `src/__pycache__/*.pyc` committed to git.
- Generated output was reorganized into three parallel `*_results/` folders (`image_results/`, `image_noncrop_results/`, `language_results/`) instead of accumulating at the repository root; the original saved its (offline-generated, not-included) figures to a single flat `figures/` folder.
- Removed `src/README.md`, a one-line junk file that contained only the text "README.md".

## 13. Annotation pass (comments only — no logic changed)

`src/mine.py` and `src/image.py`'s Gaussian-field math both carried the original author's fairly terse, one-line-per-function comments (accurate, but assuming a reader already understands the underlying method). Both files got a substantially expanded comment pass, verified against the actual code by re-running the smoke tests in §§3-4 afterward to confirm nothing was accidentally changed:

- **`src/mine.py`** gained a module-level docstring explaining the core trick the whole file implements (train a classifier to distinguish real image pairs from spliced "fake" pairs; its output approximates the log-density-ratio whose expectation is the MI), plus expanded comments on: `Model.build_model` (why there are two weight-*shared* input towers, not two separate networks), `evaluate_MI` (spelling out the `direct_mi`/`indirect_mi` formulas and what the Donsker-Varadhan correction term in `indirect_mi` actually does), `LogisticRegression`/`MINE` (what loss each one actually optimizes, and which one this project's own notebook uses), `Index`/`get_mixed_indices` (why the duplicate-pairing check exists — without it, some "marginal" samples would secretly be joint samples), `get_finite_dataset` (spelling out exactly how a "joint" batch and a "marginal" batch are constructed from the same underlying images), and `run_bipartition` (now documented as the one-call entry point that every experiment in this repo is built from, and why `algorithm` is read from a parameter rather than a global — see §4).
- **`src/image.py`** gained expanded comments on the three GMRF covariance constructors (`get_area_law_cov`/`get_diffuse_volume_cov`/`get_sparse_volume_cov`), explaining *why* each one's particular precision-matrix structure produces area-law vs. volume-law MI scaling, not just *what* the code loops over; and on the Gaussian-entropy/MI chain (`get_marginal_entropy`, `get_gaussian_mutual_information`, `get_gaussian_fit`, `get_analytic_MI`, `get_gaussian_images`), spelling out the closed-form entropy formula being computed and why `get_marginal_entropy` filters out near-zero-variance pixels (real images can have pixels, like a fixed black border, that never vary at all - a zero-variance dimension in a covariance matrix is singular and breaks the log-determinant otherwise).
