import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from MI_scaling_non_middle import pixel_pruning
from MI_scaling_non_middle import vgg_cifar10_model as vm
from MI_scaling_non_middle import vgg_conv_sliding_mi as sweep

# Prunes the fine-tuned VGG16's conv-layer activations using the tile masks derived from
# vgg_conv_sliding_mi.py's Renyi MI heatmaps (the lowest-MI tiles of each layer's own
# 7x7 grid, reusing pixel_pruning.build_tile_mask/tile_mask_to_pixel_mask unchanged - no
# new masking logic needed), and checks whether the pruned network's predictions still
# agree with the *unpruned* network's predictions - not with ground-truth labels, unlike
# every other pruning experiment in this project (train_pruned_classifiers.py's
# "does accuracy survive?" question). Two conditions:
#   - "single_layer": prune exactly one conv layer at a time (all other layers left
#     untouched), swept across all 13 layers - meant to be run first for just one layer
#     (e.g. --layers block3_conv2) to validate the mechanism before the full sweep.
#   - "block_wise": prune every one of the 13 conv layers simultaneously in a single
#     forward pass, each with its own mask - each mask is still derived from the *clean*
#     (unpruned) activations recorded by vgg_conv_sliding_mi.py, not iteratively
#     re-measured after upstream layers' pruning takes effect - the same "single upfront
#     mask" simplification train_pruned_classifiers.py already makes for pixel-space
#     pruning.
# Both conditions are evaluated at percent_kept in {0.85, 0.70} (prune 15%/30%).
#
# Run as (needs vgg_cifar10_model.py and a full vgg_conv_sliding_mi.py sweep to have been
# run first):
#     python -m MI_scaling_non_middle.vgg_conv_pruning --layers block3_conv2 --conditions single_layer
#     python -m MI_scaling_non_middle.vgg_conv_pruning

RESULTS_DIR = os.path.join(ROOT, "MI_scaling_non_middle", "results", "vgg", "pruning")

CONDITIONS = ["single_layer", "block_wise"]
DEFAULT_PERCENT_KEPT = [0.85, 0.70]  # prune 15% / 30%
BATCH_SIZE = 64


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_layer_heatmap(layer_name):
    paths = sweep.result_paths(layer_name)
    if not os.path.exists(paths["heatmap"]):
        raise FileNotFoundError(
            f"{paths['heatmap']} not found - run "
            f"`python -m MI_scaling_non_middle.vgg_conv_sliding_mi --layers {layer_name}` first.")
    heatmap = np.load(paths["heatmap"])
    if np.isnan(heatmap).any():
        raise ValueError(f"{paths['heatmap']} has unfinished (NaN) tiles - that layer's sweep hasn't completed.")
    return heatmap


def build_layer_mask(layer_name, percent_kept):

    # Reuses pixel_pruning.py's generic tile-ranking/expansion functions unchanged: the
    # heatmap is VGG-conv-layer-shaped, but build_tile_mask/tile_mask_to_pixel_mask only
    # care about a (grid_size, grid_size) array and a target pixel size, exactly like
    # they already do for train_pruned_classifiers.py's input-pixel masks. The
    # equal-division tile_mask_to_pixel_mask (not the edges-based
    # load_sliding_window_mi/pixel_mask_from_edges variant) is correct here because every
    # VGG layer's spatial size divides GRID_SIZE evenly (see vgg_cifar10_model.py's
    # module docstring) - unlike mnist/cifar10's 28px pixel grid, there's no 1px gap to
    # close with explicit edges.

    heatmap = load_layer_heatmap(layer_name)
    tile_mask = pixel_pruning.build_tile_mask(heatmap, percent_kept)
    spatial_size = vm.CONV_LAYER_SPATIAL_SIZE[layer_name]
    return pixel_pruning.tile_mask_to_pixel_mask(tile_mask, spatial_size)  # True = keep


def build_pruned_model(model, masks):

    # Rebuilds the classifier's forward pass as one flat functional graph from
    # model.input straight through to the final softmax, splicing a keep-mask multiply
    # in immediately after any conv layer named in `masks` (bool (H, W) array, True=keep
    # - pruned tiles zeroed by multiplying with the mask cast to float and broadcast over
    # the batch/channel axes). Every layer object (conv/pool layers from the backbone,
    # GlobalAveragePooling2D/Dropout/Dense from the head) is *reused* from the trained
    # model - not copied - so this shares the exact same trained weights as `model`
    # itself; Keras layers support being called again in a new graph (the same mechanism
    # that makes "shared layers" work) so this is a supported pattern, not a hack.
    #
    # The backbone's own conv/pool layers are chained manually (`layer(x, training=False)`
    # for each, in order) starting from model's own "vgg_preprocess" output tensor -
    # rather than calling `model.get_layer(backbone_name)(x)` as one unit, the way
    # vgg_cifar10_model.build_model originally does - specifically *because* that one-unit
    # call is what makes individual internal layer outputs unreachable from the outer
    # graph (see vgg_conv_sliding_mi.py's extract_layer_activations docstring for the
    # same nested-model issue in the other direction); chaining the layers individually
    # here is what makes inserting a mask *between* two of them possible at all. This
    # relies on VGG16's conv stack being a strict linear chain with no skip connections
    # (verified by direct inspection - see vgg_cifar10_model.py's module docstring).

    backbone = vm.get_backbone(model)
    x = model.get_layer("vgg_preprocess").output
    for layer in backbone.layers[1:]:  # skip the backbone's own InputLayer
        x = layer(x, training=False)
        if layer.name in masks:
            mask = masks[layer.name].astype("float32")[None, :, :, None]  # (1, H, W, 1)
            x = x * mask

    backbone_index = next(i for (i, l) in enumerate(model.layers) if l.name == vm.BACKBONE_NAME)
    for layer in model.layers[backbone_index + 1:]:  # GlobalAveragePooling2D, Dropout, Dense
        x = layer(x, training=False)

    return ks.Model(inputs=model.input, outputs=x)


def compute_output_difference(baseline_outputs, pruned_outputs):

    # The pruning-quality criterion this experiment uses instead of classification
    # accuracy: how much do the pruned network's own predictions differ from the
    # unpruned network's, on the same images - never compared against ground-truth
    # labels here.
    #   - mean_kl_divergence: mean over samples of KL(baseline || pruned), the standard
    #     divergence between two categorical (softmax) distributions - epsilon-clipped
    #     for numerical stability against exact 0 probabilities.
    #   - top1_agreement_rate: fraction of samples where the pruned network's argmax
    #     matches the baseline network's argmax (not whether either matches the true
    #     label).

    eps = 1e-12
    kl_per_sample = np.sum(
        baseline_outputs * (np.log(baseline_outputs + eps) - np.log(pruned_outputs + eps)), axis=1)
    mean_kl = float(np.mean(kl_per_sample))
    agreement = float(np.mean(np.argmax(baseline_outputs, axis=1) == np.argmax(pruned_outputs, axis=1)))
    return {"mean_kl_divergence": mean_kl, "top1_agreement_rate": agreement}


def condition_tag(condition, percent_kept, layer_name=None):
    pct = int(round(percent_kept * 100))
    if layer_name is None:
        return f"vgg_{condition}_p{pct}"
    return f"vgg_{condition}_{layer_name}_p{pct}"


def metrics_path(condition, percent_kept, layer_name=None):
    return os.path.join(RESULTS_DIR, f"{condition_tag(condition, percent_kept, layer_name)}_metrics.json")


def run_single_layer_condition(model, layer_name, percent_kept, x_eval, baseline_outputs, batch_size):
    ensure_results_dir()
    path = metrics_path("single_layer", percent_kept, layer_name)
    if os.path.exists(path):
        print(f"[single_layer/{layer_name}/p{int(round(percent_kept * 100))}] metrics already exist, skipping", flush=True)
        with open(path) as handle:
            return json.load(handle)

    mask = build_layer_mask(layer_name, percent_kept)
    pruned_model = build_pruned_model(model, {layer_name: mask})
    pruned_outputs = pruned_model.predict(x_eval, batch_size=batch_size, verbose=1)
    metrics = compute_output_difference(baseline_outputs, pruned_outputs)
    metrics.update({"condition": "single_layer", "layer": layer_name, "percent_kept": percent_kept})
    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[single_layer/{layer_name}/p{int(round(percent_kept * 100))}] "
          f"mean_kl={metrics['mean_kl_divergence']:.4f} agreement={metrics['top1_agreement_rate']:.4f}", flush=True)
    return metrics


def run_block_wise_condition(model, percent_kept, x_eval, baseline_outputs, batch_size):
    ensure_results_dir()
    path = metrics_path("block_wise", percent_kept)
    if os.path.exists(path):
        print(f"[block_wise/p{int(round(percent_kept * 100))}] metrics already exist, skipping", flush=True)
        with open(path) as handle:
            return json.load(handle)

    masks = {layer_name: build_layer_mask(layer_name, percent_kept) for layer_name in vm.CONV_LAYER_NAMES}
    pruned_model = build_pruned_model(model, masks)
    pruned_outputs = pruned_model.predict(x_eval, batch_size=batch_size, verbose=1)
    metrics = compute_output_difference(baseline_outputs, pruned_outputs)
    metrics.update({"condition": "block_wise", "percent_kept": percent_kept})
    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[block_wise/p{int(round(percent_kept * 100))}] "
          f"mean_kl={metrics['mean_kl_divergence']:.4f} agreement={metrics['top1_agreement_rate']:.4f}", flush=True)
    return metrics


def plot_single_layer_comparison(metrics_for_ratio, percent_kept):
    ensure_results_dir()
    layer_names = [m["layer"] for m in metrics_for_ratio]
    kl_values = [m["mean_kl_divergence"] for m in metrics_for_ratio]
    agreement_values = [m["top1_agreement_rate"] for m in metrics_for_ratio]

    (fig, (kl_axes, agreement_axes)) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    kl_axes.bar(layer_names, kl_values, color="#C44E52")
    kl_axes.set_ylabel("Mean KL(baseline || pruned)")
    kl_axes.set_title(f"Single-layer pruning: output divergence from baseline "
                       f"(keep {int(round(percent_kept * 100))}%, VGG16/cifar10)", fontsize=12)

    agreement_axes.bar(layer_names, agreement_values, color="#4C72B0")
    agreement_axes.set_ylabel("Top-1 agreement with baseline")
    agreement_axes.set_ylim(0, 1.05)
    agreement_axes.set_xticks(range(len(layer_names)))
    agreement_axes.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    tag = f"p{int(round(percent_kept * 100))}"
    plt.savefig(os.path.join(RESULTS_DIR, f"vgg_single_layer_comparison_{tag}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"vgg_single_layer_comparison_{tag}.png"), dpi=150)
    plt.close()


def plot_block_wise_summary(all_block_wise_metrics):
    ensure_results_dir()
    ratios = [m["percent_kept"] for m in all_block_wise_metrics]
    labels = [f"keep {int(round(r * 100))}%" for r in ratios]
    kl_values = [m["mean_kl_divergence"] for m in all_block_wise_metrics]
    agreement_values = [m["top1_agreement_rate"] for m in all_block_wise_metrics]

    (fig, (kl_axes, agreement_axes)) = plt.subplots(1, 2, figsize=(9, 4.5))
    kl_axes.bar(labels, kl_values, color="#C44E52")
    kl_axes.set_ylabel("Mean KL(baseline || pruned)")
    kl_axes.set_title("Block-wise (all 13 layers) pruning")

    agreement_axes.bar(labels, agreement_values, color="#4C72B0")
    agreement_axes.set_ylabel("Top-1 agreement with baseline")
    agreement_axes.set_ylim(0, 1.05)
    agreement_axes.set_title("Block-wise (all 13 layers) pruning")

    fig.suptitle("Block-wise MI-guided pruning vs. unpruned baseline output (VGG16/cifar10)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vgg_block_wise_summary.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "vgg_block_wise_summary.png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent-kept", type=str, default=",".join(str(p) for p in DEFAULT_PERCENT_KEPT),
                         help=f"Comma-separated fractions of tiles to keep (default: prune 15%%/30%%)")
    parser.add_argument("--layers", type=str, default=None,
                         help="Comma-separated subset of vgg_cifar10_model.CONV_LAYER_NAMES for the single_layer "
                              "condition (default: all 13) - pass one layer to validate the mechanism first, e.g. "
                              "--layers block3_conv2 --conditions single_layer")
    parser.add_argument("--conditions", type=str, default=None,
                         help="Comma-separated subset of single_layer,block_wise (default: both)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    percent_kept_values = [float(p) for p in args.percent_kept.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")] if args.conditions else CONDITIONS
    layer_names = [name.strip() for name in args.layers.split(",")] if args.layers else vm.CONV_LAYER_NAMES

    model = vm.load_trained_model()
    baseline_outputs = np.load(vm.BASELINE_OUTPUTS_PATH)
    x_eval = np.load(vm.EVAL_IMAGES_PATH)
    assert baseline_outputs.shape[0] == x_eval.shape[0], \
        "Baseline outputs/eval images count mismatch - was vgg_cifar10_model.py's eval-image file regenerated separately?"

    if "single_layer" in conditions:
        for percent_kept in percent_kept_values:
            metrics_for_ratio = []
            for layer_name in layer_names:
                metrics = run_single_layer_condition(
                    model, layer_name, percent_kept, x_eval, baseline_outputs, args.batch_size)
                metrics_for_ratio.append(metrics)
            if set(layer_names) == set(vm.CONV_LAYER_NAMES):
                # Only a full 13-layer sweep makes the per-layer comparison plot
                # meaningful - same guard vgg_conv_sliding_mi.py's plot_combined uses.
                plot_single_layer_comparison(metrics_for_ratio, percent_kept)

    if "block_wise" in conditions:
        all_block_wise_metrics = [
            run_block_wise_condition(model, percent_kept, x_eval, baseline_outputs, args.batch_size)
            for percent_kept in percent_kept_values
        ]
        plot_block_wise_summary(all_block_wise_metrics)

    print("\n=== Summary ===")
    for percent_kept in percent_kept_values:
        pct = int(round(percent_kept * 100))
        for condition in conditions:
            path = metrics_path(condition, percent_kept) if condition == "block_wise" else None
            if condition == "block_wise" and os.path.exists(path):
                with open(path) as handle:
                    metrics = json.load(handle)
                print(f"block_wise | keep={pct}% | mean_kl={metrics['mean_kl_divergence']:.4f} "
                      f"| agreement={metrics['top1_agreement_rate']:.4f}")


if __name__ == "__main__":
    main()
