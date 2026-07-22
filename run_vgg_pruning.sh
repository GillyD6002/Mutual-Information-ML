#!/usr/bin/env bash
set -euo pipefail

# Runs the full VGG16/CIFAR-10 conv-layer MI sweep + pruning experiment
# (vgg_cifar10_model.py -> vgg_conv_sliding_mi.py -> vgg_conv_pruning.py) on
# this box, in the same tensorflow/tensorflow:2.17.0-gpu docker + libcuda
# symlink fix already used by run_pruning_lfw_fer2013.sh (no sudo/venv on
# this box, system Python too old for tensorflow>=2.17 - see memory).
#
# Four steps:
#   1. vgg_cifar10_model.py - downloads VGG16's ImageNet weights (~528MB, one
#      time, needs internet), fine-tunes them into a CIFAR-10 classifier
#      (frozen-head phase then full fine-tune), and saves the trained model
#      plus a fixed 2000-image baseline softmax output every later pruned
#      condition gets diffed against.
#   2. vgg_conv_sliding_mi.py, restricted to one layer (block3_conv2) -
#      cheap, fast validation that the Renyi-MI sweep mechanism produces
#      sane output on the real trained model before spending the time on
#      all 13 layers.
#   3. vgg_conv_pruning.py, restricted to that same one layer and the
#      single_layer condition only - the "test pruning a single layer, and
#      if that works, do the rest" validation step, using the one heatmap
#      step 2 just produced.
#   4. The full runs: vgg_conv_sliding_mi.py across all 13 conv layers, then
#      vgg_conv_pruning.py across both conditions (single_layer, block_wise)
#      and both ratios (keep 85%/70%, i.e. prune 15%/30%) - step 3's
#      block3_conv2/single_layer result is reused here via the same
#      resumable-metrics-file convention every sweep script in this project
#      already uses, not recomputed.
#
# Between steps 3 and 4 is the natural point to sanity-check things by hand
# before committing to the expensive full run - watch with `docker logs -f`
# and Ctrl-C-then-`docker rm -f` this container if block3_conv2's numbers
# look wrong, rather than letting it barrel into the full 13-layer sweep.
#
# Runtime note: the Renyi MI estimator (src/renyi_mi.py) is pure numpy - it
# runs on CPU regardless of the GPU this container has, only the TF forward
# passes (fine-tuning, activation extraction) use the GPU. block1_conv1/
# block1_conv2's 224x224x64 activation maps make their 49-position sweep the
# slowest part of step 4 by a wide margin (the "outer" side of the
# window/rest split is ~3.15M-dim there); every deeper layer is much smaller
# and faster. Lower --num-images (default 500, see vgg_conv_sliding_mi.py)
# for a faster/lower-fidelity first pass if this is prohibitive.
#
# Allocates every GPU on the box (--gpus all) - this is a shared box (see
# memory); only do this if `nvidia-smi` shows it's currently idle.
#
# Usage:
#   ./run_vgg_pruning.sh
#
# Runs detached (docker -d), so it survives disconnecting; follow along with
# `docker logs -f mi-vgg-pruning`. A second, separate background process
# waits on the container and auto-commits+pushes the new results
# (MI_scaling_non_middle/results/vgg/, which excludes the large *.keras model
# weights and the eval-images array via .gitignore) once everything finishes
# - see the nohup block at the bottom. That watcher is independent of this
# shell, so it still fires even if you disconnect first; progress lands in
# mi-vgg-pruning-push.log.
#
# What lands where once this finishes:
#   - Per-layer MI heatmaps: MI_scaling_non_middle/results/vgg/cifar10_vgg_*_heatmap.png
#     and the combined 13-panel figure cifar10_vgg_conv_sliding_combined_heatmap.png
#   - Single-layer pruning bar charts (both ratios):
#     MI_scaling_non_middle/results/vgg/pruning/vgg_single_layer_comparison_p{85,70}.png
#   - Block-wise pruning summary: MI_scaling_non_middle/results/vgg/pruning/vgg_block_wise_summary.png
#   - Every condition's raw metrics (KL divergence + top-1 agreement vs. the
#     unpruned baseline): MI_scaling_non_middle/results/vgg/pruning/*_metrics.json

CONTAINER_NAME="mi-vgg-pruning"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUSH_LOG="${REPO_DIR}/mi-vgg-pruning-push.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "A container named ${CONTAINER_NAME} already exists (docker ps -a)." >&2
    echo "Remove it first if you want to start fresh: docker rm -f ${CONTAINER_NAME}" >&2
    exit 1
fi

docker run -d --gpus all --name "${CONTAINER_NAME}" \
    -v "${REPO_DIR}:/workspace" -w /workspace \
    tensorflow/tensorflow:2.17.0-gpu bash -c '
        set -euo pipefail
        mkdir -p /tmp/fixed-cuda
        # See run_pruning_lfw_fer2013.sh for why this has to match the exact
        # libcuda.so.* whose version equals the hosts actual driver version,
        # not just the image bundled default.
        DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        CUDA_LIB="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VERSION}"
        if [ ! -f "${CUDA_LIB}" ]; then
            echo "Expected ${CUDA_LIB} (matching host driver ${DRIVER_VERSION}) not found; libcuda.so.* present:" >&2
            ls /usr/lib/x86_64-linux-gnu/libcuda.so.* >&2 2>/dev/null || true
            exit 1
        fi
        ln -sf "${CUDA_LIB}" /tmp/fixed-cuda/libcuda.so.1
        ln -sf "${CUDA_LIB}" /tmp/fixed-cuda/libcuda.so
        export LD_LIBRARY_PATH="/tmp/fixed-cuda:${LD_LIBRARY_PATH:-}"

        pip install --quiet matplotlib

        python -c "import tensorflow as tf; print(\"GPUs visible:\", tf.config.list_physical_devices(\"GPU\"))"

        # Step 1: fine-tune VGG16 on CIFAR-10, save the model + baseline outputs.
        python -m MI_scaling_non_middle.vgg_cifar10_model

        # Step 2: validate the Renyi MI sweep on one layer first.
        python -m MI_scaling_non_middle.vgg_conv_sliding_mi --layers block3_conv2

        # Step 3: validate pruning + the output-divergence criterion on that
        # same one layer before scaling up to the full 13-layer sweep.
        python -m MI_scaling_non_middle.vgg_conv_pruning \
            --layers block3_conv2 --conditions single_layer

        # Step 4: the full runs (block3_conv2 from steps 2-3 is skipped here,
        # not recomputed - both scripts are resumable per-layer/per-condition).
        python -m MI_scaling_non_middle.vgg_conv_sliding_mi

        python -m MI_scaling_non_middle.vgg_conv_pruning
    '

echo "Started container ${CONTAINER_NAME} using all GPUs on the box."
echo "Follow progress:   docker logs -f ${CONTAINER_NAME}"
echo "Check it's alive:  docker ps"
echo "MI heatmaps:       MI_scaling_non_middle/results/vgg/cifar10_vgg_*_heatmap.png"
echo "Pruning results:   MI_scaling_non_middle/results/vgg/pruning/*_metrics.json"
echo "Bar charts:        MI_scaling_non_middle/results/vgg/pruning/vgg_single_layer_comparison_p{85,70}.png"
echo "                   MI_scaling_non_middle/results/vgg/pruning/vgg_block_wise_summary.png"

# See run_pruning_lfw_fer2013.sh for why this is a separate nohup'd+disowned
# background process rather than inline: keeps running independent of this
# shell/terminal, only commits on a clean (0) exit so a crashed/killed run
# isn't silently pushed as a success, and no-ops (not an error) if a rerun
# left nothing new to commit.
nohup bash -c "
    EXIT_CODE=\$(docker wait '${CONTAINER_NAME}')
    cd '${REPO_DIR}'
    if [ \"\${EXIT_CODE}\" != \"0\" ]; then
        echo \"[\$(date)] Container exited \${EXIT_CODE} (failure) - not pushing.\" >> '${PUSH_LOG}'
        exit 0
    fi
    git add MI_scaling_non_middle/results
    if git diff --cached --quiet; then
        echo \"[\$(date)] Run finished but nothing new to commit.\" >> '${PUSH_LOG}'
        exit 0
    fi
    git commit -m 'Add VGG16/CIFAR-10 conv-layer Renyi MI sweep and pruning results' >> '${PUSH_LOG}' 2>&1
    if git push >> '${PUSH_LOG}' 2>&1; then
        echo \"[\$(date)] Pushed VGG pruning results.\" >> '${PUSH_LOG}'
    else
        echo \"[\$(date)] git push FAILED - see output above. Commit is local only, fix and push manually.\" >> '${PUSH_LOG}'
    fi
" > /dev/null 2>&1 &
disown
echo "Auto-push watcher started - will commit+push results once the run finishes."
echo "Watch it:          tail -f ${PUSH_LOG}"
