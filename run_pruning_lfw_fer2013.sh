#!/usr/bin/env bash
set -euo pipefail

# Redo of the lfw_faces/fer2013_hf pruning experiment, replacing the first
# pass's approach (which squished both datasets down to 28x28 to match
# mnist/cifar10 and got poor baseline accuracy - 18% for lfw_faces, since
# identity classification across dozens of people with few images each
# doesn't survive that much downsampling or a from-scratch small CNN).
# This version:
#
#   1. Keeps each dataset at its own native (or near-native) resolution -
#      96x96 for lfw_faces (its native 125x94 crop, center-cropped/
#      edge-padded, never resized/interpolated - see
#      train_pruned_classifiers.py's _lfw_to_square), 48x48 for fer2013_hf
#      (already square natively, so untouched).
#   2. Uses a tile grid sized to tile each dataset's resolution with zero
#      leftover edge (window=8/stride=8 for lfw_faces, window=4/stride=4
#      for fer2013_hf - see TILE_WINDOW_SIZES in train_pruned_classifiers.py),
#      instead of reusing mnist/cifar10's window=3/stride=3.
#   3. Trains via transfer learning (a pretrained MobileNetV2 ImageNet
#      backbone, fine-tuned in two phases) instead of a from-scratch CNN -
#      both datasets are fundamentally data-scarce problems a bigger
#      from-scratch network doesn't fix. See
#      train_pruned_classifiers.py's build_transfer_classifier.
#   4. Measures each tile's MI over up to 70000 images instead of 30000
#      (in practice, "every image available" for both datasets, neither of
#      which actually has 70000).
#
# This takes considerably longer than the first pass: bigger images, a
# finer/larger tile grid, more images per MI estimate, and two training
# phases (frozen head + fine-tune) per condition instead of one. Old
# lfw_faces/fer2013_hf results (from the 28x28/from-scratch run) are wiped
# at the start of this script, on this box, in addition to already being
# deleted from git - a belt-and-suspenders precaution in case this box's
# checkout hasn't picked up that deletion yet by the time this runs.
#
# Two steps per dataset, same as before:
#   1. sliding_window_mi.py's sweep, now at each dataset's own window size
#      (a single combined --datasets lfw_faces,fer2013_hf invocation no
#      longer works now that they need different --stride values).
#   2. train_pruned_classifiers.py itself, at both percent_kept values
#      already used for mnist/cifar10 (0.75, 0.50).
#
# Runs inside tensorflow/tensorflow:2.17.0-gpu (this box has no sudo/venv
# and system Python is too old for tensorflow>=2.17 - see memory), with a
# libcuda symlink fix for this host's driver, same as every other GPU job
# on this box. MobileNetV2's ImageNet weights download automatically on
# first use (~14MB) - no separate pip package needed for transfer learning,
# just internet access, same as the HuggingFace/sklearn dataset downloads
# this already needed.
#
# Allocates every GPU on the box (--gpus all) rather than one device -
# train_pruned_classifiers.py's main() already wraps training in
# tf.distribute.MirroredStrategy(), which actually splits each batch across
# however many GPUs are visible, so this isn't just exposing extra devices
# for nothing. NOTE: this is a shared box (see memory) - grabbing every GPU
# will starve anyone else's job running at the same time; only do this if
# you know the box is otherwise idle right now (check `nvidia-smi`).
#
# Usage:
#   ./run_pruning_lfw_fer2013.sh
#
# Runs detached (docker -d), so it survives disconnecting; follow along
# with `docker logs -f mi-pruning-lfw-fer2013`. A second, separate
# background process waits on the container and auto-commits+pushes the
# new results (MI_scaling_non_middle/results/, which already excludes the
# large *.keras model weights via .gitignore) once training finishes - see
# the nohup block at the bottom. That watcher is also independent of this
# shell, so it still fires even if you disconnect before training ends;
# progress lands in mi-pruning-lfw-fer2013-push.log.
#
# What lands where once this finishes:
#   - Bar charts (baseline/zero/noise per dataset, both percent_kept
#     values): MI_scaling_non_middle/results/pruning/pruning_comparison_p75.png
#     and _p50.png - produced automatically by train_pruned_classifiers.py's
#     plot_comparison at the end of each run.
#   - Per-tile MI heatmaps for both datasets:
#     MI_scaling_non_middle/results/{lfw_faces,fer2013_hf}_sliding_w*_heatmap.png
#     - produced automatically by sliding_window_mi.py's plot_heatmap during
#     the sweep step.

CONTAINER_NAME="mi-pruning-lfw-fer2013"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUSH_LOG="${REPO_DIR}/mi-pruning-lfw-fer2013-push.log"

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
        # tensorflow/tensorflow:2.17.0-gpu bakes in its own libcuda.so.*
        # (a stub matching whatever CUDA version the image was built
        # against), and the images default libcuda.so symlink chain points
        # at *that* one, not at the real driver library the NVIDIA
        # container toolkit mounts in from the host. If they don'"'"'t match,
        # cuInit fails and TF reports zero GPUs even though --gpus all
        # worked. So this has to match the exact file whose version equals
        # the host'"'"'s actual driver version (from nvidia-smi) - not just
        # "the first libcuda.so.* found", which previously happened to
        # pick the right one only because its version number
        # (535.183.01) sorted alphabetically before the image'"'"'s bundled
        # one (545.23.06); that'"'"'s incidental and breaks for other version
        # pairs (e.g. "6.0.0" sorts after "10.2.0" alphabetically despite
        # being numerically smaller).
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

        pip install --quiet matplotlib scikit-learn datasets pillow

        python -c "import tensorflow as tf; print(\"GPUs visible:\", tf.config.list_physical_devices(\"GPU\"))"

        # Belt-and-suspenders: wipe any old (28x28, from-scratch) lfw_faces/
        # fer2013_hf results on this box, in case this checkout has not yet
        # picked up their deletion from git - see header comment.
        rm -f MI_scaling_non_middle/results/lfw_faces_sliding_*
        rm -f MI_scaling_non_middle/results/fer2013_hf_sliding_*
        rm -f MI_scaling_non_middle/results/pruning/lfw_faces_*
        rm -f MI_scaling_non_middle/results/pruning/fer2013_hf_*

        python -m MI_scaling_non_middle.sliding_window_mi \
            --datasets lfw_faces --window-sizes 8 --stride 8

        python -m MI_scaling_non_middle.sliding_window_mi \
            --datasets fer2013_hf --window-sizes 4 --stride 4

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.75

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.5
    '

echo "Started container ${CONTAINER_NAME} using all GPUs on the box."
echo "Follow progress:   docker logs -f ${CONTAINER_NAME}"
echo "Check it's alive:  docker ps"
echo "Results land in:   MI_scaling_non_middle/results/pruning/{lfw_faces,fer2013_hf}_*"
echo "Bar charts:        MI_scaling_non_middle/results/pruning/pruning_comparison_p{75,50}.png"
echo "MI heatmaps:       MI_scaling_non_middle/results/{lfw_faces,fer2013_hf}_sliding_w*_heatmap.png"

# `docker wait` blocks until the container exits and prints its exit code -
# run in its own nohup'd + disowned background process (not just `&`) so it
# keeps running independent of this shell/terminal, exactly like the
# detached container itself. Only commits on a clean (0) exit, so a
# crashed/killed run doesn't get silently pushed as if it succeeded; skips
# the push (not an error) if there's nothing staged, e.g. a rerun where
# every condition's metrics file already existed and was skipped.
nohup bash -c "
    EXIT_CODE=\$(docker wait '${CONTAINER_NAME}')
    cd '${REPO_DIR}'
    if [ \"\${EXIT_CODE}\" != \"0\" ]; then
        echo \"[\$(date)] Container exited \${EXIT_CODE} (failure) - not pushing.\" >> '${PUSH_LOG}'
        exit 0
    fi
    git add MI_scaling_non_middle/results
    if git diff --cached --quiet; then
        echo \"[\$(date)] Training finished but nothing new to commit.\" >> '${PUSH_LOG}'
        exit 0
    fi
    git commit -m 'Redo lfw_faces/fer2013_hf pruning at native resolution with transfer learning' >> '${PUSH_LOG}' 2>&1
    if git push >> '${PUSH_LOG}' 2>&1; then
        echo \"[\$(date)] Pushed lfw_faces/fer2013_hf pruning results.\" >> '${PUSH_LOG}'
    else
        echo \"[\$(date)] git push FAILED - see output above. Commit is local only, fix and push manually.\" >> '${PUSH_LOG}'
    fi
" > /dev/null 2>&1 &
disown
echo "Auto-push watcher started - will commit+push results once training finishes."
echo "Watch it:          tail -f ${PUSH_LOG}"
