#!/usr/bin/env bash
set -euo pipefail

# Runs the same MI-guided pixel-pruning experiment already done for
# mnist/cifar10 (MI_scaling_non_middle/train_pruned_classifiers.py) on
# lfw_faces and fer2013_hf, on the Lambda GPU box. Two steps per dataset:
#
#   1. sliding_window_mi.py's window=3/stride=3 sweep - the per-tile MI
#      ranking pixel_pruning.py's build_tile_mask needs. mnist/cifar10
#      already had this; lfw_faces/fer2013_hf don't yet.
#   2. train_pruned_classifiers.py itself, at both percent_kept values
#      already used for mnist/cifar10 (0.75, 0.50), so results/pruning/
#      ends up with the same dataset x condition x percent_kept grid for
#      all four datasets.
#
# Runs inside tensorflow/tensorflow:2.17.0-gpu (this box has no sudo/venv
# and system Python is too old for tensorflow>=2.17 - see memory), with a
# libcuda symlink fix for this host's driver, same as every other GPU job
# on this box.
#
# Usage:
#   ./run_pruning_lfw_fer2013.sh <gpu-device-index>
#   e.g. ./run_pruning_lfw_fer2013.sh 2
#
# Check `nvidia-smi` first and pick a device that's actually free - this is
# a shared box. Runs detached (docker -d), so it survives disconnecting;
# follow along with `docker logs -f mi-pruning-lfw-fer2013`.

GPU_DEVICE="${1:?usage: $0 <gpu-device-index>  (run nvidia-smi first to find a free one)}"
CONTAINER_NAME="mi-pruning-lfw-fer2013"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "A container named ${CONTAINER_NAME} already exists (docker ps -a)." >&2
    echo "Remove it first if you want to start fresh: docker rm -f ${CONTAINER_NAME}" >&2
    exit 1
fi

docker run -d --gpus "\"device=${GPU_DEVICE}\"" --name "${CONTAINER_NAME}" \
    -v "${REPO_DIR}:/workspace" -w /workspace \
    tensorflow/tensorflow:2.17.0-gpu bash -c '
        set -euo pipefail
        mkdir -p /tmp/fixed-cuda
        # Symlink whatever libcuda.so.* the host actually has, rather than
        # hardcoding a driver version - this box'"'"'s driver has moved before.
        CUDA_LIB="$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)"
        if [ -z "${CUDA_LIB}" ]; then
            echo "No /usr/lib/x86_64-linux-gnu/libcuda.so.* found - is this actually a GPU host?" >&2
            exit 1
        fi
        ln -sf "${CUDA_LIB}" /tmp/fixed-cuda/libcuda.so.1
        ln -sf "${CUDA_LIB}" /tmp/fixed-cuda/libcuda.so
        export LD_LIBRARY_PATH="/tmp/fixed-cuda:${LD_LIBRARY_PATH:-}"

        pip install --quiet matplotlib scikit-learn datasets pillow

        python -c "import tensorflow as tf; print(\"GPUs visible:\", tf.config.list_physical_devices(\"GPU\"))"

        python -m MI_scaling_non_middle.sliding_window_mi \
            --datasets lfw_faces,fer2013_hf --window-sizes 3

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.75

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.5
    '

echo "Started container ${CONTAINER_NAME} on GPU device ${GPU_DEVICE}."
echo "Follow progress:   docker logs -f ${CONTAINER_NAME}"
echo "Check it's alive:  docker ps"
echo "Results land in:   MI_scaling_non_middle/results/pruning/{lfw_faces,fer2013_hf}_*"
