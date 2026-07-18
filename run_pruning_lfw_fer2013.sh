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

        python -m MI_scaling_non_middle.sliding_window_mi \
            --datasets lfw_faces,fer2013_hf --window-sizes 3

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.75

        python -m MI_scaling_non_middle.train_pruned_classifiers \
            --datasets lfw_faces,fer2013_hf --percent-kept 0.5
    '

echo "Started container ${CONTAINER_NAME} using all GPUs on the box."
echo "Follow progress:   docker logs -f ${CONTAINER_NAME}"
echo "Check it's alive:  docker ps"
echo "Results land in:   MI_scaling_non_middle/results/pruning/{lfw_faces,fer2013_hf}_*"

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
    git commit -m 'Add lfw_faces/fer2013_hf pruning results' >> '${PUSH_LOG}' 2>&1
    if git push >> '${PUSH_LOG}' 2>&1; then
        echo \"[\$(date)] Pushed lfw_faces/fer2013_hf pruning results.\" >> '${PUSH_LOG}'
    else
        echo \"[\$(date)] git push FAILED - see output above. Commit is local only, fix and push manually.\" >> '${PUSH_LOG}'
    fi
" > /dev/null 2>&1 &
disown
echo "Auto-push watcher started - will commit+push results once training finishes."
echo "Watch it:          tail -f ${PUSH_LOG}"
