#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT=/mnt/afs/250010074/yifan/slai-manipulation
readonly ENVIRONMENT="$PROJECT/.venv-openpi"
readonly CONFIG="$PROJECT/configs/pi05_h100_jax.yaml"
readonly NUM_GPUS="${NUM_GPUS:-4}"
readonly STEPS="${STEPS:-30000}"
readonly BATCH_SIZE="${BATCH_SIZE:-16}"
readonly EXPERIMENT="${NUM_GPUS}gpu-${STEPS}step"

[[ -x "$ENVIRONMENT/bin/python" ]] || {
    echo "Missing persistent OpenPI environment: $ENVIRONMENT" >&2
    exit 1
}

export HOME="$PROJECT/.h100-home"
export OPENPI_DATA_HOME="$PROJECT/.cache/openpi"
export HF_HOME="$PROJECT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT/.cache"
export JAX_COMPILATION_CACHE_DIR="$PROJECT/.cache/jax"
export PYTHONPATH="$PROJECT/src:$PROJECT/third_party/openpi/src:$PROJECT/third_party/openpi/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SLAI_PI05_REEXEC=1

cd "$PROJECT"
actual_gpus="$($ENVIRONMENT/bin/python -c 'import jax; print(jax.device_count())')"
[[ "$actual_gpus" == "$NUM_GPUS" ]] || {
    echo "Expected $NUM_GPUS JAX GPU(s), but the scheduler exposed $actual_gpus" >&2
    exit 1
}

exec "$ENVIRONMENT/bin/python" -m slai_mi.apps.pi05 train \
    "--config=$CONFIG" \
    "--steps=$STEPS" \
    "--batch-size=$BATCH_SIZE" \
    "--experiment=$EXPERIMENT" \
    --execute
