#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT=/mnt/afs/250010074/yifan/slai-manipulation
readonly ENVIRONMENT="$PROJECT/.venv-lerobot-v3"
readonly CONFIG="$PROJECT/outputs/pi05/h100/train.yaml"
readonly NUM_GPUS="${NUM_GPUS:-4}"
readonly STEPS="${STEPS:-30000}"
readonly BATCH_SIZE="${BATCH_SIZE:-4}"
readonly OUTPUT="$PROJECT/outputs/pi05/h100/${NUM_GPUS}gpu-${STEPS}step"

[[ -x "$ENVIRONMENT/bin/accelerate" ]] || {
    echo "Missing persistent H100 environment: $ENVIRONMENT" >&2
    exit 1
}
[[ -f "$CONFIG" ]] || {
    echo "Missing generated training config: $CONFIG" >&2
    exit 1
}
[[ ! -e "$OUTPUT" ]] || {
    echo "Refusing to overwrite existing run: $OUTPUT" >&2
    exit 1
}

export HOME="$PROJECT/.h100-home"
export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$PROJECT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT/.cache"
export UV_CACHE_DIR="$PROJECT/.cache/uv"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline

cd "$PROJECT"
launcher=("$ENVIRONMENT/bin/accelerate" launch --mixed_precision bf16 --num_processes "$NUM_GPUS")
if (( NUM_GPUS > 1 )); then
    launcher+=(--multi_gpu)
fi

exec "${launcher[@]}" -m slai_mi.training.lerobot_pi05 \
    "--config_path=$CONFIG" \
    "--steps=$STEPS" \
    "--batch_size=$BATCH_SIZE" \
    "--output_dir=$OUTPUT"
