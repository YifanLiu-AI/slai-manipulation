#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:?MODE must be smoke or formal}"
TASK="${TASK:?TASK is required}"
NAMESPACE="${MODELSCOPE_NAMESPACE:?MODELSCOPE_NAMESPACE is required}"
DATASET_NAME="${DATASET_NAME:-tmp-${TASK//_/-}-dataset}"
DATASET_DIR="${DATASET_DIR:-$PROJECT/data/lerobot/$DATASET_NAME}"
CONFIG="${CONFIG:-configs/pi05_task2_9dof.yaml}"
RUN_ID="${RUN_ID:-${TASK}-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_WORKERS="${MAX_WORKERS:-8}"
RANGE_WORKERS="${RANGE_WORKERS:-4}"

case "$MODE" in
  smoke)
    NUM_GPUS=1
    STEPS=1
    ;;
  formal)
    NUM_GPUS="${NUM_GPUS:-4}"
    STEPS="${STEPS:-30000}"
    [[ "$NUM_GPUS" == 4 ]] || { echo "Formal training requires exactly 4 GPUs" >&2; exit 2; }
    ;;
  *) echo "MODE must be smoke or formal" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-1}"

cd "$PROJECT"
export MODELSCOPE_ENDPOINT="https://modelscope.cn"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

./.venv-lerobot-v3/bin/python -m slai_mi.apps.modelscope_transfer dataset-download \
  --task "$TASK" --namespace "$NAMESPACE" --repo-name "$DATASET_NAME" \
  --target "$DATASET_DIR" --max-workers "$MAX_WORKERS" --range-workers "$RANGE_WORKERS" --execute

./.venv-lerobot-v3/bin/python deploy/train-pi0.5.py \
  --mode "$MODE" --source "$DATASET_DIR" --config "$CONFIG" --run-id "$RUN_ID" \
  --num-gpus "$NUM_GPUS" --steps "$STEPS" --batch-size "$BATCH_SIZE" --execute

if [[ "$MODE" == formal ]]; then
  checkpoint="$(find "$PROJECT/outputs/pi05/$RUN_ID/full/checkpoints" -mindepth 2 -maxdepth 2 -type d -name pretrained_model -printf '%p\n' | sort -V | tail -1)"
  [[ -n "$checkpoint" ]] || { echo "Formal training produced no checkpoint" >&2; exit 1; }
  ./.venv-lerobot-v3/bin/python -m slai_mi.apps.modelscope_transfer checkpoint-upload \
    --task "$TASK" --namespace "$NAMESPACE" --source "$checkpoint" \
    --commit-message "formal 4xH100 $RUN_ID" --execute
fi
