#!/usr/bin/env bash
set -euo pipefail

# This is the single startup command shared by a preconfigured CCI App and
# copied ACP jobs. CCI must call it with --cci; formal ACP calls it without it.
MODE=formal
if [[ "${1:-}" == --cci ]]; then
  MODE=smoke
  shift
fi
[[ $# == 0 ]] || { echo "usage: sensecore_start.sh [--cci]" >&2; exit 2; }

readonly PROJECT="${PROJECT:-/mnt/afs/250010074/yifan/slai-manipulation}"
readonly ENV_FILE="${SLAI_SENSECORE_ENV:-$PROJECT/.secrets/modelscope.env}"
[[ -f "$ENV_FILE" ]] || {
  echo "Missing private SenseCore environment file: $ENV_FILE" >&2
  echo "It must define MODELSCOPE_NAMESPACE and may override TASK/CONFIG/STEPS/BATCH_SIZE." >&2
  exit 2
}
# This file is stored on shared AFS, never committed. ModelScope credentials
# themselves should be established interactively with ms-hub login.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export PROJECT MODE
if [[ "$MODE" == smoke ]]; then
  export NUM_GPUS=1 STEPS=1 BATCH_SIZE=1
else
  export NUM_GPUS=4
  export STEPS="${STEPS:-30000}"
  export BATCH_SIZE="${BATCH_SIZE:-1}"
fi

cd "$PROJECT"
exec bash deploy/cci_h100_modelscope_pipeline.sh
