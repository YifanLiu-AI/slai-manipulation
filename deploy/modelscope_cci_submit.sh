#!/usr/bin/env bash
set -euo pipefail

readonly LOCAL_PROJECT="${LOCAL_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly LOCAL_ENV="${SLAI_LOCAL_ENV:-$LOCAL_PROJECT/.env}"
if [[ -f "$LOCAL_ENV" ]]; then
  env_mode="$(stat -c '%a' "$LOCAL_ENV")"
  [[ "$env_mode" == 600 || "$env_mode" == 400 ]] || {
    echo "Refusing insecure credential file mode $env_mode: $LOCAL_ENV" >&2
    exit 2
  }
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_ENV"
  set +a
fi

ACTION="${1:-}"
shift || true
case "$ACTION" in
  cci-start|cci-stop|formal) ;;
  *)
    echo "usage: deploy/modelscope_cci_submit.sh cci-start|cci-stop|formal" >&2
    exit 2
    ;;
esac

readonly SUBSCRIPTION="${SCO_SUBSCRIPTION:-0197ee17-b6eb-7846-b2b4-a77c5f509b92}"
readonly WORKSPACE="${SCO_WORKSPACE:-share-space}"
readonly REGION="${SCO_REGION:-cn-sh-01}"
readonly CCI_APP="${SCO_CCI_APP:-app-o56pqogh}"
readonly ACP_TEMPLATE="${SCO_ACP_TEMPLATE:-lyf-challenge-4}"
readonly PROJECT="${REMOTE_PROJECT:-/mnt/afs/250010074/yifan/slai-manipulation}"
readonly TASK="${TASK:-task2}"
readonly START_SCRIPT="$PROJECT/deploy/sensecore_start.sh"
readonly JOB_NAME="${JOB_NAME:-${TASK}-pi05-4gpu-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$ACTION" in
  cci-start)
    command=(sco cci apps start "$CCI_APP" --workspace-name "$WORKSPACE" --region "$REGION" --subscription "$SUBSCRIPTION")
    purpose="Start the preconfigured CCI App. Its saved command must be: exec bash $START_SCRIPT --cci"
    ;;
  cci-stop)
    command=(sco cci apps stop "$CCI_APP" --workspace-name "$WORKSPACE" --region "$REGION" --subscription "$SUBSCRIPTION")
    purpose="Stop the CCI debug App"
    ;;
  formal)
    start_command="exec bash $START_SCRIPT"
    command=(sco acp jobs copy --workspace-name "$WORKSPACE" --copy-job-name "$ACP_TEMPLATE" --job-name "$JOB_NAME" --command="$start_command" --region "$REGION" --subscription "$SUBSCRIPTION")
    purpose="Copy the four-GPU ACP template and run formal training (no --cci flag)"
    ;;
esac

printf 'action=%s\npurpose=%s\nsubscription=%s\nworkspace=%s\nregion=%s\n' \
  "$ACTION" "$purpose" "$SUBSCRIPTION" "$WORKSPACE" "$REGION"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${EXECUTE:-0}" != 1 ]]; then
  echo "dry-run: run sco init && sco doctor, then set EXECUTE=1"
  exit 0
fi
command -v sco >/dev/null || { echo "sco is not installed or not in PATH" >&2; exit 2; }
sco doctor
exec "${command[@]}"
