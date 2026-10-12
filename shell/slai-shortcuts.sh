# SLaI commands. Every runner works from any current directory.
export SLAI_ROOT=/home/user/shiyi/slai-manipulation

_srun() (
    cd "$SLAI_ROOT" || exit
    uv run "$@"
)

_slai_python() {
    if [ -n "${SLAI_PYTHON:-}" ] && [ -x "$SLAI_PYTHON" ]; then
        printf '%s\n' "$SLAI_PYTHON"
    elif [ -x "$SLAI_ROOT/.venv-lerobot-v3/bin/python" ]; then
        printf '%s\n' "$SLAI_ROOT/.venv-lerobot-v3/bin/python"
    elif [ -x "$SLAI_ROOT/.venv/bin/python" ]; then
        printf '%s\n' "$SLAI_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        command -v python
    fi
}

_scollect() (
    cd "$SLAI_ROOT" || exit
    collect_python="$SLAI_ROOT/.venv-lerobot-v3/bin/python"
    if [ ! -x "$collect_python" ]; then
        printf '采集环境不存在: %s\n' "$collect_python" >&2
        exit 1
    fi
    if ! "$collect_python" -c 'import lerobot, slai_mi' >/dev/null 2>&1; then
        printf '采集环境缺少 lerobot/slai_mi: %s\n' "$collect_python" >&2
        exit 1
    fi
    exec "$collect_python" -m slai_mi.apps.collect_real "$@"
)

s() { cd "$SLAI_ROOT" || return; }

sr()   { _srun slai-teleop-real "$@"; }
srx()  { _srun slai-teleop-real "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }
sim()  { _srun slai-teleop-sim "$@"; }
simx() { _srun slai-teleop-sim "$@" --run; }

sc()   { _scollect "$@"; }
scw()  {
    _scollect \
        --strategy ur5e_wrist_8dof_collection \
        --task configs/tasks/block_into_box.yaml \
        "$@"
}
sc9()  {
    _scollect \
        --strategy ur5e_wrist_gripper_9dof_collection \
        --task configs/tasks/task2.yaml \
        "$@"
}
scx()  {
    local arg episode_mode=
    for arg in "$@"; do
        case "$arg" in
            --episodes|--episodes=*|--continuous) episode_mode=1 ;;
        esac
    done
    if [ -n "$episode_mode" ]; then
        _scollect "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    else
        _scollect "$@" --continuous --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    fi
}
scc()  { _scollect "$@" --continuous --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }
sc9x()  (
    # The read-only preview and real collector share RealSense devices and
    # dashboard port 8765. Hand ownership over explicitly and restore preview
    # after collection exits, including failures and Ctrl-C.
    restore_ui=
    if systemctl --user is-active --quiet slai-collection-ui.service; then
        systemctl --user stop slai-collection-ui.service || exit
        restore_ui=1
    fi
    trap 'if [ -n "$restore_ui" ]; then systemctl --user start slai-collection-ui.service; fi' EXIT

    episode_mode=
    for arg in "$@"; do
        case "$arg" in
            --episodes|--episodes=*|--continuous) episode_mode=1 ;;
        esac
    done
    if [ -n "$episode_mode" ]; then
        _scollect \
            --strategy ur5e_wrist_gripper_9dof_collection \
            --task configs/tasks/task2.yaml \
            "$@" \
            --execute-real \
            --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    else
        _scollect \
            --strategy ur5e_wrist_gripper_9dof_collection \
            --task configs/tasks/task2.yaml \
            --continuous \
            "$@" \
            --execute-real \
            --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    fi
)
scwx() (
    restore_ui=
    if systemctl --user is-active --quiet slai-collection-ui.service; then
        systemctl --user stop slai-collection-ui.service || exit
        restore_ui=1
    fi
    trap 'if [ -n "$restore_ui" ]; then systemctl --user start slai-collection-ui.service; fi' EXIT
    episode_args=--continuous
    for arg in "$@"; do
        case "$arg" in
            --episodes|--episodes=*|--continuous) episode_args= ;;
        esac
    done
    _scollect \
        --strategy ur5e_wrist_gripper_9dof_collection \
        --task configs/tasks/task2.yaml \
        ${episode_args:+$episode_args} \
        "$@" \
        --execute-real \
        --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
)
scs()  { _srun slai-collect-sim "$@"; }
scsx() { _srun slai-collect-sim "$@" --run; }

su()   { _srun slai-collection-ui "$@"; }
sul()  { _srun slai-collection-ui "$@" --live; }
st()   { _srun slai-train "$@"; }
si()   { _srun slai-mi-infer "$@"; }
se()   { _srun slai-evaluate "$@"; }
sp()   { _srun slai-pi05 "$@"; }
sd()   { _srun slai-deploy-real "$@"; }
sdx()  { _srun slai-deploy-real "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }

sph()  { _srun --extra pose-hub slai-pose-hub "$@"; }
sphx() { _srun --extra pose-hub slai-pose-hub "$@" --run; }
spb()  { _srun --extra pose-hub slai-pose-hub-bridge "$@"; }
spbx() { _srun --extra pose-hub slai-pose-hub-bridge "$@" --run; }

sv()   { _srun pytest "$@"; }
sl()   { _srun ruff check . "$@"; }
rec()  (
    cd "$SLAI_ROOT" || exit
    # Read-only TASK2 9DoF snapshot; never homes or commands hardware.
    # Resolve the interpreter from the environment, not from the caller's cwd.
    task2_python="$(_slai_python)" || exit 1
    if ! PYTHONPATH="$SLAI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$task2_python" -m slai_mi.apps.task2_state "$@"; then
        printf 'rec 需要可导入 pyserial、PyYAML、numpy 的 Python 环境；可设置 SLAI_PYTHON=/path/to/python\n' >&2
        exit 1
    fi
)
wuji-check() (
    cd "$SLAI_ROOT" || exit
    ./.venv/bin/python -m slai_mi.apps.check_wuji_retarget "$@" --run
)
wrist-park() (
    cd "$SLAI_ROOT" || exit
    uv run slai-park-wrist "$@"
)
Home() (
    exec "$SLAI_ROOT/shell/Home" "$@"
)

export SLAI_JAX_ROOT=/home/user/shiyi/Sensecore_H100
# Enable grasp-gated wrist jump protection by default. Set to 0 only for
# deliberate diagnostic runs.
export SLAI_JAX_WRIST_JUMP_PROTECT="${SLAI_JAX_WRIST_JUMP_PROTECT:-1}"
# lark-cli is installed with the pi Lark package but is not always on PATH
# in a plain conda shell.  Keep Feishu episode sync available to all JAX IDs.
export PATH="/home/user/.pi/agent/npm/node_modules/@tunglam/pi-lark-cli/bin:$PATH"

_slai_jax_task2() (
    mode="${1:-}"
    shift || exit 2
    # Continuous TASK2 rollouts are capped at 1500 control steps (about
    # 100 seconds at 15 Hz). Keep this fixed for jax-slow, jax-normal, and
    # every generated Pxx-JAXx-01 shortcut.
    export SLAI_JAX_MAX_STEPS=1500
    dof=2dof
    dry_run=
    episode_id=
    while [ "$#" -gt 0 ]; do
        case "$1" in
            2dof|3dof) dof="$1" ;;
            --dry-run) dry_run=1 ;;
            -P?*) episode_id="${1#-P}" ;;
            *) printf '用法: jax-home [--dry-run] | jax-slow|jax-normal [2dof|3dof] [-P<EpisodeID>] [--dry-run]\n' >&2; exit 2 ;;
        esac
        shift
    done
    if [ -n "$episode_id" ]; then
        case "$episode_id" in
            P[0-9][0-9]-JAX2-01|P[0-9][0-9]-JAX3-01) ;;
            *) printf '无效 Episode ID: %s（必须是 P01-P99-JAX2/3-01）\n' "$episode_id" >&2; exit 2 ;;
        esac
        episode_position="${episode_id%%-*}"
        episode_number=$((10#${episode_position#P}))
        if [ "$episode_number" -lt 1 ] || [ "$episode_number" -gt 99 ]; then
            printf '无效 Episode 位置: %s（允许 P01-P99）\n' "$episode_position" >&2
            exit 2
        fi
        episode_variant="${episode_id#*-}"
        episode_variant="${episode_variant%%-*}"
        expected_variant="JAX${dof%dof}"
        if [ "$episode_variant" != "$expected_variant" ]; then
            printf 'Episode 与 DoF 不匹配: %s 使用 %s\n' "$episode_id" "$dof" >&2
            exit 2
        fi
    fi
    case "$mode" in
        home|slow) task=task2_continuous_jax_ultraslow.yaml ;;
        normal) task=task2_continuous_jax_normal.yaml ;;
        *) printf '未知 JAX TASK2 模式: %s\n' "$mode" >&2; exit 2 ;;
    esac

    if pgrep -f '[s]lai_mi.apps.collect_real.*--execute-real' >/dev/null; then
        printf '正在安全释放采集控制权，保留只读相机监控供 JAX 使用。\n' >&2
        systemctl --user stop slai-wrist-collection.service 2>/dev/null || true
        sleep 3
    fi
    # JAX owns UR5/wrist/gripper; the read-only UI owns camera streaming.
    if ! systemctl --user is-active --quiet slai-collection-ui.service; then
        systemctl --user start slai-collection-ui.service 2>/dev/null || true
    fi
    camera_ready=
    for _camera_wait in $(seq 1 20); do
        if curl -fsS --max-time 1 http://127.0.0.1:8765/api/cameras >/dev/null 2>&1; then
            camera_ready=1
            break
        fi
        sleep 1
    done
    if [ -z "$camera_ready" ]; then
        printf 'JAX_CAMERA_PRECHECK_FAILED endpoint=http://127.0.0.1:8765/api/cameras\n' >&2
        exit 1
    fi

    vendor=/home/user/haitao_files/robowrist/wrist_teleop_firmaware/02_Python_Client_CLI
    export SLAI_PROJECT_ROOT="$SLAI_JAX_ROOT"
    openpi_client=/home/user/shiyi/slai-manipulation/third_party/openpi/packages/openpi-client/src
    export PYTHONPATH="$SLAI_JAX_ROOT/src:$openpi_client:$vendor:$vendor/open_loop_record${PYTHONPATH:+:$PYTHONPATH}"
    real_args=()
    if [ -z "$dry_run" ]; then
        real_args=(--execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION)
        gripper_port="${SLAI_JAX_GRIPPER_PORT:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}"
        # CH340 adapters can take a few seconds to recreate their by-id link
        # after a USB bus reset. Wait for enumeration before HOME opens the
        # port, but never substitute another tty or reconnect an active device.
        gripper_ready_streak=0
        for _gripper_wait in $(seq 1 15); do
            if [ -e "$gripper_port" ]; then
                gripper_ready_streak=$((gripper_ready_streak + 1))
                [ "$gripper_ready_streak" -ge 2 ] && break
            else
                gripper_ready_streak=0
            fi
            udevadm settle --timeout=1 2>/dev/null || true
            sleep 1
        done
        if [ ! -e "$gripper_port" ]; then
            printf 'JAX_GRIPPER_PORT_MISSING port=%s; reconnect the Hiwonder USB adapter before starting HOME\n' "$gripper_port" >&2
            exit 1
        fi
        printf 'JAX_GRIPPER_PORT_READY port=%s\n' "$gripper_port"
    fi

    "$SLAI_JAX_ROOT/.venv/bin/python" \
        -P \
        -m slai_mi.apps.task2_home \
        --hardware-config "$SLAI_JAX_ROOT/configs/hardware.yaml" \
        --task "$SLAI_JAX_ROOT/configs/tasks/$task" \
        --standalone \
        --timeout 30 \
        "${real_args[@]}" || exit

    [ "$mode" = home ] && exit 0
    case "$dof" in
        2dof) inference_config="$SLAI_JAX_ROOT/configs/pi05_jax_2dof_32ep_continuous_refine_10k_inference.yaml" ;;
        3dof) inference_config="$SLAI_JAX_ROOT/configs/pi05_jax_3dof_30ep_continuous_refine_10k_inference.yaml" ;;
    esac
    # All TASK2 episode IDs and both jax-slow/jax-normal modes use the
    # original validated 10k checkpoint for this rollout phase.
    printf 'JAX_POLICY_VARIANT=%s episode=%s profile=validated_10k config=%s\n' "$dof" "${episode_id:-none}" "$inference_config"
    printf 'JAX_ROLLOUT_STEP_LIMIT=%s\n' "$SLAI_JAX_MAX_STEPS"

    _feishu_warn() { printf 'FEISHU_WARN %s\n' "$*" >&2; }
    _feishu_mark_failed() {
        local record_id="${SLAI_FEISHU_RECORD_ID:-}" payload
        [ -n "$record_id" ] || return 0
        payload=$(jq -nc --arg rid "$record_id" --arg note "$1" \
            '{update_records:{($rid):{"上传状态":["失败"],"备注":$note}}}') || return 0
        lark-cli base +record-batch-update --base-token PB8ibt4lHaPdQSsbIl0crUq6nFb \
            --table-id tblYdIji2EnsQkof --json "$payload" --as user >/dev/null 2>&1 || true
    }
    _feishu_prepare() {
        [ -n "$episode_id" ] || return 0
        command -v lark-cli >/dev/null 2>&1 || { _feishu_warn lark_cli_missing; return 1; }
        local base=PB8ibt4lHaPdQSsbIl0crUq6nFb table=tblYdIji2EnsQkof
        local search record_id payload result_json
        search=$(lark-cli base +record-search --base-token "$base" --table-id "$table" \
            --keyword "$episode_id" --search-field 'Episode ID' --field-id 'Episode ID' \
            --format json --as user 2>/dev/null) || { _feishu_warn search_failed; return 1; }
        record_id=$(printf '%s' "$search" | jq -r '.data.record_id_list[0] // empty' 2>/dev/null)
        if [ -z "$record_id" ]; then
            payload=$(jq -nc --arg id "$episode_id" --arg cmd "jax-$dof -P$episode_id" \
                '{create_records:[{"Episode ID":$id,"运行命令":$cmd,"运行状态":["运行中"]}]}')
            result_json=$(lark-cli base +record-batch-create --base-token "$base" --table-id "$table" \
                --json "$payload" --as user 2>/dev/null) || { _feishu_warn create_failed; return 1; }
            record_id=$(printf '%s' "$result_json" | jq -r '.data.record_id_list[0] // empty' 2>/dev/null)
        fi
        [ -n "$record_id" ] || { _feishu_warn record_id_missing; return 1; }
        export SLAI_FEISHU_RECORD_ID="$record_id"
        printf 'FEISHU_RECORD id=%s\n' "$record_id"
        payload=$(jq -nc --arg rid "$record_id" --arg cmd "jax-$dof -P$episode_id" \
            '{update_records:{($rid):{"运行命令":$cmd,"运行状态":["运行中"],"上传状态":["上传中"]}}}') || return 1
        lark-cli base +record-batch-update --base-token "$base" --table-id "$table" \
            --json "$payload" --as user >/dev/null 2>&1 || { _feishu_warn record_prepare_update_failed; return 1; }
    }
    _feishu_sync() {
        [ -n "$episode_id" ] || return 0
        command -v lark-cli >/dev/null 2>&1 || { _feishu_warn lark_cli_missing; return 1; }
        local base=PB8ibt4lHaPdQSsbIl0crUq6nFb table=tblYdIji2EnsQkof
        local record_id payload log_file photo video gif content_ok readback upload_output
        record_id="${SLAI_FEISHU_RECORD_ID:-}"
        [ -n "$record_id" ] || { _feishu_warn record_not_bound; return 0; }
        log_file=$(mktemp)
        # The caller places the command output in this file before this sync hook.
        if [ -n "${SLAI_JAX_LOG_FILE:-}" ] && [ -f "$SLAI_JAX_LOG_FILE" ]; then log_file="$SLAI_JAX_LOG_FILE"; fi
        local step runtime start end run_started
        # Parse only canonical rollout telemetry. Generic s=/t= matching is
        # unsafe because unrelated diagnostics can contain those fragments.
        step=$(grep -oE 'STEP [0-9]+' "$log_file" 2>/dev/null | tail -1 | awk '{print $2}')
        if [ -z "$step" ]; then
            step=$(grep -oE 'JAX_ROLLOUT_LIMIT_REACHED steps=[0-9]+' "$log_file" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        fi
        step=${step:-0}
        runtime=$(awk '/^STEP [0-9]+ dt=[[:space:]]*[0-9.]+ms/{gsub(/^STEP [0-9]+ dt=[[:space:]]*/, ""); gsub(/ms.*/, ""); total += $1} END{if (total > 0) printf "%.3f", total/1000}' "$log_file")
        # Prefer the actual OBS recording interval when available; this is the
        # user-facing inference/episode duration, not startup or HOME time.
        obs_start=$(grep -oE 'OBS_RECORD_START epoch=[0-9.]+' "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*=//')
        obs_stop=$(grep -oE 'OBS_RECORD_STOP epoch=[0-9.]+' "$log_file" 2>/dev/null | tail -1 | sed -E 's/.*=//')
        if [ -n "$obs_start" ] && [ -n "$obs_stop" ]; then
            runtime=$(awk -v a="$obs_start" -v b="$obs_stop" 'BEGIN {printf "%.3f", b-a}')
        fi
        runtime=${runtime:-0}
        # Use the paths emitted by this invocation.  Searching by mtime can
        # mix a stale photo with a new recording, especially after a retry.
        photo=$(grep -oE 'HOME_PHOTO_OK path=[^[:space:]]+' "$log_file" 2>/dev/null | tail -1 | sed 's/^.*path=//')
        video=$(grep -oE 'OBS_RECORD_OK video=[^[:space:]]+' "$log_file" 2>/dev/null | tail -1 | sed 's/^.*video=//')
        gif=$(grep -oE 'OBS_RECORD_OK video=[^[:space:]]+ gif=[^[:space:]]+' "$log_file" 2>/dev/null | tail -1 | sed -E 's/^.* gif=//')
        content_ok=$(grep -oE 'OBS_RECORD_CONTENT_OK video=[^[:space:]]+' "$log_file" 2>/dev/null | tail -1 | sed 's/^.*video=//')
        # Older deploy runs may have written the MP4 before GIF conversion
        # failed, so recover the exact newest recording directory and retry the
        # deterministic conversion before declaring the upload unavailable.
        run_started="${SLAI_JAX_RUN_STARTED_EPOCH:-0}"
        if [ -z "$video" ]; then
            video=$(find "$SLAI_JAX_ROOT/outputs/pi05/jax/home-photos" -type f -name episode.mp4 \
                -newermt "@$run_started" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        fi
        if [ -n "$video" ] && [ -z "$gif" ]; then
            gif="${video%/*}/episode.gif"
            if [ ! -s "$gif" ]; then
                if ffmpeg -y -i "$video" -vf 'fps=10,scale=640:-1:flags=lanczos' -loop 0 "$gif" >/dev/null 2>&1; then
                    printf 'FEISHU_GIF_REGENERATED path=%s\n' "$gif"
                else
                    gif=''
                fi
            fi
        fi
        [ -s "$photo" ] || { _feishu_warn home_photo_missing; _feishu_mark_failed "${episode_id} 位置照片缺失"; return 1; }
        [ -n "$content_ok" ] || { _feishu_warn video_content_unverified; _feishu_mark_failed "${episode_id} 视频内容未通过验收"; return 1; }
        [ -s "$video" ] || { _feishu_warn video_missing; _feishu_mark_failed "${episode_id} 视频缺失"; return 1; }
        [ -s "$gif" ] || { _feishu_warn gif_missing; _feishu_mark_failed "${episode_id} GIF缺失"; return 1; }

        _feishu_clear_attachments() {
            local field="$1" data token
            data=$(lark-cli base +record-get --base-token "$base" --table-id "$table" --record-id "$record_id" --field-id "$field" --format json --as user) || return 1
            while IFS= read -r token; do
                [ -n "$token" ] || continue
                lark-cli base +record-remove-attachment --base-token "$base" --table-id "$table" --record-id "$record_id" --field-id "$field" --file-token "$token" --yes --as user >/dev/null 2>&1 || return 1
            done < <(printf '%s' "$data" | jq -r '.. | objects | .file_token? // empty')
        }
        _feishu_clear_attachments fldkorMSGF || { _feishu_warn position_clear_failed; _feishu_mark_failed "${episode_id} 位置附件清理失败"; return 1; }
        _feishu_clear_attachments fld7Z7YP0t || { _feishu_warn video_clear_failed; _feishu_mark_failed "${episode_id} 视频附件清理失败"; return 1; }
        start=$(date -Iseconds); end=$(date -Iseconds)
        payload=$(jq -nc --arg rid "$record_id" --arg cmd "jax-$dof -P$episode_id" --arg cp "$inference_config" \
            --arg started_at "$start" --arg finished_at "$end" --arg rt "$runtime" --arg step "$step" \
            '({"update_records":{}} | .update_records[$rid] = {"运行命令":$cmd,"Checkpoint 路径":$cp,"执行时间":$started_at,"录制开始时间":$started_at,"录制结束时间":$finished_at,"Episode 时长(秒)":($rt|tonumber),"Episode Step":($step|tonumber),"Episode 时间与步长":("JAX / 15Hz / "+$step+" steps / "+$rt+"s"),"运行状态":["待判定"]})' )
        lark-cli base +record-batch-update --base-token "$base" --table-id "$table" --json "$payload" --as user >/dev/null 2>&1 || { _feishu_warn update_failed; _feishu_mark_failed "${episode_id} 记录字段更新失败"; return 1; }
        # lark-cli deliberately rejects absolute --file paths.  Run from the
        # project root and pass safe relative paths for every attachment.
        local upload_root="$SLAI_JAX_ROOT"
        local photo_rel="${photo#"$upload_root/"}" video_rel="${video#"$upload_root/"}" gif_rel="${gif#"$upload_root/"}"
        (cd "$upload_root" && lark-cli base +record-upload-attachment --base-token "$base" --table-id "$table" --record-id "$record_id" --field-id fldkorMSGF --file "$photo_rel" --as user >/dev/null 2>&1) || { _feishu_warn photo_upload_failed; _feishu_mark_failed "${episode_id} 位置照片上传失败"; return 1; }
        printf 'FEISHU_UPLOAD_OK episode=%s kind=photo file=%s\n' "$episode_id" "$(basename "$photo")"
        (cd "$upload_root" && lark-cli base +record-upload-attachment --base-token "$base" --table-id "$table" --record-id "$record_id" --field-id fld7Z7YP0t --file "$video_rel" --as user >/dev/null 2>&1) || { _feishu_warn video_upload_failed; _feishu_mark_failed "${episode_id} MP4上传失败"; return 1; }
        printf 'FEISHU_UPLOAD_OK episode=%s kind=video file=%s\n' "$episode_id" "$(basename "$video")"
        (cd "$upload_root" && lark-cli base +record-upload-attachment --base-token "$base" --table-id "$table" --record-id "$record_id" --field-id fld7Z7YP0t --file "$gif_rel" --as user >/dev/null 2>&1) || { _feishu_warn gif_upload_failed; _feishu_mark_failed "${episode_id} GIF上传失败"; return 1; }
        printf 'FEISHU_UPLOAD_OK episode=%s kind=gif file=%s\n' "$episode_id" "$(basename "$gif")"
        payload=$(jq -nc --arg rid "$record_id" --arg status "成功" '{update_records:{($rid):{"上传状态":[$status]}}}')
        lark-cli base +record-batch-update --base-token "$base" --table-id "$table" --json "$payload" --as user >/dev/null 2>&1 || { _feishu_warn upload_status_update_failed; _feishu_mark_failed "${episode_id} 上传状态写入失败"; return 1; }
        readback=$(lark-cli base +record-get --base-token "$base" --table-id "$table" --record-id "$record_id" --format json --as user) || { _feishu_warn readback_failed; _feishu_mark_failed "${episode_id} 上传回读失败"; return 1; }
        printf '%s' "$readback" | jq -e --arg video "$(basename "$video")" --arg gif "$(basename "$gif")" --arg photo "$(basename "$photo")" '(.data.data[0] | tostring) | contains($video) and contains($gif) and contains($photo)' >/dev/null || { _feishu_warn readback_artifact_mismatch; _feishu_mark_failed "${episode_id} 上传回读缺少本轮附件"; return 1; }
    }
    if [ -n "$episode_id" ] && [ -z "$dry_run" ]; then
        _feishu_prepare || return 1
        export SLAI_JAX_LOG_FILE="$(mktemp)"
        export SLAI_JAX_RUN_STARTED_EPOCH="$(date +%s)"
        _feishu_on_exit() {
            local command_rc=$? sync_rc
            _feishu_sync || sync_rc=$?
            if [ "$command_rc" -ne 0 ]; then return "$command_rc"; fi
            return "${sync_rc:-0}"
        }
        trap '_feishu_on_exit' EXIT
        exec > >(tee -a "$SLAI_JAX_LOG_FILE") 2>&1
    fi
    /home/user/shiyi/slai-manipulation/third_party/openpi/.venv/bin/python \
        -P \
        -m slai_mi.apps.deploy_real_jax \
        --config "$inference_config" \
        --hardware-config "$SLAI_JAX_ROOT/configs/hardware.yaml" \
        --task "$SLAI_JAX_ROOT/configs/tasks/$task" \
        --max-steps 0 \
        --gripper-progress \
        "${real_args[@]}"
)

jax-home() { _slai_jax_task2 home "$@"; }
jax-slow() { _slai_jax_task2 slow "$@"; }
jax-normal() { _slai_jax_task2 normal "$@"; }

# TASK2 episode shortcuts.  The episode ID is the public command name and
# dispatches to the corresponding normal-speed JAX variant, e.g.
#   P01-JAX3-01  ==  jax-normal 3dof -PP01-JAX3-01
# Keep these as shell functions so they work from any directory and inherit
# the same safety checks, HOME sequence, Feishu binding, and cleanup as the
# canonical jax-normal command.
for _slai_index in $(seq 1 99); do
    printf -v _slai_position '%02d' "$_slai_index"
    for _slai_dof in 2 3; do
        _slai_episode="P${_slai_position}-JAX${_slai_dof}-01"
        eval "${_slai_episode}() { jax-normal ${_slai_dof}dof -P${_slai_episode} \"\$@\"; }"
    done
done
unset _slai_index _slai_position _slai_dof _slai_episode

# Backwards-compatible shortcut from the previous setup.
sm() {
    case "${1:-}" in
        collect) shift; scc "$@" ;;
        *) printf '用法: sm collect\n' >&2; return 2 ;;
    esac
}
