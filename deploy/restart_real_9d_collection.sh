#!/usr/bin/env bash
set -euo pipefail

SERVICE="${COLLECTION_SERVICE:-slai-wrist-collection.service}"
UI_SERVICE="${COLLECTION_UI_SERVICE:-slai-collection-ui.service}"
UR5_HOST="${UR5_HOST:-192.168.1.106}"
DASHBOARD="${COLLECTION_DASHBOARD:-http://127.0.0.1:8765/api/status}"
WAIT_SECONDS="${COLLECTION_START_TIMEOUT_S:-120}"

fail() {
  echo "COLLECTION_RECOVERY_FAILED: $*" >&2
  systemctl --user --no-pager --full status "$SERVICE" >&2 || true
  journalctl --user -u "$SERVICE" -n 160 --no-pager >&2 || true
  exit 1
}

command -v systemctl >/dev/null || fail "systemctl is unavailable"
command -v curl >/dev/null || fail "curl is unavailable"

healthy_status() {
  local status dataset
  status="$(curl -fsS --max-time 3 "$DASHBOARD" 2>/dev/null || true)"
  [[ -n "$status" ]] || return 1
  dataset="$(python3 -c '
import json, sys
x=json.load(sys.stdin)
assert x.get("phase") == "ready"
assert x.get("can_record") is True
assert x.get("camera_count") == 3 and x.get("camera_online") == 3
assert all(c.get("valid") and not c.get("error") and float(c.get("age_ms", 1e9)) < 1000 for c in x["cameras"])
assert all(x.get("devices", {}).get(k, {}).get("state") == "active" for k in ("ur5", "wrist", "gripper"))
assert x.get("spacemouse", {}).get("connected") is True
print(x["dataset_path"])
' <<<"$status" 2>/dev/null)" || return 1
  printf '%s' "$dataset"
}

# Idempotence matters for real hardware: do not home or reconnect a healthy collector.
if systemctl --user is-active --quiet "$SERVICE"; then
  if dataset="$(healthy_status)"; then
    echo "COLLECTION_ALREADY_READY service=$SERVICE dashboard=${DASHBOARD%/api/status} dataset=$dataset"
    exit 0
  fi
fi

# Never run the read-only preview and real collector against the same cameras.
systemctl --user stop "$UI_SERVICE" 2>/dev/null || true

# Refuse to start when the robot is not reachable. These checks do not send motion.
ping -c 1 -W 2 "$UR5_HOST" >/dev/null || fail "UR5 is unreachable at $UR5_HOST"
python3 - "$UR5_HOST" <<'PY' || exit $?
import socket, sys
host = sys.argv[1]
for port in (30001, 30002, 30003, 30004):
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, port))
    except OSError as exc:
        raise SystemExit(f"UR5 port preflight failed: {host}:{port}: {exc}")
    finally:
        sock.close()
print("UR5_PREFLIGHT_OK")
PY

# Required local inputs. Camera identity is validated by the collector itself.
test -e /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  || fail "Hiwonder gripper serial device is missing"
test -e /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_E261BD20503059384C2E3120FF08262F-if00 \
  || fail "OpenRB wrist serial device is missing"
test -e /dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_14:C1:9F:26:4A:A8-if00 \
  || fail "ESP32 wrist sensor serial device is missing"
systemctl is-active --quiet spacenavd.service || fail "spacenavd is not active"

systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
systemctl --user restart "$SERVICE"

deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  active="$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)"
  [[ "$active" == failed ]] && fail "collector entered failed state"
  if [[ "$active" == active ]] && dataset="$(healthy_status)"; then
    echo "COLLECTION_READY service=$SERVICE dashboard=${DASHBOARD%/api/status} dataset=$dataset"
    exit 0
  fi
  sleep 2
done

fail "collector did not reach a fresh, three-camera ready state within ${WAIT_SECONDS}s"
