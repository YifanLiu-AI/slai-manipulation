"""Read-only HTTP dashboard for collection device status."""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

import yaml

from slai_mi.collection.history import build_collection_history
from slai_mi.datasets.lerobot_v3.schema import RECORDED_BUTTON_NAMES
from slai_mi.input_schema import enabled_cameras, load_input_schema
from slai_mi.runtime import StrategyProfileError, load_strategy_profile

STATIC_ROOT = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@runtime_checkable
class StatusProvider(Protocol):
    """Read-only data source consumed by the dashboard.

    Implementations may monitor devices, but this interface deliberately exposes
    no commands that can move a robot or change device configuration.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def camera_jpeg(self, key: str) -> bytes | None: ...


class OfflineStatusProvider:
    """In-memory provider used when live monitoring is not explicitly injected."""

    def __init__(self, status: dict[str, Any]) -> None:
        self._status = copy.deepcopy(status)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def status(self) -> dict[str, Any]:
        return copy.deepcopy(self._status)

    def camera_jpeg(self, key: str) -> bytes | None:
        return None


class DashboardRuntime:
    """Serialize provider access and make its lifecycle idempotent."""

    def __init__(self, provider: StatusProvider | dict[str, Any]) -> None:
        self._provider: StatusProvider = (
            OfflineStatusProvider(provider) if isinstance(provider, dict) else provider
        )
        self._lock = threading.RLock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._provider.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._provider.stop()
            finally:
                self._started = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._provider.status())

    def spacemouse_status(self) -> dict[str, Any]:
        """Return only the latest input snapshot for the low-latency stream."""
        with self._lock:
            provider_method = getattr(self._provider, "spacemouse_status", None)
            if callable(provider_method):
                return copy.deepcopy(provider_method())
            return copy.deepcopy(self._provider.status().get("spacemouse", {}))

    def camera_jpeg(self, key: str) -> bytes | None:
        with self._lock:
            frame = self._provider.camera_jpeg(key)
            return bytes(frame) if frame is not None else None

    def collection_history(self) -> dict[str, Any]:
        with self._lock:
            provider_method = getattr(self._provider, "collection_history", None)
            if callable(provider_method):
                return copy.deepcopy(provider_method())
            return build_collection_history(
                PROJECT_ROOT / "data" / "lerobot",
                current_status=self._provider.status(),
            )


def load_hardware_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Unable to load hardware config {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Hardware config must contain a mapping: {candidate}")
    return payload


def offline_status(config: dict[str, Any], *, task: str = "等待采集任务") -> dict[str, Any]:
    """Build status without importing or opening any device SDK."""
    status = dashboard_status_template(config, task=task)
    camera_enabled = bool(config.get("cameras", {}).get("enabled"))
    for camera in status["cameras"]:
        camera["error"] = "设备监测未启用" if camera_enabled else "配置中已禁用"
    status["spacemouse"]["error"] = (
        "设备监测未启用" if config.get("spacemouse", {}).get("enabled") else "配置中已禁用"
    )
    return status


def dashboard_status_template(
    config: dict[str, Any],
    *,
    task: str = "等待采集任务",
) -> dict[str, Any]:
    """Build the schema-driven status contract shared by live and collection providers."""
    schema = load_input_schema(config.get("input_schema"))
    identities = {
        str(item.get("role")): item
        for item in config.get("cameras", {}).get("devices", [])
        if isinstance(item, dict)
    }
    height, width, _channels = (int(value) for value in schema["capture"]["image_shape"])
    cameras = []
    for camera in enabled_cameras(schema):
        role = str(camera["role"])
        identity = identities.get(role, {})
        cameras.append(
            {
                "key": role,
                "role": role,
                "label": str(camera.get("label") or role),
                "dataset_key": str(camera.get("dataset_key") or ""),
                "serial": str(identity.get("serial") or ""),
                "model": "RealSense",
                "connected": False,
                "valid": False,
                "fps": 0.0,
                "resolution": [width, height],
                "sequence": None,
                "age_ms": None,
                "drops": 0,
                "error": None,
            }
        )
    state_names = [
        name
        for component in schema["capture"]["state"]["components"]
        if "constant" not in component and component.get("display", True)
        for name in component["names"]
    ]
    source_names = [
        *(str(camera["role"]) for camera in enabled_cameras(schema)),
        *(str(channel["name"]) for channel in schema["synchronization"]["state_channels"]),
        str(schema["synchronization"]["command_channel"]["name"]),
    ]
    return {
        "schema_version": 1,
        "read_only": True,
        "phase": "starting",
        "phase_label": "正在启动",
        "recording": False,
        "can_record": False,
        "task": task,
        "dataset_path": None,
        "mode": "combined",
        "devices": {
            "ur5": {"state": "starting" if config.get("ur5", {}).get("enabled") else "inactive"},
            "wuji": {
                "state": ("starting" if config.get("wujihand", {}).get("enabled") else "inactive")
            },
            "wrist": {
                "state": (
                    "starting" if config.get("wrist_sensor", {}).get("enabled") else "inactive"
                )
            },
        },
        "episode": {
            "index": 1,
            "attempt": 0,
            "valid_frames": 0,
            "rejected_frames": 0,
            "elapsed_s": 0.0,
        },
        "camera_count": len(cameras),
        "camera_online": 0,
        "cameras": cameras,
        "dof": {
            "names": state_names,
            "values": [0.0] * len(state_names),
            "valid": False,
            "age_ms": None,
        },
        "spacemouse": {
            "connected": False,
            "device": "SpaceMouse Pro",
            "motion": [0.0] * 6,
            "buttons": {name: False for name in RECORDED_BUTTON_NAMES},
            "active": False,
            "valid": False,
            "age_ms": None,
            "last_activity_ms": None,
            "error": None,
        },
        "temperature": {
            "available": False,
            "values": [],
            "max_c": None,
            "level": "unknown",
            "warning_c": 70.0,
            "critical_c": 75.0,
            "limit_c": 80.0,
        },
        "sync": {
            "ready": False,
            "valid_ratio": 0.0,
            "source_names": source_names,
            "source_age_ms": [None] * len(source_names),
            "source_drops": [0] * len(source_names),
            "source_restarts": [0] * len(source_names),
            "validity_mask": [0] * len(source_names),
            "camera_skew_ms": {},
            "fallback": "相机按主时间线配对；状态线性对齐；SpaceMouse 命令零阶保持",
        },
        "events": [],
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def build_handler(
    provider: StatusProvider | DashboardRuntime | dict[str, Any],
) -> type[BaseHTTPRequestHandler]:
    runtime = provider if isinstance(provider, DashboardRuntime) else DashboardRuntime(provider)

    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/spacemouse/events":
                self._send_spacemouse_stream(runtime)
                return
            if route == "/api/events":
                self._send_event_stream(runtime)
                return
            if route == "/api/status":
                try:
                    self._send_json(runtime.status())
                except Exception as exc:  # noqa: BLE001 - provider failures become HTTP status
                    self._send_json(
                        {"error": f"status unavailable: {type(exc).__name__}"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if route == "/api/collection-history":
                try:
                    self._send_json(runtime.collection_history())
                except Exception as exc:  # noqa: BLE001 - history failures become HTTP status
                    self._send_json(
                        {"error": f"collection history unavailable: {type(exc).__name__}"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if route in {"/api/spacemouse", "/api/cameras", "/api/recording", "/api/devices"}:
                try:
                    status = runtime.status()
                    if route == "/api/spacemouse":
                        payload = status.get("spacemouse", {})
                    elif route == "/api/cameras":
                        payload = {
                            str(camera["key"]): camera for camera in status.get("cameras", [])
                        }
                    elif route == "/api/devices":
                        payload = {
                            "mode": status.get("mode", "combined"),
                            "devices": status.get("devices", {}),
                        }
                    else:
                        episode = status.get("episode", {})
                        if status.get("recording"):
                            detail = (
                                f"{episode.get('valid_frames', 0)} 帧 · "
                                f"{float(episode.get('elapsed_s', 0.0)):.1f} 秒"
                            )
                        elif status.get("can_record"):
                            detail = "等待 MENU"
                        else:
                            detail = ""
                        payload = {
                            "state": {
                                "code": status.get("phase", "starting"),
                                "label": status.get("phase_label", "准备中"),
                                "detail": detail,
                            },
                            # The provider stores newest first; the legacy journal is chronological.
                            "events": list(reversed(status.get("events", []))),
                            "temperature": status.get("temperature", {}),
                        }
                    self._send_json(payload)
                except Exception as exc:  # noqa: BLE001 - provider failures become HTTP status
                    self._send_json(
                        {"error": f"status unavailable: {type(exc).__name__}"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if route.startswith("/frame/") and route.endswith(".jpg"):
                key = unquote(route[len("/frame/") : -len(".jpg")])
                if not key or "/" in key or key in {".", ".."}:
                    self._send_json({"error": "invalid camera key"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_camera_frame(runtime, key)
                return
            if route.startswith("/stream/") and route.endswith(".mjpg"):
                key = unquote(route[len("/stream/") : -len(".mjpg")])
                if not key or "/" in key or key in {".", ".."}:
                    self._send_json({"error": "invalid camera key"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_camera_stream(runtime, key)
                return
            prefix, suffix = "/api/cameras/", "/frame.jpg"
            if route.startswith(prefix) and route.endswith(suffix):
                key = unquote(route[len(prefix) : -len(suffix)])
                if not key or "/" in key or key in {".", ".."}:
                    self._send_json({"error": "invalid camera key"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_camera_frame(runtime, key)
                return
            requested = "index.html" if route == "/" else unquote(route.lstrip("/"))
            target = (STATIC_ROOT / requested).resolve()
            if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            cache_control = "no-cache" if target.suffix in {".html", ".js", ".css", ".svg"} else "public, max-age=3600"
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)

        def _send_camera_frame(self, source: DashboardRuntime, key: str) -> None:
            try:
                frame = source.camera_jpeg(key)
            except KeyError:
                self._send_json({"error": "unknown camera"}, HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:  # noqa: BLE001 - isolate provider failure from server
                self._send_json(
                    {"error": f"frame unavailable: {type(exc).__name__}"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            if frame is None:
                self._send_json({"error": "frame unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._send_bytes(frame, "image/jpeg")

        def _send_event_stream(self, source: DashboardRuntime) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 500\n\n")
                self.wfile.flush()
                while True:
                    payload = json.dumps(source.status(), ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    # Keep read-only input feedback close to SpaceMouse sample
                    # time without tying it to the camera frame rate.
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def _send_spacemouse_stream(self, source: DashboardRuntime) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 250\n\n")
                self.wfile.flush()
                while True:
                    payload = json.dumps(
                        source.spacemouse_status(), ensure_ascii=False, separators=(",", ":")
                    )
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def _send_camera_stream(self, source: DashboardRuntime, key: str) -> None:
            try:
                source.camera_jpeg(key)
            except KeyError:
                self._send_json({"error": "unknown camera"}, HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    frame = source.camera_jpeg(key)
                    if frame is not None:
                        header = (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                        )
                        self.wfile.write(header)
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.12)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument(
        "--strategy",
        help="Show one configured hardware strategy without opening its devices",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--live", action="store_true", help="Open read-only physical input monitors"
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="With --live, leave SpaceMouse to the active teleoperation process",
    )
    parser.add_argument(
        "--spacemouse-collection-launch",
        action="store_true",
        help="Start the wrist collection service after two physical Menu+Fit press cycles",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if args.camera_only and not args.live:
        raise SystemExit("--camera-only requires --live")
    if args.spacemouse_collection_launch and (not args.live or args.camera_only):
        raise SystemExit(
            "--spacemouse-collection-launch requires --live and SpaceMouse monitoring"
        )
    hardware = load_hardware_config(args.hardware_config)
    task_label = "等待采集任务"
    if args.strategy:
        try:
            strategy = load_strategy_profile(args.strategy)
        except StrategyProfileError as exc:
            raise SystemExit(f"Dashboard strategy failed: {exc}") from exc
        hardware = strategy.configure_hardware(hardware)
        task_label = strategy.label
    if args.live:
        from slai_mi.ui.live_provider import factory

        provider: StatusProvider = factory(
            hardware,
            monitor_spacemouse=not args.camera_only,
            collection_service=(
                "slai-wrist-collection.service"
                if args.spacemouse_collection_launch
                else None
            ),
        )
    else:
        provider = OfflineStatusProvider(offline_status(hardware, task=task_label))
    runtime = DashboardRuntime(provider)
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runtime))
    server.daemon_threads = True
    address, port = server.server_address[:2]
    mode = "live monitors" if args.live else "devices offline"
    print(f"Collection dashboard: http://{address}:{port} (read-only, {mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
