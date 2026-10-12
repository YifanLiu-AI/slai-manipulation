"""In-process collection dashboard fed by the real recording workflow."""

from __future__ import annotations

import copy
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image

from slai_mi.datasets.lerobot_v3.schema import RECORDED_BUTTON_NAMES
from slai_mi.devices.spacemouse.buttons import BUTTON_NAME_BY_CODE
from slai_mi.input_schema import compose_capture_vector, enabled_cameras, load_input_schema
from slai_mi.ui.collection_frontend import (
    DashboardRuntime,
    build_handler,
    dashboard_status_template,
)


class CollectionDashboardProvider:
    """Keep a bounded live snapshot without owning or commanding hardware."""

    def __init__(self, hardware: dict[str, Any], task: str) -> None:
        self.schema = load_input_schema(hardware.get("input_schema"))
        self._status = dashboard_status_template(hardware, task=task)
        self._lock = threading.RLock()
        self._frames: dict[str, bytes] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=80)
        self._started_at = time.monotonic()
        self._recording_started_at: float | None = None
        self._last_preview_at = 0.0
        self._sync_samples = 0
        self._sync_valid = 0
        self._last_sync_ready = False
        self._temperature_level = "unknown"
        self._camera_counts = {str(item["role"]): 0 for item in enabled_cameras(self.schema)}

    def start(self) -> None:
        self.event("数采监控台已启动，等待真实设备数据")

    def stop(self) -> None:
        return

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = copy.deepcopy(self._status)
        if self._recording_started_at is not None:
            status["episode"]["elapsed_s"] = round(
                time.monotonic() - self._recording_started_at, 1
            )
        return status

    def camera_jpeg(self, key: str) -> bytes | None:
        if key not in self._camera_counts:
            raise KeyError(key)
        with self._lock:
            frame = self._frames.get(key)
        return bytes(frame) if frame is not None else None

    def event(self, message: str, *, level: str = "info", code: str = "collection") -> None:
        item = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": level,
            "code": code,
            "message": message,
        }
        with self._lock:
            self._events.appendleft(item)
            self._status["events"] = list(self._events)
            self._touch()

    def set_phase(
        self,
        phase: str,
        label: str,
        *,
        can_record: bool | None = None,
        recording: bool | None = None,
    ) -> None:
        with self._lock:
            self._status["phase"] = phase
            self._status["phase_label"] = label
            if can_record is not None:
                self._status["can_record"] = can_record
            if recording is not None:
                self._status["recording"] = recording
            self._touch()

    def set_dataset_path(self, path: str | Path) -> None:
        with self._lock:
            self._status["dataset_path"] = str(path)
            self._touch()

    def start_episode(self, *, index: int, attempt: int) -> None:
        with self._lock:
            self._status["episode"] = {
                "index": int(index),
                "attempt": int(attempt),
                "valid_frames": 0,
                "rejected_frames": 0,
                "elapsed_s": 0.0,
            }
            self._recording_started_at = time.monotonic()
        self.set_phase("recording", "正在录制", can_record=False, recording=True)
        self.event(f"Episode {index} 开始录制", code="episode_start")

    def finish_episode(self, action: str, *, index: int) -> None:
        self._recording_started_at = None
        if action == "save":
            self.set_phase("ready", "已保存，可以继续", can_record=True, recording=False)
            self.event(f"Episode {index} 保存成功", level="success", code="episode_save")
        else:
            self.set_phase("ready", "已丢弃，可以重录", can_record=True, recording=False)
            self.event(f"Episode {index} 已丢弃", level="warning", code="episode_discard")

    def record_frame(self, _frame: dict[str, object]) -> None:
        key = "valid_frames" if self._last_sync_ready else "rejected_frames"
        with self._lock:
            self._status["episode"][key] += 1
            self._touch()

    def observe_spacemouse(self, motion: Any, buttons: dict[Any, Any]) -> None:
        """Publish all physical buttons without changing the recorded 12-button contract."""
        axes = np.asarray(motion, dtype=np.float32)
        if axes.shape != (6,) or not np.isfinite(axes).all():
            return
        named_buttons = {name: False for name in BUTTON_NAME_BY_CODE.values()}
        for key, pressed in buttons.items():
            try:
                name = BUTTON_NAME_BY_CODE.get(int(key), str(key))
            except (TypeError, ValueError):
                name = str(key)
            named_buttons[name] = bool(pressed)
        with self._lock:
            self._status["spacemouse"] = {
                **self._status["spacemouse"],
                "connected": True,
                "motion": axes.astype(float).tolist(),
                "buttons": named_buttons,
                "active": bool(np.any(np.abs(axes) > 0.0)),
                "valid": True,
                "age_ms": 0.0,
                "last_activity_ms": 0.0,
                "error": None,
            }
            self._touch()

    def observe_inputs(self, inputs: Any) -> None:
        now = time.monotonic()
        capture = self.schema["capture"]
        synchronization = self.schema["synchronization"]
        cameras = enabled_cameras(self.schema)
        camera_roles = [str(camera["role"]) for camera in cameras]
        state_names = [str(item["name"]) for item in synchronization["state_channels"]]
        command_name = str(synchronization["command_channel"]["name"])
        samples = [
            *(inputs.cameras[role] for role in camera_roles),
            *(inputs.channels[name] for name in state_names),
            inputs.channels[command_name],
        ]
        ages = [max(0.0, now - float(sample.host_timestamp_s)) * 1000.0 for sample in samples]
        camera_limit = float(synchronization["max_camera_age_ms"])
        state_limit = float(synchronization["max_state_age_ms"])
        command_limit = float(synchronization["max_command_age_ms"])
        limits = [
            *([camera_limit] * len(camera_roles)),
            *([state_limit] * len(state_names)),
            command_limit,
        ]
        validity = [
            int(bool(sample.valid) and np.isfinite(age) and age <= limit)
            for sample, age, limit in zip(samples, ages, limits, strict=True)
        ]
        primary_role = str(capture["primary_timeline_role"])
        primary_time = float(inputs.cameras[primary_role].host_timestamp_s)
        skew = {
            role: abs(float(inputs.cameras[role].host_timestamp_s) - primary_time) * 1000.0
            for role in camera_roles
            if role != primary_role
        }
        skew_ready = all(
            value <= float(synchronization["max_camera_skew_ms"]) for value in skew.values()
        )
        sync_ready = bool(all(validity) and skew_ready)
        state = compose_capture_vector(
            self.schema,
            "state",
            {name: inputs.channels[name].value for name in state_names},
        )
        mouse = inputs.channels[command_name].value
        temperature = getattr(inputs.channels["wuji"].value, "temperature", None)
        axes = np.asarray(mouse.axes, dtype=np.float32)
        buttons = np.asarray(mouse.buttons, dtype=bool)
        recorded_buttons = {
            name: bool(buttons[index]) if index < len(buttons) else False
            for index, name in enumerate(RECORDED_BUTTON_NAMES)
        }
        camera_status = []
        elapsed = max(now - self._started_at, 1e-6)
        encode_previews = now - self._last_preview_at >= 0.10
        encoded: dict[str, bytes] = {}
        for index, camera in enumerate(cameras):
            role = str(camera["role"])
            sample = inputs.cameras[role]
            self._camera_counts[role] += 1
            if encode_previews:
                output = BytesIO()
                Image.fromarray(np.asarray(sample.value, dtype=np.uint8)).save(
                    output, format="JPEG", quality=78
                )
                encoded[role] = output.getvalue()
            current = next(item for item in self._status["cameras"] if item["key"] == role)
            camera_status.append(
                {
                    **current,
                    "connected": True,
                    "valid": bool(validity[index])
                    and (role == primary_role or skew.get(role, 0.0) <= float(synchronization["max_camera_skew_ms"])),
                    "fps": self._camera_counts[role] / elapsed,
                    "sequence": int(sample.sequence),
                    "age_ms": round(ages[index], 1),
                    "drops": int(sample.dropped_before),
                    "error": None,
                }
            )
        if encode_previews:
            self._last_preview_at = now
        self._sync_samples += 1
        self._sync_valid += int(sync_ready)
        self._last_sync_ready = sync_ready
        state_offset = len(camera_roles)
        state_ages = ages[state_offset : state_offset + len(state_names)]
        mouse_age = ages[-1]
        temperature_transition: tuple[str, float] | None = None
        with self._lock:
            self._frames.update(encoded)
            self._status["cameras"] = camera_status
            self._status["camera_online"] = sum(item["connected"] for item in camera_status)
            self._status["dof"] = {
                **self._status["dof"],
                "values": state.astype(float).tolist(),
                "valid": bool(all(validity[state_offset : state_offset + len(state_names)])),
                "age_ms": round(max(state_ages), 1),
            }
            self._status["spacemouse"] = {
                **self._status["spacemouse"],
                "connected": True,
                "motion": axes.astype(float).tolist(),
                "buttons": {
                    **self._status["spacemouse"]["buttons"],
                    **recorded_buttons,
                },
                "active": bool(np.any(np.abs(axes) > 0.0)),
                "valid": bool(validity[-1]),
                "age_ms": round(mouse_age, 1),
                "last_activity_ms": round(mouse_age, 1),
                "error": None,
            }
            device_offset = len(camera_roles)
            self._status["devices"] = {
                "ur5": {
                    "state": "active" if inputs.channels["ur5"].valid else "error",
                    "age_ms": round(ages[device_offset + state_names.index("ur5")], 1),
                },
                "wuji": {
                    "state": "active" if inputs.channels["wuji"].valid else "error",
                    "age_ms": round(ages[device_offset + state_names.index("wuji")], 1),
                },
            }
            self._status["sync"] = {
                **self._status["sync"],
                "ready": sync_ready,
                "valid_ratio": self._sync_valid / self._sync_samples,
                "source_age_ms": [round(value, 1) for value in ages],
                "source_drops": [int(sample.dropped_before) for sample in samples],
                "source_restarts": [int(sample.restart_count) for sample in samples],
                "validity_mask": validity,
                "camera_skew_ms": {key: round(value, 2) for key, value in skew.items()},
            }
            if isinstance(temperature, dict):
                level = str(temperature.get("level", "unknown"))
                maximum = float(temperature.get("max_c"))
                self._status["temperature"] = {
                    "available": True,
                    "values": [float(value) for value in temperature.get("values", ())],
                    "max_c": maximum,
                    "level": level,
                    "warning_c": float(temperature.get("warning_c", 70.0)),
                    "critical_c": float(temperature.get("critical_c", 75.0)),
                    "limit_c": float(temperature.get("limit_c", 80.0)),
                }
                if level != self._temperature_level:
                    self._temperature_level = level
                    temperature_transition = (level, maximum)
            self._touch()
        if temperature_transition is not None:
            level, maximum = temperature_transition
            if level in {"warning", "critical"}:
                self.event(
                    f"Wuji 温度{('临界' if level == 'critical' else '警告')}：{maximum:.1f} °C",
                    level="error" if level == "critical" else "warning",
                    code="wuji_temperature",
                )
            elif level == "normal":
                self.event(
                    f"Wuji 温度正常：{maximum:.1f} °C",
                    level="success",
                    code="wuji_temperature",
                )

    def _touch(self) -> None:
        self._status["updated_at"] = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )


class DashboardSynchronizer:
    """Publish the exact synchronized inputs consumed by the recorder."""

    def __init__(self, synchronizer: Any, provider: CollectionDashboardProvider) -> None:
        self._synchronizer = synchronizer
        self._provider = provider

    def read(self, timeout_s: float = 1.0) -> Any:
        inputs = self._synchronizer.read(timeout_s=timeout_s)
        self._provider.observe_inputs(inputs)
        return inputs


class CollectionDashboard:
    """Own the local HTTP dashboard while collection owns the hardware sources."""

    def __init__(
        self,
        hardware: dict[str, Any],
        task: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = True,
    ) -> None:
        self.provider = CollectionDashboardProvider(hardware, task)
        self.runtime = DashboardRuntime(self.provider)
        self.server = ThreadingHTTPServer((host, port), build_handler(self.runtime))
        self.server.daemon_threads = True
        address, actual_port = self.server.server_address[:2]
        browser_host = "127.0.0.1" if address in {"0.0.0.0", "::"} else str(address)
        self.url = f"http://{browser_host}:{actual_port}"
        self.open_browser = open_browser
        self._thread: threading.Thread | None = None
        self._opener: threading.Timer | None = None

    def start(self) -> str:
        self.runtime.start()
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="collection-dashboard",
            daemon=True,
        )
        self._thread.start()
        if self.open_browser:
            self._opener = threading.Timer(0.5, webbrowser.open, args=(self.url,))
            self._opener.daemon = True
            self._opener.start()
        return self.url

    def stop(self) -> None:
        if self._opener is not None:
            self._opener.cancel()
        self.server.shutdown()
        self.server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.runtime.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
