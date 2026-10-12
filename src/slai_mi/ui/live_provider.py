"""Read-only live camera and SpaceMouse provider for the collection dashboard."""

from __future__ import annotations

import threading
import time
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from slai_mi.devices.cameras import CameraConfig, RealSenseCapture, validate_camera_set
from slai_mi.devices.spacemouse.buttons import BUTTON_NAME_BY_CODE
from slai_mi.devices.spacemouse.client import SpaceMouseProcess
from slai_mi.input_schema import enabled_cameras, load_input_schema
from slai_mi.ui.collection_frontend import dashboard_status_template


class LiveStatusProvider:
    """Monitor configured input devices without exposing any motion command."""

    def __init__(self, hardware: dict[str, Any]) -> None:
        self.hardware = hardware
        self.schema = load_input_schema(hardware.get("input_schema"))
        image_height, image_width, _channels = self.schema["capture"]["image_shape"]
        identities = {str(item["role"]): item for item in hardware["cameras"]["devices"]}
        cameras = enabled_cameras(self.schema)
        configs = validate_camera_set(
            (
                CameraConfig(
                    str(camera["role"]),
                    str(identities[str(camera["role"])]["serial"]),
                    int(image_width),
                    int(image_height),
                    int(self.schema["capture"]["fps"]),
                )
                for camera in cameras
            ),
            expected_count=len(cameras),
        )
        self.capture = RealSenseCapture(configs)
        self.mouse = SpaceMouseProcess()
        self._frames: dict[str, Any] = {}
        self._counts = {item.name: 0 for item in configs}
        self._errors: dict[str, str] = {}
        self._started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_mouse_activity: float | None = None

    def start(self) -> None:
        self.capture.start()
        try:
            self.mouse.start()
        except (OSError, RuntimeError) as exc:
            self._errors["spacemouse"] = str(exc)
        self._started_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="dashboard-inputs", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.capture.read(0.5)
                with self._lock:
                    self._frames.update(frames)
                    for role in frames:
                        self._counts[role] += 1
            except TimeoutError:
                continue
            except (OSError, RuntimeError) as exc:
                self._errors["cameras"] = str(exc)
                self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.capture.stop()
        self.mouse.stop()

    def status(self) -> dict[str, Any]:
        status = dashboard_status_template(self.hardware, task="独立设备监测")
        status.update({"phase": "monitoring", "phase_label": "只读设备监测"})
        elapsed = max(time.monotonic() - self._started_at, 1e-6)
        with self._lock:
            frames = dict(self._frames)
            counts = dict(self._counts)
        cameras = []
        for camera in enabled_cameras(self.schema):
            role = str(camera["role"])
            frame = frames.get(role)
            cameras.append(
                {
                    "key": role,
                    "role": role,
                    "label": str(camera.get("label") or role),
                    "serial": next(
                        str(item["serial"])
                        for item in self.hardware["cameras"]["devices"]
                        if item["role"] == role
                    ),
                    "model": "RealSense",
                    "connected": frame is not None,
                    "valid": frame is not None,
                    "fps": counts[role] / elapsed,
                    "resolution": list(self.schema["capture"]["image_shape"][:2][::-1]),
                    "error": self._errors.get("cameras"),
                    "sequence": int(frame.sequence) if frame is not None else None,
                    "age_ms": (
                        max(0.0, time.monotonic() - frame.host_timestamp_s) * 1000.0
                        if frame is not None
                        else None
                    ),
                    "drops": 0,
                }
            )
        try:
            motion, buttons = self.mouse.state()
            mouse_error = self._errors.get("spacemouse")
        except RuntimeError as exc:
            motion, buttons, mouse_error = np.zeros(6), {}, str(exc)
        active = bool(np.any(np.asarray(motion)))
        if active:
            self._last_mouse_activity = time.monotonic()
        status.update(
            {
                "camera_count": len(cameras),
                "camera_online": sum(item["connected"] for item in cameras),
                "cameras": cameras,
            }
        )
        status["spacemouse"].update(
            {
                "connected": mouse_error is None,
                "device": "SpaceMouse Pro",
                "motion": np.asarray(motion, dtype=float).tolist(),
                "buttons": {
                    BUTTON_NAME_BY_CODE.get(int(key), str(key)): bool(value)
                    for key, value in buttons.items()
                },
                "active": active,
                "valid": mouse_error is None,
                "age_ms": (
                    None
                    if self._last_mouse_activity is None
                    else (time.monotonic() - self._last_mouse_activity) * 1000.0
                ),
                "last_activity_ms": (
                    None
                    if self._last_mouse_activity is None
                    else (time.monotonic() - self._last_mouse_activity) * 1000.0
                ),
                "error": mouse_error,
            }
        )
        return status

    def camera_jpeg(self, key: str) -> bytes | None:
        with self._lock:
            frame = self._frames.get(key)
        if key not in self._counts:
            raise KeyError(key)
        if frame is None:
            return None
        output = BytesIO()
        Image.fromarray(np.asarray(frame.color, dtype=np.uint8)).save(output, format="JPEG", quality=85)
        return output.getvalue()


def factory(hardware: dict[str, Any]) -> LiveStatusProvider:
    return LiveStatusProvider(hardware)
