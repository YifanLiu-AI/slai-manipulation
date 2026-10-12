"""Live camera and physical SpaceMouse provider for the collection dashboard."""

from __future__ import annotations

import subprocess
import threading
import time
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from slai_mi.devices.cameras import CameraConfig, RealSenseCapture, validate_camera_set
from slai_mi.devices.spacemouse.buttons import BUTTON_NAME_BY_CODE, Button
from slai_mi.devices.spacemouse.client import SpaceMouseProcess
from slai_mi.devices.spacemouse.gestures import RepeatedChordGesture
from slai_mi.input_schema import enabled_cameras, load_input_schema
from slai_mi.ui.collection_frontend import dashboard_status_template
from slai_mi.ui.webrtc_preview import H264PreviewPublisher


class LiveStatusProvider:
    """Monitor configured input devices without exposing any motion command."""

    def __init__(
        self,
        hardware: dict[str, Any],
        *,
        monitor_spacemouse: bool = True,
        collection_service: str | None = None,
    ) -> None:
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
        self.preview = H264PreviewPublisher(
            (str(camera["role"]) for camera in cameras),
            width=int(image_width),
            height=int(image_height),
            fps=int(self.schema["capture"]["fps"]),
        )
        self._monitor_spacemouse = monitor_spacemouse
        self.collection_service = collection_service
        self.mouse = self._new_mouse() if monitor_spacemouse else None
        self._frames: dict[str, Any] = {}
        self._jpeg_cache: dict[str, tuple[float, bytes]] = {}
        self._counts = {item.name: 0 for item in configs}
        self._errors: dict[str, str] = {}
        self._started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mouse_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._mouse_motion = np.zeros(6, dtype=np.float32)
        self._mouse_buttons: dict[int, bool] = {}
        self._mouse_sample_at = 0.0
        self._last_mouse_activity: float | None = None
        self._collection_start_requested = False
        self._collection_start_error: str | None = None
        self._collection_gesture = RepeatedChordGesture(
            (int(Button.MENU), int(Button.FIT)), repeats=2, timeout_s=5.0
        )

    @staticmethod
    def _new_mouse() -> SpaceMouseProcess:
        # Use the same spacenavd-backed input path as teleoperation and collection.
        return SpaceMouseProcess(
            deadzone=0.12,
            stale_timeout=0.05,
            rate_hz=250.0,
            backend="spnav",
        )

    def start(self) -> None:
        self.capture.start()
        self.preview.start()
        try:
            if self.mouse is not None:
                self.mouse.start()
                self._mouse_thread = threading.Thread(
                    target=self._read_mouse_loop,
                    name="dashboard-spacemouse",
                    daemon=True,
                )
                self._mouse_thread.start()
        except (OSError, RuntimeError) as exc:
            self._errors["spacemouse"] = str(exc)
        self._started_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="dashboard-inputs", daemon=True)
        self._thread.start()

    def _read_mouse_loop(self) -> None:
        while not self._stop.is_set() and self.mouse is not None:
            try:
                motion, buttons = self.mouse.state()
                now = time.monotonic()
                with self._lock:
                    self._mouse_motion = np.asarray(motion, dtype=np.float32)
                    self._mouse_buttons = {int(key): bool(value) for key, value in buttons.items()}
                    self._mouse_sample_at = now
                cap_active = bool(np.any(np.abs(self._mouse_motion) > 0.0))
                if cap_active:
                    self._last_mouse_activity = now
                    self._collection_gesture.reset()
                if (
                    self.collection_service
                    and not cap_active
                    and not self._collection_start_requested
                    and self._collection_gesture.update(buttons, now)
                ):
                    self._request_collection_start()
            except RuntimeError as exc:
                self._errors["spacemouse"] = str(exc)
                return
            time.sleep(0.001)

    def _request_collection_start(self) -> None:
        if self.collection_service is None:
            return
        with self._lock:
            self._collection_start_requested = True
            self._collection_start_error = None
        try:
            subprocess.run(
                ["systemctl", "--user", "start", "--no-block", self.collection_service],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            with self._lock:
                self._collection_start_requested = False
                self._collection_start_error = f"数采服务启动失败: {exc}"
            self._collection_gesture.reset()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.capture.read(0.5)
                for role, frame in frames.items():
                    try:
                        self.preview.publish(role, frame.color)
                    except (KeyError, TypeError, ValueError):
                        pass
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
        if self._mouse_thread is not None:
            self._mouse_thread.join(timeout=2.0)
            self._mouse_thread = None
        self.capture.stop()
        self.preview.stop()
        with self._lock:
            mouse = self.mouse
            self.mouse = None
        if mouse is not None:
            mouse.stop()

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
        with self._lock:
            mouse = self.mouse
            collection_start_requested = self._collection_start_requested
            collection_start_error = self._collection_start_error
        if mouse is None:
            motion, buttons = np.zeros(6), {}
            mouse_sample_at = 0.0
            mouse_error = collection_start_error
        else:
            with self._lock:
                motion = self._mouse_motion.copy()
                buttons = dict(self._mouse_buttons)
                mouse_sample_at = self._mouse_sample_at
            mouse_error = self._errors.get("spacemouse")
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
        if collection_start_requested:
            status.update({"phase": "switching", "phase_label": "正在切换到腕部数采"})
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
                    if mouse_sample_at <= 0.0
                    else max(0.0, time.monotonic() - mouse_sample_at) * 1000.0
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

    def spacemouse_status(self) -> dict[str, Any]:
        with self._lock:
            mouse = self.mouse
            collection_start_error = self._collection_start_error
        if mouse is None:
            motion, buttons = np.zeros(6), {}
            mouse_sample_at = 0.0
            mouse_error = collection_start_error
        else:
            with self._lock:
                motion = self._mouse_motion.copy()
                buttons = dict(self._mouse_buttons)
                mouse_sample_at = self._mouse_sample_at
            mouse_error = self._errors.get("spacemouse")
        now = time.monotonic()
        active = bool(np.any(np.asarray(motion)))
        return {
            "connected": mouse_error is None,
            "device": "SpaceMouse Pro",
            "motion": np.asarray(motion, dtype=float).tolist(),
            "buttons": {
                BUTTON_NAME_BY_CODE.get(int(key), str(key)): bool(value)
                for key, value in buttons.items()
            },
            "active": active,
            "valid": mouse_error is None,
            "age_ms": None if mouse_sample_at <= 0.0 else max(0.0, now - mouse_sample_at) * 1000.0,
            "last_activity_ms": (
                None
                if self._last_mouse_activity is None
                else max(0.0, now - self._last_mouse_activity) * 1000.0
            ),
            "error": mouse_error,
        }

    def camera_jpeg(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            frame = self._frames.get(key)
            cached = self._jpeg_cache.get(key)
        if key not in self._counts:
            raise KeyError(key)
        if frame is None:
            return None
        if cached is not None and now - cached[0] < 0.1:
            return cached[1]
        output = BytesIO()
        Image.fromarray(np.asarray(frame.color, dtype=np.uint8)).save(output, format="JPEG", quality=68)
        encoded = output.getvalue()
        with self._lock:
            self._jpeg_cache[key] = (now, encoded)
        return encoded


def factory(
    hardware: dict[str, Any],
    *,
    monitor_spacemouse: bool = True,
    collection_service: str | None = None,
) -> LiveStatusProvider:
    return LiveStatusProvider(
        hardware,
        monitor_spacemouse=monitor_spacemouse,
        collection_service=collection_service,
    )
