"""Threaded multi-RealSense capture with host-time frame pairing."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import suppress
from itertools import product

import numpy as np

from .models import CameraConfig, CameraFrame


class FrameSynchronizer:
    def __init__(
        self,
        camera_names: tuple[str, ...],
        queue_size: int = 8,
        *,
        max_skew_ms: float | None = None,
    ):
        if len(camera_names) < 2 or len(set(camera_names)) != len(camera_names) or queue_size < 2:
            raise ValueError("synchronizer needs unique cameras and queue_size >= 2")
        if max_skew_ms is not None and max_skew_ms <= 0.0:
            raise ValueError("max_skew_ms must be positive")
        self._queues = {name: deque(maxlen=queue_size) for name in camera_names}
        self._max_skew_s = None if max_skew_ms is None else max_skew_ms / 1000.0
        self._condition = threading.Condition()

    def add(self, frame: CameraFrame) -> None:
        if frame.camera not in self._queues:
            raise ValueError(f"unknown camera: {frame.camera}")
        with self._condition:
            self._queues[frame.camera].append(frame)
            self._condition.notify_all()

    def read(self, timeout_s: float = 1.0) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                chosen = self._choose_frames() if all(self._queues.values()) else None
                if chosen is not None:
                    for name, frame in chosen.items():
                        queue = self._queues[name]
                        while queue:
                            current = queue.popleft()
                            if current is frame:
                                break
                    return chosen
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    latest = {
                        name: queue[-1].host_timestamp_s if queue else None
                        for name, queue in self._queues.items()
                    }
                    available = [value for value in latest.values() if value is not None]
                    latest_skew_ms = (
                        (max(available) - min(available)) * 1000.0
                        if len(available) == len(latest)
                        else None
                    )
                    raise TimeoutError(
                        "timed out waiting for synchronized camera frames; "
                        f"latest_skew_ms={latest_skew_ms!r} latest={latest}"
                    )
                self._condition.wait(remaining)

    def _choose_frames(self) -> dict[str, CameraFrame] | None:
        if self._max_skew_s is None:
            reference = min(queue[-1].host_timestamp_s for queue in self._queues.values())
            return {
                name: min(queue, key=lambda frame: abs(frame.host_timestamp_s - reference))
                for name, queue in self._queues.items()
            }
        names = tuple(self._queues)
        feasible: list[tuple[float, float, tuple[CameraFrame, ...]]] = []
        for frames in product(*(self._queues[name] for name in names)):
            times = [frame.host_timestamp_s for frame in frames]
            skew = max(times) - min(times)
            if skew <= self._max_skew_s:
                feasible.append((min(times), -skew, frames))
        if not feasible:
            return None
        # Prefer the newest complete group, using lower skew only as a tie-breaker.
        frames = max(feasible, key=lambda item: item[:2])[2]
        return dict(zip(names, frames, strict=True))


class RealSenseCapture:
    """Capture configured cameras; pyrealsense2 is imported only by start()."""

    def __init__(
        self,
        configs: tuple[CameraConfig, ...],
        *,
        rs_module=None,
        clock=time.monotonic,
        max_skew_ms: float | None = None,
    ):
        if not configs:
            raise ValueError("at least one camera is required")
        self.configs, self._rs, self._clock = configs, rs_module, clock
        self.synchronizer = (
            FrameSynchronizer(
                tuple(item.name for item in configs), max_skew_ms=max_skew_ms
            )
            if len(configs) > 1
            else None
        )
        self._latest: dict[str, CameraFrame] = {}
        self._pipelines: list[object] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._errors: dict[str, BaseException] = {}

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("RealSense capture is already running")
        if self._rs is None:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError("RealSense support requires pyrealsense2") from exc
            self._rs = rs
        self._stop.clear()
        for config in self.configs:
            thread = threading.Thread(target=self._run, args=(config,), daemon=True, name=f"RealSense-{config.name}")
            self._threads.append(thread)
            thread.start()

    def _run(self, camera: CameraConfig) -> None:
        rs, pipeline = self._rs, self._rs.pipeline()
        self._pipelines.append(pipeline)
        started = False
        try:
            config = rs.config()
            config.enable_device(camera.serial)
            config.enable_stream(rs.stream.color, camera.width, camera.height, rs.format.rgb8, camera.fps)
            if camera.enable_depth:
                config.enable_stream(rs.stream.depth, camera.width, camera.height, rs.format.z16, camera.fps)
            pipeline.start(config)
            started = True
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(1500)
                color = frames.get_color_frame()
                if not color:
                    continue
                depth_frame = frames.get_depth_frame() if camera.enable_depth else None
                frame = CameraFrame(camera.name, int(color.get_frame_number()), float(color.get_timestamp()) / 1000, self._clock(), np.ascontiguousarray(color.get_data()).copy(), None if depth_frame is None else np.asanyarray(depth_frame.get_data()).copy())
                self._latest[camera.name] = frame
                if self.synchronizer is not None:
                    self.synchronizer.add(frame)
        except Exception as exc:  # noqa: BLE001 - surface backend failures to the reader thread
            self._errors[camera.name] = exc
        finally:
            if started:
                with suppress(RuntimeError):
                    pipeline.stop()

    def read(self, timeout_s: float = 1.0) -> dict[str, CameraFrame]:
        if self._errors:
            name, error = next(iter(self._errors.items()))
            raise RuntimeError(f"RealSense {name} failed: {error}") from error
        if self.synchronizer is not None:
            return self.synchronizer.read(timeout_s)
        deadline = self._clock() + timeout_s
        name = self.configs[0].name
        while name not in self._latest:
            if self._clock() >= deadline:
                raise TimeoutError("timed out waiting for RealSense frame")
            time.sleep(0.005)
        return {name: self._latest.pop(name)}

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3)
        self._threads.clear()
