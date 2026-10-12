"""Timestamp normalization and bounded-buffer synchronization for real collection."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    value: T
    monotonic_s: float
    device_s: float | None = None
    sequence: int | None = None
    valid: bool = True
    dropped_before: int = 0


class ClockMapper:
    """Estimate ``host = slope * device + offset`` with reset and drift gates."""

    def __init__(
        self,
        window: int = 180,
        *,
        slope_range: tuple[float, float] = (0.98, 1.02),
        min_fit_samples: int = 8,
        min_fit_span_s: float = 0.5,
    ) -> None:
        if (
            window < 2
            or min_fit_samples < 2
            or min_fit_span_s < 0.0
            or not 0.0 < slope_range[0] <= slope_range[1]
        ):
            raise ValueError("invalid clock mapper window or slope range")
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)
        self._last_device_time: float | None = None
        self._last_mapped: float | None = None
        self.slope_range = slope_range
        self.min_fit_samples = int(min_fit_samples)
        self.min_fit_span_s = float(min_fit_span_s)
        self.slope = 1.0
        self.offset = 0.0

    def update(self, device_s: float, host_s: float) -> float:
        if not np.isfinite((device_s, host_s)).all():
            raise ValueError("clock samples must be finite")
        device_s, host_s = float(device_s), float(host_s)
        if self._last_device_time is not None and device_s <= self._last_device_time:
            self._pairs.clear()
            self.slope = 1.0
        self._last_device_time = device_s
        self._pairs.append((device_s, host_s))
        values = np.asarray(self._pairs, dtype=np.float64)
        device, host = values[:, 0], values[:, 1]
        self.offset = float(np.median(host - device))
        if len(values) >= self.min_fit_samples and device[-1] - device[0] >= self.min_fit_span_s:
            centered = device - device.mean()
            denominator = float(centered @ centered)
            if denominator > 0.0:
                candidate = float(centered @ (host - host.mean()) / denominator)
                if self.slope_range[0] <= candidate <= self.slope_range[1]:
                    self.slope = candidate
                    self.offset = float(host.mean() - candidate * device.mean())
        mapped = self.map(device_s)
        if self._last_mapped is not None:
            mapped = max(mapped, self._last_mapped)
        self._last_mapped = mapped
        return mapped

    def map(self, device_s: float) -> float:
        return self.slope * float(device_s) + self.offset


class TimeSeries(Generic[T]):
    """Thread-safe retention-window series used by asynchronous device workers."""

    def __init__(self, retain_s: float = 5.0) -> None:
        if retain_s <= 0.0:
            raise ValueError("retain_s must be positive")
        self.retain_s = float(retain_s)
        self._samples: deque[TimedSample[T]] = deque()
        self._lock = threading.Lock()
        self.dropped_before = 0

    def append(self, sample: TimedSample[T]) -> None:
        with self._lock:
            if self._samples and sample.monotonic_s < self._samples[-1].monotonic_s:
                raise ValueError("samples must be appended in timestamp order")
            self._samples.append(sample)
            cutoff = sample.monotonic_s - self.retain_s
            while self._samples and self._samples[0].monotonic_s < cutoff:
                self._samples.popleft()
                self.dropped_before += 1

    def snapshot(self) -> tuple[TimedSample[T], ...]:
        with self._lock:
            return tuple(self._samples)

    def latest(self) -> TimedSample[T] | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def closest(self, target_s: float, max_delta_s: float) -> TimedSample[T] | None:
        samples = self.snapshot()
        if not samples:
            return None
        sample = min(samples, key=lambda item: abs(item.monotonic_s - target_s))
        return sample if sample.valid and abs(sample.monotonic_s - target_s) <= max_delta_s else None

    def before(self, target_s: float, max_age_s: float) -> TimedSample[T] | None:
        samples = [item for item in self.snapshot() if item.valid and item.monotonic_s <= target_s]
        if not samples:
            return None
        sample = samples[-1]
        return sample if target_s - sample.monotonic_s <= max_age_s else None

    def bracket(
        self, target_s: float, max_span_s: float
    ) -> tuple[TimedSample[T], TimedSample[T]] | None:
        samples = self.snapshot()
        before = next(
            (item for item in reversed(samples) if item.valid and item.monotonic_s <= target_s),
            None,
        )
        after = next(
            (item for item in samples if item.valid and item.monotonic_s >= target_s),
            None,
        )
        if before is None or after is None or after.monotonic_s - before.monotonic_s > max_span_s:
            return None
        return before, after


class DeviceClockFit:
    """Fit ``monotonic = slope * device + offset`` over a bounded window."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 2:
            raise ValueError("clock fit capacity must be at least two")
        self._mapper = ClockMapper(window=capacity, min_fit_samples=2, min_fit_span_s=0.0)

    def observe(self, device_s: float, monotonic_s: float) -> float:
        return self._mapper.update(device_s, monotonic_s)

    def to_monotonic(self, device_s: float) -> float:
        if not self._mapper._pairs:
            raise RuntimeError("clock fit has no observations")
        return self._mapper.map(device_s)


class BoundedBuffer(Generic[T]):
    """Timestamp-ordered, fixed-capacity buffer with nearest/ZOH/interpolation reads."""

    def __init__(self, capacity: int = 256, *, retain_s: float | None = None) -> None:
        if capacity < 2:
            raise ValueError("buffer capacity must be at least two")
        self.capacity = capacity
        self.retain_s = retain_s
        self._items: deque[TimedSample[T]] = deque()
        self.dropped_before = 0
        self.sequence_gaps = 0

    def append(self, sample: TimedSample[T]) -> None:
        if self._items and sample.monotonic_s < self._items[-1].monotonic_s:
            raise ValueError("samples must be appended in monotonic timestamp order")
        if (
            self._items
            and sample.sequence is not None
            and self._items[-1].sequence is not None
            and sample.sequence > self._items[-1].sequence + 1
        ):
            self.sequence_gaps += sample.sequence - self._items[-1].sequence - 1
        self._items.append(sample)
        while len(self._items) > self.capacity:
            self._items.popleft()
            self.dropped_before += 1
        if self.retain_s is not None:
            cutoff = sample.monotonic_s - self.retain_s
            while self._items and self._items[0].monotonic_s < cutoff:
                self._items.popleft()
                self.dropped_before += 1

    def snapshot(self) -> tuple[TimedSample[T], ...]:
        return tuple(self._items)

    def latest(self) -> TimedSample[T] | None:
        return self._items[-1] if self._items else None

    def nearest(self, timestamp_s: float) -> TimedSample[T]:
        self._require_items()
        return min(self._items, key=lambda item: abs(item.monotonic_s - timestamp_s))

    def zoh(self, timestamp_s: float) -> TimedSample[T]:
        self._require_items()
        candidates = [item for item in self._items if item.monotonic_s <= timestamp_s]
        if not candidates:
            raise LookupError("no sample exists at or before requested timestamp")
        return candidates[-1]

    def interpolate(self, timestamp_s: float) -> TimedSample[np.ndarray]:
        self._require_items()
        before = [item for item in self._items if item.monotonic_s <= timestamp_s]
        after = [item for item in self._items if item.monotonic_s >= timestamp_s]
        if not before or not after:
            raise LookupError("requested timestamp is not bracketed")
        left, right = before[-1], after[0]
        left_value = np.asarray(left.value, dtype=np.float64)
        right_value = np.asarray(right.value, dtype=np.float64)
        if left_value.shape != right_value.shape:
            raise ValueError("interpolation sample shapes differ")
        span = right.monotonic_s - left.monotonic_s
        alpha = 0.0 if span == 0.0 else (timestamp_s - left.monotonic_s) / span
        value = left_value + alpha * (right_value - left_value)
        return TimedSample(value=value, monotonic_s=timestamp_s)

    def _require_items(self) -> None:
        if not self._items:
            raise LookupError("buffer is empty")


@dataclass(frozen=True)
class SourceDiagnostic:
    skew_ms: float
    age_ms: float
    valid: bool
    reason: str | None = None
    dropped_before: int = 0
    sequence_gaps: int = 0


@dataclass(frozen=True)
class SynchronizedFrame:
    timestamp_s: float
    cameras: Mapping[str, Any]
    states: Mapping[str, np.ndarray]
    command: Any
    diagnostics: Mapping[str, SourceDiagnostic]

    @property
    def valid(self) -> bool:
        return all(item.valid for item in self.diagnostics.values())

    @property
    def ur5_state(self) -> np.ndarray:
        return self.states["ur5"]

    @property
    def wuji_state(self) -> np.ndarray:
        return self.states["wuji"]


class RealFrameSynchronizer:
    """Use the primary camera as timeline and align all other real sources."""

    def __init__(
        self,
        *,
        camera_roles: Sequence[str] = ("primary", "secondary", "wrist"),
        primary_role: str = "primary",
        state_channels: Sequence[str] = ("ur5", "wuji"),
        capacity: int = 256,
        retain_s: float | None = 5.0,
        max_camera_skew_ms: float = 20.0,
        max_camera_age_ms: float = 100.0,
        max_state_age_ms: float = 100.0,
        max_command_age_ms: float = 250.0,
    ) -> None:
        roles = tuple(camera_roles)
        channels = tuple(state_channels)
        if not roles or len(set(roles)) != len(roles) or primary_role not in roles:
            raise ValueError("camera roles must be unique and contain primary_role")
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("state channels must be non-empty and unique")
        self.camera_roles, self.primary_role = roles, primary_role
        self.cameras = {role: BoundedBuffer[Any](capacity, retain_s=retain_s) for role in roles}
        self.states = {
            name: BoundedBuffer[np.ndarray](capacity, retain_s=retain_s) for name in channels
        }
        self.ur5 = self.states.get("ur5")
        self.wuji = self.states.get("wuji")
        self.commands = BoundedBuffer[Any](capacity, retain_s=retain_s)
        self.max_camera_skew_ms = max_camera_skew_ms
        self.max_camera_age_ms = max_camera_age_ms
        self.max_state_age_ms = max_state_age_ms
        self.max_command_age_ms = max_command_age_ms

    @classmethod
    def from_input_schema(cls, schema: Mapping[str, Any]) -> RealFrameSynchronizer:
        capture = schema["capture"]
        sync = schema["synchronization"]
        roles = tuple(
            str(camera["role"])
            for camera in capture["cameras"]
            if camera.get("enabled", True)
        )
        channel_specs = sync["state_channels"]
        return cls(
            camera_roles=roles,
            primary_role=str(capture["primary_timeline_role"]),
            state_channels=tuple(
                str(item["name"]) if isinstance(item, Mapping) else str(item)
                for item in channel_specs
            ),
            retain_s=float(sync["retain_s"]),
            max_camera_skew_ms=float(sync["max_camera_skew_ms"]),
            max_camera_age_ms=float(sync["max_camera_age_ms"]),
            max_state_age_ms=float(sync["max_state_age_ms"]),
            max_command_age_ms=float(sync["max_command_age_ms"]),
        )

    def synchronize(self, primary: TimedSample[Any], *, now_s: float) -> SynchronizedFrame:
        target = primary.monotonic_s
        camera_samples = {self.primary_role: primary}
        diagnostics: dict[str, SourceDiagnostic] = {}
        for role in self.camera_roles:
            if role == self.primary_role:
                continue
            camera_samples[role] = self.cameras[role].nearest(target)
        for role, sample in camera_samples.items():
            skew = abs(sample.monotonic_s - target) * 1000.0
            buffer = self.cameras[role]
            diagnostics[f"camera.{role}"] = self._diagnostic(
                skew,
                max(0.0, now_s - sample.monotonic_s) * 1000.0,
                self.max_camera_skew_ms,
                self.max_camera_age_ms,
                "camera skew or age exceeded",
                sample.dropped_before + buffer.dropped_before,
                buffer.sequence_gaps,
            )
        states = {name: buffer.interpolate(target) for name, buffer in self.states.items()}
        for name, buffer in self.states.items():
            nearest = buffer.nearest(target)
            age = abs(nearest.monotonic_s - target) * 1000.0
            diagnostics[name] = self._diagnostic(
                age,
                age,
                self.max_state_age_ms,
                self.max_state_age_ms,
                "state too old",
                nearest.dropped_before + buffer.dropped_before,
                buffer.sequence_gaps,
            )
        command = self.commands.zoh(target)
        command_age = (target - command.monotonic_s) * 1000.0
        diagnostics["command"] = self._diagnostic(
            0.0,
            command_age,
            float("inf"),
            self.max_command_age_ms,
            "command too old",
            command.dropped_before + self.commands.dropped_before,
            self.commands.sequence_gaps,
        )
        return SynchronizedFrame(
            timestamp_s=target,
            cameras={role: sample.value for role, sample in camera_samples.items()},
            states={name: np.asarray(sample.value) for name, sample in states.items()},
            command=command.value,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostic(
        skew: float,
        age: float,
        skew_limit: float,
        age_limit: float,
        reason: str,
        dropped_before: int = 0,
        sequence_gaps: int = 0,
    ) -> SourceDiagnostic:
        valid = bool(
            np.isfinite((skew, age)).all()
            and skew <= skew_limit
            and age <= age_limit
        )
        return SourceDiagnostic(
            skew,
            age,
            valid,
            None if valid else reason,
            int(dropped_before),
            int(sequence_gaps),
        )
