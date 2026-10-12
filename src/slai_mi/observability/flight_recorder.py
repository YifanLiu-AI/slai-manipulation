"""Bounded in-memory control-loop trace with an on-stop disk dump."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def default_flight_recorder_directory() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "slai-mi" / "flight-recorder"


@dataclass(frozen=True)
class ControlCycleRecord:
    """One physical control-loop iteration."""

    monotonic_s: float
    spacemouse_motion: tuple[float, float, float, float, float, float]
    spacemouse_buttons: tuple[int, ...]
    spacemouse_sample_age_s: float | None
    ur5_commanded_twist: tuple[float, float, float, float, float, float]
    ur5_tcp_z_m: float | None
    floor_guard_active: bool
    phase: str
    note: str = ""


class FlightRecorder:
    """Fixed-capacity ring buffer for post-incident control-loop inspection."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("flight recorder capacity must be positive")
        self._buffer: deque[ControlCycleRecord] = deque(maxlen=capacity)

    def record(self, entry: ControlCycleRecord) -> None:
        self._buffer.append(entry)

    def __len__(self) -> int:
        return len(self._buffer)

    def snapshot(self) -> list[ControlCycleRecord]:
        return list(self._buffer)

    def dump(
        self,
        *,
        reason: str,
        directory: Path | None = None,
        clock: Any = time.time,
    ) -> Path:
        target_dir = directory if directory is not None else default_flight_recorder_directory()
        target_dir.mkdir(parents=True, exist_ok=True)
        dumped_at = clock()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(dumped_at))
        path = target_dir / f"flight-{stamp}-{os.getpid()}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"type": "failure", "reason": reason, "dumped_at": dumped_at})
                + "\n"
            )
            for record in self._buffer:
                stream.write(json.dumps({"type": "cycle", **asdict(record)}) + "\n")
        return path


@dataclass
class NullFlightRecorder:
    """No-op recorder for call sites that have not opted into tracing."""

    _dummy: list[Any] = field(default_factory=list)

    def record(self, _entry: ControlCycleRecord) -> None:
        return

    def __len__(self) -> int:
        return 0

    def snapshot(self) -> list[ControlCycleRecord]:
        return []

    def dump(
        self,
        *,
        reason: str,
        directory: Path | None = None,
        clock: Any = time.time,
    ) -> Path | None:
        return None
