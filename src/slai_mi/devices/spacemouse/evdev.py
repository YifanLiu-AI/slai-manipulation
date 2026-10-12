"""Minimal Linux evdev reader for six-axis SpaceMouse events."""

from __future__ import annotations

import os
import struct
from collections import deque
from pathlib import Path

from . import spnav

EV_SYN = 0
EV_KEY = 1
EV_REL = 2
SYN_REPORT = 0
REL_AXIS_COUNT = 6
INPUT_EVENT = struct.Struct("@llHHi")


class EvdevReader:
    """Translate kernel input events into the existing spnav event types."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._buffer = bytearray()
        self._events: deque[spnav.SpnavMotionEvent | spnav.SpnavButtonEvent] = deque()
        self._motion = [0] * REL_AXIS_COUNT
        self._motion_seen = False

    def open(self) -> None:
        if self._fd is not None:
            raise RuntimeError("evdev reader is already open")
        self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def feed(self, data: bytes) -> None:
        """Decode complete input_event records, retaining an incomplete tail."""
        self._buffer.extend(data)
        while len(self._buffer) >= INPUT_EVENT.size:
            record = bytes(self._buffer[: INPUT_EVENT.size])
            del self._buffer[: INPUT_EVENT.size]
            _seconds, _microseconds, event_type, code, value = INPUT_EVENT.unpack(record)
            if event_type == EV_REL and 0 <= code < REL_AXIS_COUNT:
                self._motion[code] = value
                self._motion_seen = True
            elif event_type == EV_KEY:
                # Linux evdev exposes BTN_0.. as 256+N, while the rest of
                # this project (and spnav) uses the logical bnum N.
                logical_bnum = code - 256 if code >= 256 else code
                self._events.append(
                    spnav.SpnavButtonEvent(logical_bnum, bool(value))
                )
            elif event_type == EV_SYN and code == SYN_REPORT and self._motion_seen:
                self._events.append(
                    spnav.SpnavMotionEvent(
                        tuple(self._motion[:3]),
                        tuple(self._motion[3:]),
                        0,
                    )
                )
                self._motion[:] = [0] * REL_AXIS_COUNT
                self._motion_seen = False

    def poll_event(self) -> spnav.SpnavMotionEvent | spnav.SpnavButtonEvent | None:
        if self._events:
            return self._events.popleft()
        if self._fd is None:
            raise RuntimeError("evdev reader is not open")
        try:
            data = os.read(self._fd, INPUT_EVENT.size * 64)
        except BlockingIOError:
            return None
        if not data:
            raise RuntimeError(f"SpaceMouse evdev device closed: {self.path}")
        self.feed(data)
        return self._events.popleft() if self._events else None

