"""Dependency-isolated SpaceMouse process for Isaac simulation."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
WORKER_PYTHON = Path(
    os.environ.get(
        "SPACEMOUSE_WORKER_PYTHON",
        sys.executable,
    )
)
WORKER_MODULE = "slai_mi.devices.spacemouse.workers.events"
# Keep only the newest state while bounding work if a producer floods the socket.
MAX_DRAIN_PACKETS = 4096


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{current}" if current else str(SOURCE_ROOT)
    )
    return environment


class SpaceMouseProcess:
    """Expose an external SpaceMouse reader through the regular state interface."""

    def __init__(
        self,
        *,
        deadzone: float = 0.12,
        stale_timeout: float = 0.10,
        rate_hz: float = 125.0,
        backend: str = "spnav",
        event_device: Path | None = None,
    ) -> None:
        self.deadzone = deadzone
        self.stale_timeout = stale_timeout
        self.rate_hz = rate_hz
        self.backend = backend
        self.event_device = Path(event_device) if event_device is not None else None
        self._sock: socket.socket | None = None
        self._process: subprocess.Popen[Any] | None = None
        self._motion = np.zeros(6, dtype=np.float32)
        self._buttons: dict[int, bool] = {}
        self._pending_button_states: deque[dict[int, bool]] = deque()
        self._error: str | None = None
        self._packet_count = 0
        self._last_packet_time: float | None = None
        self._last_generated_time: float | None = None
        self._stale_since: float | None = None
        self._last_event_time: float | None = None
        self._worker_event_count = 0
        self._worker_reconnect_count = 0
        self._worker_connected = False

    def start(self) -> None:
        if self._process is not None or self._sock is not None:
            raise RuntimeError("SpaceMouse worker is already running")
        if not WORKER_PYTHON.is_file():
            raise FileNotFoundError(f"SpaceMouse worker Python not found: {WORKER_PYTHON}")
        self._motion.fill(0.0)
        self._buttons.clear()
        self._pending_button_states.clear()
        self._error = None
        self._packet_count = 0
        self._last_packet_time = None
        self._last_generated_time = None
        self._stale_since = None
        self._last_event_time = None
        self._worker_event_count = 0
        self._worker_reconnect_count = 0
        self._worker_connected = False
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        parent_sock.setblocking(False)
        command = [
            str(WORKER_PYTHON),
            "-m",
            WORKER_MODULE,
            "--socket-fd",
            str(child_sock.fileno()),
            "--deadzone",
            str(self.deadzone),
            "--stale-timeout",
            str(self.stale_timeout),
            "--rate-hz",
            str(self.rate_hz),
            "--backend",
            self.backend,
        ]
        if self.event_device is not None:
            command.extend(("--event-device", str(self.event_device)))
        try:
            self._process = subprocess.Popen(
                command,
                pass_fds=(child_sock.fileno(),),
                cwd=PROJECT_ROOT,
                env=_worker_environment(),
                start_new_session=True,
            )
        except Exception:
            parent_sock.close()
            child_sock.close()
            raise
        child_sock.close()
        self._sock = parent_sock

        deadline = time.monotonic() + 2.0
        while self._packet_count == 0 and time.monotonic() < deadline:
            self._drain_packets()
            if self._error is not None:
                self.stop()
                raise RuntimeError(self._error)
            if self._process.poll() is not None:
                self.stop()
                raise RuntimeError(
                    f"SpaceMouse worker exited with code {self._process.returncode} before ready"
                )
            time.sleep(0.01)
        if self._packet_count == 0:
            self.stop()
            raise RuntimeError("SpaceMouse worker produced no state packets within two seconds")

    def _drain_packets(self) -> None:
        if self._sock is None:
            raise RuntimeError("SpaceMouse worker is not running")
        for _index in range(MAX_DRAIN_PACKETS):
            try:
                packet = json.loads(self._sock.recv(65536).decode("utf-8"))
            except BlockingIOError:
                return
            packet_type = packet.get("type")
            if packet_type == "state":
                received = time.monotonic()
                generated = float(packet.get("generated_monotonic_s", float("nan")))
                if not np.isfinite(generated) or generated > received + 0.01:
                    self._error = "SpaceMouse worker returned an invalid generation timestamp"
                    continue
                self._packet_count += 1
                self._last_packet_time = received
                self._last_generated_time = generated
                self._last_event_time = packet.get("last_event_time")
                self._worker_event_count = int(packet.get("event_count", 0))
                self._worker_reconnect_count = int(packet.get("reconnect_count", 0))
                self._worker_connected = bool(packet.get("connected", True))
                packet_timeout = max(0.10, 4.0 / self.rate_hz)
                if received - generated > packet_timeout:
                    self._motion.fill(0.0)
                    if self._stale_since is None:
                        self._stale_since = received
                    continue
                motion = np.asarray(packet.get("motion"), dtype=np.float32)
                if motion.shape != (6,) or not np.isfinite(motion).all():
                    self._error = "SpaceMouse worker returned invalid motion"
                else:
                    self._stale_since = None
                    self._motion = motion
                    buttons = {
                        int(button): bool(pressed)
                        for button, pressed in packet.get("buttons", {}).items()
                    }
                    if buttons != self._buttons:
                        self._buttons = buttons
                        self._pending_button_states.append(buttons.copy())
            elif packet_type == "error":
                self._error = str(packet.get("message", "unknown SpaceMouse worker error"))
        try:
            self._sock.recv(65536, socket.MSG_PEEK)
        except BlockingIOError:
            return
        self._motion.fill(0.0)
        if self._stale_since is None:
            self._stale_since = time.monotonic()

    def state(self) -> tuple[np.ndarray, dict[int, bool]]:
        self._drain_packets()
        if self._process is not None and self._process.poll() is not None and self._error is None:
            self._error = f"SpaceMouse worker exited with code {self._process.returncode}"
        now = time.monotonic()
        if self._stale_since is not None and self._error is None:
            stale_duration = now - self._stale_since
            self._motion.fill(0.0)
            if stale_duration > 1.0:
                self._error = (
                    "SpaceMouse worker remained stale for "
                    f"{stale_duration:.3f}s; motion stayed cleared"
                )
        if self._error is not None:
            raise RuntimeError(self._error)
        buttons = (
            self._pending_button_states.popleft()
            if self._pending_button_states
            else self._buttons.copy()
        )
        return self._motion.copy(), buttons

    def diagnostics(
        self, *, refresh: bool = False
    ) -> dict[str, int | float | bool | None | str]:
        """Return counters; refresh only for out-of-band diagnostic callers."""
        if refresh and self._sock is not None:
            self._drain_packets()
        return {
            "worker_pid": self._process.pid if self._process is not None else None,
            "packet_count": self._packet_count,
            "last_packet_time": self._last_packet_time,
            "last_generated_time": self._last_generated_time,
            "stale_since": self._stale_since,
            "last_event_time": self._last_event_time,
            "event_count": self._worker_event_count,
            "reconnect_count": self._worker_reconnect_count,
            "connected": self._worker_connected,
            "error": self._error,
        }

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            if self._sock is not None:
                with suppress(OSError):
                    self._sock.send(b"stop")
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
        self._process = None
        self._sock = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
