"""Lifecycle client for a hardware driver running under a selected Python."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .process_ipc import envelope, receive_message, send_message, validate_message


class DriverProcess:
    def __init__(self, *, device_id: str, python: Path, module: str, arguments: list[str], startup_timeout_s: float = 8.0, request_timeout_s: float = 1.0):
        self.device_id = device_id
        self.python = Path(python)
        self.module = module
        self.arguments = arguments
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.process: subprocess.Popen[bytes] | None = None
        self.socket: socket.socket | None = None
        self.sequence = 0
        self.armed = False
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._request_lock = threading.RLock()

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError(f"{self.device_id} driver is already started")
        if not self.python.exists():
            raise FileNotFoundError(self.python)
        self._temporary = tempfile.TemporaryDirectory(prefix=f"slai-{self.device_id}-")
        socket_path = Path(self._temporary.name) / "driver.sock"
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH"))))
        self.process = subprocess.Popen(
            [str(self.python), "-m", self.module, "--socket", str(socket_path), "--device-id", self.device_id, *self.arguments],
            env=env,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError(f"{self.device_id} driver exited during startup")
            if socket_path.exists():
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.request_timeout_s)
                try:
                    sock.connect(str(socket_path))
                    self.socket = sock
                    response = self.request("hello")
                    if response.get("type") != "ready":
                        raise RuntimeError(f"invalid ready handshake: {response}")
                    return
                except (ConnectionRefusedError, FileNotFoundError):
                    sock.close()
                    self.socket = None
            time.sleep(0.02)
        self.stop()
        raise TimeoutError(f"{self.device_id} driver startup timed out")

    def request(self, message_type: str, **payload: Any) -> dict[str, Any]:
        with self._request_lock:
            if self.process is None or self.process.poll() is not None or self.socket is None:
                raise RuntimeError(f"{self.device_id} driver is not healthy")
            self.sequence += 1
            sequence = self.sequence
            send_message(self.socket, envelope(self.device_id, sequence, message_type, **payload))
            try:
                response = receive_message(self.socket)
            except TimeoutError:
                # A late command acknowledgement cannot be safely associated with a
                # later request. Closing IPC makes the worker disconnect and disable.
                sock, self.socket = self.socket, None
                self.armed = False
                sock.close()
                raise TimeoutError(f"{self.device_id} driver request timed out and was disabled")
            validate_message(response, device_id=self.device_id)
            if response["sequence"] != sequence:
                raise RuntimeError("driver IPC response sequence mismatch")
            if response.get("error_code") != "OK":
                raise RuntimeError(f"{self.device_id} driver {response['error_code']}: {response.get('message', '')}")
            return response

    def heartbeat(self) -> int:
        response = self.request("heartbeat")
        if self.armed and response.get("armed") is not True:
            self.armed = False
            raise RuntimeError(f"{self.device_id} worker disarmed unexpectedly")
        return int(response["monotonic_ns"])

    def arm(self) -> None:
        self.request("arm")
        self.armed = True

    def disable(self) -> None:
        if self.process is not None and self.process.poll() is None and self.socket is not None:
            try:
                self.request("disable")
            except (ConnectionError, OSError, RuntimeError):
                pass
        self.armed = False

    def stop(self) -> None:
        process, sock = self.process, self.socket
        self.process = None
        self.socket = None
        self.armed = False
        if process is not None and process.poll() is None and sock is not None:
            try:
                self.sequence += 1
                send_message(sock, envelope(self.device_id, self.sequence, "shutdown"))
            except OSError:
                pass
        if sock is not None:
            sock.close()
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.disable()
        self.stop()
