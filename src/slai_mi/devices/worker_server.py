"""Single-client fail-closed server shared by hardware driver workers."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .process_ipc import envelope, receive_message, send_message, validate_message


class WorkerBackend(Protocol):
    def handle(self, message_type: str, payload: Mapping[str, Any], armed: bool) -> Mapping[str, Any]: ...
    def disable(self) -> None: ...
    def close(self) -> None: ...


def serve(*, socket_path: Path, device_id: str, backend_factory: Callable[[], WorkerBackend], watchdog_s: float) -> None:
    backend = backend_factory()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    armed = False
    try:
        server.bind(str(socket_path))
        server.listen(1)
        connection, _ = server.accept()
        connection.settimeout(min(0.25, watchdog_s))
        last_contact = time.monotonic()
        with connection:
            while True:
                try:
                    request = receive_message(connection)
                except TimeoutError:
                    if armed and time.monotonic() - last_contact > watchdog_s:
                        backend.disable()
                        armed = False
                    continue
                except (ConnectionError, OSError):
                    break
                last_contact = time.monotonic()
                sequence = request.get("sequence", -1)
                try:
                    validate_message(request, device_id=device_id)
                    message_type = str(request.get("type"))
                    if message_type == "hello":
                        result: Mapping[str, Any] = {"type": "ready"}
                    elif message_type == "heartbeat":
                        result = {"type": "heartbeat", "armed": armed}
                    elif message_type == "arm":
                        armed = True
                        result = {"type": "armed"}
                    elif message_type == "disable":
                        backend.disable()
                        armed = False
                        result = {"type": "disabled"}
                    elif message_type == "shutdown":
                        backend.disable()
                        break
                    else:
                        result = backend.handle(message_type, request, armed)
                    send_message(connection, envelope(device_id, sequence, str(result.get("type", message_type)), **{k: v for k, v in result.items() if k != "type"}))
                except Exception as exc:  # noqa: BLE001 - vendor failures cross IPC safely
                    backend.disable()
                    armed = False
                    send_message(connection, {**envelope(device_id, sequence, "error", message=str(exc)), "error_code": type(exc).__name__.upper()})
    finally:
        try:
            backend.disable()
        finally:
            backend.close()
            server.close()
            socket_path.unlink(missing_ok=True)
