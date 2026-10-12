"""Local one-shot home commands shared by collection and hardware runtimes."""

from __future__ import annotations

import fcntl
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

HOME_PACKET = b"slai-manipulation-home-v1"
PING_PACKET = b"slai-manipulation-home-ping-v1"
MAX_PACKET_BYTES = 256


def default_task2_home_socket() -> Path:
    """Return the stable per-user endpoint owned by the active TASK2 controller."""
    configured = os.environ.get("SLAI_TASK2_HOME_SOCKET")
    if configured:
        return Path(configured).expanduser()
    runtime_root = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime_root) if runtime_root else Path("/tmp") / f"slai-{os.getuid()}"
    return base / "slai-manipulation" / "task2-home.sock"


def default_station_control_lock() -> Path:
    return default_task2_home_socket().with_name("station-control.lock")


class StationControlBusyError(RuntimeError):
    pass


@contextmanager
def exclusive_station_control(
    path: Path | str | None = None,
) -> Iterator[None]:
    """Prevent independent processes from competing for station hardware."""
    lock_path = Path(path) if path is not None else default_station_control_lock()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StationControlBusyError(
                "station hardware is already owned by another controller"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def station_control_is_busy(path: Path | str | None = None) -> bool:
    try:
        with exclusive_station_control(path):
            return False
    except StationControlBusyError:
        return True


def _endpoint_accepts_datagrams(path: Path) -> bool:
    if not path.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        try:
            sender.sendto(PING_PACKET, str(path))
        except (ConnectionRefusedError, FileNotFoundError):
            return False
    return True


class HomeCommandReceiver:
    """Receive non-blocking home requests on a process-specific Unix socket."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._socket is not None:
            raise RuntimeError("home command receiver is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if _endpoint_accepts_datagrams(self.path):
                raise FileExistsError(f"home command socket already exists: {self.path}")
            self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(str(self.path))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self._socket = sock

    def poll(self) -> bool:
        if self._socket is None:
            raise RuntimeError("home command receiver is not running")
        requested = False
        while True:
            try:
                packet = self._socket.recv(MAX_PACKET_BYTES)
            except BlockingIOError:
                break
            requested |= packet == HOME_PACKET
        return requested

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def request_home(paths: tuple[Path | str, ...]) -> None:
    """Deliver one request to every controller and report all failed targets."""
    failed: list[str] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        for path in paths:
            try:
                sender.sendto(HOME_PACKET, str(path))
            except OSError as exc:
                failed.append(f"{path}: {exc}")
    if failed:
        raise RuntimeError("failed to request coordinated home: " + "; ".join(failed))


def request_home_if_active(path: Path | str) -> bool:
    """Request HOME from an existing owner without opening any hardware."""
    endpoint = Path(path)
    if not endpoint.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        try:
            sender.sendto(HOME_PACKET, str(endpoint))
        except (ConnectionRefusedError, FileNotFoundError):
            return False
    return True
