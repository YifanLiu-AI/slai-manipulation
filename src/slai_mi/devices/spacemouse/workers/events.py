"""Publish normalized SpaceMouse state to an Isaac parent process."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import suppress
from pathlib import Path

from slai_mi.devices.spacemouse.device import SpaceMouse


def _send(sock: socket.socket, payload: dict) -> None:
    with suppress(OSError):
        sock.send(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--stale-timeout", type=float, default=0.10)
    parser.add_argument("--rate-hz", type=float, default=125.0)
    parser.add_argument("--backend", choices=("spnav", "evdev"), default="spnav")
    parser.add_argument("--event-device", type=Path)
    args = parser.parse_args()
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")

    parent_pid = os.getppid()
    sock = socket.socket(fileno=args.socket_fd)
    sock.setblocking(False)
    period = 1.0 / args.rate_hz
    packet_count = 0
    try:
        with SpaceMouse(
            deadzone=args.deadzone,
            stale_timeout=args.stale_timeout,
            backend=args.backend,
            event_device=args.event_device,
        ) as spacemouse:
            while os.getppid() == parent_pid:
                started = time.monotonic()
                try:
                    if sock.recv(64) in {b"", b"stop"}:
                        break
                except BlockingIOError:
                    pass
                motion, buttons, event_time = spacemouse.timestamped_state()
                diagnostics = spacemouse.diagnostics()
                packet_count += 1
                _send(
                    sock,
                    {
                        "type": "state",
                        "generated_monotonic_s": time.monotonic(),
                        "motion": motion.tolist(),
                        "buttons": buttons,
                        "packet_count": packet_count,
                        "last_event_time": event_time or None,
                        "event_count": diagnostics["event_count"],
                        "reconnect_count": diagnostics["reconnect_count"],
                        "connected": diagnostics["connected"],
                    },
                )
                elapsed = time.monotonic() - started
                if elapsed < period:
                    time.sleep(period - elapsed)
    except Exception as exc:
        _send(sock, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        sock.close()


if __name__ == "__main__":
    main()
