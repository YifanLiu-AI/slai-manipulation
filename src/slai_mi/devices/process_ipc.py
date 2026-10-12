"""Versioned, length-prefixed MessagePack IPC for isolated hardware drivers."""

from __future__ import annotations

import socket
import struct
import time
from collections.abc import Mapping
from typing import Any

import msgpack

PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 1024 * 1024


def envelope(device_id: str, sequence: int, message_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "device_id": device_id,
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "type": message_type,
        "error_code": "OK",
        **payload,
    }


def send_message(sock: socket.socket, message: Mapping[str, Any]) -> None:
    payload = msgpack.packb(dict(message), use_bin_type=True)
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("driver IPC packet exceeds safety limit")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("driver IPC peer disconnected")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def receive_message(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack(">I", _read_exact(sock, 4))[0]
    if size > MAX_PACKET_BYTES:
        raise RuntimeError("driver IPC packet exceeds safety limit")
    value = msgpack.unpackb(_read_exact(sock, size), raw=False)
    if not isinstance(value, dict):
        raise TypeError("driver IPC message is not a map")
    return value


def validate_message(message: Mapping[str, Any], *, device_id: str) -> None:
    if message.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("driver IPC protocol version mismatch")
    if message.get("device_id") != device_id:
        raise RuntimeError("driver IPC device identity mismatch")
    if not isinstance(message.get("sequence"), int):
        raise TypeError("driver IPC sequence is missing")
    if not isinstance(message.get("monotonic_ns"), int):
        raise TypeError("driver IPC monotonic timestamp is missing")
