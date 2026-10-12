"""Hiwonder/LewanSoul serial-bus gripper (one physical DoF)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import serial

MOVE_TIME_WRITE = 1
TEMP_READ = 26
VOLTAGE_READ = 27
POSITION_READ = 28
LOAD_WRITE = 31
LOAD_READ = 32


@dataclass(frozen=True)
class GripperState:
    """One measured sample timestamped in the host monotonic clock domain."""

    position: int
    normalized_position: float
    target_position: int
    temperature_c: int
    voltage_v: float
    load_enabled: bool
    host_timestamp_s: float
    sequence: int
    binary_threshold: int = 500

    @property
    def actual_q(self) -> tuple[float]:
        """Binary gripper state: 0=open, 1=closed."""
        return (float(self.position >= self.binary_threshold),)

    @property
    def target_q(self) -> tuple[float]:
        """Binary gripper action: 0=open, 1=closed."""
        return (float(self.target_position >= self.binary_threshold),)

    @property
    def binary_state(self) -> int:
        return int(self.position >= self.binary_threshold)


class HiwonderGripper:
    """Persistent, thread-safe serial connection for servo ID 1 by default."""

    def __init__(
        self,
        port: str | Path,
        *,
        servo_id: int = 1,
        baud: int = 115200,
        move_time_ms: int = 1000,
        max_temperature_c: int = 60,
        binary_threshold: int = 500,
        serial_factory: Any = serial.Serial,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be in [0, 253]")
        if move_time_ms <= 0 or move_time_ms > 65535:
            raise ValueError("move_time_ms must be in [1, 65535]")
        self.port = str(port)
        self.servo_id = int(servo_id)
        self.baud = int(baud)
        self.move_time_ms = int(move_time_ms)
        if not 0 <= binary_threshold <= 1000:
            raise ValueError("binary_threshold must be in [0, 1000]")
        self.max_temperature_c = int(max_temperature_c)
        self.binary_threshold = int(binary_threshold)
        self._serial_factory = serial_factory
        self._monotonic = monotonic
        self._serial: Any | None = None
        self._lock = threading.Lock()
        self._target = 0
        self._sequence = 0

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open(self) -> None:
        if self._serial is not None:
            raise RuntimeError("gripper is already open")
        self._serial = self._serial_factory(self.port, self.baud, timeout=0.3)
        self._target = self._read_register(POSITION_READ, 2, signed=True)

    def close(self) -> None:
        with self._lock:
            port, self._serial = self._serial, None
            if port is not None:
                try:
                    port.write(self._packet(LOAD_WRITE, 0))
                    port.flush()
                finally:
                    port.close()

    def command(self, value: int) -> None:
        """Command normalized binary action: 0 -> position 0, 1 -> position 1000."""
        if value not in (0, 1):
            raise ValueError("gripper command must be 0 or 1")
        target = value * 1000
        with self._lock:
            port = self._require_open()
            port.write(self._packet(LOAD_WRITE, 1))
            duration = self.move_time_ms
            port.write(
                self._packet(
                    MOVE_TIME_WRITE,
                    target & 0xFF,
                    target >> 8,
                    duration & 0xFF,
                    duration >> 8,
                )
            )
            port.flush()
            self._target = target

    def unload(self) -> None:
        with self._lock:
            port = self._require_open()
            port.write(self._packet(LOAD_WRITE, 0))
            port.flush()

    def read_state(self) -> GripperState:
        """Read telemetry and timestamp completion with ``time.monotonic()``."""
        with self._lock:
            position = self._read_register_unlocked(POSITION_READ, 2, signed=True)
            temperature = self._read_register_unlocked(TEMP_READ, 1)
            voltage_mv = self._read_register_unlocked(VOLTAGE_READ, 2)
            load = self._read_register_unlocked(LOAD_READ, 1)
            received_at = float(self._monotonic())
            self._sequence += 1
            if temperature >= self.max_temperature_c and load:
                port = self._require_open()
                port.write(self._packet(LOAD_WRITE, 0))
                port.flush()
                raise RuntimeError(
                    f"gripper temperature {temperature}C reached safety limit; torque disabled"
                )
            return GripperState(
                position=position,
                normalized_position=min(1.0, max(0.0, position / 1000.0)),
                target_position=self._target,
                temperature_c=temperature,
                voltage_v=voltage_mv / 1000.0,
                load_enabled=bool(load),
                host_timestamp_s=received_at,
                sequence=self._sequence,
                binary_threshold=self.binary_threshold,
            )

    def _read_register(self, command: int, size: int, *, signed: bool = False) -> int:
        with self._lock:
            return self._read_register_unlocked(command, size, signed=signed)

    def _read_register_unlocked(self, command: int, size: int, *, signed: bool = False) -> int:
        port = self._require_open()
        port.reset_input_buffer()
        port.write(self._packet(command))
        port.flush()
        reply = port.read(size + 6)
        if len(reply) != size + 6 or reply[:2] != b"\x55\x55":
            raise RuntimeError(f"invalid gripper reply: {reply.hex(' ')}")
        if reply[2] != self.servo_id or reply[4] != command:
            raise RuntimeError(f"unexpected gripper reply: {reply.hex(' ')}")
        if reply[-1] != (~sum(reply[2:-1]) & 0xFF):
            raise RuntimeError("invalid gripper checksum")
        return int.from_bytes(reply[5:-1], "little", signed=signed)

    def _packet(self, command: int, *data: int) -> bytes:
        body = bytes((self.servo_id, len(data) + 3, command, *data))
        return b"\x55\x55" + body + bytes((~sum(body) & 0xFF,))

    def _require_open(self) -> Any:
        if self._serial is None:
            raise RuntimeError("gripper is not open")
        return self._serial
