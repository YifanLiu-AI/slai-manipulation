"""Hiwonder/LewanSoul serial-bus gripper with normalized continuous control."""

from __future__ import annotations

import math
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


class HiwonderError(RuntimeError):
    """Hiwonder communication, protocol, or safety failure."""


class HiwonderTimeout(HiwonderError):
    """The configured servo did not return a complete reply."""


def _normalize_position(position: int, open_position: int, closed_position: int) -> float:
    return float(min(1.0, max(0.0, (position - open_position) / (closed_position - open_position))))


def _denormalize_position(value: float, open_position: int, closed_position: int) -> int:
    return round(open_position + value * (closed_position - open_position))


@dataclass(frozen=True)
class HiwonderGripperState:
    """One Hiwonder sample expressed through the shared gripper state contract."""

    position: int
    normalized_position: float
    target_position: int
    normalized_target_position: float
    temperature_c: int
    voltage_v: float
    load_enabled: bool
    host_timestamp_s: float
    sequence: int
    binary_threshold: int
    closing_increases_position: bool
    feedback_position_bias: int = 0
    speed_raw: int | None = None
    load_raw: int | None = None
    current_raw: int | None = None
    moving: bool | None = None

    def _closed_at(self, position: int) -> bool:
        if self.closing_increases_position:
            return position >= self.binary_threshold
        return position <= self.binary_threshold

    @property
    def actual_q(self) -> tuple[float]:
        return (float(self._closed_at(self.position - self.feedback_position_bias)),)

    @property
    def target_q(self) -> tuple[float]:
        return (float(self._closed_at(self.target_position)),)

    @property
    def continuous_actual_q(self) -> tuple[float]:
        return (self.normalized_position,)

    @property
    def continuous_target_q(self) -> tuple[float]:
        return (self.normalized_target_position,)

    @property
    def binary_state(self) -> int:
        return int(self._closed_at(self.position - self.feedback_position_bias))


class HiwonderGripper:
    """Persistent Hiwonder adapter implementing the shared normalized-position API."""

    driver_name = "hiwonder"

    def __init__(
        self,
        port: str | Path,
        *,
        servo_id: int = 1,
        baud: int = 115_200,
        open_position: int = 0,
        closed_position: int = 1000,
        feedback_position_bias: int = 0,
        binary_threshold: int | None = None,
        full_stroke_time_ms: int = 1000,
        min_move_time_ms: int = 20,
        max_temperature_c: int = 60,
        open_unload_delay_s: float | None = None,
        timeout_s: float = 0.3,
        serial_factory: Any = serial.Serial,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be in [0, 253]")
        if not 0 <= open_position <= 1000 or not 0 <= closed_position <= 1000:
            raise ValueError("Hiwonder gripper endpoints must be in [0, 1000]")
        if open_position == closed_position:
            raise ValueError("open_position and closed_position must differ")
        if not -1000 <= feedback_position_bias <= 1000:
            raise ValueError("feedback_position_bias must be in [-1000, 1000]")
        low, high = sorted((open_position, closed_position))
        threshold = round((open_position + closed_position) / 2)
        if binary_threshold is not None:
            threshold = int(binary_threshold)
        if not low <= threshold <= high:
            raise ValueError("binary_threshold must lie between the open and closed positions")
        if not 1 <= full_stroke_time_ms <= 65_535:
            raise ValueError("full_stroke_time_ms must be in [1, 65535]")
        if not 1 <= min_move_time_ms <= full_stroke_time_ms:
            raise ValueError("min_move_time_ms must be in [1, full_stroke_time_ms]")
        if max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be positive")
        if open_unload_delay_s is not None and (
            not math.isfinite(open_unload_delay_s) or open_unload_delay_s <= 0.0
        ):
            raise ValueError("open_unload_delay_s must be positive when configured")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")

        self.port = str(port)
        self.servo_id = int(servo_id)
        self.baud = int(baud)
        self.open_position = int(open_position)
        self.closed_position = int(closed_position)
        self.feedback_position_bias = int(feedback_position_bias)
        self.binary_threshold = threshold
        self.full_stroke_time_ms = int(full_stroke_time_ms)
        self.min_move_time_ms = int(min_move_time_ms)
        self.max_temperature_c = int(max_temperature_c)
        self.open_unload_delay_s = open_unload_delay_s
        self.timeout_s = float(timeout_s)
        self._serial_factory = serial_factory
        self._monotonic = monotonic
        self._serial: Any | None = None
        self._lock = threading.Lock()
        self._target = self.open_position
        self._target_commanded_at_s: float | None = None
        self._sequence = 0

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open(self) -> None:
        with self._lock:
            if self._serial is not None:
                raise RuntimeError("gripper is already open")
            port = self._serial_factory(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout_s,
                write_timeout=self.timeout_s,
            )
            self._serial = port
            try:
                self._target = self._read_register_unlocked(
                    POSITION_READ, 2, signed=True
                )
            except BaseException:
                self._serial = None
                port.close()
                raise

    def close(self) -> None:
        """Close the serial link without changing the commanded torque state."""
        with self._lock:
            port, self._serial = self._serial, None
            if port is not None:
                port.close()

    def command(self, value: int) -> None:
        if value not in (0, 1):
            raise ValueError("gripper command must be 0 or 1")
        self.command_position(float(value))

    def command_position(self, value: float) -> None:
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("normalized gripper position must be in [0, 1]")
        target = _denormalize_position(
            normalized,
            self.open_position,
            self.closed_position,
        )
        span = abs(self.closed_position - self.open_position)
        with self._lock:
            port = self._require_open()
            distance = abs(target - self._target)
            duration = max(
                self.min_move_time_ms,
                round(distance / span * self.full_stroke_time_ms),
            )
            port.write(self._packet(LOAD_WRITE, 1))
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
            self._target_commanded_at_s = float(self._monotonic())

    def unload(self) -> None:
        with self._lock:
            port = self._require_open()
            port.write(self._packet(LOAD_WRITE, 0))
            port.flush()

    def read_state(self, *, enforce_safety: bool = True) -> HiwonderGripperState:
        with self._lock:
            position = self._read_register_unlocked(POSITION_READ, 2, signed=True)
            temperature = self._read_register_unlocked(TEMP_READ, 1)
            voltage_mv = self._read_register_unlocked(VOLTAGE_READ, 2)
            load = self._read_register_unlocked(LOAD_READ, 1)
            received_at = float(self._monotonic())
            self._sequence += 1
            if enforce_safety and temperature >= self.max_temperature_c:
                if load:
                    port = self._require_open()
                    port.write(self._packet(LOAD_WRITE, 0))
                    port.flush()
                raise HiwonderError(
                    f"Hiwonder safety stop: temperature={temperature}C; torque disabled"
                )
            should_unload_open = bool(
                self.open_unload_delay_s is not None
                and load
                and self._target == self.open_position
                and self._target_commanded_at_s is not None
                and received_at - self._target_commanded_at_s >= self.open_unload_delay_s
            )
            if should_unload_open:
                port = self._require_open()
                port.write(self._packet(LOAD_WRITE, 0))
                port.flush()
                self._target_commanded_at_s = None
                load = 0
            return HiwonderGripperState(
                position=position,
                normalized_position=_normalize_position(
                    position - self.feedback_position_bias,
                    self.open_position,
                    self.closed_position,
                ),
                target_position=self._target,
                normalized_target_position=_normalize_position(
                    self._target, self.open_position, self.closed_position
                ),
                temperature_c=temperature,
                voltage_v=voltage_mv / 1000.0,
                load_enabled=bool(load),
                host_timestamp_s=received_at,
                sequence=self._sequence,
                binary_threshold=self.binary_threshold,
                closing_increases_position=self.closed_position > self.open_position,
                feedback_position_bias=self.feedback_position_bias,
            )

    @classmethod
    def packet(cls, servo_id: int, command: int, *data: int) -> bytes:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be in [0, 253]")
        if any(not 0 <= value <= 255 for value in (command, *data)):
            raise ValueError("Hiwonder packet bytes must be in [0, 255]")
        body = bytes((servo_id, len(data) + 3, command, *data))
        return b"\x55\x55" + body + bytes((~sum(body) & 0xFF,))

    def _packet(self, command: int, *data: int) -> bytes:
        return self.packet(self.servo_id, command, *data)

    def _read_register_unlocked(
        self,
        command: int,
        size: int,
        *,
        signed: bool = False,
    ) -> int:
        port = self._require_open()
        port.reset_input_buffer()
        port.write(self._packet(command))
        port.flush()
        reply = port.read(size + 6)
        if len(reply) != size + 6:
            raise HiwonderTimeout(
                f"Hiwonder ID {self.servo_id} did not return {size} bytes on {self.port}"
            )
        if reply[:2] != b"\x55\x55" or reply[2] != self.servo_id or reply[4] != command:
            raise HiwonderError(f"unexpected Hiwonder reply: {reply.hex(' ')}")
        if reply[-1] != (~sum(reply[2:-1]) & 0xFF):
            raise HiwonderError("invalid Hiwonder checksum")
        return int.from_bytes(reply[5:-1], "little", signed=signed)

    def _require_open(self) -> Any:
        if self._serial is None:
            raise RuntimeError("gripper is not open")
        return self._serial
