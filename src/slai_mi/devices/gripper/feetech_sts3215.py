"""FEETECH STS3215 serial-bus gripper (one physical DoF)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import serial

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACCELERATION = 41
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63
ADDR_MOVING = 66
ADDR_PRESENT_CURRENT = 69

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
BROADCAST_ID = 0xFE


class ServoError(RuntimeError):
    """FEETECH communication, protocol, or safety failure."""


class ServoTimeout(ServoError):
    """No valid status packet arrived before the configured deadline."""


@dataclass(frozen=True)
class ServoStatus:
    """Raw feedback from the STS3215 control table."""

    servo_id: int
    position: int
    speed_raw: int
    load_raw: int
    voltage_v: float
    temperature_c: int
    moving: bool
    current_raw: int
    torque_enabled: bool


@dataclass(frozen=True)
class GripperState:
    """One gripper sample timestamped in the host monotonic clock domain."""

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
    speed_raw: int
    load_raw: int
    current_raw: int
    moving: bool

    def _closed_at(self, position: int) -> bool:
        if self.closing_increases_position:
            return position >= self.binary_threshold
        return position <= self.binary_threshold

    @property
    def actual_q(self) -> tuple[float]:
        """Binary gripper state: 0=open, 1=closed."""
        return (float(self._closed_at(self.position)),)

    @property
    def target_q(self) -> tuple[float]:
        """Binary gripper target: 0=open, 1=closed."""
        return (float(self._closed_at(self.target_position)),)

    @property
    def continuous_actual_q(self) -> tuple[float]:
        """Measured gripper position, normalized as 0=open and 1=closed."""
        return (self.normalized_position,)

    @property
    def continuous_target_q(self) -> tuple[float]:
        """Commanded gripper position, normalized as 0=open and 1=closed."""
        return (self.normalized_target_position,)

    @property
    def binary_state(self) -> int:
        return int(self._closed_at(self.position))

    @property
    def load_magnitude(self) -> int:
        return _signed_feedback_magnitude(self.load_raw)

    @property
    def current_magnitude(self) -> int:
        return _signed_feedback_magnitude(self.current_raw)


class STSBus:
    """Minimal FEETECH STS/SCS half-duplex packet bus."""

    def __init__(
        self,
        port: str | Path,
        baudrate: int = 1_000_000,
        timeout: float = 0.08,
        *,
        serial_factory: Any = serial.Serial,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if timeout <= 0.0:
            raise ValueError("timeout must be positive")
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self._monotonic = monotonic
        self._sleeper = sleeper
        self.serial = serial_factory(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,
            write_timeout=self.timeout,
        )

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def packet(servo_id: int, instruction: int, parameters: Iterable[int] = ()) -> bytes:
        params = list(parameters)
        if not 0 <= servo_id <= BROADCAST_ID:
            raise ValueError("servo_id must be in [0, 254]")
        if not 0 <= instruction <= 0xFF:
            raise ValueError("instruction must be a byte")
        if any(not 0 <= item <= 0xFF for item in params):
            raise ValueError("packet parameters must be bytes")
        body = [servo_id, len(params) + 2, instruction, *params]
        return bytes((0xFF, 0xFF, *body, (~sum(body)) & 0xFF))

    @staticmethod
    def extract_packets(data: bytes) -> list[bytes]:
        packets: list[bytes] = []
        for start in range(max(0, len(data) - 5)):
            if data[start : start + 2] != b"\xff\xff":
                continue
            length = data[start + 3]
            end = start + 4 + length
            if length < 2 or end > len(data):
                continue
            packet = data[start:end]
            if (sum(packet[2:]) & 0xFF) == 0xFF:
                packets.append(packet)
        return packets

    def _exchange(self, request: bytes, *, expect_response: bool = True) -> bytes | None:
        self.serial.reset_input_buffer()
        self.serial.write(request)
        self.serial.flush()
        if not expect_response:
            return None

        deadline = self._monotonic() + self.timeout
        received = bytearray()
        while self._monotonic() < deadline:
            waiting = self.serial.in_waiting
            if waiting:
                received.extend(self.serial.read(waiting))
                for packet in self.extract_packets(bytes(received)):
                    # Several USB adapters echo the transmitted instruction.
                    if packet != request:
                        return packet
            self._sleeper(0.001)
        raise ServoTimeout(f"no valid FEETECH response; received={received.hex(' ')}")

    def ping(self, servo_id: int) -> bool:
        try:
            response = self._exchange(self.packet(servo_id, INST_PING))
        except ServoTimeout:
            return False
        return response is not None and response[2] == servo_id and response[4] == 0

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        if not 0 <= address <= 0xFF:
            raise ValueError("address must be a byte")
        if not 1 <= size <= 0xFF:
            raise ValueError("size must be in [1, 255]")
        response = self._exchange(self.packet(servo_id, INST_READ, (address, size)))
        assert response is not None
        if response[2] != servo_id:
            raise ServoError(f"unexpected FEETECH response ID {response[2]}")
        if response[4]:
            raise ServoError(f"servo {servo_id} returned error 0x{response[4]:02x}")
        data = response[5:-1]
        if len(data) != size:
            raise ServoError(f"expected {size} FEETECH data bytes, got {len(data)}")
        return data

    def write(self, servo_id: int, address: int, data: Iterable[int]) -> None:
        payload = [address, *data]
        response = self._exchange(
            self.packet(servo_id, INST_WRITE, payload),
            expect_response=servo_id != BROADCAST_ID,
        )
        if response is None:
            return
        if response[2] != servo_id:
            raise ServoError(f"unexpected FEETECH response ID {response[2]}")
        if response[4]:
            raise ServoError(f"servo {servo_id} returned error 0x{response[4]:02x}")

    def read_u8(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[0]

    def read_u16(self, servo_id: int, address: int) -> int:
        return int.from_bytes(self.read(servo_id, address, 2), "little")

    def torque_enable(self, servo_id: int, enabled: bool) -> None:
        self.write(servo_id, ADDR_TORQUE_ENABLE, (int(enabled),))

    def set_position(
        self,
        servo_id: int,
        position: int,
        *,
        speed: int = 250,
        acceleration: int = 8,
        movement_time: int = 0,
    ) -> None:
        for name, value, upper in (
            ("position", position, 4095),
            ("speed", speed, 65535),
            ("movement_time", movement_time, 65535),
            ("acceleration", acceleration, 255),
        ):
            if not 0 <= value <= upper:
                raise ValueError(f"{name} must be in [0, {upper}]")
        self.write(
            servo_id,
            ADDR_GOAL_ACCELERATION,
            (
                acceleration,
                position & 0xFF,
                (position >> 8) & 0xFF,
                movement_time & 0xFF,
                (movement_time >> 8) & 0xFF,
                speed & 0xFF,
                (speed >> 8) & 0xFF,
            ),
        )

    def get_position(self, servo_id: int) -> int:
        return self.read_u16(servo_id, ADDR_PRESENT_POSITION)

    def get_status(self, servo_id: int) -> ServoStatus:
        # Addresses 56..70 form one contiguous telemetry block, so one bus
        # transaction gives a coherent and low-latency feedback sample.
        data = self.read(servo_id, ADDR_PRESENT_POSITION, 15)
        return ServoStatus(
            servo_id=servo_id,
            position=int.from_bytes(data[0:2], "little"),
            speed_raw=int.from_bytes(data[2:4], "little"),
            load_raw=int.from_bytes(data[4:6], "little"),
            voltage_v=data[6] / 10.0,
            temperature_c=data[7],
            moving=bool(data[10]),
            current_raw=int.from_bytes(data[13:15], "little"),
            torque_enabled=bool(self.read_u8(servo_id, ADDR_TORQUE_ENABLE)),
        )


def _signed_feedback_magnitude(raw: int) -> int:
    # STS speed/load/current feedback stores direction separately from magnitude.
    return raw & 0x03FF


def _normalize_position(position: int, open_position: int, closed_position: int) -> float:
    value = (position - open_position) / (closed_position - open_position)
    return min(1.0, max(0.0, value))


def _denormalize_position(value: float, open_position: int, closed_position: int) -> int:
    return round(open_position + value * (closed_position - open_position))


class FeetechSTS3215Gripper:
    """Thread-safe STS3215 gripper with binary compatibility and continuous position control."""

    driver_name = "feetech_sts3215"

    def __init__(
        self,
        port: str | Path,
        *,
        servo_id: int = 7,
        baud: int = 1_000_000,
        open_position: int = 100,
        closed_position: int = 3995,
        binary_threshold: int | None = None,
        speed: int = 250,
        acceleration: int = 8,
        max_temperature_c: int = 70,
        max_load_raw: int = 750,
        max_current_raw: int = 600,
        timeout_s: float = 0.08,
        bus_factory: Any = STSBus,
        serial_factory: Any = serial.Serial,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be in [0, 253]")
        if not 0 <= open_position <= 4095 or not 0 <= closed_position <= 4095:
            raise ValueError("gripper endpoints must be in [0, 4095]")
        if open_position == closed_position:
            raise ValueError("open_position and closed_position must differ")
        low, high = sorted((open_position, closed_position))
        threshold = round((open_position + closed_position) / 2)
        if binary_threshold is not None:
            threshold = int(binary_threshold)
        if not low <= threshold <= high:
            raise ValueError("binary_threshold must lie between the open and closed positions")
        if not 0 <= speed <= 65535:
            raise ValueError("speed must be in [0, 65535]")
        if not 0 <= acceleration <= 255:
            raise ValueError("acceleration must be in [0, 255]")
        if max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be positive")
        if not 1 <= max_load_raw <= 1023 or not 1 <= max_current_raw <= 1023:
            raise ValueError("load and current safety limits must be in [1, 1023]")

        self.port = str(port)
        self.servo_id = int(servo_id)
        self.baud = int(baud)
        self.open_position = int(open_position)
        self.closed_position = int(closed_position)
        self.binary_threshold = threshold
        self.speed = int(speed)
        self.acceleration = int(acceleration)
        self.max_temperature_c = int(max_temperature_c)
        self.max_load_raw = int(max_load_raw)
        self.max_current_raw = int(max_current_raw)
        self.timeout_s = float(timeout_s)
        self._bus_factory = bus_factory
        self._serial_factory = serial_factory
        self._monotonic = monotonic
        self._bus: STSBus | None = None
        self._lock = threading.Lock()
        self._target = self.open_position
        self._sequence = 0

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open(self) -> None:
        with self._lock:
            if self._bus is not None:
                raise RuntimeError("gripper is already open")
            bus = self._bus_factory(
                self.port,
                baudrate=self.baud,
                timeout=self.timeout_s,
                serial_factory=self._serial_factory,
            )
            try:
                if not bus.ping(self.servo_id):
                    raise ServoTimeout(
                        f"FEETECH STS3215 ID {self.servo_id} did not respond on {self.port}"
                    )
                self._target = int(bus.get_position(self.servo_id))
            except BaseException:
                bus.close()
                raise
            self._bus = bus

    def close(self) -> None:
        """Close the serial link without changing the servo's torque state."""
        with self._lock:
            bus, self._bus = self._bus, None
            if bus is not None:
                bus.close()

    def command(self, value: int) -> None:
        """Command the logical binary action: 0=open, 1=closed."""
        if value not in (0, 1):
            raise ValueError("gripper command must be 0 or 1")
        self.command_position(float(value))

    def command_position(self, value: float) -> None:
        """Command an absolute normalized position: 0=open and 1=closed."""
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("normalized gripper position must be in [0, 1]")
        target = _denormalize_position(
            normalized,
            self.open_position,
            self.closed_position,
        )
        with self._lock:
            bus = self._require_open()
            status = bus.get_status(self.servo_id)
            self._enforce_safety(status, disable_torque=True)
            if not status.torque_enabled:
                bus.torque_enable(self.servo_id, True)
            bus.set_position(
                self.servo_id,
                target,
                speed=self.speed,
                acceleration=self.acceleration,
            )
            self._target = target

    def unload(self) -> None:
        with self._lock:
            self._require_open().torque_enable(self.servo_id, False)

    def read_state(self, *, enforce_safety: bool = True) -> GripperState:
        with self._lock:
            status = self._require_open().get_status(self.servo_id)
            if enforce_safety:
                self._enforce_safety(status, disable_torque=status.torque_enabled)
            self._sequence += 1
            return GripperState(
                position=status.position,
                normalized_position=_normalize_position(
                    status.position, self.open_position, self.closed_position
                ),
                target_position=self._target,
                normalized_target_position=_normalize_position(
                    self._target, self.open_position, self.closed_position
                ),
                temperature_c=status.temperature_c,
                voltage_v=status.voltage_v,
                load_enabled=status.torque_enabled,
                host_timestamp_s=float(self._monotonic()),
                sequence=self._sequence,
                binary_threshold=self.binary_threshold,
                closing_increases_position=self.closed_position > self.open_position,
                speed_raw=status.speed_raw,
                load_raw=status.load_raw,
                current_raw=status.current_raw,
                moving=status.moving,
            )

    def _enforce_safety(self, status: ServoStatus, *, disable_torque: bool) -> None:
        load = _signed_feedback_magnitude(status.load_raw)
        current = _signed_feedback_magnitude(status.current_raw)
        failures = []
        if status.temperature_c >= self.max_temperature_c:
            failures.append(f"temperature={status.temperature_c}C")
        if load >= self.max_load_raw:
            failures.append(f"load={load}")
        if current >= self.max_current_raw:
            failures.append(f"current={current}")
        if not failures:
            return
        if disable_torque:
            self._require_open().torque_enable(self.servo_id, False)
        raise ServoError("FEETECH safety stop: " + ", ".join(failures))

    def _require_open(self) -> STSBus:
        if self._bus is None:
            raise RuntimeError("gripper is not open")
        return self._bus
