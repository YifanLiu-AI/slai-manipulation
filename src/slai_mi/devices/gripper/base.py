"""Brand-independent interfaces for one-DoF serial grippers."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class GripperStateLike(Protocol):
    """State fields required by teleoperation, collection, HOME, and inference."""

    position: int
    normalized_position: float
    target_position: int
    normalized_target_position: float
    temperature_c: int
    voltage_v: float
    load_enabled: bool
    host_timestamp_s: float
    sequence: int

    @property
    def actual_q(self) -> tuple[float]: ...

    @property
    def target_q(self) -> tuple[float]: ...

    @property
    def continuous_actual_q(self) -> tuple[float]: ...

    @property
    def continuous_target_q(self) -> tuple[float]: ...

    @property
    def binary_state(self) -> int: ...


@runtime_checkable
class GripperDevice(Protocol):
    """Stable normalized-position contract implemented by every servo adapter."""

    driver_name: str
    port: str
    servo_id: int
    open_position: int
    closed_position: int

    def __enter__(self) -> Self: ...

    def __exit__(self, *_args: object) -> None: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def command(self, value: int) -> None: ...

    def command_position(self, value: float) -> None: ...

    def unload(self) -> None: ...

    def read_state(self, *, enforce_safety: bool = True) -> GripperStateLike: ...

