from __future__ import annotations

import json

import pytest

from slai_mi.observability.flight_recorder import (
    ControlCycleRecord,
    FlightRecorder,
    NullFlightRecorder,
)


def _cycle(monotonic_s: float, *, vz: float = 0.0, floor: bool = False) -> ControlCycleRecord:
    return ControlCycleRecord(
        monotonic_s=monotonic_s,
        spacemouse_motion=(0.0, 0.0, vz, 0.0, 0.0, 0.0),
        spacemouse_buttons=(),
        spacemouse_sample_age_s=0.004,
        ur5_commanded_twist=(0.0, 0.0, vz, 0.0, 0.0, 0.0),
        ur5_tcp_z_m=0.10,
        floor_guard_active=floor,
        phase="recording",
    )


def test_recorder_bounds_memory_and_evicts_oldest_first() -> None:
    recorder = FlightRecorder(capacity=3)
    for index in range(5):
        recorder.record(_cycle(float(index)))

    assert len(recorder) == 3
    assert [entry.monotonic_s for entry in recorder.snapshot()] == [2.0, 3.0, 4.0]


def test_recorder_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        FlightRecorder(capacity=0)


def test_dump_writes_reason_and_every_buffered_cycle_in_order(tmp_path) -> None:
    recorder = FlightRecorder(capacity=8)
    recorder.record(_cycle(1.0, vz=-0.35))
    recorder.record(_cycle(1.008, vz=-0.35, floor=True))

    path = recorder.dump(
        reason="UR5 unhealthy", directory=tmp_path, clock=lambda: 1700000000.0
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header == {
        "type": "failure",
        "reason": "UR5 unhealthy",
        "dumped_at": 1700000000.0,
    }
    first_cycle = json.loads(lines[1])
    second_cycle = json.loads(lines[2])
    assert first_cycle["monotonic_s"] == 1.0
    assert first_cycle["ur5_commanded_twist"] == [0.0, 0.0, -0.35, 0.0, 0.0, 0.0]
    assert second_cycle["floor_guard_active"] is True


def test_dump_creates_missing_directory(tmp_path) -> None:
    recorder = FlightRecorder(capacity=1)
    recorder.record(_cycle(0.0))
    target = tmp_path / "nested" / "flight-recorder"

    path = recorder.dump(reason="test", directory=target, clock=lambda: 1700000000.0)

    assert path.parent == target
    assert path.exists()


def test_null_recorder_is_a_no_op() -> None:
    recorder = NullFlightRecorder()
    recorder.record(_cycle(0.0))

    assert len(recorder) == 0
    assert recorder.snapshot() == []
    assert recorder.dump(reason="unused") is None
