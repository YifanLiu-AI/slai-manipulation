from __future__ import annotations

import numpy as np
import pytest

from slai_mi.collection.synchronization import (
    BoundedBuffer,
    ClockMapper,
    DeviceClockFit,
    RealFrameSynchronizer,
    TimedSample,
    TimeSeries,
)


def sample(value, timestamp: float) -> TimedSample:
    return TimedSample(value=value, monotonic_s=timestamp)


def test_device_clock_fit_maps_drifted_device_time_to_monotonic() -> None:
    fit = DeviceClockFit(capacity=8)
    for device in range(6):
        fit.observe(float(device), 50.0 + 1.002 * device)
    assert fit.to_monotonic(8.0) == pytest.approx(58.016)


def test_clock_mapper_adopts_only_plausible_drift_and_resets() -> None:
    mapper = ClockMapper(window=16)
    for index in range(8):
        mapper.update(index * 0.1, 10.0 + index * 0.1002)
    assert mapper.slope == pytest.approx(1.002)
    mapper.update(0.0, 20.0)
    assert mapper.slope == 1.0
    assert mapper.map(1.0) == pytest.approx(21.0)


def test_time_series_retention_and_bounded_outage_gates() -> None:
    series = TimeSeries[np.ndarray](retain_s=1.0)
    series.append(sample(np.array([0.0]), 0.0))
    series.append(sample(np.array([1.0]), 0.5))
    series.append(sample(np.array([2.0]), 1.1))
    assert [item.monotonic_s for item in series.snapshot()] == [0.5, 1.1]
    assert series.dropped_before == 1
    assert series.before(1.2, max_age_s=0.2) is not None
    assert series.before(1.5, max_age_s=0.2) is None
    assert series.bracket(0.8, max_span_s=0.7) is not None
    assert series.bracket(0.8, max_span_s=0.5) is None


def test_bounded_buffer_interpolates_and_zoh_does_not_look_ahead() -> None:
    buffer = BoundedBuffer[np.ndarray](capacity=2)
    buffer.append(sample(np.array([0.0, 2.0]), 1.0))
    buffer.append(sample(np.array([2.0, 4.0]), 3.0))
    np.testing.assert_allclose(buffer.interpolate(2.0).value, [1.0, 3.0])
    assert buffer.zoh(2.0).monotonic_s == 1.0
    buffer.append(sample(np.array([4.0, 6.0]), 4.0))
    with pytest.raises(LookupError, match="at or before"):
        buffer.zoh(0.5)


def test_primary_camera_timeline_interpolates_state_and_holds_command() -> None:
    sync = RealFrameSynchronizer(max_camera_age_ms=50.0)
    primary = sample("p", 1.0)
    sync.cameras["secondary"].append(sample("s", 0.995))
    sync.cameras["wrist"].append(sample("w", 1.006))
    for buffer, low, high in (
        (sync.ur5, [0.0, 2.0], [2.0, 4.0]),
        (sync.wuji, [10.0], [14.0]),
    ):
        buffer.append(sample(np.asarray(low), 0.9))
        buffer.append(sample(np.asarray(high), 1.1))
    sync.commands.append(sample({"target": 1}, 0.98))

    frame = sync.synchronize(primary, now_s=1.01)

    assert frame.cameras == {"primary": "p", "secondary": "s", "wrist": "w"}
    np.testing.assert_allclose(frame.ur5_state, [1.0, 3.0])
    np.testing.assert_allclose(frame.wuji_state, [12.0])
    assert frame.command == {"target": 1}
    assert frame.valid
    assert frame.diagnostics["camera.wrist"].skew_ms == pytest.approx(6.0)
    assert frame.diagnostics["command"].age_ms == pytest.approx(20.0)


def test_sync_marks_excessive_camera_skew_and_stale_command_invalid() -> None:
    sync = RealFrameSynchronizer(max_camera_skew_ms=10, max_command_age_ms=20)
    primary = sample("p", 1.0)
    sync.cameras["secondary"].append(sample("s", 0.98))
    sync.cameras["wrist"].append(sample("w", 1.0))
    for buffer in (sync.ur5, sync.wuji):
        buffer.append(sample(np.array([0.0]), 0.99))
        buffer.append(sample(np.array([1.0]), 1.01))
    sync.commands.append(sample("old", 0.9))

    frame = sync.synchronize(primary, now_s=1.0)

    assert not frame.valid
    assert not frame.diagnostics["camera.secondary"].valid
    assert not frame.diagnostics["command"].valid


def test_sync_roles_and_drop_accounting_come_from_schema() -> None:
    schema = {
        "capture": {
            "primary_timeline_role": "front",
            "cameras": [
                {"role": "front", "enabled": True},
                {"role": "hand", "enabled": True},
            ],
        },
        "synchronization": {
            "state_channels": ["arm"],
            "retain_s": 0.5,
            "max_camera_skew_ms": 20,
            "max_camera_age_ms": 100,
            "max_state_age_ms": 100,
            "max_command_age_ms": 250,
        },
    }
    sync = RealFrameSynchronizer.from_input_schema(schema)
    sync.cameras["hand"].append(TimedSample("old", 0.0, sequence=1))
    sync.cameras["hand"].append(TimedSample("new", 1.0, sequence=3))
    sync.states["arm"].append(sample(np.array([0.0]), 0.9))
    sync.states["arm"].append(sample(np.array([2.0]), 1.1))
    sync.commands.append(sample("hold", 0.95))
    frame = sync.synchronize(sample("front", 1.0), now_s=1.01)
    assert frame.cameras == {"front": "front", "hand": "new"}
    np.testing.assert_allclose(frame.states["arm"], [1.0])
    assert frame.diagnostics["camera.hand"].dropped_before == 1
    assert frame.diagnostics["camera.hand"].sequence_gaps == 1
