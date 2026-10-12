"""Detect physical continuity boundaries in canonical VLA v3 telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema

_INPUT_SCHEMA = load_input_schema()
CAMERA_SOURCE_NAMES = tuple(str(camera["role"]) for camera in enabled_cameras(_INPUT_SCHEMA))
STATE_SOURCE_NAMES = tuple(
    str(channel["name"]) for channel in _INPUT_SCHEMA["synchronization"]["state_channels"]
)
COMMAND_SOURCE_NAME = str(_INPUT_SCHEMA["synchronization"]["command_channel"]["name"])
CRITICAL_SOURCE_NAMES = (*CAMERA_SOURCE_NAMES, *STATE_SOURCE_NAMES)
CRITICAL_SOURCE_COUNT = len(CRITICAL_SOURCE_NAMES)
SOURCE_COUNT = CRITICAL_SOURCE_COUNT + 1  # Command input is audited but not continuity-critical.


@dataclass(frozen=True)
class ContinuitySegment:
    source_episode_index: int
    start: int
    end: int
    boundary_reasons_before: tuple[str, ...]

    @property
    def length(self) -> int:
        return self.end - self.start


def _row_is_valid(validity: np.ndarray, index: int) -> bool:
    return bool(np.all(validity[index, :CRITICAL_SOURCE_COUNT] == 1))


def _boundary_reasons(
    sequences: np.ndarray,
    device_timestamps_s: np.ndarray,
    restarts: np.ndarray,
    index: int,
    *,
    max_camera_period_s: float,
    camera_source_names: tuple[str, ...] = CAMERA_SOURCE_NAMES,
    critical_source_names: tuple[str, ...] = CRITICAL_SOURCE_NAMES,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for camera_index, camera_name in enumerate(camera_source_names):
        sequence_step = int(sequences[index, camera_index] - sequences[index - 1, camera_index])
        if sequence_step != 1:
            reasons.append(f"{camera_name}_sequence_step={sequence_step}")
        source_dt = float(
            device_timestamps_s[index, camera_index] - device_timestamps_s[index - 1, camera_index]
        )
        if not 0.0 < source_dt <= max_camera_period_s:
            reasons.append(f"{camera_name}_source_dt_s={source_dt:.9f}")
    changed = np.flatnonzero(
        restarts[index, : len(critical_source_names)]
        != restarts[index - 1, : len(critical_source_names)]
    )
    reasons.extend(f"{critical_source_names[int(i)]}_restart_changed" for i in changed)
    return tuple(reasons)


def split_continuous_segments(
    episode_indices: object,
    sequences: object,
    device_timestamps_s: object,
    restarts: object,
    validity: object,
    *,
    max_camera_period_s: float,
    camera_source_names: tuple[str, ...] = CAMERA_SOURCE_NAMES,
    state_source_names: tuple[str, ...] = STATE_SOURCE_NAMES,
    command_source_name: str = COMMAND_SOURCE_NAME,
) -> tuple[list[ContinuitySegment], list[dict[str, object]]]:
    """Partition rows without bridging camera gaps, restarts, or invalid samples."""
    episodes = np.asarray(episode_indices, dtype=np.int64)
    seq = np.asarray(sequences, dtype=np.int64)
    source_time = np.asarray(device_timestamps_s, dtype=np.float64)
    restart_counts = np.asarray(restarts, dtype=np.int64)
    valid = np.asarray(validity, dtype=np.int64)
    row_count = len(episodes)
    critical_source_names = (*camera_source_names, *state_source_names)
    critical_source_count = len(critical_source_names)
    source_count = critical_source_count + int(bool(command_source_name))
    expected_shape = (row_count, source_count)
    for name, value in (
        ("sequences", seq),
        ("device_timestamps_s", source_time),
        ("restarts", restart_counts),
        ("validity", valid),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape is {value.shape}, expected {expected_shape}")
    if max_camera_period_s <= 0.0:
        raise ValueError("max_camera_period_s must be positive")
    if row_count == 0:
        return [], []

    segments: list[ContinuitySegment] = []
    events: list[dict[str, object]] = []
    start: int | None = None
    reasons_before: tuple[str, ...] = ("source_episode_start",)

    def close(end: int) -> None:
        nonlocal start
        if start is not None and end > start:
            segments.append(
                ContinuitySegment(
                    source_episode_index=int(episodes[start]),
                    start=start,
                    end=end,
                    boundary_reasons_before=reasons_before,
                )
            )
        start = None

    for index in range(row_count):
        if not bool(np.all(valid[index, :critical_source_count] == 1)):
            close(index)
            invalid_sources = np.flatnonzero(valid[index, :critical_source_count] != 1)
            reasons = tuple(f"critical_source_{int(i)}_invalid" for i in invalid_sources)
            events.append({"row": index, "reasons": list(reasons), "row_skipped": True})
            reasons_before = reasons
            continue

        if index == 0 or episodes[index] != episodes[index - 1]:
            close(index)
            reasons_before = ("source_episode_start",)
            start = index
            continue

        if start is None:
            reasons_before = ("after_invalid_row",)
            start = index
            continue

        reasons = _boundary_reasons(
            seq,
            source_time,
            restart_counts,
            index,
            max_camera_period_s=max_camera_period_s,
            camera_source_names=camera_source_names,
            critical_source_names=critical_source_names,
        )
        if reasons:
            close(index)
            events.append({"row": index, "reasons": list(reasons), "row_skipped": False})
            reasons_before = reasons
            start = index

    close(row_count)
    return segments, events
