import numpy as np

from slai_mi.datasets.lerobot_v3.continuity import split_continuous_segments


def test_camera_sequence_gap_starts_a_new_segment() -> None:
    rows = 4
    sequences = np.tile(np.arange(rows, dtype=np.int64)[:, None], (1, 6))
    sequences[2:, 0] += 1
    timestamps = np.tile((np.arange(rows) / 30.0)[:, None], (1, 6))
    segments, events = split_continuous_segments(
        np.zeros(rows, dtype=np.int64),
        sequences,
        timestamps,
        np.zeros((rows, 6), dtype=np.int64),
        np.ones((rows, 6), dtype=np.int64),
        max_camera_period_s=0.05,
    )
    assert [(segment.start, segment.end) for segment in segments] == [(0, 2), (2, 4)]
    assert events[0]["row"] == 2


def test_invalid_critical_source_skips_row() -> None:
    rows = 3
    validity = np.ones((rows, 6), dtype=np.int64)
    validity[1, 4] = 0
    segments, events = split_continuous_segments(
        np.zeros(rows, dtype=np.int64),
        np.tile(np.arange(rows)[:, None], (1, 6)),
        np.tile((np.arange(rows) / 30.0)[:, None], (1, 6)),
        np.zeros((rows, 6), dtype=np.int64),
        validity,
        max_camera_period_s=0.05,
    )
    assert [(segment.start, segment.end) for segment in segments] == [(0, 1), (2, 3)]
    assert events[0]["row_skipped"] is True


def test_continuity_column_count_follows_configured_sources() -> None:
    rows = 3
    source_count = 4  # Two cameras, one state channel, one command channel.
    segments, events = split_continuous_segments(
        np.zeros(rows, dtype=np.int64),
        np.tile(np.arange(rows)[:, None], (1, source_count)),
        np.tile((np.arange(rows) / 30.0)[:, None], (1, source_count)),
        np.zeros((rows, source_count), dtype=np.int64),
        np.ones((rows, source_count), dtype=np.int64),
        max_camera_period_s=0.05,
        camera_source_names=("front", "wrist"),
        state_source_names=("robot",),
        command_source_name="operator",
    )
    assert [(segment.start, segment.end) for segment in segments] == [(0, 3)]
    assert events == []
