from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from slai_mi.datasets.lerobot_v3.auto_merge import (
    classify_wrist_episodes,
    classify_wrist_session,
)


def _dataset(
    tmp_path: Path, fe: list[float], ru: list[float], episodes: list[int] | None = None
) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "features": {
            "action": {
                "names": [
                    "ur5.tcp.vx",
                    "ur5.tcp.vy",
                    "ur5.tcp.vz",
                    "ur5.tcp.wx",
                    "ur5.tcp.wy",
                    "ur5.tcp.wz",
                    "wrist.fe.target_position",
                    "wrist.ru.target_position",
                    "gripper.target_position",
                ]
            }
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    actions = [[0.0] * 6 + [a, b, 0.0] for a, b in zip(fe, ru, strict=True)]
    episode_indices = episodes if episodes is not None else [0] * len(actions)
    pq.write_table(
        pa.table(
            {
                "action": pa.array(actions, type=pa.list_(pa.float32(), 9)),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    return root


def test_wrist_session_below_five_degrees_is_3dof(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [0.0, 0.08], [0.0, 0.0])
    group, ranges = classify_wrist_session(root, threshold_deg=5.0)
    assert group == "3DoF-only"
    assert ranges[0] == pytest.approx(0.08)


def test_wrist_session_above_five_degrees_is_2dof(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [0.0, 0.09], [0.0, 0.0])
    group, _ranges = classify_wrist_session(root, threshold_deg=5.0)
    assert group == "2DoF-only"


def test_mixed_session_is_classified_per_episode(tmp_path: Path) -> None:
    root = _dataset(
        tmp_path,
        [0.0, 0.0, 0.0, 0.23],
        [0.0, 0.0, 0.0, 0.0],
        episodes=[0, 0, 1, 1],
    )
    groups, ranges = classify_wrist_episodes(root, threshold_deg=5.0)
    assert groups == {"2DoF-only": [1], "3DoF-only": [0]}
    assert ranges[0] == pytest.approx((0.0, 0.0))
    assert ranges[1] == pytest.approx((0.23, 0.0))
