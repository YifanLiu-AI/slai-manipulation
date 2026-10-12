"""Classify TASK2 episodes and atomically append them to aggregate datasets."""

from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import load_input_schema

from .configured import ConfiguredDatasetContract
from .merge import merge_vla_v3_datasets

WRIST_MOTOR_TELEMETRY_KEYS = (
    "telemetry.wrist_motor_position_raw",
    "telemetry.wrist_motor_goal_raw",
    "telemetry.wrist_motor_rate_raw_s",
)
GROUPS = ("2DoF-only", "3DoF-only")


def wrist_episode_action_ranges_rad(root: Path) -> dict[int, tuple[float, float]]:
    """Return FE/RU target peak-to-peak ranges for every finalized episode."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["action"]["names"]
    indices = [names.index("wrist.fe.target_position"), names.index("wrist.ru.target_position")]
    bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["action", "episode_index"])
        actions = np.asarray(table["action"].to_pylist())
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        if actions.size == 0:
            continue
        wrist = actions[:, indices]
        for episode in np.unique(episodes):
            values = wrist[episodes == episode]
            minimum, maximum = bounds.get(
                int(episode),
                (np.full(2, np.inf, dtype=np.float64), np.full(2, -np.inf, dtype=np.float64)),
            )
            bounds[int(episode)] = (
                np.minimum(minimum, values.min(axis=0)),
                np.maximum(maximum, values.max(axis=0)),
            )
    if not bounds:
        raise ValueError(f"dataset contains no action rows: {root}")
    return {
        episode: (float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]))
        for episode, (minimum, maximum) in sorted(bounds.items())
    }


def wrist_action_range_rad(root: Path) -> tuple[float, float]:
    """Return FE/RU target ranges across a session for diagnostics."""
    ranges = wrist_episode_action_ranges_rad(root).values()
    return tuple(max(values[index] for values in ranges) for index in range(2))  # type: ignore[return-value]


def classify_wrist_episodes(
    root: Path, *, threshold_deg: float
) -> tuple[dict[str, list[int]], dict[int, tuple[float, float]]]:
    if not math.isfinite(threshold_deg) or threshold_deg <= 0:
        raise ValueError("wrist classification threshold must be positive and finite")
    threshold_rad = math.radians(threshold_deg)
    ranges = wrist_episode_action_ranges_rad(root)
    groups = {name: [] for name in GROUPS}
    for episode, values in ranges.items():
        group = "2DoF-only" if max(values) >= threshold_rad else "3DoF-only"
        groups[group].append(episode)
    return groups, ranges


def classify_wrist_session(root: Path, *, threshold_deg: float) -> tuple[str, tuple[float, float]]:
    """Compatibility helper that classifies a whole session by its largest episode."""
    groups, ranges = classify_wrist_episodes(root, threshold_deg=threshold_deg)
    group = "2DoF-only" if groups["2DoF-only"] else "3DoF-only"
    combined = tuple(max(values[index] for values in ranges.values()) for index in range(2))
    return group, combined  # type: ignore[return-value]


def update_wrist_aggregate(
    session_root: Path,
    *,
    dataset_config: Mapping[str, Any],
    input_schema: str | Path,
) -> tuple[Path, ...]:
    """Split one session by episode and atomically update non-empty aggregates."""
    settings = dataset_config.get("wrist_auto_merge", {})
    if not isinstance(settings, Mapping) or not settings.get("enabled", False):
        return ()
    threshold_deg = float(settings.get("action_range_threshold_deg", 5.0))
    groups, ranges = classify_wrist_episodes(session_root, threshold_deg=threshold_deg)

    schema = load_input_schema(input_schema)
    aggregate_schema = copy.deepcopy(schema)
    aggregate_schema.setdefault("telemetry", {}).setdefault("wrist_motor", {})["enabled"] = False
    contract = ConfiguredDatasetContract(aggregate_schema)

    workspace = session_root.parent / f".{session_root.name}.episode-auto-{uuid.uuid4().hex}"
    normalized = workspace / "normalized"
    split_root = workspace / "splits"
    workspace.mkdir(parents=True)
    updated: list[Path] = []
    try:
        _copy_without_motor_telemetry(session_root, normalized, contract)
        subsets = _split_episode_groups(normalized, groups, split_root, contract)
        for group in GROUPS:
            subset = subsets.get(group)
            if subset is None:
                continue
            configured_outputs = settings.get("output_roots", {})
            if not isinstance(configured_outputs, Mapping):
                raise ValueError("wrist_auto_merge.output_roots must be a mapping")
            configured_root = configured_outputs.get(group)
            aggregate_root = (
                Path(str(configured_root))
                if configured_root is not None
                else Path(str(dataset_config.get("root", "data/lerobot"))) / group
            )
            _append_aggregate(subset, aggregate_root, group, contract, workspace)
            updated.append(aggregate_root)
            details = ", ".join(
                f"ep={episode}:FE={math.degrees(ranges[episode][0]):.2f}deg/"
                f"RU={math.degrees(ranges[episode][1]):.2f}deg"
                for episode in groups[group]
            )
            print(
                f"Wrist episode auto-merge: group={group} threshold={threshold_deg:.2f}deg "
                f"episodes={groups[group]} aggregate={aggregate_root} ({details})",
                flush=True,
            )
        return tuple(updated)
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def _split_episode_groups(
    source: Path,
    groups: Mapping[str, list[int]],
    output: Path,
    contract: ConfiguredDatasetContract,
) -> dict[str, Path]:
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    requested = {group: episodes for group, episodes in groups.items() if episodes}
    dataset = LeRobotDataset(repo_id="local/wrist-episode-source", root=source)
    split_dataset(dataset, requested, output_dir=output)
    result = {}
    for group in requested:
        root = output / group
        contract.write_manifest(root)
        contract.validate_root(root)
        result[group] = root
    return result


def _append_aggregate(
    subset: Path,
    aggregate_root: Path,
    group: str,
    contract: ConfiguredDatasetContract,
    workspace: Path,
) -> None:
    merged = workspace / f"merged-{group}"
    backup = workspace / f"previous-{group}"
    if not aggregate_root.exists():
        subset.replace(aggregate_root)
        return
    merge_vla_v3_datasets(
        [aggregate_root, subset],
        merged,
        repo_id=f"local/{group.lower()}",
        contract=contract,
    )
    aggregate_root.replace(backup)
    try:
        merged.replace(aggregate_root)
    except BaseException:
        backup.replace(aggregate_root)
        raise
    shutil.rmtree(backup)


def _copy_without_motor_telemetry(
    source: Path, target: Path, contract: ConfiguredDatasetContract
) -> None:
    """Create a merge-compatible copy while preserving the raw source dataset."""
    import pyarrow.parquet as pq

    shutil.copytree(source, target)
    for path in sorted((target / "data").glob("**/*.parquet")):
        table = pq.read_table(path)
        keep = [name for name in table.column_names if name not in WRIST_MOTOR_TELEMETRY_KEYS]
        pq.write_table(table.select(keep), path)
    for path in sorted((target / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        keep = [
            name
            for name in table.column_names
            if not any(
                name.startswith(f"stats/{key}/") for key in WRIST_MOTOR_TELEMETRY_KEYS
            )
        ]
        pq.write_table(table.select(keep), path)

    info_path = target / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in WRIST_MOTOR_TELEMETRY_KEYS:
        info.get("features", {}).pop(key, None)
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    stats_path = target / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in WRIST_MOTOR_TELEMETRY_KEYS:
        stats.pop(key, None)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    contract.write_manifest(target)
    contract.validate_root(target)
