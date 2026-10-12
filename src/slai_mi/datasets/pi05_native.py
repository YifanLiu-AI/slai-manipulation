"""Build schema-driven LeRobot views consumed by the PyTorch PI0.5 backend."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema

TRAINING_VECTOR_KEYS = ("observation.state", "action")
MAX_NORMALIZED_ABS = 10.0
QUANTILE_EPSILON = 1e-6


def _pad(value: object, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or len(array) > size:
        raise ValueError(f"{label} shape {array.shape} cannot fit configured model size {size}")
    return np.pad(array, (0, size - len(array)))


def build_native_v21(
    source: Path,
    target: Path,
    *,
    source_repo_id: str,
    target_repo_id: str,
    schema_path: Path,
) -> Path:
    """Rename/pad one OpenPI v2.1 view for LeRobot PI0.5 training."""
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"native PI0.5 view requires LeRobot v2.1, found {CODEBASE_VERSION}")
    source, target = source.resolve(), target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite native PI0.5 dataset: {target}")
    source_ds = LeRobotDataset(source_repo_id, root=source, video_backend="pyav")
    schema = load_input_schema(schema_path)
    cameras = enabled_cameras(schema)
    policy = schema["pi05"]
    capture_channels = {str(item["channel"]) for item in schema["capture"]["state"]["components"]}
    if {"ur5", "wrist", "gripper"}.issubset(capture_channels):
        robot_type = "ur5_wrist_gripper_pi05"
    elif {"ur5", "wrist"}.issubset(capture_channels):
        robot_type = "ur5_wrist_pi05"
    else:
        robot_type = "ur5_wujihand_pi05"
    state_size = int(policy["state"]["model_pad_to"])
    action_size = int(policy["action"]["model_pad_to"])
    image = {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    }
    features = {str(camera["policy_key"]): dict(image) for camera in cameras}
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_size,),
                "names": ["state"],
            },
            "action": {"dtype": "float32", "shape": (action_size,), "names": ["action"]},
        }
    )
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        root=target,
        robot_type=robot_type,
        fps=int(source_ds.fps),
        features=features,
        use_videos=True,
        tolerance_s=1e-6,
        image_writer_threads=4,
    )
    previous_episode: int | None = None
    try:
        for index in range(len(source_ds)):
            item = source_ds[index]
            episode = int(item["episode_index"].item())
            if previous_episode is None:
                if episode != 0:
                    raise ValueError(f"source episode_index starts at {episode}, expected 0")
                previous_episode = episode
            elif episode != previous_episode:
                if episode != previous_episode + 1:
                    raise ValueError(
                        f"source episode_index jumps from {previous_episode} to {episode}"
                    )
                target_ds.save_episode()
                previous_episode = episode
            policy_images = {}
            for camera in cameras:
                source_key = str(camera["openpi_key"])
                if source_key not in item:
                    raise ValueError(
                        f"configured camera {camera['role']} is missing from v2.1 item: {source_key}"
                    )
                image_value = np.asarray(item[source_key])
                if image_value.ndim == 3 and image_value.shape[0] in (1, 3, 4):
                    image_value = np.moveaxis(image_value, 0, -1)
                policy_images[str(camera["policy_key"])] = np.ascontiguousarray(image_value)
            target_ds.add_frame(
                {
                    **policy_images,
                    "observation.state": _pad(item["state"], state_size, "state"),
                    "action": _pad(item["actions"], action_size, "action"),
                    "task": str(item["task"]),
                }
            )
        if previous_episode is None:
            raise ValueError("source native PI0.5 dataset contains no frames")
        target_ds.save_episode()
    finally:
        target_ds.stop_image_writer()
    info = json.loads((target / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info["total_episodes"]) != int(source_ds.num_episodes) or int(
        info["total_frames"]
    ) != len(source_ds):
        raise ValueError(
            "native PI0.5 view changed episode/frame counts: "
            f"source={source_ds.num_episodes}/{len(source_ds)}, "
            f"target={info['total_episodes']}/{info['total_frames']}"
        )
    return target


def upgrade_native_v21_to_v30(source: Path, target: Path, *, repo_id: str) -> Path:
    """Run LeRobot's official in-place converter on an isolated copy."""
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    source, target = source.resolve(), target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite native PI0.5 v3 dataset: {target}")
    if not source.is_dir():
        raise FileNotFoundError(f"native PI0.5 v2.1 dataset is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slai_pi05_v30_", dir=target.parent) as temporary:
        working = Path(temporary) / "dataset"
        shutil.copytree(source, working)
        convert_dataset(repo_id=repo_id, root=working, push_to_hub=False)
        add_training_quantiles(working, repo_id=repo_id)
        shutil.move(str(working), target)
    return target


def add_training_quantiles(root: Path, *, repo_id: str, batch_size: int = 4096) -> None:
    """Add the quantiles required by LeRobot PI0.5 normalization."""
    from lerobot.datasets.compute_stats import RunningQuantileStats
    from lerobot.datasets.io_utils import load_stats, write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if batch_size < 1:
        raise ValueError("quantile batch size must be positive")
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    stats = load_stats(root)
    if stats is None:
        raise ValueError(f"native PI0.5 v3 stats are missing: {root}")
    for key in TRAINING_VECTOR_KEYS:
        if key not in dataset.hf_dataset.column_names:
            raise ValueError(f"native PI0.5 v3 dataset is missing training vector: {key}")
        running = RunningQuantileStats()
        column = dataset.hf_dataset[key]
        for start in range(0, len(column), batch_size):
            batch = np.stack(
                [
                    np.asarray(value, dtype=np.float32)
                    for value in column[start : start + batch_size]
                ]
            )
            running.update(batch)
        if len(column) == 1:
            value = np.asarray(column[0], dtype=np.float32)
            computed = {name: value for name in ("q01", "q10", "q50", "q90", "q99")}
        else:
            computed = running.get_statistics()
        stats.setdefault(key, {}).update(
            {name: computed[name] for name in ("q01", "q10", "q50", "q90", "q99")}
        )
    stabilize_training_quantiles(stats)
    write_stats(stats, root)
    validate_native_v30_stats(root)


def _vector_stats(values: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"min", "max", "q01", "q99"}
    if missing := sorted(required - values.keys()):
        raise ValueError(f"native PI0.5 v3 {key} stats are missing: {', '.join(missing)}")
    arrays = tuple(np.asarray(values[name], dtype=np.float64) for name in ("min", "max", "q01", "q99"))
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError(f"native PI0.5 v3 {key} stats have inconsistent vector shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"native PI0.5 v3 {key} stats contain non-finite values")
    return arrays


def _normalized_extrema(
    minimum: np.ndarray, maximum: np.ndarray, q01: np.ndarray, q99: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    denominator = q99 - q01 + QUANTILE_EPSILON
    return (minimum - q01) / denominator * 2.0 - 1.0, (maximum - q01) / denominator * 2.0 - 1.0


def stabilize_training_quantiles(stats: dict) -> list[dict[str, float | int | str]]:
    """Expand dangerous quantile ranges so training vectors cannot be amplified catastrophically."""
    repairs: list[dict[str, float | int | str]] = []
    for key in TRAINING_VECTOR_KEYS:
        values = stats.get(key)
        if not isinstance(values, dict):
            raise TypeError(f"native PI0.5 v3 stats are missing feature: {key}")
        minimum, maximum, q01, q99 = _vector_stats(values, key)
        low, high = _normalized_extrema(minimum, maximum, q01, q99)
        variable = maximum > minimum
        dangerous = variable & ((np.abs(low) > MAX_NORMALIZED_ABS) | (np.abs(high) > MAX_NORMALIZED_ABS))
        for index in np.flatnonzero(dangerous):
            repairs.append(
                {
                    "feature": key,
                    "dimension": int(index),
                    "old_q01": float(q01[index]),
                    "old_q99": float(q99[index]),
                    "normalized_min": float(low[index]),
                    "normalized_max": float(high[index]),
                }
            )
            q01[index] = minimum[index]
            q99[index] = maximum[index]
        values["q01"] = q01.astype(np.float32)
        values["q99"] = q99.astype(np.float32)
    return repairs


def validate_native_v30_stats(root: Path) -> None:
    """Fail closed when PI0.5 quantile normalization would amplify a training vector dangerously."""
    path = root / "meta" / "stats.json"
    if not path.is_file():
        raise ValueError(f"native PI0.5 v3 stats are missing: {path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    for key in TRAINING_VECTOR_KEYS:
        values = stats.get(key)
        if not isinstance(values, dict):
            raise TypeError(f"native PI0.5 v3 stats are missing feature: {key}")
        minimum, maximum, q01, q99 = _vector_stats(values, key)
        if np.any(q99 < q01):
            raise ValueError(f"native PI0.5 v3 {key} has q99 below q01")
        low, high = _normalized_extrema(minimum, maximum, q01, q99)
        variable = maximum > minimum
        dangerous = np.flatnonzero(
            variable & ((np.abs(low) > MAX_NORMALIZED_ABS) | (np.abs(high) > MAX_NORMALIZED_ABS))
        )
        if len(dangerous):
            details = ", ".join(
                f"dim {int(index)} range=[{low[index]:.3g},{high[index]:.3g}]"
                for index in dangerous
            )
            raise ValueError(
                f"native PI0.5 v3 {key} quantile normalization exceeds "
                f"abs({MAX_NORMALIZED_ABS:g}): {details}; rebuild stats before training"
            )
