"""Executable contract for canonical UR5 + Wuji LeRobot Dataset v3 data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import transformed_vector_dimension, vector_indices
from slai_mi.rotation import rotation6d_columns_to_matrix

from .schema import (
    ACTION_DIM,
    CAMERA_SCHEMAS,
    CAMERA_SKEW_MS,
    DEVICE_TIMESTAMPS_S,
    FPS,
    HOST_RECEIVE_TIMESTAMPS_S,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_SCHEMA,
    OBSERVATION_TCP_POSE,
    SOURCE_AGE_MS,
    SOURCE_DROPS,
    SOURCE_RESTARTS,
    SOURCE_SEQUENCES,
    SPACEMOUSE_BUTTONS,
    VALIDITY_MASK,
    lerobot_features,
)

CONTRACT_ID = "robot_teleoperation.ur5_wuji.three_rgb_cartesian_rot6d_columns.v5"
CONTRACT_FILENAME = "robot_teleoperation_contract.json"
LEROBOT_CODEBASE_VERSION = "v3.0"
IMAGE_KEYS = tuple(str(camera["dataset_key"]) for camera in CAMERA_SCHEMAS)
NONNEGATIVE_KEYS = (
    CAMERA_SKEW_MS,
    SOURCE_AGE_MS,
    DEVICE_TIMESTAMPS_S,
    HOST_RECEIVE_TIMESTAMPS_S,
    SOURCE_SEQUENCES,
    SOURCE_DROPS,
    SOURCE_RESTARTS,
)
BINARY_KEYS = (VALIDITY_MASK, SPACEMOUSE_BUTTONS)
STANDARD_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _canonical_schema() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "dtype": feature["dtype"],
            "shape": list(feature["shape"]),
            "names": list(feature["names"]),
        }
        for key, feature in lerobot_features().items()
    }


def schema_sha256() -> str:
    encoded = json.dumps(
        _canonical_schema(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_manifest() -> dict[str, Any]:
    features = lerobot_features()
    policy_state_dim = sum(
        len(
            vector_indices(
                source,
                transformed_vector_dimension(
                    source,
                    int(features[str(source["key"])]["shape"][0]),
                    f"pi05.state.sources[{index}]",
                ),
                f"pi05.state.sources[{index}]",
            )
        )
        for index, source in enumerate(INPUT_SCHEMA["pi05"]["state"]["sources"])
    )
    return {
        "contract_id": CONTRACT_ID,
        "schema_sha256": schema_sha256(),
        "lerobot_codebase_version": LEROBOT_CODEBASE_VERSION,
        "fps": FPS,
        "capture": {
            "images": {
                "keys": list(IMAGE_KEYS),
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
            },
            "state": list(INPUT_SCHEMA["capture"]["state"]["components"]),
            "tcp_pose": dict(INPUT_SCHEMA["capture"]["tcp_pose"]),
            "action": list(INPUT_SCHEMA["capture"]["action"]["components"]),
        },
        "pi05_training_view": {
            "state_dim": policy_state_dim,
            "physical_action_dim": ACTION_DIM,
            "model_state_dim": int(INPUT_SCHEMA["pi05"]["state"]["model_pad_to"]),
            "model_action_dim": int(INPUT_SCHEMA["pi05"]["action"]["model_pad_to"]),
            "padding": "OpenPI model transform only; never stored in the dataset",
        },
    }


def write_contract_manifest(root: Path) -> Path:
    path = root / "meta" / CONTRACT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_frame(frame: dict[str, Any]) -> None:
    """Reject frames that cannot be interpreted under the canonical contract."""
    expected = set(lerobot_features()) | {"task"}
    actual = set(frame)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"canonical frame keys differ: missing={missing}, extra={extra}")
    if not isinstance(frame["task"], str) or not frame["task"].strip():
        raise ValueError("task must be a non-empty string")

    for key, feature in lerobot_features().items():
        value = np.asarray(frame[key])
        expected_shape = tuple(feature["shape"])
        expected_dtype = np.dtype("uint8" if feature["dtype"] == "video" else feature["dtype"])
        if value.shape != expected_shape:
            raise ValueError(f"{key} shape is {value.shape}, expected {expected_shape}")
        if value.dtype != expected_dtype:
            raise ValueError(f"{key} dtype is {value.dtype}, expected {expected_dtype}")
        if key not in IMAGE_KEYS and not np.isfinite(value).all():
            raise ValueError(f"{key} contains a non-finite value")

    for key in NONNEGATIVE_KEYS:
        if np.any(np.asarray(frame[key]) < 0):
            raise ValueError(f"{key} must be nonnegative")
    for key in BINARY_KEYS:
        if not np.isin(np.asarray(frame[key]), (0, 1)).all():
            raise ValueError(f"{key} must contain only 0 or 1")
    rotation6d_columns_to_matrix(np.asarray(frame[OBSERVATION_TCP_POSE])[3:])


def _validate_manifest(root: Path) -> None:
    path = root / "meta" / CONTRACT_FILENAME
    if not path.is_file():
        raise ValueError(f"missing canonical contract manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = contract_manifest()
    if manifest != expected:
        raise ValueError(
            "dataset contract manifest does not match the current canonical schema"
        )


def _validate_metadata(root: Path) -> dict[str, Any]:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != LEROBOT_CODEBASE_VERSION:
        raise ValueError(
            f"codebase_version is {info.get('codebase_version')!r}, "
            f"expected {LEROBOT_CODEBASE_VERSION!r}"
        )
    if info.get("fps") != FPS:
        raise ValueError(f"dataset fps is {info.get('fps')}, expected {FPS}")

    expected = {**_canonical_schema(), **STANDARD_FEATURES}
    actual = info.get("features", {})
    if set(actual) != set(expected):
        raise ValueError(
            "dataset feature keys differ from canonical contract: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for key, feature in expected.items():
        stored = actual[key]
        for attribute in ("dtype", "shape", "names"):
            if stored.get(attribute) != feature[attribute]:
                raise ValueError(
                    f"{key}.{attribute} is {stored.get(attribute)!r}, "
                    f"expected {feature[attribute]!r}"
                )
    for key in IMAGE_KEYS:
        stored_info = actual[key].get("info", {})
        if stored_info.get("is_depth_map") is not False:
            raise ValueError(f"{key}.info.is_depth_map must be false")
    if info.get("robot_type") != "ur5_wuji_hand1":
        raise ValueError(
            f"robot_type is {info.get('robot_type')!r}, expected 'ur5_wuji_hand1'"
        )
    if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("dataset must contain at least one committed, non-empty episode")
    if int(info.get("total_tasks", 0)) <= 0:
        raise ValueError("dataset must contain at least one task")
    return info


def _validate_parquet(root: Path, expected_frames: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError("dataset contains no Parquet data files")
    expected_features = lerobot_features()
    numeric_keys = [key for key in expected_features if key not in IMAGE_KEYS]
    standard_keys = list(STANDARD_FEATURES)
    rows = 0
    all_frame_indices: list[np.ndarray] = []
    all_episode_indices: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        required_columns = set(numeric_keys) | {
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        missing = sorted(required_columns - set(schema.names))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for key in numeric_keys:
            feature = expected_features[key]
            scalar = {
                "float32": pa.float32(),
                "float64": pa.float64(),
                "int64": pa.int64(),
            }[feature["dtype"]]
            shape = tuple(feature["shape"])
            allowed_types = (
                (scalar,)
                if shape == (1,)
                else (pa.list_(scalar, shape[0]), pa.list_(scalar))
            )
            if schema.field(key).type not in allowed_types:
                raise ValueError(
                    f"{path}:{key} type is {schema.field(key).type}, "
                    f"expected one of {allowed_types}"
                )

        for batch in parquet.iter_batches(columns=numeric_keys + standard_keys):
            for key in numeric_keys:
                values = np.asarray(batch.column(key).to_pylist())
                feature_shape = tuple(expected_features[key]["shape"])
                expected_shape = (
                    (batch.num_rows,)
                    if feature_shape == (1,)
                    else (batch.num_rows, *feature_shape)
                )
                if values.shape != expected_shape:
                    raise ValueError(
                        f"{path}:{key} values have shape {values.shape}, "
                        f"expected {expected_shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}:{key} contains a non-finite value")
                if key in NONNEGATIVE_KEYS and np.any(values < 0):
                    raise ValueError(f"{path}:{key} contains a negative value")
                if key in BINARY_KEYS and not np.isin(values, (0, 1)).all():
                    raise ValueError(f"{path}:{key} contains a value outside {{0, 1}}")
            timestamp = np.asarray(batch.column("timestamp"), dtype=np.float64)
            frame_index = np.asarray(batch.column("frame_index"), dtype=np.int64)
            episode_index = np.asarray(batch.column("episode_index"), dtype=np.int64)
            index = np.asarray(batch.column("index"), dtype=np.int64)
            task_index = np.asarray(batch.column("task_index"), dtype=np.int64)
            if not np.isfinite(timestamp).all():
                raise ValueError(f"{path}:timestamp contains a non-finite value")
            if np.any(frame_index < 0) or np.any(episode_index < 0) or np.any(task_index < 0):
                raise ValueError(f"{path} contains a negative LeRobot index")
            if not np.allclose(timestamp, frame_index / FPS, rtol=0.0, atol=1e-5):
                raise ValueError(f"{path} timestamps do not equal frame_index / {FPS}")
            all_frame_indices.append(frame_index)
            all_episode_indices.append(episode_index)
            all_indices.append(index)
            rows += batch.num_rows
    if rows != expected_frames:
        raise ValueError(f"Parquet row count is {rows}, metadata reports {expected_frames}")
    indices = np.concatenate(all_indices)
    if not np.array_equal(indices, np.arange(rows, dtype=np.int64)):
        raise ValueError("global dataset index is not contiguous from zero")
    episode_indices = np.concatenate(all_episode_indices)
    frame_indices = np.concatenate(all_frame_indices)
    for episode in np.unique(episode_indices):
        episode_frames = frame_indices[episode_indices == episode]
        if not np.array_equal(episode_frames, np.arange(len(episode_frames), dtype=np.int64)):
            raise ValueError(f"episode {episode} frame_index is not contiguous from zero")
    return len(paths)


def _validate_videos(root: Path) -> int:
    import av

    paths = sorted((root / "videos").rglob("*.mp4"))
    unexpected = [path for path in paths if not any(key in path.parts for key in IMAGE_KEYS)]
    if unexpected:
        raise ValueError(
            f"dataset contains videos outside the canonical image features: {unexpected}"
        )
    for key in IMAGE_KEYS:
        if not any(key in path.parts for path in paths):
            raise ValueError(f"dataset contains no video file for {key}")
    for path in paths:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"video has no video stream: {path}")
            stream = container.streams.video[0]
            if (stream.height, stream.width) != (480, 640):
                raise ValueError(
                    f"video dimensions are {stream.width}x{stream.height}, expected 640x480: {path}"
                )
            try:
                next(container.decode(stream))
            except StopIteration as exc:
                raise ValueError(f"video contains no decodable frame: {path}") from exc
    return len(paths)


def validate_dataset_root(root: Path) -> dict[str, Any]:
    """Validate a finalized dataset without applying subjective quality gates."""
    root = root.resolve()
    _validate_manifest(root)
    info = _validate_metadata(root)
    parquet_files = _validate_parquet(root, int(info["total_frames"]))
    video_files = _validate_videos(root)
    return {
        "contract_id": CONTRACT_ID,
        "schema_sha256": schema_sha256(),
        "fps": FPS,
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "parquet_files": parquet_files,
        "video_files": video_files,
    }
