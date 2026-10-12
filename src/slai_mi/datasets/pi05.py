"""Convert canonical LeRobot v3 demonstrations into the PI0.5 v2.1 view."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import (
    enabled_cameras,
    load_input_schema,
    resolve_camera_keys,
    select_transformed_vector,
    select_vector,
    transformed_vector_dimension,
    vector_indices,
)

TARGET_IMAGE_SHAPE = (224, 224, 3)
SUPPORTED_CONTRACTS = {
    "robot_teleoperation.ur5_wuji.pi05_cartesian.v1",
    "robot_teleoperation.ur5_wuji.three_rgb_cartesian.v3",
    "robot_teleoperation.ur5_wuji.three_rgb_cartesian_rot6d_columns.v5",
}


def discover_source_roots(source: Path) -> tuple[Path, ...]:
    """Return one dataset root or all direct descendants carrying v3 metadata."""
    source = source.resolve()
    if (source / "meta" / "info.json").is_file():
        return (source,)
    roots = sorted(path.parent.parent for path in source.glob("*/meta/info.json"))
    if not roots:
        raise ValueError(f"no LeRobot dataset roots found under {source}")
    return tuple(roots)


def validate_pi05_source(
    root: Path, schema: dict[str, Any] | None = None
) -> dict[str, object]:
    """Validate a PI0.5 source against the configured camera and vector schema."""
    info_path = root / "meta" / "info.json"
    contract_path = root / "meta" / "robot_teleoperation_contract.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    schema = schema or load_input_schema()
    capture, policy = schema["capture"], schema["pi05"]
    if info.get("codebase_version") != "v3.0" or info.get("fps") != capture["fps"]:
        raise ValueError(
            f"PI0.5 source must be LeRobot v3 at configured {capture['fps']} Hz: {root}"
        )
    if contract.get("contract_id") not in SUPPORTED_CONTRACTS:
        raise ValueError(f"unsupported PI0.5 source contract: {contract.get('contract_id')}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError(f"dataset features are missing: {root}")
    cameras = resolve_camera_keys(features, schema)
    image_shape = list(capture["image_shape"])
    for camera, key in cameras:
        if features[key].get("shape") != image_shape:
            raise ValueError(f"{root}:{key} for camera {camera['role']} must have shape {image_shape}")
    state_dim = 0
    for index, source in enumerate(policy["state"]["sources"]):
        key = source["key"]
        shape = features.get(key, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"{root}:{key} must be a vector feature")
        source_dim = transformed_vector_dimension(
            source, int(shape[0]), f"pi05.state.sources[{index}]"
        )
        state_dim += len(vector_indices(source, source_dim, f"pi05.state.sources[{index}]"))
    action_spec = policy["action"]
    action_key = action_spec["key"]
    action_shape = features.get(action_key, {}).get("shape")
    if not isinstance(action_shape, list) or len(action_shape) != 1:
        raise ValueError(f"{root}:{action_key} must be a vector feature")
    action_dim = len(vector_indices(action_spec, int(action_shape[0]), "pi05.action"))
    episodes, frames = int(info.get("total_episodes", 0)), int(info.get("total_frames", 0))
    if episodes <= 0 or frames <= 0:
        raise ValueError(f"PI0.5 source must contain committed frames: {root}")
    return {
        "root": str(root),
        "contract_id": contract["contract_id"],
        "episodes": episodes,
        "frames": frames,
        "camera_keys": {str(camera["role"]): key for camera, key in cameras},
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def policy_rgb(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {image.shape}")
    if image.shape != TARGET_IMAGE_SHAPE:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("PI0.5 image conversion requires Pillow") from exc
        image = np.asarray(
            Image.fromarray(image).resize(TARGET_IMAGE_SHAPE[:2][::-1], Image.Resampling.LANCZOS)
        )
    return np.ascontiguousarray(image, dtype=np.uint8)


def stage_episode(
    dataset: Any,
    start: int,
    end: int,
    stride: int,
    path: Path,
    *,
    schema: dict[str, Any] | None = None,
    camera_keys: dict[str, str] | None = None,
) -> int:
    if stride < 1:
        raise ValueError("PI0.5 sampling stride must be positive")
    schema = schema or load_input_schema()
    policy = schema["pi05"]
    cameras = enabled_cameras(schema)
    images: dict[str, list[np.ndarray]] = {str(camera["openpi_key"]): [] for camera in cameras}
    states, actions = [], []
    task = ""
    for index in range(start, end, stride):
        item = dataset[index]
        for camera in cameras:
            role = str(camera["role"])
            key = camera_keys.get(role) if camera_keys else next(
                (str(key) for key in camera["source_keys"] if key in item), None
            )
            if key is None:
                raise ValueError(f"configured camera {role} is absent from episode item")
            images[str(camera["openpi_key"])].append(policy_rgb(item[key]))
        parts = [
            select_transformed_vector(
                item[source["key"]], source, f"pi05.state.sources[{source_index}]"
            )
            for source_index, source in enumerate(policy["state"]["sources"])
        ]
        states.append(np.concatenate(parts, dtype=np.float32))
        actions.append(select_vector(item[policy["action"]["key"]], policy["action"], "pi05.action"))
        task = str(item["task"])
    if not states:
        raise ValueError("episode has no frames after PI0.5 sampling")
    np.savez(
        path,
        **{key: np.stack(values) for key, values in images.items()},
        state=np.stack(states),
        actions=np.stack(actions),
        task=np.asarray(task),
    )
    return len(states)


def convert_v3_to_v21(
    source: Path,
    target: Path,
    *,
    repo_id: str,
    source_fps: int,
    policy_fps: int,
    v21_python: Path,
    schema_path: Path | None = None,
) -> Path:
    """Validate v3, stage downsampled episodes, then write with LeRobot v2.1."""
    from lerobot.datasets import LeRobotDataset

    source, target = source.resolve(), target.resolve()
    # Resolving a venv Python symlink bypasses that venv's site-packages.
    v21_python = Path(os.path.abspath(v21_python.expanduser()))
    if target.exists():
        raise FileExistsError(f"refusing to overwrite PI0.5 dataset: {target}")
    if not v21_python.is_file():
        raise FileNotFoundError(f"LeRobot v2.1 Python is missing: {v21_python}")
    schema = load_input_schema(schema_path)
    configured_source_fps = int(schema["capture"]["fps"])
    configured_policy_fps = int(schema["pi05"]["fps"])
    if (source_fps, policy_fps) != (configured_source_fps, configured_policy_fps):
        raise ValueError("PI0.5 FPS arguments must match the input schema")
    if source_fps <= 0 or policy_fps <= 0 or source_fps % policy_fps:
        raise ValueError("source FPS must be a positive multiple of policy FPS")
    roots = discover_source_roots(source)
    writer = Path(__file__).with_name("pi05_writer.py")
    with tempfile.TemporaryDirectory(prefix="slai_pi05_stage_") as temporary:
        staging = Path(temporary)
        output_episode = 0
        for source_index, root in enumerate(roots):
            source_contract = validate_pi05_source(root, schema)
            dataset = LeRobotDataset(
                f"local/pi05-v3-source-{source_index}",
                root=root,
                return_uint8=True,
                video_backend="pyav",
            )
            if int(dataset.fps) != source_fps:
                raise ValueError(
                    f"configured source FPS {source_fps} differs from {root}: {dataset.fps}"
                )
            required = {
                str(source["key"]) for source in schema["pi05"]["state"]["sources"]
            }
            required.add(str(schema["pi05"]["action"]["key"]))
            if missing := sorted(required - set(dataset.features)):
                raise ValueError(f"{root} is missing features: " + ", ".join(missing))
            for episode in dataset.meta.episodes or []:
                stage_episode(
                    dataset,
                    int(episode["dataset_from_index"]),
                    int(episode["dataset_to_index"]),
                    source_fps // policy_fps,
                    staging / f"episode-{output_episode:06d}.npz",
                    schema=schema,
                    camera_keys=dict(source_contract["camera_keys"]),
                )
                output_episode += 1
        subprocess.run(
            [
                str(v21_python),
                str(writer),
                str(staging),
                str(target),
                "--repo-id",
                repo_id,
                "--schema",
                str((schema_path or Path("configs/input_schema.yaml")).resolve()),
            ],
            check=True,
        )
    return target
