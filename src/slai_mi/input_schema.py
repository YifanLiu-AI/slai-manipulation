"""Validated YAML-driven input mappings shared by collection and policy tooling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from slai_mi.rotation import normalize_ur_base_tcp_pose_to_rotation6d_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs/input_schema.yaml"


def load_input_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema_path = Path(path).expanduser().resolve() if path else DEFAULT_SCHEMA_PATH
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"input schema must be a version 1 mapping: {schema_path}")
    capture = _mapping(data, "capture")
    cameras = capture.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("input schema capture.cameras must be a non-empty list")
    roles: set[str] = set()
    policy_keys: set[str] = set()
    for index, raw in enumerate(cameras):
        if not isinstance(raw, dict):
            raise TypeError(f"capture.cameras[{index}] must be a mapping")
        role = str(raw.get("role") or "").strip()
        if not role or role in roles:
            raise ValueError(f"camera role must be non-empty and unique: {role!r}")
        roles.add(role)
        keys = raw.get("source_keys")
        if not isinstance(keys, list) or not keys or not all(str(key).strip() for key in keys):
            raise ValueError(f"camera {role} source_keys must be a non-empty list")
        policy_key = str(raw.get("policy_key") or "").strip()
        if raw.get("enabled", True) and (not policy_key or policy_key in policy_keys):
            raise ValueError(f"enabled camera {role} requires a unique policy_key")
        policy_keys.add(policy_key)
    if str(capture.get("primary_timeline_role")) not in roles:
        raise ValueError("capture.primary_timeline_role must name a configured camera")
    tcp_pose = _mapping(capture, "tcp_pose")
    if not str(tcp_pose.get("key") or "").strip():
        raise ValueError("capture.tcp_pose.key must be configured")
    if tcp_pose.get("source_representation") != "xyz_rotvec_in_ur_base_frame":
        raise ValueError("capture.tcp_pose source must be UR base-frame xyz+rotvec")
    if tcp_pose.get("dataset_representation") != "xyz_rotation6d_columns_in_ur_base_frame":
        raise ValueError(
            "capture.tcp_pose dataset representation must be column-contiguous rotation6d"
        )
    if not isinstance(tcp_pose.get("names"), list) or len(tcp_pose["names"]) != 9:
        raise ValueError("capture.tcp_pose.names must declare xyz plus six rotation values")
    for vector_name in ("state", "action"):
        vector = _mapping(capture, vector_name)
        components = vector.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"capture.{vector_name}.components must be a non-empty list")
        for index, component in enumerate(components):
            if not isinstance(component, dict) or not str(component.get("channel") or "").strip():
                raise ValueError(f"capture.{vector_name}.components[{index}] requires a channel")
            names = component.get("names")
            if not isinstance(names, list) or not names or not all(str(name).strip() for name in names):
                raise ValueError(f"capture.{vector_name}.components[{index}] requires names")
    policy = _mapping(data, "pi05")
    action = _mapping(policy, "action")
    _vector_spec(action, "pi05.action")
    slots = policy.get("model_image_slots")
    if not isinstance(slots, list) or not slots or len(set(slots)) != len(slots):
        raise ValueError("pi05.model_image_slots must be a non-empty unique list")
    configured_slots = {
        str(camera["policy_key"]).removeprefix("observation.images.")
        for camera in cameras
        if camera.get("enabled", True)
    }
    if unknown := sorted(configured_slots - {str(slot) for slot in slots}):
        raise ValueError(f"enabled cameras use undeclared PI0.5 model image slots: {unknown}")
    delta = _mapping(action, "delta_from_state")
    action_indices = delta.get("action_indices")
    state_indices = delta.get("state_indices")
    if (
        not isinstance(action_indices, list)
        or not isinstance(state_indices, list)
        or len(action_indices) != len(state_indices)
        or not action_indices
        or not all(isinstance(index, int) and index >= 0 for index in (*action_indices, *state_indices))
    ):
        raise ValueError("pi05.action.delta_from_state requires equal non-negative index lists")
    state = _mapping(policy, "state")
    sources = state.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pi05.state.sources must be a non-empty list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TypeError(f"pi05.state.sources[{index}] must be a mapping")
        _vector_spec(source, f"pi05.state.sources[{index}]")
    synchronization = _mapping(data, "synchronization")
    channels = synchronization.get("state_channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("synchronization.state_channels must be a non-empty list")
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict) or not str(channel.get("name") or "").strip():
            raise ValueError(f"synchronization.state_channels[{index}] requires a name")
        _vector_spec(channel, f"synchronization.state_channels[{index}]")
    command = synchronization.get("command_channel")
    if not isinstance(command, dict) or not str(command.get("name") or "").strip():
        raise ValueError("synchronization.command_channel requires a name")
    _vector_spec(command, "synchronization.command_channel")
    return data


def enabled_cameras(schema: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    capture = _mapping(schema, "capture")
    return tuple(camera for camera in capture["cameras"] if camera.get("enabled", True))


def resolve_camera_keys(
    features: Mapping[str, Any], schema: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], str], ...]:
    resolved = []
    for camera in enabled_cameras(schema):
        candidates = tuple(str(key) for key in camera["source_keys"])
        key = next((candidate for candidate in candidates if candidate in features), None)
        if key is None:
            available = sorted(name for name in features if name.startswith("observation.images."))
            raise ValueError(
                f"camera {camera['role']} is missing; configured candidates={list(candidates)}, "
                f"available={available}"
            )
        resolved.append((camera, key))
    return tuple(resolved)


def vector_indices(spec: Mapping[str, Any], dimension: int, label: str) -> np.ndarray:
    indices = spec.get("indices", "all")
    mask = spec.get("mask")
    if mask is not None:
        if indices != "all":
            raise ValueError(f"{label} cannot define both indices and mask")
        if not isinstance(mask, list) or len(mask) != dimension or not all(
            isinstance(item, bool) for item in mask
        ):
            raise ValueError(f"{label}.mask must contain {dimension} booleans")
        result = np.flatnonzero(mask)
    elif indices == "all":
        result = np.arange(dimension, dtype=np.int64)
    elif isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
        result = np.asarray(indices, dtype=np.int64)
    else:
        raise ValueError(f"{label}.indices must be 'all' or an integer list")
    if result.ndim != 1 or len(result) == 0 or len(set(result.tolist())) != len(result):
        raise ValueError(f"{label} selection must be non-empty and unique")
    if np.any(result < 0) or np.any(result >= dimension):
        raise ValueError(f"{label} selection exceeds source dimension {dimension}")
    return result


def select_vector(value: object, spec: Mapping[str, Any], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector, got {array.shape}")
    return np.ascontiguousarray(array[vector_indices(spec, len(array), label)])


def transformed_vector_dimension(
    spec: Mapping[str, Any], source_dimension: int, label: str
) -> int:
    transform = spec.get("transform")
    if transform is None:
        return source_dimension
    if transform != "ur_base_tcp_pose_to_rotation6d_columns":
        raise ValueError(f"{label} uses unsupported transform: {transform}")
    if source_dimension not in (6, 9):
        raise ValueError(f"{label} TCP pose source dimension must be 6 or 9")
    output_dimension = int(spec.get("output_dim", 0))
    if output_dimension != 9:
        raise ValueError(f"{label}.output_dim must be 9 for column-contiguous rotation6d")
    return output_dimension


def select_transformed_vector(
    value: object, spec: Mapping[str, Any], label: str
) -> np.ndarray:
    transform = spec.get("transform")
    transformed = (
        normalize_ur_base_tcp_pose_to_rotation6d_columns(value)
        if transform == "ur_base_tcp_pose_to_rotation6d_columns"
        else value
    )
    if transform not in (None, "ur_base_tcp_pose_to_rotation6d_columns"):
        raise ValueError(f"{label} uses unsupported transform: {transform}")
    return select_vector(transformed, spec, label)


def capture_vector_names(schema: Mapping[str, Any], vector_name: str) -> tuple[str, ...]:
    vector = _mapping(_mapping(schema, "capture"), vector_name)
    return tuple(str(name) for component in vector["components"] for name in component["names"])


def compose_capture_vector(
    schema: Mapping[str, Any], vector_name: str, channels: Mapping[str, Any]
) -> np.ndarray:
    vector = _mapping(_mapping(schema, "capture"), vector_name)
    parts = []
    for index, component in enumerate(vector["components"]):
        channel_name = str(component["channel"])
        if channel_name not in channels:
            raise ValueError(f"capture.{vector_name} channel is unavailable: {channel_name}")
        source = channels[channel_name]
        attribute = component.get("attribute")
        value = getattr(source, str(attribute)) if attribute else source
        part = select_vector(value, component, f"capture.{vector_name}.components[{index}]")
        if len(part) != len(component["names"]):
            raise ValueError(
                f"capture.{vector_name}.components[{index}] selects {len(part)} values but "
                f"declares {len(component['names'])} names"
            )
        parts.append(part)
    return np.concatenate(parts, dtype=np.float32)


def split_capture_vector(
    schema: Mapping[str, Any], vector_name: str, value: object
) -> dict[str, dict[str, np.ndarray]]:
    vector = _mapping(_mapping(schema, "capture"), vector_name)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"capture {vector_name} must be a finite vector")
    expected = sum(len(component["names"]) for component in vector["components"])
    if len(array) != expected:
        raise ValueError(f"capture {vector_name} has {len(array)} values, expected {expected}")
    result: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for component in vector["components"]:
        length = len(component["names"])
        channel = str(component["channel"])
        attribute = str(component.get("attribute") or "value")
        result.setdefault(channel, {})[attribute] = np.ascontiguousarray(
            array[offset : offset + length]
        )
        offset += length
    return result


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"input schema {key} must be a mapping")
    return value


def _vector_spec(spec: Mapping[str, Any], label: str) -> None:
    if not str(spec.get("key") or spec.get("field") or "").strip():
        raise ValueError(f"{label}.key or field must be configured")
    if "mask" not in spec and "indices" not in spec:
        raise ValueError(f"{label} must define indices or mask")
