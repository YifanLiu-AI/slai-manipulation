from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from slai_mi.apps.deploy_real import _observation
from slai_mi.input_schema import load_input_schema


def test_live_observation_matches_pi05_model_contract() -> None:
    torch = pytest.importorskip("torch")
    schema = load_input_schema()
    now = time.monotonic()
    frames = {
        camera["role"]: SimpleNamespace(
            color=np.zeros((480, 640, 3), dtype=np.uint8), host_timestamp_s=now
        )
        for camera in schema["capture"]["cameras"]
    }
    batch = _observation(
        schema,
        frames,
        {"tcp_pose": np.zeros(6), "joints": np.zeros(6)},
        np.zeros(20),
        "task",
        torch,
    )
    assert batch["observation.state"].shape == (1, 32)
    assert batch["observation.images.base_0_rgb"].shape == (1, 3, 224, 224)
    assert batch["task"] == ["task"]


def test_task2_pi05_state_places_gripper_at_model_index_11() -> None:
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_gripper_9dof.yaml")
    names = []
    capture_names = [
        name
        for component in schema["capture"]["state"]["components"]
        for name in component["names"]
    ]
    for source in schema["pi05"]["state"]["sources"]:
        if source["key"] == "observation.tcp_pose":
            names.extend(["tcp"] * 9)
        else:
            names.extend([capture_names[index] for index in source["indices"]])
    assert names[9:12] == [
        "wrist.fe.position",
        "wrist.ru.position",
        "gripper.position",
    ]


def test_task2_live_observation_uses_wrist_and_gripper_state() -> None:
    torch = pytest.importorskip("torch")
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_gripper_9dof.yaml")
    now = time.monotonic()
    frames = {
        camera["role"]: SimpleNamespace(
            color=np.zeros((480, 640, 3), dtype=np.uint8), host_timestamp_s=now
        )
        for camera in schema["capture"]["cameras"]
    }
    batch = _observation(
        schema,
        frames,
        {"tcp_pose": np.zeros(6), "joints": np.arange(6, dtype=np.float32)},
        np.asarray([0.1, -0.2, 1.0], dtype=np.float32),
        "task2",
        torch,
    )
    assert batch["observation.state"].shape == (1, 32)
    # 9D TCP pose + wrist FE/RU + binary gripper are the unpadded model state.
    np.testing.assert_allclose(batch["observation.state"][0, 9:12].numpy(), [0.1, -0.2, 1.0])
    assert batch["observation.state"][0, 11].item() == 1.0


def test_live_observation_rejects_camera_skew() -> None:
    torch = pytest.importorskip("torch")
    schema = load_input_schema()
    now = time.monotonic()
    frames = {
        camera["role"]: SimpleNamespace(
            color=np.zeros((480, 640, 3), dtype=np.uint8),
            host_timestamp_s=now - index * 0.1,
        )
        for index, camera in enumerate(schema["capture"]["cameras"])
    }
    with pytest.raises(RuntimeError, match="skew"):
        _observation(schema, frames, {"tcp_pose": np.zeros(6), "joints": np.zeros(6)}, np.zeros(20), "task", torch)
