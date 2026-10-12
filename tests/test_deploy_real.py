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
