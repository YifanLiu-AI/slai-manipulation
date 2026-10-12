"""Live-provider contract tests for the read-only collection dashboard."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np

from slai_mi.collection.vla_recorder import SourceSample, SynchronizedInputs
from slai_mi.ui.collection_dashboard import CollectionDashboardProvider
from slai_mi.ui.collection_frontend import DashboardRuntime, build_handler


class FakeStatusProvider:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.status_reads = 0
        self.frame_reads: list[str] = []

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def status(self) -> dict[str, object]:
        self.status_reads += 1
        return {
            "read_only": True,
            "camera_count": 1,
            "camera_online": 1,
            "cameras": [
                {
                    "key": "camera-wrist-key",
                    "role": "wrist",
                    "serial": "not-used-as-a-route",
                    "connected": True,
                    "model": "fake",
                    "fps": 30.0,
                    "resolution": [640, 480],
                    "error": None,
                }
            ],
            "spacemouse": {
                "connected": True,
                "device": "fake SpaceMouse",
                "motion": [0.1, 0.0, 0.0, 0.0, 0.0, -0.2],
                "buttons": {"home": True},
                "active": True,
                "last_activity_ms": 2,
                "error": None,
            },
        }

    def camera_jpeg(self, key: str) -> bytes | None:
        self.frame_reads.append(key)
        if key != "camera-wrist-key":
            raise KeyError(key)
        return b"\xff\xd8fake-jpeg\xff\xd9"


def test_runtime_lifecycle_is_idempotent() -> None:
    provider = FakeStatusProvider()
    runtime = DashboardRuntime(provider)
    runtime.start()
    runtime.start()
    runtime.stop()
    runtime.stop()
    assert (provider.starts, provider.stops) == (1, 1)


def test_live_provider_status_and_role_key_camera_endpoint() -> None:
    provider = FakeStatusProvider()
    runtime = DashboardRuntime(provider)
    runtime.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(runtime))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        status = json.loads(response.read())
        assert response.status == 200
        assert status["spacemouse"]["motion"][0] == 0.1

        connection.request("GET", "/api/cameras/camera-wrist-key/frame.jpg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.read() == b"\xff\xd8fake-jpeg\xff\xd9"
        assert provider.frame_reads == ["camera-wrist-key"]

        connection.request("GET", "/frame/camera-wrist-key.jpg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"\xff\xd8fake-jpeg\xff\xd9"
        assert provider.frame_reads == ["camera-wrist-key", "camera-wrist-key"]

        connection.request("GET", "/api/cameras/not-a-serial/frame.jpg")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        server.shutdown()
        server.server_close()
        runtime.stop()
        thread.join(timeout=2)


def test_collection_provider_publishes_schema_driven_inputs() -> None:
    hardware = {
        "input_schema": "configs/input_schema.yaml",
        "cameras": {
            "devices": [
                {"role": "primary", "serial": "primary-serial"},
                {"role": "wrist", "serial": "wrist-serial"},
                {"role": "secondary", "serial": "secondary-serial"},
            ]
        },
    }
    provider = CollectionDashboardProvider(hardware, "Pick up the block.")
    now = 100.0
    ur5 = SimpleNamespace(actual_q=np.arange(6, dtype=np.float32))
    wuji = SimpleNamespace(
        actual_q=np.arange(20, dtype=np.float32),
        temperature={
            "values": np.full(20, 72.0),
            "max_c": 72.0,
            "level": "warning",
            "warning_c": 70.0,
            "critical_c": 75.0,
            "limit_c": 80.0,
        },
    )
    mouse = SimpleNamespace(
        axes=np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32),
        buttons=np.asarray([1, *([0] * 11)], dtype=np.int64),
    )

    def sample(value, host: float, sequence: int) -> SourceSample:
        return SourceSample(value, host, host, sequence)

    inputs = SynchronizedInputs(
        cameras={
            "primary": sample(np.zeros((480, 640, 3), dtype=np.uint8), now, 10),
            "wrist": sample(np.ones((480, 640, 3), dtype=np.uint8), now + 0.005, 11),
            "secondary": sample(np.full((480, 640, 3), 2, dtype=np.uint8), now - 0.004, 12),
        },
        channels={
            "ur5": sample(ur5, now, 20),
            "wuji": sample(wuji, now, 21),
            "spacemouse": sample(mouse, now, 22),
        },
    )
    provider._started_at = now  # Align the deterministic sample clock with the provider.
    provider.observe_spacemouse(
        np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32),
        {2: True, 8: True, 26: True},
    )
    provider.observe_inputs(inputs)
    status = provider.status()

    assert [camera["key"] for camera in status["cameras"]] == [
        "primary",
        "wrist",
        "secondary",
    ]
    assert len(status["dof"]["values"]) == 26
    assert status["dof"]["values"][:6] == list(range(6))
    assert status["spacemouse"]["buttons"]["menu"] is True
    assert status["spacemouse"]["active"] is True
    assert status["spacemouse"]["buttons"]["t"] is True
    assert status["spacemouse"]["buttons"]["roll_cw"] is True
    assert status["spacemouse"]["buttons"]["rotation_lock"] is True
    assert status["devices"]["ur5"]["state"] == "active"
    assert status["devices"]["wuji"]["state"] == "active"
    assert status["temperature"]["level"] == "warning"
    assert status["temperature"]["max_c"] == 72.0
    assert status["temperature"]["values"] == [72.0] * 20
    assert any(event["code"] == "wuji_temperature" for event in status["events"])
    assert set(status["sync"]["camera_skew_ms"]) == {"wrist", "secondary"}
    assert provider.camera_jpeg("wrist").startswith(b"\xff\xd8")
