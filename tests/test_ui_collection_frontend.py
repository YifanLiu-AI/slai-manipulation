"""Tests for the read-only collection dashboard."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml

from slai_mi.ui import collection_frontend
from slai_mi.ui.collection_frontend import build_handler, offline_status


def test_offline_status_uses_schema_camera_and_dof_names(tmp_path: Path) -> None:
    schema = yaml.safe_load(Path("configs/input_schema.yaml").read_text(encoding="utf-8"))
    wrist = next(camera for camera in schema["capture"]["cameras"] if camera["role"] == "wrist")
    schema["capture"]["cameras"] = [wrist]
    schema["capture"]["primary_timeline_role"] = "wrist"
    schema_path = tmp_path / "input_schema.yaml"
    schema_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    status = offline_status(
        {
            "input_schema": str(schema_path),
            "cameras": {"enabled": False, "devices": [{"name": "wrist", "serial": None}]},
            "spacemouse": {"enabled": False},
        }
    )
    assert status["read_only"] is True
    assert status["camera_online"] == 0
    assert status["cameras"][0]["key"] == "wrist"
    assert status["cameras"][0]["connected"] is False
    assert len(status["dof"]["names"]) == 26
    assert status["temperature"]["level"] == "unknown"


def test_dashboard_serves_status_and_static_page() -> None:
    status = offline_status({"cameras": {"devices": []}, "spacemouse": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(status))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["read_only"] is True

        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        page = response.read().decode("utf-8")
        assert "SLAI-teleop" in page
        assert "Input Map" in page
        assert "Dataset Overview" in page
        assert 'id="spacemouse-container"' in page
        assert 'class="camera-grid"' in page
        assert 'class="temperature-status"' in page
        assert "WujiHand</span><strong>--</strong>" in page
        assert 'id="collection-task"' in page
        assert 'class="dataset-history"' in page
        assert 'data-collection-active="false"' in page

        connection.request("GET", "/styles.css")
        response = connection.getresponse()
        assert response.status == 200
        style = response.read().decode("utf-8")
        assert "--surface-base: #f2f2f7" in style
        assert "#spacemouse-puck.active" in style

        connection.request("GET", "/app.js")
        response = connection.getresponse()
        assert response.status == 200
        script = response.read().decode("utf-8")
        assert 'secondary: "CAM 01 - LEFT"' in script
        assert 'primary: "CAM 00 - CENTER MAIN"' in script
        assert 'wrist: "CAM 02 - RIGHT"' in script
        assert 'new EventSource("/api/events")' in script
        assert 'fetch("/api/status"' in script
        assert "RTCPeerConnection" in script
        assert "/whep" in script
        assert "function renderTemperature(temperature)" in script
        assert 'fetch("/api/collection-history"' in script
        assert 'eventPanel.dataset.collectionActive = String(collectionActive)' in script
        assert "navigator.hid" not in script
        assert "__STITCH_CONFIG__" not in script

        connection.request("GET", "/spacemouse-input-map.svg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/svg+xml"
        input_map = response.read().decode("utf-8")
        assert 'data-button="menu"' in input_map
        assert 'data-button="rotation_lock"' in input_map
        assert 'data-pressed="true"' in input_map
        assert 'id="key-t" data-button="roll_cw"' in input_map
        assert 'id="key-front" data-button="t"' in input_map
        assert 'id="key-roll" data-button="rotation_lock"' in input_map
        assert 'id="key-rear" data-button="front"' in input_map
        assert 'id="key-lock" data-button="rear"' in input_map

        connection.request("GET", "/api/cameras")
        response = connection.getresponse()
        cameras = json.loads(response.read())
        assert response.status == 200
        assert list(cameras) == ["primary", "wrist", "secondary"]

        connection.request("GET", "/api/recording")
        response = connection.getresponse()
        recording = json.loads(response.read())
        assert response.status == 200
        assert recording["state"]["code"] == "starting"
        assert recording["temperature"]["level"] == "unknown"

        connection.request("GET", "/api/devices")
        response = connection.getresponse()
        devices = json.loads(response.read())
        assert response.status == 200
        assert set(devices["devices"]) == {"ur5", "wuji", "wrist"}

        connection.request("GET", "/api/collection-history")
        response = connection.getresponse()
        history = json.loads(response.read())
        assert response.status == 200
        assert history["dataset_root"].endswith("/data/lerobot/real-9D")
        assert isinstance(history["sessions"], list)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_streams_status_and_camera_without_short_polling() -> None:
    status = offline_status({"cameras": {"devices": []}, "spacemouse": {}})

    class Provider:
        def start(self):
            return

        def stop(self):
            return

        def status(self):
            return status

        def camera_jpeg(self, key):
            if key != "primary":
                raise KeyError(key)
            return b"\xff\xd8preview\xff\xd9"

    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(Provider()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        events = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        events.request("GET", "/api/events")
        response = events.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream; charset=utf-8"
        assert response.readline() == b"retry: 500\n"
        assert response.readline() == b"\n"
        event = response.readline().decode("utf-8")
        assert event.startswith("data: ")
        assert json.loads(event.removeprefix("data: "))["read_only"] is True
        events.close()

        stream = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        stream.request("GET", "/stream/primary.mjpg")
        response = stream.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == (
            "multipart/x-mixed-replace; boundary=frame"
        )
        assert response.readline() == b"--frame\r\n"
        assert response.readline() == b"Content-Type: image/jpeg\r\n"
        stream.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_strategy_filters_devices_without_opening_hardware(
    monkeypatch, capsys
) -> None:
    captured = {}

    class Runtime:
        def __init__(self, provider):
            captured["status"] = provider.status()

        def start(self):
            return

        def stop(self):
            return

    class Server:
        server_address = ("127.0.0.1", 8765)
        daemon_threads = False

        def __init__(self, *_args):
            return

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            return

    monkeypatch.setattr(collection_frontend, "DashboardRuntime", Runtime)
    monkeypatch.setattr(collection_frontend, "ThreadingHTTPServer", Server)

    assert (
        collection_frontend.main(
            ["--strategy", "ur5e_wrist_8dof_teleop", "--port", "8765"]
        )
        == 0
    )
    assert captured["status"]["task"].startswith("SpaceMouse UR5e")
    assert "Collection dashboard" in capsys.readouterr().out


def test_camera_only_requires_live_mode() -> None:
    try:
        collection_frontend.main(["--camera-only"])
    except SystemExit as exc:
        assert str(exc) == "--camera-only requires --live"
    else:
        raise AssertionError("camera-only mode must reject an offline server")
