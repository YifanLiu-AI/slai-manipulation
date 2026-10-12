"""Tests for the read-only collection dashboard."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml

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
        assert "采集相机预览" in page
        assert 'class="device-shell"' in page
        assert "采集日志" in page
        assert "Wuji 温度" in page
        assert 'data-temperature-level="unknown"' in page

        connection.request("GET", "/styles.css")
        response = connection.getresponse()
        assert response.status == 200
        style = response.read().decode("utf-8")
        assert "background: #111" in style
        assert ".puck.active" in style

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
        assert set(devices["devices"]) == {"ur5", "wuji"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
