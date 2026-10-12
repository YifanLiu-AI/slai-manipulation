import threading
import time

import pytest

from slai_mi.devices.cameras import (
    CameraConfig,
    CameraFrame,
    FrameSynchronizer,
    validate_camera_set,
)


def test_three_camera_contract_requires_unique_serials():
    cameras = [CameraConfig(f"cam{i}", str(i), 640, 480, 30) for i in range(3)]
    assert len(validate_camera_set(cameras)) == 3
    cameras[2] = CameraConfig("cam2", "1", 640, 480, 30)
    with pytest.raises(ValueError, match="serials"):
        validate_camera_set(cameras)


def test_frame_synchronizer_pairs_nearest_host_times():
    sync = FrameSynchronizer(("left", "right"))
    sync.add(CameraFrame("left", 1, 1.0, 10.0, object()))
    sync.add(CameraFrame("right", 2, 1.0, 10.01, object()))
    result = sync.read(0.01)
    assert set(result) == {"left", "right"}


def test_frame_synchronizer_timeout_reports_each_camera_and_latest_skew():
    sync = FrameSynchronizer(("primary", "secondary", "wrist"), max_skew_ms=20.0)
    sync.add(CameraFrame("primary", 1, 1.0, 10.000, object()))
    sync.add(CameraFrame("secondary", 1, 1.0, 10.010, object()))
    sync.add(CameraFrame("wrist", 1, 1.0, 10.035, object()))

    with pytest.raises(TimeoutError) as failure:
        sync.read(0.001)

    message = str(failure.value)
    assert "latest_skew_ms=" in message
    assert all(name in message for name in ("primary", "secondary", "wrist"))


def test_frame_synchronizer_waits_for_frames_within_required_skew():
    sync = FrameSynchronizer(("primary", "secondary", "wrist"), max_skew_ms=20.0)
    sync.add(CameraFrame("primary", 1, 1.0, 10.005, object()))
    sync.add(CameraFrame("secondary", 1, 1.0, 10.000, object()))
    sync.add(CameraFrame("wrist", 1, 1.0, 10.035, object()))

    def add_next_frames():
        time.sleep(0.01)
        sync.add(CameraFrame("secondary", 2, 1.0, 10.033, object()))
        sync.add(CameraFrame("primary", 2, 1.0, 10.038, object()))

    producer = threading.Thread(target=add_next_frames)
    producer.start()
    result = sync.read(0.1)
    producer.join()

    host_times = [frame.host_timestamp_s for frame in result.values()]
    assert (max(host_times) - min(host_times)) * 1000.0 <= 20.0
    assert result["wrist"].sequence == 1
    assert result["primary"].sequence == 2
    assert result["secondary"].sequence == 2
