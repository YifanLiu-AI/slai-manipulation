from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from slai_mi.devices.ur5.worker import FakeRTDE, UR5WorkerBackend, VendorRTDE


class FakeReceiver:
    def __init__(self, host: str) -> None:
        self.host = host
        self.disconnected = False

    def isConnected(self) -> bool:
        return True

    def isEmergencyStopped(self) -> bool:
        return False

    def isProtectiveStopped(self) -> bool:
        return False

    def getActualTCPPose(self) -> list[float]:
        return [0.0] * 6

    def disconnect(self) -> None:
        self.disconnected = True


class FakeControl:
    def __init__(self, host: str) -> None:
        self.host = host
        self.reuploads = 0
        self.stopped = False
        self.disconnected = False
        self.speed_result = True
        self.speed_calls = 0
        self.stop_l_calls = 0

    def isConnected(self) -> bool:
        return True

    def isProgramRunning(self) -> bool:
        return True

    def reuploadScript(self) -> bool:
        self.reuploads += 1
        return True

    def speedL(self, *_args: object) -> bool:
        self.speed_calls += 1
        return self.speed_result

    def speedStop(self, _deceleration: float) -> bool:
        return True

    def stopL(self, _deceleration: float) -> None:
        self.stop_l_calls += 1

    def isPoseWithinSafetyLimits(self, _pose: list[float]) -> bool:
        return True

    def stopScript(self) -> None:
        self.stopped = True

    def disconnect(self) -> None:
        self.disconnected = True


def install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=FakeReceiver)
    )
    monkeypatch.setitem(
        sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=FakeControl)
    )


def test_worker_floor_blocks_descent_but_allows_upward_recovery() -> None:
    rtde = FakeRTDE()
    rtde.pose = [0.0, 0.0, 0.13093, 0.0, 0.0, 0.0]
    backend = UR5WorkerBackend(
        rtde,
        max_linear=0.25,
        max_angular=0.60,
        min_tcp_z_m=0.13093,
    )
    payload = {
        "acceleration": 0.5,
        "duration_s": 0.008,
    }

    backend.handle(
        "write_twist", {**payload, "twist": [0.01, 0.0, -0.08, 0.0, 0.0, 0.0]}, armed=True
    )
    assert rtde.twist == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]

    backend.handle(
        "write_twist", {**payload, "twist": [0.0, 0.0, 0.08, 0.0, 0.0, 0.0]}, armed=True
    )
    assert rtde.twist == [0.0, 0.0, 0.08, 0.0, 0.0, 0.0]


def test_worker_floor_blocks_a_command_projected_below_the_floor() -> None:
    rtde = FakeRTDE()
    rtde.pose = [0.0, 0.0, 0.14, 0.0, 0.0, 0.0]
    backend = UR5WorkerBackend(
        rtde,
        max_linear=0.25,
        max_angular=0.60,
        min_tcp_z_m=0.13093,
    )

    backend.handle(
        "write_twist",
        {
            "twist": [0.0, 0.0, -0.08, 0.0, 0.0, 0.0],
            "acceleration": 0.5,
            "duration_s": 0.008,
        },
        armed=True,
    )

    assert rtde.twist == [0.0] * 6


def test_worker_clips_float_noise_but_rejects_meaningful_joint_overspeed() -> None:
    rtde = FakeRTDE()
    backend = UR5WorkerBackend(rtde, max_linear=0.25, max_angular=0.60)
    payload = {
        "velocity": [0.6000000000000001, 0.0, 0.0, 0.0, 0.0, 0.0],
        "acceleration": 0.5,
        "duration_s": 0.008,
    }

    assert backend.handle("write_joint_velocity", payload, armed=True) == {
        "type": "command_ack"
    }
    with pytest.raises(ValueError, match="joint speed"):
        backend.handle(
            "write_joint_velocity",
            {**payload, "velocity": [0.601, 0.0, 0.0, 0.0, 0.0, 0.0]},
            armed=True,
        )


def test_vendor_replaces_existing_control_script_and_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    vendor = VendorRTDE("robot")
    vendor.speed([0.0] * 6, 0.1, 0.1)
    control = vendor.control
    assert control.reuploads == 1
    vendor.stop()
    assert vendor.control is None
    assert control.stop_l_calls == 1
    assert control.stopped
    assert control.disconnected
    vendor.close()
    assert control.stop_l_calls == 1
    assert vendor.receiver.disconnected


def test_vendor_prepare_starts_and_checks_control_without_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    vendor = VendorRTDE("robot")
    vendor.prepare([0.0] * 6)
    assert vendor.control.reuploads == 1
    assert vendor.control.speed_calls == 0
    vendor.close()


def test_vendor_reconnects_when_uploaded_script_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    vendor = VendorRTDE("robot")
    vendor.prepare([0.0] * 6)
    stale = vendor.control
    stale.isProgramRunning = lambda: False
    vendor.prepare([0.0] * 6)
    assert vendor.control is not stale
    assert stale.disconnected
    assert vendor.control.reuploads == 1
    vendor.close()


def test_vendor_rejects_false_speed_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(monkeypatch)
    vendor = VendorRTDE("robot")
    vendor._start_control()
    vendor.control.speed_result = False
    try:
        with pytest.raises(RuntimeError, match="speedL command returned false"):
            vendor.speed([0.0] * 6, 0.1, 0.1)
    finally:
        vendor.close()
