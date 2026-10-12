from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from slai_mi.devices.ur5.worker import VendorRTDE


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
