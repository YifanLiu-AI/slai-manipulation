from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slai_mi.devices.wujihand.vendor_wujihandpy import WujiHandPyBackend


class FakeHand:
    def __init__(self, *, serial_number: str) -> None:
        self.serial_number = serial_number
        self.target = None
        self.enabled = None
        self.enabled_writes = []
        self.effort = np.ones((5, 4), dtype=float)
        self.temperature = np.full((5, 4), 25.0)

    def get_product_sn(self):
        return "product-id"

    def read_joint_actual_position(self, timeout=None):
        return np.zeros((5, 4), dtype=float)

    def read_joint_lower_limit(self, timeout=None):
        return np.full((5, 4), -2.0)

    def read_joint_upper_limit(self, timeout=None):
        return np.full((5, 4), 2.0)

    def read_joint_temperature(self, timeout=None):
        return self.temperature.copy()

    def read_joint_error_code(self, timeout=None):
        return np.zeros((5, 4), dtype=np.uint32)

    def read_joint_effort_limit(self, timeout=None):
        return self.effort.copy()

    def write_joint_target_position(self, value, timeout=None) -> None:
        self.target = np.asarray(value)

    def write_joint_effort_limit(self, value, timeout=None) -> None:
        self.effort = np.asarray(value)

    def write_joint_enabled(self, value, timeout=None) -> None:
        self.enabled = value
        self.enabled_writes.append(value)


def test_vendor_adapter_uses_sdk_joint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    backend = WujiHandPyBackend("usb-id")

    assert backend.read_positions().shape == (20,)
    backend.write_positions(np.full(20, 0.005, dtype=float))
    assert backend._hand.target.shape == (5, 4)
    assert backend._hand.enabled is True
    backend.write_positions(np.full(20, 0.006, dtype=float))
    assert backend._hand.enabled_writes == [True]
    backend.disable()
    assert backend._hand.enabled is False
    assert backend._hand.effort.tolist() == np.ones((5, 4)).tolist()


def test_vendor_adapter_rejects_invalid_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    backend = WujiHandPyBackend("usb-id")
    with pytest.raises(ValueError, match="20 finite"):
        backend.write_positions([0.0] * 19)


def test_vendor_adapter_falls_back_to_sdk_joint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    class ShapeStrictHand(FakeHand):
        def write_joint_enabled(self, value, timeout=None) -> None:
            array = np.asarray(value)
            if array.shape != (5, 4):
                raise RuntimeError("Array shape must be {5, 4}!")
            self.enabled = array
            self.enabled_writes.append(array)

    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=ShapeStrictHand))
    backend = WujiHandPyBackend("usb-id")
    backend.write_positions(np.full(20, 0.005, dtype=float))
    assert backend._hand.enabled.shape == (5, 4)
    assert backend._hand.enabled.all()
    backend.disable()
    assert backend._hand.enabled.shape == (5, 4)
    assert not backend._hand.enabled.any()


def test_vendor_adapter_rejects_wrong_product_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    with pytest.raises(RuntimeError, match="product serial mismatch"):
        WujiHandPyBackend("usb-id", expected_product_serial="another-product")


def test_vendor_adapter_limits_each_target_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    backend = WujiHandPyBackend("usb-id", max_velocity_rad_s=0.01)
    backend.write_positions(np.ones(20))
    assert np.max(backend._hand.target) <= 0.001001


def test_vendor_adapter_reports_temperature_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    backend = WujiHandPyBackend("usb-id")
    assert backend.temperature_status()["level"] == "normal"

    backend._hand.temperature.fill(72.0)
    backend._last_temperature_check = 0.0
    backend._check_health(2.1)
    assert backend.temperature_status()["level"] == "warning"

    backend._hand.temperature.fill(76.0)
    backend._check_health(4.2)
    status = backend.temperature_status()
    assert status["level"] == "critical"
    assert status["max_c"] == 76.0
    assert status["warning_c"] == 70.0
    assert status["critical_c"] == 75.0
    assert status["limit_c"] == 80.0


def test_vendor_adapter_fails_closed_above_max_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=FakeHand))
    backend = WujiHandPyBackend("usb-id")
    backend._hand.temperature.fill(80.1)
    backend._last_temperature_check = 0.0
    with pytest.raises(RuntimeError, match="temperature 80.1C exceeds 80.0C"):
        backend._check_health(2.1)
