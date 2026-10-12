import sys
from pathlib import Path

import pytest

from slai_mi.devices.wujihand.process import WujiHandDriverProcess


def test_fake_wuji_process_round_trip_and_arm_gate():
    driver = WujiHandDriverProcess(python=Path(sys.executable), usb_serial="fake", fake=True)
    with driver:
        assert driver.read_positions() == (0.0,) * 20
        temperature = driver.read_temperature()
        assert temperature == {
            "values": (25.0,) * 20,
            "max_c": 25.0,
            "level": "normal",
            "warning_c": 70.0,
            "critical_c": 75.0,
            "limit_c": 80.0,
        }
        lower, upper = driver.read_limits()
        assert lower == (-3.14,) * 20
        assert upper == (3.14,) * 20
        with pytest.raises(RuntimeError, match="must be armed"):
            driver.write_positions([1.0] * 20)
        driver.arm()
        driver.write_positions([1.0] * 20)
        assert driver.read_positions() == (1.0,) * 20
        assert driver.heartbeat() > 0
        driver.disable()
        assert not driver.armed
