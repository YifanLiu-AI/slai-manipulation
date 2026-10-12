import sys
from pathlib import Path

import pytest

from slai_mi.devices.ur5.process import UR5DriverProcess


def test_fake_ur5_process_state_motion_limits_and_disable():
    driver = UR5DriverProcess(python=Path(sys.executable), host="fake", fake=True)
    with driver:
        assert driver.read_state()["robot_mode"] == 7
        driver.prepare_control()
        with pytest.raises(RuntimeError, match="must be armed"):
            driver.write_twist([0.0] * 6)
        driver.arm()
        driver.write_twist([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert driver.read_state()["tcp_speed"][0] == 0.01
        driver.write_joint_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
        with pytest.raises(RuntimeError, match="linear speed"):
            driver.write_twist([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
        driver.stop_motion()
        assert driver.read_state()["tcp_speed"] == [0.0] * 6
