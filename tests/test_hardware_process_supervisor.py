import sys
import time
from pathlib import Path

import pytest

from slai_mi.devices.ur5.process import UR5DriverProcess
from slai_mi.devices.wujihand.process import WujiHandDriverProcess
from slai_mi.runtime.hardware_supervisor import HardwareProcessSupervisor


def test_supervisor_arms_both_and_fails_both_closed():
    ur5 = UR5DriverProcess(python=Path(sys.executable), host="fake", fake=True)
    wuji = WujiHandDriverProcess(python=Path(sys.executable), usb_serial="fake", fake=True)
    supervisor = HardwareProcessSupervisor({"ur5": ur5, "wujihand": wuji})
    with supervisor:
        supervisor.arm()
        assert ur5.armed and wuji.armed
        ur5.process.kill()
        ur5.process.wait()
        with pytest.raises(RuntimeError, match="failed closed"):
            supervisor.check()
        assert not supervisor.armed
        assert not wuji.armed
        with pytest.raises(RuntimeError, match="cannot be re-armed"):
            supervisor.arm()


def test_blocking_ur5_call_keeps_wuji_watchdog_alive():
    ur5 = UR5DriverProcess(
        # The sleep models time spent inside the real worker's synchronous RTDE
        # handler, where the UR watchdog loop cannot run until the call returns.
        python=Path(sys.executable), host="fake", fake=True, watchdog_s=2.0
    )
    wuji = WujiHandDriverProcess(
        python=Path(sys.executable), usb_serial="fake", fake=True, watchdog_s=0.5
    )
    supervisor = HardwareProcessSupervisor({"ur5": ur5, "wujihand": wuji})
    with supervisor:
        supervisor.arm()
        supervisor.call_with_peer_heartbeats("ur5", lambda: time.sleep(0.7))
        supervisor.check()
        assert supervisor.armed
        assert ur5.armed and wuji.armed


def test_background_heartbeats_cover_armed_setup_gaps():
    ur5 = UR5DriverProcess(
        python=Path(sys.executable), host="fake", fake=True, watchdog_s=0.25
    )
    wuji = WujiHandDriverProcess(
        python=Path(sys.executable), usb_serial="fake", fake=True, watchdog_s=0.5
    )
    supervisor = HardwareProcessSupervisor({"ur5": ur5, "wujihand": wuji})
    with supervisor:
        supervisor.arm()
        time.sleep(0.7)
        supervisor.check()
        assert supervisor.armed
        assert ur5.armed and wuji.armed
