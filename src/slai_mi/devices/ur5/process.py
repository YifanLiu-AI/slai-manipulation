"""Main-process proxy for the isolated UR5 RTDE driver."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slai_mi.devices.driver_process import DriverProcess


class UR5DriverProcess(DriverProcess):
    def __init__(self, *, python: Path, host: str, fake: bool = False, watchdog_s: float = 0.25, max_linear_m_s: float = 0.02, max_angular_rad_s: float = 0.1):
        arguments = ["--host", host, "--watchdog-s", str(watchdog_s), "--max-linear", str(max_linear_m_s), "--max-angular", str(max_angular_rad_s)]
        if fake:
            arguments.append("--fake")
        # The first control command uploads the RTDE control script and can take
        # longer than steady-state requests. Worker-side speed duration limits
        # and the watchdog remain the physical motion boundary.
        super().__init__(device_id="ur5", python=python, module="slai_mi.devices.ur5.worker", arguments=arguments, request_timeout_s=5.0)

    def read_state(self) -> dict[str, Any]:
        return self.request("read_state")["state"]

    def prepare_control(self) -> None:
        """Connect and validate RTDE control before the driver is armed."""
        self.request("prepare_control")

    def write_twist(self, twist: Sequence[float], *, acceleration: float = 0.2, duration_s: float = 0.02) -> None:
        if not self.armed:
            raise RuntimeError("UR5 driver must be armed before motion")
        self.request("write_twist", twist=[float(value) for value in twist], acceleration=float(acceleration), duration_s=float(duration_s))

    def write_joint_velocity(
        self,
        velocity: Sequence[float],
        *,
        acceleration: float = 0.2,
        duration_s: float = 0.02,
    ) -> None:
        if not self.armed:
            raise RuntimeError("UR5 driver must be armed before motion")
        self.request(
            "write_joint_velocity",
            velocity=[float(value) for value in velocity],
            acceleration=float(acceleration),
            duration_s=float(duration_s),
        )

    def stop_motion(self) -> None:
        self.request("stop_motion")
