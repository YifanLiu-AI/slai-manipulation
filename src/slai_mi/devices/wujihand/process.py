"""Main-process proxy for the isolated WujiHand driver."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from slai_mi.devices.driver_process import DriverProcess


class WujiHandDriverProcess(DriverProcess):
    def __init__(
        self,
        *,
        python: Path,
        usb_serial: str,
        product_serial: str = "",
        fake: bool = False,
        watchdog_s: float = 0.5,
        max_temperature_c: float = 80.0,
        thermal_warning_temperature_c: float = 70.0,
        thermal_critical_temperature_c: float = 75.0,
        limit_margin_rad: float = 0.03,
        max_effort_fraction: float = 0.65,
        max_velocity_rad_s: float = 3.0,
    ):
        arguments = [
            "--usb-serial",
            usb_serial,
            "--product-serial",
            product_serial,
            "--watchdog-s",
            str(watchdog_s),
            "--max-temperature-c",
            str(max_temperature_c),
            "--thermal-warning-temperature-c",
            str(thermal_warning_temperature_c),
            "--thermal-critical-temperature-c",
            str(thermal_critical_temperature_c),
            "--limit-margin-rad",
            str(limit_margin_rad),
            "--max-effort-fraction",
            str(max_effort_fraction),
            "--max-velocity-rad-s",
            str(max_velocity_rad_s),
        ]
        if fake:
            arguments.append("--fake")
        super().__init__(device_id="wujihand", python=python, module="slai_mi.devices.wujihand.worker", arguments=arguments)

    def read_positions(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.request("read_positions")["positions"])

    def read_limits(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        result = self.request("read_limits")
        return (
            tuple(float(value) for value in result["lower"]),
            tuple(float(value) for value in result["upper"]),
        )

    def read_temperature(self) -> dict[str, object]:
        result = self.request("read_temperature")
        return {
            "values": tuple(float(value) for value in result["values"]),
            "max_c": float(result["max_c"]),
            "level": str(result["level"]),
            "warning_c": float(result["warning_c"]),
            "critical_c": float(result["critical_c"]),
            "limit_c": float(result["limit_c"]),
        }

    def write_positions(self, positions: Sequence[float]) -> None:
        if not self.armed:
            raise RuntimeError("WujiHand driver must be armed before motion")
        self.request("write_positions", positions=[float(value) for value in positions])
