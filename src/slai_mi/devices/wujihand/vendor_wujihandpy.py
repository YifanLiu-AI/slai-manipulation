"""Adapter for the installed WujiHand vendor Python binding.

The SDK identifies USB devices by their USB serial (not a device path).
Commands are deliberately explicit and are not issued during construction.
"""

from __future__ import annotations

import time

import numpy as np

JOINT_COUNT = 20


def _joint_array(value, *, dtype=float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.shape != (JOINT_COUNT,):
        raise RuntimeError(f"WujiHand feedback dimension is {result.shape}, expected (20,)")
    return result


class WujiHandPyBackend:
    def __init__(
        self,
        usb_serial: str,
        *,
        expected_product_serial: str = "",
        max_temperature_c: float = 80.0,
        thermal_warning_temperature_c: float = 70.0,
        thermal_critical_temperature_c: float = 75.0,
        limit_margin_rad: float = 0.03,
        max_effort_fraction: float = 0.65,
        max_velocity_rad_s: float = 3.0,
    ) -> None:
        import wujihandpy

        self._hand = wujihandpy.Hand(serial_number=usb_serial)
        self._max_temperature_c = float(max_temperature_c)
        self._thermal_warning_temperature_c = float(thermal_warning_temperature_c)
        self._thermal_critical_temperature_c = float(thermal_critical_temperature_c)
        self._max_velocity_rad_s = float(max_velocity_rad_s)
        if not 0.0 < max_effort_fraction <= 1.0:
            raise ValueError("WujiHand effort fraction must be in (0, 1]")
        if (
            self._max_temperature_c <= 0.0
            or self._max_velocity_rad_s <= 0.0
            or not 0.0 < self._thermal_warning_temperature_c
            < self._thermal_critical_temperature_c
            < self._max_temperature_c
        ):
            raise ValueError("WujiHand temperature and velocity limits must be positive")

        product_serial = str(self._hand.get_product_sn())
        if expected_product_serial and product_serial != expected_product_serial:
            raise RuntimeError(
                "WujiHand product serial mismatch: "
                f"expected {expected_product_serial}, got {product_serial}"
            )
        position = _joint_array(self._hand.read_joint_actual_position(timeout=2.0))
        lower = _joint_array(self._hand.read_joint_lower_limit(timeout=2.0))
        upper = _joint_array(self._hand.read_joint_upper_limit(timeout=2.0))
        temperature = _joint_array(self._hand.read_joint_temperature(timeout=2.0))
        errors = _joint_array(self._hand.read_joint_error_code(timeout=2.0), dtype=np.uint32)
        effort = _joint_array(self._hand.read_joint_effort_limit(timeout=2.0))
        if not all(np.isfinite(values).all() for values in (position, lower, upper, temperature, effort)):
            raise RuntimeError("WujiHand diagnostics contain non-finite feedback")
        if np.any(lower >= upper):
            raise RuntimeError("WujiHand diagnostics contain invalid joint limits")
        if np.any(errors):
            index = int(np.flatnonzero(errors)[0])
            raise RuntimeError(f"WujiHand joint {index} error: 0x{int(errors[index]):08x}")
        if np.any(temperature > self._max_temperature_c):
            index = int(np.argmax(temperature))
            raise RuntimeError(
                f"WujiHand joint {index} temperature {temperature[index]:.1f}C exceeds "
                f"{self._max_temperature_c:.1f}C"
            )
        if float(np.ptp(effort)) > 1e-6:
            raise RuntimeError("WujiHand effort limits must be uniform for safe restoration")

        self._lower = lower + float(limit_margin_rad)
        self._upper = upper - float(limit_margin_rad)
        if np.any(self._lower >= self._upper):
            raise RuntimeError("WujiHand limit margin leaves an empty joint range")
        self._original_effort = float(effort[0])
        self._operating_effort = self._original_effort * float(max_effort_fraction)
        self._command = np.clip(position, self._lower, self._upper)
        self._last_write_at = time.monotonic()
        self._last_error_check = self._last_write_at
        self._last_temperature_check = self._last_write_at
        self._latest_temperature = temperature.copy()
        self._enabled = False

    def read_limits(self):
        return self._lower.copy(), self._upper.copy()

    def read_positions(self):
        return _joint_array(self._hand.read_joint_actual_position(timeout=1.0))

    def read_temperature(self):
        return self._latest_temperature.copy()

    def temperature_status(self) -> dict[str, object]:
        maximum = float(np.max(self._latest_temperature))
        level = (
            "critical"
            if maximum >= self._thermal_critical_temperature_c
            else "warning"
            if maximum >= self._thermal_warning_temperature_c
            else "normal"
        )
        return {
            "values": self._latest_temperature.copy(),
            "max_c": maximum,
            "level": level,
            "warning_c": self._thermal_warning_temperature_c,
            "critical_c": self._thermal_critical_temperature_c,
            "limit_c": self._max_temperature_c,
        }

    def _write_enabled(self, enabled: bool) -> None:
        try:
            self._hand.write_joint_enabled(enabled, timeout=1.0)
        except RuntimeError as exc:
            if "Array shape" not in str(exc):
                raise
            self._hand.write_joint_enabled(
                np.full((5, 4), enabled, dtype=bool), timeout=1.0
            )

    def _enable_at_current_position(self) -> None:
        actual = np.clip(self.read_positions(), self._lower, self._upper)
        try:
            self._hand.write_joint_target_position(actual.reshape(5, 4), timeout=1.0)
            self._hand.write_joint_effort_limit(
                np.full((5, 4), self._operating_effort), timeout=1.0
            )
            self._write_enabled(True)
            time.sleep(0.2)
        except (RuntimeError, OSError):
            try:
                self._write_enabled(False)
            finally:
                raise
        self._command = actual
        self._last_write_at = time.monotonic() - 0.1
        self._enabled = True

    def _check_health(self, now: float) -> None:
        if now - self._last_error_check >= 0.5:
            errors = _joint_array(
                self._hand.read_joint_error_code(timeout=0.5), dtype=np.uint32
            )
            if np.any(errors):
                index = int(np.flatnonzero(errors)[0])
                raise RuntimeError(
                    f"WujiHand runtime joint {index} error: 0x{int(errors[index]):08x}"
                )
            self._last_error_check = now
        if now - self._last_temperature_check >= 2.0:
            temperature = _joint_array(self._hand.read_joint_temperature(timeout=0.5))
            self._latest_temperature = temperature.copy()
            if np.any(temperature > self._max_temperature_c):
                index = int(np.argmax(temperature))
                raise RuntimeError(
                    f"WujiHand runtime joint {index} temperature "
                    f"{temperature[index]:.1f}C exceeds {self._max_temperature_c:.1f}C"
                )
            self._last_temperature_check = now

    def write_positions(self, positions) -> None:
        target = np.asarray(positions, dtype=float)
        if target.shape != (20,) or not np.isfinite(target).all():
            raise ValueError("WujiHand target must contain 20 finite positions")
        if not self._enabled:
            self._enable_at_current_position()
        now = time.monotonic()
        elapsed = min(max(now - self._last_write_at, 0.001), 0.1)
        requested = np.clip(target, self._lower, self._upper)
        max_delta = self._max_velocity_rad_s * elapsed
        command = self._command + np.clip(requested - self._command, -max_delta, max_delta)
        self._hand.write_joint_target_position(command.reshape(5, 4), timeout=0.5)
        self._command = command
        self._last_write_at = now
        self._check_health(now)

    def disable(self) -> None:
        try:
            self._write_enabled(False)
        finally:
            self._enabled = False
            self._hand.write_joint_effort_limit(
                np.full((5, 4), self._original_effort), timeout=1.0
            )


def backend_factory(usb_serial: str) -> WujiHandPyBackend:
    return WujiHandPyBackend(usb_serial)
