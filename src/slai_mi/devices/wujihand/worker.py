"""Python 3.11 worker owning the WujiHand vendor SDK and USB connection."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slai_mi.devices.worker_server import serve


class FakeBackend:
    def __init__(self) -> None:
        self.positions = [0.0] * 20
        self.disabled = False

    def read_positions(self) -> Sequence[float]:
        return self.positions

    def read_temperature(self) -> Sequence[float]:
        return [25.0] * 20

    def temperature_status(self) -> Mapping[str, Any]:
        return {
            "values": self.read_temperature(),
            "max_c": 25.0,
            "level": "normal",
            "warning_c": 70.0,
            "critical_c": 75.0,
            "limit_c": 80.0,
        }

    def read_limits(self) -> tuple[Sequence[float], Sequence[float]]:
        return [-3.14] * 20, [3.14] * 20

    def write_positions(self, positions: Sequence[float]) -> None:
        self.positions = list(positions)
        self.disabled = False

    def disable(self) -> None:
        self.disabled = True


class WujiWorkerBackend:
    def __init__(self, backend: Any):
        self.backend = backend

    def handle(self, message_type: str, payload: Mapping[str, Any], armed: bool) -> Mapping[str, Any]:
        if message_type == "read_positions":
            positions = [float(value) for value in self.backend.read_positions()]
            if len(positions) != 20:
                raise RuntimeError("WujiHand feedback dimension is not 20")
            return {"type": "state", "positions": positions}
        if message_type == "read_limits":
            lower, upper = self.backend.read_limits()
            lower = [float(value) for value in lower]
            upper = [float(value) for value in upper]
            if len(lower) != 20 or len(upper) != 20:
                raise RuntimeError("WujiHand limit dimension is not 20")
            return {"type": "limits", "lower": lower, "upper": upper}
        if message_type == "read_temperature":
            status = self.backend.temperature_status()
            return {
                "type": "temperature",
                "values": [float(value) for value in status["values"]],
                "max_c": float(status["max_c"]),
                "level": str(status["level"]),
                "warning_c": float(status["warning_c"]),
                "critical_c": float(status["critical_c"]),
                "limit_c": float(status["limit_c"]),
            }
        if message_type == "write_positions":
            if not armed:
                raise RuntimeError("WujiHand worker is not armed")
            positions = [float(value) for value in payload.get("positions", [])]
            if len(positions) != 20:
                raise ValueError("WujiHand command dimension is not 20")
            self.backend.write_positions(positions)
            return {"type": "command_ack"}
        raise ValueError(f"unsupported WujiHand command: {message_type}")

    def disable(self) -> None:
        self.backend.disable()

    def close(self) -> None:
        self.backend.disable()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--usb-serial", required=True)
    parser.add_argument("--product-serial", default="")
    parser.add_argument("--watchdog-s", type=float, default=0.5)
    parser.add_argument("--max-temperature-c", type=float, default=80.0)
    parser.add_argument("--thermal-warning-temperature-c", type=float, default=70.0)
    parser.add_argument("--thermal-critical-temperature-c", type=float, default=75.0)
    parser.add_argument("--limit-margin-rad", type=float, default=0.03)
    parser.add_argument("--max-effort-fraction", type=float, default=0.65)
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    def factory() -> WujiWorkerBackend:
        if args.fake:
            backend = FakeBackend()
        else:
            # This is intentionally the only hardware path importing the ABI-bound SDK.
            from .vendor_wujihandpy import WujiHandPyBackend

            backend = WujiHandPyBackend(
                args.usb_serial,
                expected_product_serial=args.product_serial,
                max_temperature_c=args.max_temperature_c,
                thermal_warning_temperature_c=args.thermal_warning_temperature_c,
                thermal_critical_temperature_c=args.thermal_critical_temperature_c,
                limit_margin_rad=args.limit_margin_rad,
                max_effort_fraction=args.max_effort_fraction,
                max_velocity_rad_s=args.max_velocity_rad_s,
            )
        return WujiWorkerBackend(backend)

    serve(socket_path=args.socket, device_id=args.device_id, backend_factory=factory, watchdog_s=args.watchdog_s)


if __name__ == "__main__":
    main()
