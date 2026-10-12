"""Configuration factory for interchangeable station gripper adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import GripperDevice
from .feetech_sts3215 import FeetechSTS3215Gripper
from .hiwonder import HiwonderGripper

_DRIVER_ALIASES = {
    "feetech_sts3215": "feetech_sts3215",
    "sts3215": "feetech_sts3215",
    "hiwonder": "hiwonder",
    "lewansoul": "hiwonder",
    "lobot": "hiwonder",
}


def available_gripper_drivers() -> tuple[str, ...]:
    """Return the canonical driver names accepted by the factory."""
    return tuple(sorted(set(_DRIVER_ALIASES.values())))


def resolve_gripper_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a selected nested adapter while accepting the legacy flat layout."""
    requested = str(config.get("driver", "feetech_sts3215")).strip().lower()
    try:
        driver = _DRIVER_ALIASES[requested]
    except KeyError as exc:
        available = ", ".join(available_gripper_drivers())
        raise ValueError(
            f"unsupported gripper driver: {requested!r}; available: {available}"
        ) from exc

    resolved = {key: value for key, value in config.items() if key != "adapters"}
    adapters = config.get("adapters")
    if adapters is not None:
        if not isinstance(adapters, Mapping):
            raise TypeError("gripper.adapters must be a mapping")
        selected = adapters.get(requested, adapters.get(driver))
        if not isinstance(selected, Mapping):
            raise ValueError(f"gripper.adapters has no configuration for driver {driver!r}")
        resolved.update(selected)
    resolved["driver"] = driver
    return resolved


def create_gripper(config: Mapping[str, Any]) -> GripperDevice:
    """Build the configured adapter behind the normalized-position contract."""
    resolved = resolve_gripper_config(config)
    driver = str(resolved["driver"])
    if driver == "feetech_sts3215":
        return FeetechSTS3215Gripper(
            str(resolved["port"]),
            servo_id=int(resolved.get("servo_id", 7)),
            baud=int(resolved.get("baud", 1_000_000)),
            open_position=int(resolved.get("open_position", 100)),
            closed_position=int(resolved.get("closed_position", 3995)),
            binary_threshold=(
                int(resolved["binary_threshold"])
                if "binary_threshold" in resolved
                else None
            ),
            speed=int(resolved.get("speed", 250)),
            acceleration=int(resolved.get("acceleration", 8)),
            max_temperature_c=int(resolved.get("max_temperature_c", 70)),
            max_load_raw=int(resolved.get("max_load_raw", 750)),
            max_current_raw=int(resolved.get("max_current_raw", 600)),
            timeout_s=float(resolved.get("timeout_s", 0.08)),
        )
    if driver == "hiwonder":
        return HiwonderGripper(
            str(resolved["port"]),
            servo_id=int(resolved.get("servo_id", 1)),
            baud=int(resolved.get("baud", 115_200)),
            open_position=int(resolved.get("open_position", 0)),
            closed_position=int(resolved.get("closed_position", 1000)),
            feedback_position_bias=int(resolved.get("feedback_position_bias", 0)),
            binary_threshold=(
                int(resolved["binary_threshold"])
                if "binary_threshold" in resolved
                else None
            ),
            full_stroke_time_ms=int(resolved.get("full_stroke_time_ms", 1000)),
            min_move_time_ms=int(resolved.get("min_move_time_ms", 20)),
            max_temperature_c=int(resolved.get("max_temperature_c", 60)),
            open_unload_delay_s=(
                float(resolved["open_unload_delay_s"])
                if resolved.get("open_unload_delay_s") is not None
                else None
            ),
            timeout_s=float(resolved.get("timeout_s", 0.3)),
        )
    raise AssertionError(f"unhandled gripper driver: {driver}")
