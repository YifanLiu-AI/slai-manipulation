"""Commission the selected gripper adapter without opening other robot hardware."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from slai_mi.apps._common import load_yaml
from slai_mi.devices.gripper import (
    ContinuousSpaceMouseGripperControl,
    GripperDevice,
    create_gripper,
)
from slai_mi.devices.spacemouse.device import SpaceMouse


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        help="normalized position in [0,1], or one of: off, status, buttons",
    )
    value.add_argument("--hardware-config", default="configs/hardware.yaml")
    value.add_argument("--duration", type=float, default=15.0)
    return value


def _gripper(config: dict) -> GripperDevice:
    section = config.get("gripper")
    if not isinstance(section, dict) or section.get("enabled") is not True:
        raise SystemExit("hardware.gripper must be enabled")
    return create_gripper(section)


def _optional(value: object) -> str:
    return "n/a" if value is None else str(value)


def _print_state(gripper: GripperDevice) -> None:
    state = gripper.read_state()
    print(
        f"driver={gripper.driver_name} position={state.position} binary={state.binary_state} "
        f"normalized={state.normalized_position:.3f} "
        f"target={state.target_position} "
        f"target_normalized={state.normalized_target_position:.3f} "
        f"temperature={state.temperature_c}C "
        f"voltage={state.voltage_v:.3f}V torque={int(state.load_enabled)} "
        f"speed_raw={_optional(getattr(state, 'speed_raw', None))} "
        f"load_raw={_optional(getattr(state, 'load_raw', None))} "
        f"current_raw={_optional(getattr(state, 'current_raw', None))} "
        f"moving={_optional(getattr(state, 'moving', None))} "
        f"host_monotonic_s={state.host_timestamp_s:.6f} sequence={state.sequence}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    hardware = load_yaml(args.hardware_config)
    with _gripper(hardware) as gripper:
        if args.command not in {"off", "status", "buttons"}:
            try:
                position = float(args.command)
            except ValueError as exc:
                raise SystemExit("command must be a normalized position in [0,1]") from exc
            gripper.command_position(position)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                _print_state(gripper)
                time.sleep(0.2)
        elif args.command == "off":
            gripper.unload()
            _print_state(gripper)
        elif args.command == "status":
            _print_state(gripper)
        else:
            section = hardware["gripper"]
            control = ContinuousSpaceMouseGripperControl(
                gripper,
                rate_per_s=float(section.get("teleop_rate_per_s", 1.0)),
                command_hz=float(section.get("teleop_command_hz", 30.0)),
            )
            control.synchronize()
            print(
                "SpaceMouse test: hold button 3 to close gradually; "
                "hold button 4 to open gradually; release to hold",
                flush=True,
            )
            deadline = time.monotonic() + args.duration
            next_status = 0.0
            with SpaceMouse() as mouse:
                while time.monotonic() < deadline:
                    _motion, buttons = mouse.state()
                    command = control.update(buttons)
                    now = time.monotonic()
                    if command is not None:
                        print(f"button command={command}", flush=True)
                    if command is not None or now >= next_status:
                        _print_state(gripper)
                        next_status = now + 0.5
                    time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
