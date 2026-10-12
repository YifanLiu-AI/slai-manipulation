"""Commission the Hiwonder gripper without opening any other robot hardware."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from slai_mi.apps._common import load_yaml
from slai_mi.devices.gripper import HiwonderGripper, SpaceMouseGripperControl
from slai_mi.devices.spacemouse.device import SpaceMouse


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("0", "1", "off", "status", "buttons"))
    value.add_argument("--hardware-config", default="configs/hardware.yaml")
    value.add_argument("--duration", type=float, default=15.0)
    return value


def _gripper(config: dict) -> HiwonderGripper:
    section = config.get("gripper")
    if not isinstance(section, dict) or section.get("enabled") is not True:
        raise SystemExit("hardware.gripper must be enabled")
    return HiwonderGripper(
        str(section["port"]),
        servo_id=int(section.get("servo_id", 1)),
        baud=int(section.get("baud", 115200)),
        move_time_ms=int(section.get("move_time_ms", 1000)),
        max_temperature_c=int(section.get("max_temperature_c", 60)),
        binary_threshold=int(section.get("binary_threshold", 500)),
    )


def _print_state(gripper: HiwonderGripper) -> None:
    state = gripper.read_state()
    print(
        f"position={state.position} binary={state.binary_state} "
        f"normalized={state.normalized_position:.3f} "
        f"target={state.target_position} temperature={state.temperature_c}C "
        f"voltage={state.voltage_v:.3f}V load={int(state.load_enabled)} "
        f"host_monotonic_s={state.host_timestamp_s:.6f} sequence={state.sequence}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    hardware = load_yaml(args.hardware_config)
    with _gripper(hardware) as gripper:
        if args.command in ("0", "1"):
            gripper.command(int(args.command))
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
            control = SpaceMouseGripperControl(gripper)
            print("SpaceMouse test: button 3 = close (1), button 4 = open (0)", flush=True)
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
