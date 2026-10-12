"""Move the TASK2 UR5, wrist, and gripper to the commissioned start pose."""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from slai_mi.apps._common import load_yaml, project_path, require_real_robot_confirmation
from slai_mi.collection.home_control import (
    StationControlBusyError,
    default_task2_home_socket,
    request_home_if_active,
    station_control_is_busy,
)
from slai_mi.devices.gripper import create_gripper
from slai_mi.devices.ur5.geometry import joint_home_velocity
from slai_mi.devices.wrist_sensor.teleop import WristMasterSlaveController
from slai_mi.site_adapter import UR5OnlySession, _task_home_joints


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--task", default="configs/tasks/task2_continuous.yaml")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--active-wait",
        type=float,
        default=45.0,
        help="Seconds to wait for a starting hardware owner to expose HOME",
    )
    parser.add_argument(
        "--active-socket",
        type=Path,
        default=default_task2_home_socket(),
        help="Low-level endpoint owned by an active TASK2 controller",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Do not dispatch to an active controller; open hardware directly",
    )
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    require_real_robot_confirmation(args.execute_real, args.confirm)
    if not args.execute_real:
        print(
            {
                "app": "task2_home",
                "mode": "dry-run",
                "task": args.task,
                "active_socket": str(args.active_socket),
                "dispatch": "standalone" if args.standalone else "active-controller-first",
            }
        )
        return 0
    if not args.standalone:
        if request_home_if_active(args.active_socket):
            print(f"TASK2 HOME accepted by active low-level controller: {args.active_socket}")
            return 0
        deadline = time.monotonic() + args.active_wait
        while station_control_is_busy() and time.monotonic() < deadline:
            time.sleep(0.05)
            if request_home_if_active(args.active_socket):
                print(
                    f"TASK2 HOME accepted by starting low-level controller: "
                    f"{args.active_socket}"
                )
                return 0
        if station_control_is_busy():
            raise StationControlBusyError(
                "station is controlled by a process that does not expose the TASK2 HOME endpoint"
            )
    hardware = load_yaml(args.hardware_config)
    task = load_yaml(args.task)
    target = _task_home_joints(task)[:6]
    session = UR5OnlySession(hardware, task)
    wrist_cfg = hardware["wrist_sensor"]
    wrist = WristMasterSlaveController(
        project_path(wrist_cfg["config"]),
        teleop_port=str(wrist_cfg.get("teleop_port", "auto")),
        openrb_port=str(wrist_cfg.get("openrb_port", "auto")),
        baud=int(wrist_cfg.get("baud", 115200)),
    )
    gripper = create_gripper(hardware["gripper"])
    with session.lease(arm=False), wrist, gripper:
        gripper.command_position(0.0)
        wrist.request_home()
        session.prepare_ur5_control()
        session.arm()
        started = time.monotonic()
        while True:
            state = session.read_ur5_state()
            current = np.asarray(state["joints"], dtype=np.float64)
            velocity, ur5_done = joint_home_velocity(
                current, target, session.home_joint_speed, tolerance=0.01
            )
            wrist_done = bool(wrist.home_status().get("at_home"))
            grip = gripper.read_state()
            grip_done = grip.normalized_position <= float(
                hardware["gripper"].get("home_tolerance_normalized", 0.02)
            )
            if ur5_done:
                session.write_ur5_joint_velocity(
                    np.zeros(6), acceleration=session.ur5_acceleration,
                    duration_s=session.control_period_s,
                )
            else:
                session.write_ur5_joint_velocity(
                    velocity, acceleration=session.ur5_acceleration,
                    duration_s=session.control_period_s,
                )
            if ur5_done and wrist_done and grip_done:
                print("TASK2 HOME complete")
                print(f"ur5_error_rad={float(np.max(np.abs(target-current))):.6f}")
                print(f"wrist={wrist.home_status()}")
                print(
                    f"gripper_position={grip.position} "
                    f"normalized_position={grip.normalized_position:.4f}"
                )
                return 0
            if time.monotonic() - started >= args.timeout:
                raise TimeoutError(
                    f"TASK2 HOME timeout: ur5_done={ur5_done}, "
                    f"wrist_done={wrist_done}, gripper_done={grip_done}"
                )
            time.sleep(session.control_period_s)


if __name__ == "__main__":
    raise SystemExit(main())
