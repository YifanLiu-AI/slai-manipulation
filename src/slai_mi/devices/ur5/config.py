"""Typed command-line configuration for real UR5 teleoperation."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from slai_mi.devices.spacemouse.mapping import SpeedSettings

ROBOT_HOST_DEFAULT = ""
HARDWARE_CONFIRMATION = "MOVE_UR5_WITH_SPACEMOUSE"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ZERO_POSE_FILE_DEFAULT = PROJECT_ROOT / "configs" / "poses" / "ur5_zero_pose.json"
TASK_START_FILE_DEFAULT = PROJECT_ROOT / "configs" / "tasks" / "remove_objects_from_box.yaml"


@dataclass(frozen=True)
class UR5TeleopConfig:
    robot_host: str = ROBOT_HOST_DEFAULT
    enable_hardware: bool = False
    confirmation: str = ""
    speeds: SpeedSettings = field(default_factory=SpeedSettings)
    acceleration: float = 0.50
    stop_deceleration: float = 0.25
    rate_hz: float = 125.0
    deadzone: float = 0.12
    input_timeout: float = 0.10
    input_arm_timeout: float = 60.0
    show_input: bool = False
    zero_pose_file: Path = ZERO_POSE_FILE_DEFAULT
    task_start_file: Path = TASK_START_FILE_DEFAULT
    home_command_socket: Path | None = None
    home_translation_speed: float = 0.25
    home_rotation_speed: float = 0.60
    wrist_3_jog_speed: float = 0.20
    max_offset_mm: float = 0.0
    max_rotation_deg: float = 5.0
    max_runtime_s: float = 600.0

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def max_offset_m(self) -> float:
        return self.max_offset_mm / 1000.0

    @property
    def max_rotation_rad(self) -> float:
        return math.radians(self.max_rotation_deg)

    def validate(self) -> None:
        if not self.robot_host.strip():
            raise ValueError("robot host must not be empty")
        if self.enable_hardware and self.confirmation != HARDWARE_CONFIRMATION:
            raise ValueError(f"hardware motion requires --confirm-text {HARDWARE_CONFIRMATION}")

        speed = self.speeds
        _range(speed.translation, 0.0, 0.25, "translation speed", lower_open=True)
        _range(speed.rotation, 0.0, 1.0, "rotation speed", lower_open=True)
        _range(
            speed.boost_translation,
            speed.translation,
            0.25,
            "Ctrl translation speed",
        )
        _range(
            speed.boost_rotation,
            speed.rotation,
            1.0,
            "Ctrl rotation speed",
        )
        _range(self.acceleration, 0.0, 1.0, "acceleration", lower_open=True)
        _range(
            self.stop_deceleration,
            0.0,
            2.0,
            "stop deceleration",
            lower_open=True,
        )
        _range(self.rate_hz, 5.0, 125.0, "control rate")
        _range(self.deadzone, 0.0, 0.99, "deadzone")
        _range(self.input_timeout, 0.05, 0.25, "input timeout")
        _range(self.input_arm_timeout, 5.0, 300.0, "input arming timeout")
        if not str(self.zero_pose_file):
            raise ValueError("zero pose file must not be empty")
        if not str(self.task_start_file):
            raise ValueError("task start file must not be empty")
        if self.home_command_socket is not None and not str(self.home_command_socket):
            raise ValueError("home command socket path must not be empty")
        _range(
            self.home_translation_speed,
            0.0,
            0.25,
            "home translation speed",
            lower_open=True,
        )
        _range(
            self.home_rotation_speed,
            0.0,
            1.0,
            "home rotation speed",
            lower_open=True,
        )
        _range(
            self.wrist_3_jog_speed,
            0.0,
            0.50,
            "wrist_3 jog speed",
            lower_open=True,
        )
        _zero_or_range(self.max_offset_mm, 1.0, 1000.0, "maximum offset")
        _zero_or_range(self.max_rotation_deg, 1.0, 180.0, "maximum rotation")
        _range(self.max_runtime_s, 1.0, 1800.0, "maximum runtime")


def _range(
    value: float,
    minimum: float,
    maximum: float,
    name: str,
    *,
    lower_open: bool = False,
) -> None:
    valid_lower = value > minimum if lower_open else value >= minimum
    if not math.isfinite(value) or not valid_lower or value > maximum:
        bracket = "(" if lower_open else "["
        raise ValueError(f"{name} must be in {bracket}{minimum}, {maximum}]")


def _zero_or_range(value: float, minimum: float, maximum: float, name: str) -> None:
    if not math.isfinite(value) or (value != 0.0 and not minimum <= value <= maximum):
        raise ValueError(f"{name} must be 0 or in [{minimum}, {maximum}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Control a real UR5 with isolated SpaceMouse XYZ/TCP motion over RTDE."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    connection = parser.add_argument_group("robot connection")
    connection.add_argument("--robot-host", default=ROBOT_HOST_DEFAULT)
    connection.add_argument("--enable-hardware", action="store_true")
    connection.add_argument("--confirm-text", default="")

    speed = parser.add_argument_group("Cartesian speed")
    speed.add_argument("--translation-speed", type=float, default=0.080, help="m/s")
    speed.add_argument("--rotation-speed", type=float, default=0.45, help="rad/s")
    speed.add_argument(
        "--ctrl-translation-speed", type=float, default=0.25, help="m/s while Ctrl is held"
    )
    speed.add_argument(
        "--ctrl-rotation-speed", type=float, default=0.60, help="rad/s while Ctrl is held"
    )
    speed.add_argument("--acceleration", type=float, default=0.50, help="m/s^2")
    speed.add_argument("--stop-deceleration", type=float, default=0.25, help="m/s^2")

    input_group = parser.add_argument_group("SpaceMouse input")
    input_group.add_argument("--rate-hz", type=float, default=125.0)
    input_group.add_argument("--deadzone", type=float, default=0.12)
    input_group.add_argument("--input-timeout", type=float, default=0.10, help="seconds")
    input_group.add_argument(
        "--input-arm-timeout",
        type=float,
        default=60.0,
        help="seconds allowed for the move-and-center input handshake",
    )
    input_group.add_argument(
        "--show-input",
        action="store_true",
        help="Open a live six-axis and Shift/Ctrl operator window",
    )
    input_group.add_argument(
        "--wrist-3-jog-speed",
        type=float,
        default=0.20,
        help="rad/s while SpaceMouse Button 1 or 2 is held",
    )

    home = parser.add_argument_group("task-owned recorded states")
    home.add_argument("--zero-pose-file", type=Path, default=ZERO_POSE_FILE_DEFAULT)
    home.add_argument(
        "--task-start-file",
        type=Path,
        default=TASK_START_FILE_DEFAULT,
        help="Task YAML defining the T-key UR5/Wuji zero and Wuji state 1",
    )
    home.add_argument(
        "--home-command-socket",
        type=Path,
        help="Unix datagram socket for recorder-triggered coordinated home requests",
    )
    home.add_argument("--home-translation-speed", type=float, default=0.25, help="m/s")
    home.add_argument("--home-rotation-speed", type=float, default=0.60, help="rad/s")

    safety = parser.add_argument_group("safety envelope")
    safety.add_argument("--max-offset-mm", type=float, default=0.0)
    safety.add_argument("--max-rotation-deg", type=float, default=5.0)
    safety.add_argument("--max-runtime-s", type=float, default=600.0)
    return parser


def config_from_namespace(args: argparse.Namespace) -> UR5TeleopConfig:
    return UR5TeleopConfig(
        robot_host=args.robot_host,
        enable_hardware=args.enable_hardware,
        confirmation=args.confirm_text,
        speeds=SpeedSettings(
            translation=args.translation_speed,
            rotation=args.rotation_speed,
            boost_translation=args.ctrl_translation_speed,
            boost_rotation=args.ctrl_rotation_speed,
        ),
        acceleration=args.acceleration,
        stop_deceleration=args.stop_deceleration,
        rate_hz=args.rate_hz,
        deadzone=args.deadzone,
        input_timeout=args.input_timeout,
        input_arm_timeout=args.input_arm_timeout,
        show_input=args.show_input,
        wrist_3_jog_speed=args.wrist_3_jog_speed,
        zero_pose_file=args.zero_pose_file,
        task_start_file=args.task_start_file,
        home_command_socket=args.home_command_socket,
        home_translation_speed=args.home_translation_speed,
        home_rotation_speed=args.home_rotation_speed,
        max_offset_mm=args.max_offset_mm,
        max_rotation_deg=args.max_rotation_deg,
        max_runtime_s=args.max_runtime_s,
    )


def parse_config(argv: Sequence[str] | None = None) -> UR5TeleopConfig:
    parser = build_parser()
    config = config_from_namespace(parser.parse_args(argv))
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config
