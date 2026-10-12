"""Hardware orchestration for guarded real-UR5 SpaceMouse control."""

from __future__ import annotations

import fcntl
import math
import os
import signal
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.collection.home_control import HomeCommandReceiver
from slai_mi.collection.task_start_pose import load_task_start
from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.device import SpaceMouse
from slai_mi.devices.spacemouse.diagnostics import (
    validate_service_binding,
    wait_for_live_input,
)
from slai_mi.devices.spacemouse.mapping import (
    MotionMode,
    SpeedProfile,
    build_hardware_twist,
    required_released_buttons,
    select_speed_limits,
    wrist_3_jog_direction,
)
from slai_mi.devices.spacemouse.monitor import SpaceMouseMonitor

from .config import UR5TeleopConfig
from .geometry import (
    apply_relative_workspace_guard,
    home_twist,
    joint_home_velocity,
    rotation_offset_rad,
)
from .zero_pose import load_zero_pose

NORMAL_SAFETY_MODE = 1
RUNNING_ROBOT_MODE = 7
LOCK_PATH = Path("/tmp/robot_teleoperation_ur5_spacemouse.lock")


@dataclass(frozen=True)
class RuntimeDependencies:
    receiver_factory: Callable[[str], Any]
    control_factory: Callable[[str], Any]
    spacemouse_factory: Callable[..., AbstractContextManager[Any]] = SpaceMouse
    input_preflight: Callable[[], str | None] = validate_service_binding
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    emit: Callable[[str], None] = print
    lock_factory: Callable[[], AbstractContextManager[None]] | None = None
    monitor_factory: Callable[[], AbstractContextManager[Any]] = SpaceMouseMonitor


def default_dependencies() -> RuntimeDependencies:
    import rtde_control
    import rtde_receive

    return RuntimeDependencies(
        receiver_factory=rtde_receive.RTDEReceiveInterface,
        control_factory=rtde_control.RTDEControlInterface,
    )


def validate_robot_health(receiver: Any) -> None:
    if receiver.isEmergencyStopped():
        raise RuntimeError("UR5 emergency stop is active")
    if receiver.isProtectiveStopped():
        raise RuntimeError("UR5 protective stop is active")
    robot_mode = int(receiver.getRobotMode())
    if robot_mode != RUNNING_ROBOT_MODE:
        raise RuntimeError(
            f"UR5 robot mode is {robot_mode}, expected {RUNNING_ROBOT_MODE} (RUNNING)"
        )
    safety_mode = int(receiver.getSafetyMode())
    if safety_mode != NORMAL_SAFETY_MODE:
        raise RuntimeError(
            f"UR5 safety mode is {safety_mode}, expected {NORMAL_SAFETY_MODE} (NORMAL)"
        )


def read_tcp_pose(receiver: Any) -> np.ndarray:
    pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError(f"invalid UR5 TCP feedback: {pose}")
    return pose


def read_tcp_speed(receiver: Any) -> np.ndarray:
    speed = np.asarray(receiver.getActualTCPSpeed(), dtype=np.float64)
    if speed.shape != (6,) or not np.isfinite(speed).all():
        raise RuntimeError(f"invalid UR5 speed feedback: {speed}")
    return speed


def read_joint_positions(receiver: Any) -> np.ndarray:
    joints = np.asarray(receiver.getActualQ(), dtype=np.float64)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise RuntimeError(f"invalid UR5 joint feedback: {joints}")
    return joints


def ensure_stationary(receiver: Any) -> None:
    speed = read_tcp_speed(receiver)
    if np.linalg.norm(speed[:3]) > 0.002 or np.linalg.norm(speed[3:]) > 0.02:
        raise RuntimeError(f"UR5 is already moving; refusing control: {speed.tolist()}")


def wait_for_stationary_pose(
    receiver: Any,
    *,
    timeout_s: float = 30.0,
    hold_s: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Return feedback only after the TCP remains stationary for a hold period."""
    deadline = monotonic() + timeout_s
    stationary_since: float | None = None
    while monotonic() < deadline:
        speed = read_tcp_speed(receiver)
        now = monotonic()
        stationary = bool(np.linalg.norm(speed[:3]) <= 0.002 and np.linalg.norm(speed[3:]) <= 0.02)
        if stationary:
            stationary_since = stationary_since or now
            if now - stationary_since >= hold_s:
                return read_tcp_pose(receiver)
        else:
            stationary_since = None
        sleep(0.02)
    raise RuntimeError(f"UR5 did not remain stationary for {hold_s:.1f}s")


def ensure_control_program(
    control: Any,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timeout_s: float = 3.0,
) -> None:
    """Require the UR controller to execute the uploaded RTDE control script."""
    if control.isProgramRunning():
        return
    if not control.reuploadScript():
        raise RuntimeError("UR5 RTDE control script is not running and reupload failed")
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if control.isProgramRunning():
            return
        sleep(0.05)
    raise RuntimeError("UR5 RTDE control script did not start after reupload")


@contextmanager
def exclusive_controller_lock(path: Path = LOCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another SpaceMouse UR5 controller is already running") from exc
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


def safe_stop(control: Any | None, deceleration: float) -> None:
    if control is None:
        return
    with suppress(Exception):
        control.speedStop(deceleration)
    with suppress(Exception):
        control.stopL(deceleration)
    with suppress(Exception):
        control.stopScript()


def configuration_messages(config: UR5TeleopConfig) -> list[str]:
    return [
        (
            "Controls: cap=XYZ; hold Shift=TCP-local rotation; "
            "Button 1/2=wrist_3 clockwise/counterclockwise; hold Button 4=return to zero."
        ),
        "XYZ and TCP rotation are isolated to suppress cross-axis motion.",
        (
            "Training maxima="
            f"{config.speeds.translation * 1000.0:.1f} mm/s,"
            f"{config.speeds.rotation:.2f} rad/s."
        ),
        (
            "Hold SpaceMouse Ctrl for per-axis maxima="
            f"{config.speeds.boost_translation * 1000.0:.0f} mm/s,"
            f"{config.speeds.boost_rotation:.2f} rad/s."
        ),
        (
            "Button 4 home maxima="
            f"{config.home_translation_speed * 1000.0:.1f} mm/s,"
            f"{config.home_rotation_speed:.2f} rad/s; release stops physical home."
        ),
        (
            f"Wrist_3 jog={config.wrist_3_jog_speed:.2f} rad/s; "
            f"control loop={config.rate_hz:.0f} Hz; acceleration={config.acceleration:.2f}."
        ),
        "Low-latency path active; the UR controller's safety system remains active.",
    ]


class UR5TeleopRuntime:
    def __init__(
        self,
        config: UR5TeleopConfig,
        dependencies: RuntimeDependencies | None = None,
        stop_event: threading.Event | None = None,
        *,
        install_signal_handlers: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.dependencies = dependencies or default_dependencies()
        self.stop_event = stop_event or threading.Event()
        self.install_signal_handlers = install_signal_handlers

    def run(self) -> None:
        if self.install_signal_handlers:
            signal.signal(signal.SIGINT, self._request_stop)
            signal.signal(signal.SIGTERM, self._request_stop)

        lock_factory = self.dependencies.lock_factory or exclusive_controller_lock
        with lock_factory():
            self._run_locked()

    def _request_stop(self, _signum: int, _frame: Any) -> None:
        self.stop_event.set()

    def _run_locked(self) -> None:
        receiver = None
        control = None
        emit = self.dependencies.emit
        try:
            receiver = self.dependencies.receiver_factory(self.config.robot_host)
            if not receiver.isConnected():
                raise RuntimeError(f"RTDE receive did not connect to {self.config.robot_host}")
            validate_robot_health(receiver)
            ensure_stationary(receiver)
            checked_pose = read_tcp_pose(receiver)
            zero_pose = load_zero_pose(self.config.zero_pose_file, self.config.robot_host)
            task_start = load_task_start(self.config.task_start_file)
            zero_joints = task_start.ur5_zero
            emit(
                f"UR5 read-only check passed: host={self.config.robot_host} "
                f"tcp={_rounded_pose(checked_pose)}"
            )
            emit(
                "Button 4 task zero loaded: "
                f"task={task_start.task_id} path={task_start.path} "
                f"q={_rounded_pose(zero_joints)}"
            )
            if not self.config.enable_hardware:
                emit("DRY RUN: RTDE control and SpaceMouse input were not opened")
                return

            preflight_message = self.dependencies.input_preflight()
            if preflight_message:
                emit(preflight_message)

            monitor_context = (
                self.dependencies.monitor_factory() if self.config.show_input else nullcontext(None)
            )
            home_context = (
                HomeCommandReceiver(self.config.home_command_socket)
                if self.config.home_command_socket is not None
                else nullcontext(None)
            )
            with (
                monitor_context as monitor,
                self.dependencies.spacemouse_factory(
                    deadzone=self.config.deadzone,
                    stale_timeout=self.config.input_timeout,
                ) as spacemouse,
                home_context as home_receiver,
            ):

                def observe_input(motion: np.ndarray, buttons: dict[int, bool]) -> None:
                    if monitor is not None and not self._update_monitor(monitor, motion, buttons):
                        raise RuntimeError("SpaceMouse input window was closed")

                wait_for_live_input(
                    spacemouse,
                    required_released_buttons(),
                    timeout_s=self.config.input_arm_timeout,
                    monotonic=self.dependencies.monotonic,
                    sleep=self.dependencies.sleep,
                    emit=emit,
                    on_state=observe_input,
                )
                start_pose = checked_pose
                control = self.dependencies.control_factory(self.config.robot_host)
                if not control.isConnected():
                    raise RuntimeError(f"RTDE control did not connect to {self.config.robot_host}")
                ensure_control_program(
                    control,
                    monotonic=self.dependencies.monotonic,
                    sleep=self.dependencies.sleep,
                )
                validate_robot_health(receiver)
                ensure_stationary(receiver)
                start_pose = read_tcp_pose(receiver)
                if not control.isPoseWithinSafetyLimits(start_pose.tolist()):
                    raise RuntimeError("current UR5 TCP pose is outside configured safety limits")
                if zero_pose is not None and not control.isPoseWithinSafetyLimits(
                    zero_pose.tolist()
                ):
                    raise RuntimeError("recorded UR5 zero pose is outside configured safety limits")
                if zero_joints is not None and not control.isJointsWithinSafetyLimits(
                    zero_joints.tolist()
                ):
                    raise RuntimeError(
                        "recorded UR5 joint zero is outside configured safety limits"
                    )
                emit(f"REAL UR5 CONTROL ENABLED: armed_tcp={_rounded_pose(start_pose)}")

                for message in configuration_messages(self.config):
                    emit(message)
                self._control_loop(
                    receiver,
                    control,
                    spacemouse,
                    start_pose,
                    monitor,
                    zero_pose=zero_pose,
                    zero_joints=zero_joints,
                    home_receiver=home_receiver,
                )
        finally:
            safe_stop(control, self.config.stop_deceleration)
            if control is not None:
                with suppress(Exception):
                    control.disconnect()
            if receiver is not None:
                with suppress(Exception):
                    receiver.disconnect()
            emit("UR5 SpaceMouse controller stopped.")

    def _control_loop(
        self,
        receiver: Any,
        control: Any | None,
        spacemouse: Any,
        start_pose: np.ndarray,
        monitor: Any | None = None,
        *,
        zero_pose: np.ndarray | None = None,
        zero_joints: np.ndarray | None = None,
        home_receiver: Any | None = None,
    ) -> None:
        config = self.config
        clock = self.dependencies.monotonic
        started_at = clock()
        next_report = started_at
        next_program_check = started_at
        last_mode = MotionMode.TRANSLATION_XYZ
        last_profile = select_speed_limits({}, config.speeds).profile
        remote_home_active = False
        home_pressed_last = False
        home_reached_last = False

        while not self.stop_event.is_set():
            loop_started = clock()
            if loop_started - started_at >= config.max_runtime_s:
                self.dependencies.emit("Maximum test duration reached; stopping.")
                break

            validate_robot_health(receiver)
            current_pose = read_tcp_pose(receiver)
            current_speed = read_tcp_speed(receiver)
            if control is not None and loop_started >= next_program_check:
                if not control.isProgramRunning():
                    raise RuntimeError("UR5 RTDE control script stopped during teleoperation")
                next_program_check = loop_started + 0.5
            offset_m = float(np.linalg.norm(current_pose[:3] - start_pose[:3]))
            angle_rad = rotation_offset_rad(start_pose, current_pose)

            motion, buttons = spacemouse.state()
            speed = select_speed_limits(buttons, config.speeds)
            motion_command = build_hardware_twist(motion, buttons, speed)
            wrist_direction = wrist_3_jog_direction(buttons)
            wrist_buttons_pressed = bool(
                buttons.get(int(Button.ONE), False) or buttons.get(int(Button.TWO), False)
            )
            effective_mode = (
                MotionMode.WRIST_3_JOINT if wrist_buttons_pressed else motion_command.mode
            )
            if monitor is not None and not self._update_monitor(
                monitor, motion, buttons, mode=effective_mode
            ):
                self.dependencies.emit("SpaceMouse input window closed; stopping.")
                break
            home_pressed = bool(buttons.get(int(Button.HOME), False))
            cap_active = bool(np.any(np.abs(motion) > 0.0))
            home_available = zero_joints is not None or zero_pose is not None
            if home_receiver is not None and home_receiver.poll():
                if not home_available:
                    self.dependencies.emit(
                        f"Recorder home ignored: record a zero pose at {config.zero_pose_file}"
                    )
                else:
                    remote_home_active = True
                    self.dependencies.emit("Recorder requested UR5 home.")
            if home_pressed and not home_pressed_last:
                if not home_available:
                    self.dependencies.emit(
                        f"Button 4 ignored: record a zero pose at {config.zero_pose_file}"
                    )
                elif cap_active:
                    self.dependencies.emit("Button 4 ignored: center the SpaceMouse cap first.")
                elif wrist_buttons_pressed:
                    self.dependencies.emit("Button 4 ignored: release Button 1 and 2 first.")
                else:
                    self.dependencies.emit("Button 4 home started; release Button 4 to stop.")
            elif not home_pressed and home_pressed_last and not remote_home_active:
                self.dependencies.emit("Button 4 released; hold position.")
            home_pressed_last = home_pressed

            if remote_home_active and cap_active:
                remote_home_active = False
                self.dependencies.emit("Recorder home canceled by SpaceMouse cap input.")
            if remote_home_active and wrist_buttons_pressed:
                remote_home_active = False
                self.dependencies.emit("Recorder home canceled by wrist_3 jog input.")

            physical_home_active = bool(
                home_pressed
                and home_available
                and not cap_active
                and not wrist_buttons_pressed
            )
            home_active = remote_home_active or physical_home_active

            boundary = None
            joint_velocity: np.ndarray | None = None
            if wrist_buttons_pressed:
                joint_velocity = np.zeros(6, dtype=np.float64)
                if wrist_direction == 0:
                    boundary = "Button 1 and 2 conflict"
                elif cap_active:
                    boundary = "center the SpaceMouse cap for wrist_3 jog"
                else:
                    joint_velocity[5] = wrist_direction * config.wrist_3_jog_speed
                    if control is not None:
                        predicted_joints = read_joint_positions(receiver) + joint_velocity * max(
                            0.25, 2.0 * config.period_s
                        )
                        if not control.isJointsWithinSafetyLimits(predicted_joints.tolist()):
                            joint_velocity[:] = 0.0
                            boundary = "wrist_3 joint safety limit"
                    start_pose[3:] = current_pose[3:]
                    angle_rad = 0.0
                twist = np.zeros(6, dtype=np.float64)
            elif home_active and zero_joints is not None:
                current_joints = read_joint_positions(receiver)
                joint_velocity, reached = joint_home_velocity(
                    current_joints,
                    zero_joints,
                    config.home_rotation_speed,
                )
                if control is not None:
                    horizon_s = max(0.25, 2.0 * config.period_s)
                    predicted_joints = current_joints + joint_velocity * horizon_s
                    if not control.isJointsWithinSafetyLimits(predicted_joints.tolist()):
                        joint_velocity[:] = 0.0
                        boundary = "Button 4 joint safety limit"
                twist = np.zeros(6, dtype=np.float64)
                if reached:
                    remote_home_active = False
                    start_pose = current_pose.copy()
                    offset_m = 0.0
                    angle_rad = 0.0
                    if not home_reached_last:
                        self.dependencies.emit("UR5 joint home reached; zero reference restored.")
                home_reached_last = reached
            elif home_active:
                assert zero_pose is not None
                twist, reached = home_twist(
                    current_pose,
                    zero_pose,
                    config.home_translation_speed,
                    config.home_rotation_speed,
                )
                if reached:
                    remote_home_active = False
                    start_pose = zero_pose.copy()
                    offset_m = 0.0
                    angle_rad = 0.0
                    if not home_reached_last:
                        self.dependencies.emit("UR5 home reached; zero reference restored.")
                home_reached_last = reached
            else:
                home_reached_last = False
                self._enforce_hard_envelope(offset_m, angle_rad)
                twist = motion_command.twist

                horizon_s = max(0.25, 2.0 * config.period_s)
                twist, boundary = apply_relative_workspace_guard(
                    twist,
                    current_pose,
                    start_pose,
                    config.max_offset_m,
                    config.max_rotation_rad,
                    horizon_s,
                )
            moving = bool(np.any(np.abs(twist) > 0.0))

            if control is not None:
                if joint_velocity is not None:
                    if not control.speedJ(
                        joint_velocity.tolist(), config.acceleration, config.period_s
                    ):
                        raise RuntimeError("UR speedJ command returned false")
                elif moving:
                    acceleration = config.acceleration
                    command = twist
                    if not control.speedL(command.tolist(), acceleration, config.period_s):
                        raise RuntimeError("UR speedL command returned false")
                else:
                    acceleration = config.stop_deceleration
                    command = np.zeros(6, dtype=np.float64)
                    if not control.speedL(command.tolist(), acceleration, config.period_s):
                        raise RuntimeError("UR speedL command returned false")

            if effective_mode is not last_mode:
                self.dependencies.emit(f"mode={effective_mode.value}")
                last_mode = effective_mode
            if speed.profile is not last_profile:
                self.dependencies.emit(f"speed={speed.profile.value}")
                last_profile = speed.profile
            if loop_started >= next_report:
                self._report_status(
                    effective_mode,
                    speed.profile,
                    offset_m,
                    angle_rad,
                    current_speed,
                    motion,
                    boundary,
                )
                next_report = loop_started + 0.5

            elapsed = clock() - loop_started
            if elapsed < config.period_s:
                self.dependencies.sleep(config.period_s - elapsed)

    def _update_monitor(
        self,
        monitor: Any,
        motion: np.ndarray,
        buttons: dict[int, bool],
        mode: MotionMode | None = None,
    ) -> bool:
        speed = select_speed_limits(buttons, self.config.speeds)
        command = build_hardware_twist(motion, buttons, speed)
        return bool(
            monitor.update(
                motion,
                buttons,
                mode=(mode or command.mode).value,
                speed=speed.profile.value,
            )
        )

    def _enforce_hard_envelope(self, offset_m: float, angle_rad: float) -> None:
        if self.config.max_offset_mm > 0.0 and offset_m > self.config.max_offset_m + 0.003:
            raise RuntimeError(f"UR5 exceeded startup envelope: {offset_m * 1000.0:.2f} mm")
        if self.config.max_rotation_deg > 0.0 and angle_rad > math.radians(
            self.config.max_rotation_deg + 0.5
        ):
            raise RuntimeError(
                f"UR5 exceeded startup rotation envelope: {math.degrees(angle_rad):.2f} deg"
            )

    def _report_status(
        self,
        mode: MotionMode,
        profile: SpeedProfile,
        offset_m: float,
        angle_rad: float,
        current_speed: np.ndarray,
        motion: np.ndarray,
        boundary: str | None,
    ) -> None:
        linear_speed = np.linalg.norm(current_speed[:3]) * 1000.0
        angular_speed = math.degrees(np.linalg.norm(current_speed[3:]))
        boundary_text = f" blocked={boundary}" if boundary else ""
        input_text = [round(float(value), 2) for value in motion]
        self.dependencies.emit(
            f"status mode={mode.value} speed={profile.value} "
            f"offset={offset_m * 1000.0:.2f} mm "
            f"rotation={math.degrees(angle_rad):.2f} deg "
            f"actual_speed={linear_speed:.2f} mm/s,{angular_speed:.2f} deg/s "
            f"input={input_text}{boundary_text}"
        )


def _rounded_pose(pose: np.ndarray) -> list[float]:
    return [round(float(value), 5) for value in pose]


def run_real_ur5(config: UR5TeleopConfig) -> None:
    UR5TeleopRuntime(config).run()
