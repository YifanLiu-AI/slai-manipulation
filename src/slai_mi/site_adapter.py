"""Default adapter for the local UR5/WujiHand/three-RealSense station."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import partial
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from slai_mi.collection.home_control import (
    HomeCommandReceiver,
    default_task2_home_socket,
    exclusive_station_control,
)
from slai_mi.collection.vla_recorder import (
    EpisodeRecorder,
    SourceSample,
    SynchronizedInputs,
    assemble_configured_frame,
)
from slai_mi.datasets.lerobot_v3.configured import (
    ConfiguredDatasetContract,
    create_configured_dataset,
)
from slai_mi.datasets.lerobot_v3.schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES
from slai_mi.datasets.lerobot_v3.writer import create_dataset
from slai_mi.datasets.pi05_incremental import IncrementalV21Sidecar
from slai_mi.devices.cameras import CameraConfig, RealSenseCapture, validate_camera_set
from slai_mi.devices.gripper import (
    ContinuousSpaceMouseGripperControl,
    GripperDevice,
    SpaceMouseGripperControl,
    create_gripper,
    resolve_gripper_config,
)
from slai_mi.devices.spacemouse.buttons import RECORDED_BUTTONS, Button
from slai_mi.devices.spacemouse.client import SpaceMouseProcess
from slai_mi.devices.spacemouse.mapping import (
    SpeedSettings,
    build_hardware_twist,
    select_speed_limits,
    wrist_3_jog_direction,
)
from slai_mi.devices.ur5.geometry import (
    apply_relative_workspace_guard,
    joint_home_velocity,
)
from slai_mi.devices.ur5.process import UR5DriverProcess
from slai_mi.devices.wrist_sensor.openrb_v2 import OpenRBWristV2
from slai_mi.devices.wrist_sensor.teleop import WristMasterSlaveController
from slai_mi.devices.wujihand.filters import OneEuroFilter
from slai_mi.devices.wujihand.manual_control import (
    AccelerationLimitedTrajectory,
    ManualHandSettings,
    ManualWujiController,
)
from slai_mi.devices.wujihand.process import WujiHandDriverProcess
from slai_mi.input_schema import (
    CONTINUOUS_GRIPPER_CONTRACT,
    enabled_cameras,
    gripper_value_contract,
    load_input_schema,
    split_capture_vector,
)
from slai_mi.observability.flight_recorder import (
    ControlCycleRecord,
    FlightRecorder,
    NullFlightRecorder,
)
from slai_mi.runtime import CollectionDependencies, TeleopDependencies
from slai_mi.runtime.hardware_supervisor import HardwareProcessSupervisor

MAX_LINEAR_M_S = 0.25
MAX_ANGULAR_RAD_S = 0.60
# Normal motion stays below the worker's fail-closed hard boundary. The worker
# limits remain unchanged and still reject genuinely unsafe commands.
UR5_OPERATIONAL_LIMIT_RATIO = 0.90
CONTROL_PERIOD_S = 1.0 / 125.0
WUJI_PERIOD_S = 1.0 / 30.0
WUJI_MAX_VELOCITY_RAD_S = 3.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _operational_limit(hard_limit: float) -> float:
    return float(hard_limit) * UR5_OPERATIONAL_LIMIT_RATIO


def _bounded_speed_settings(
    motion: dict[str, Any], linear_hard_limit: float, angular_hard_limit: float
) -> tuple[SpeedSettings, float, float]:
    configured = SpeedSettings(
        translation=float(motion["translation_speed"]),
        rotation=float(motion["rotation_speed"]),
        boost_translation=float(motion["precision_translation_speed"]),
        boost_rotation=float(motion["precision_rotation_speed"]),
    )
    if (
        max(configured.translation, configured.boost_translation) > linear_hard_limit
        or max(configured.rotation, configured.boost_rotation) > angular_hard_limit
    ):
        raise ValueError("task SpaceMouse speed exceeds the hardware safety ceiling")
    linear = _operational_limit(linear_hard_limit)
    angular = _operational_limit(angular_hard_limit)
    return (
        SpeedSettings(
            translation=min(configured.translation, linear),
            rotation=min(configured.rotation, angular),
            boost_translation=min(configured.boost_translation, linear),
            boost_rotation=min(configured.boost_rotation, angular),
        ),
        linear,
        angular,
    )


def _clamp_command(values: Any, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Apply the final operational envelope, including floating-point overshoot."""
    command = np.asarray(values, dtype=np.float64).reshape(-1)
    if command.shape != (6,) or not np.isfinite(command).all():
        raise ValueError("UR5 command must contain six finite values")
    return np.clip(command, lower, upper)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"hardware.{name} must be a mapping")
    return value


def _load_task_ref(reference: str) -> dict[str, Any]:
    path = (PROJECT_ROOT / "configs" / "tasks" / reference).resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load task control reference {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"task reference must contain a mapping: {path}")
    return payload


def _configured_pose(
    task: dict[str, Any],
    reference: str,
    *,
    kind: str,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    payload = _load_task_ref(reference)
    label = str(payload.get("name") or reference)
    if payload.get("configured") is not True:
        raise ValueError(f"task pose {label} is not commissioned (configured must be true)")
    if payload.get("kind") != kind or payload.get("units") != "rad":
        raise ValueError(f"task pose {label} must be a {kind} pose in radians")
    if payload.get("joint_names") != list(joint_names):
        raise ValueError(
            f"task pose {label} must use the canonical {len(joint_names)} DoF joint order"
        )
    positions = np.asarray(payload.get("joint_positions"), dtype=float)
    if positions.shape != (len(joint_names),) or not np.isfinite(positions).all():
        raise ValueError(
            f"task pose {label} must contain {len(joint_names)} finite joint positions"
        )
    if kind == "task_start":
        schema_ref = str(payload.get("schema_ref") or "")
        expected_schema = str(task.get("state_schema") or "")
        if expected_schema in {"real_v1", "task1_v1"}:
            valid = schema_ref.endswith("#schemas.real_v1")
        elif expected_schema == "task2_9dof":
            valid = schema_ref.endswith("input_schemas/ur5e_wrist_gripper_9dof.yaml")
        elif expected_schema == "task2_9dof_continuous":
            valid = schema_ref.endswith(
                "input_schemas/ur5e_wrist_gripper_9dof_continuous.yaml"
            )
        else:
            valid = False
        if not valid:
            raise ValueError(f"task pose {label} does not match task schema {expected_schema!r}")
    return positions.copy()


def _control_profile(task: dict[str, Any]) -> dict[str, Any]:
    reference = str(task.get("control_profile_ref") or "")
    if not reference:
        raise ValueError("task.control_profile_ref is required for real control")
    return _load_task_ref(reference)


def _hand_preset(task: dict[str, Any], name: str) -> np.ndarray:
    presets = task.get("hand_presets")
    if not isinstance(presets, dict) or not str(presets.get(name) or ""):
        raise ValueError(f"task.hand_presets.{name} is required for manual Wuji control")
    return _configured_pose(
        task,
        str(presets[name]),
        kind="hand_preset",
        joint_names=WUJI_JOINT_NAMES,
    )


def _task_home_joints(task: dict[str, Any]) -> np.ndarray:
    reference = str(task.get("start_pose_ref") or "")
    if not reference:
        raise ValueError("task.start_pose_ref is required for T-key home")
    joint_names = (
        (*UR5_JOINT_NAMES, "wrist_pitch_joint", "wrist_yaw_joint", "gripper_joint")
        if str(task.get("state_schema")) in {"task2_9dof", "task2_9dof_continuous"}
        else (*UR5_JOINT_NAMES, *WUJI_JOINT_NAMES)
    )
    return _configured_pose(task, reference, kind="task_start", joint_names=joint_names)


class StationSession:
    """Own both isolated drivers so one failure always disables the pair."""

    def __init__(self, hardware: dict[str, Any], task: dict[str, Any]) -> None:
        ur5, wuji = _section(hardware, "ur5"), _section(hardware, "wujihand")
        profile = _control_profile(task)
        motion = _section(profile, "motion")
        hand = _section(profile, "hand")
        linear = min(float(ur5.get("max_linear_m_s", MAX_LINEAR_M_S)), MAX_LINEAR_M_S)
        angular = min(float(ur5.get("max_angular_rad_s", MAX_ANGULAR_RAD_S)), MAX_ANGULAR_RAD_S)
        (
            self.speed_settings,
            self.linear_command_limit,
            self.angular_command_limit,
        ) = _bounded_speed_settings(motion, linear, angular)
        self.control_period_s = 1.0 / float(motion["control_hz"])
        self.ur5_acceleration = float(motion["acceleration"])
        self.wrist_3_jog_speed = float(motion["wrist_3_jog_speed"])
        self.home_joint_speed = float(motion["home_joint_speed"])
        self.max_offset_m = float(motion["max_offset_mm"]) / 1000.0
        self.max_rotation_rad = math.radians(float(motion["max_rotation_deg"]))
        self.spacemouse_deadzone = float(motion["deadzone"])
        self.hand_settings = ManualHandSettings(
            command_hz=float(hand["command_hz"]),
            grasp_speed=float(hand["grasp_speed"]),
            grasp_acceleration=float(hand["grasp_acceleration"]),
            release_speed=float(hand["release_speed"]),
            release_acceleration=float(hand["release_acceleration"]),
        )
        self.retarget_speed = float(hand["retarget_speed"])
        self.retarget_acceleration = float(hand["retarget_acceleration"])
        self.lost_hand_return_delay = float(hand["lost_hand_return_delay"])
        self.retarget_filter_settings = (
            float(hand["one_euro_min_cutoff"]),
            float(hand["one_euro_beta"]),
        )
        task_home_joints = _task_home_joints(task)
        self.ur5_home_joints = task_home_joints[:6].copy()
        self.wuji_home_joints = task_home_joints[6:].copy()
        self.wrist_3_jog_speed = min(
            self.wrist_3_jog_speed, self.angular_command_limit
        )
        self.home_joint_speed = min(self.home_joint_speed, self.angular_command_limit)
        self.open_hand_target = _hand_preset(task, "open")
        self.grasp_hand_target = _hand_preset(task, "grasp")
        presets = task.get("hand_presets")
        has_auxiliary = isinstance(presets, dict) and all(
            str(presets.get(name) or "") for name in ("auxiliary_open", "auxiliary_grasp")
        )
        self.auxiliary_open_hand_target = (
            _hand_preset(task, "auxiliary_open") if has_auxiliary else self.open_hand_target.copy()
        )
        self.auxiliary_grasp_hand_target = (
            _hand_preset(task, "auxiliary_grasp")
            if has_auxiliary
            else self.grasp_hand_target.copy()
        )
        self.ur5 = UR5DriverProcess(
            python=Path(ur5["driver_python"]),
            host=str(ur5["host"]),
            watchdog_s=float(ur5.get("driver_watchdog_s", 0.25)),
            max_linear_m_s=linear,
            max_angular_rad_s=angular,
            min_tcp_z_m=float(ur5["min_tcp_z_m"]),
        )
        self.wuji = WujiHandDriverProcess(
            python=Path(wuji["driver_python"]),
            usb_serial=str(wuji["usb_serial"]),
            product_serial=str(wuji.get("product_serial", "")),
            watchdog_s=float(wuji.get("driver_watchdog_s", 0.5)),
            max_temperature_c=min(float(wuji.get("max_temperature_c", 80.0)), 80.0),
            thermal_warning_temperature_c=float(wuji.get("thermal_warning_temperature_c", 70.0)),
            thermal_critical_temperature_c=float(wuji.get("thermal_critical_temperature_c", 75.0)),
            limit_margin_rad=max(float(wuji.get("limit_margin_rad", 0.03)), 0.03),
            max_effort_fraction=min(float(wuji.get("max_effort_fraction", 0.65)), 0.65),
            max_velocity_rad_s=min(
                float(wuji.get("max_velocity_rad_s", WUJI_MAX_VELOCITY_RAD_S)),
                WUJI_MAX_VELOCITY_RAD_S,
            ),
        )
        wrist = _section(hardware, "wrist_sensor")
        self.wrist = (
            OpenRBWristV2(
                wrist.get(
                    "config",
                    "third_party/02_Python_Client_CLI/closed_loop_record/wrist_output_v2.yaml",
                ),
                port=str(wrist.get("openrb_port", "auto")),
                baud=int(wrist.get("baud", 115200)),
                mode=str(wrist.get("mode", "closed_loop")),
            )
            if bool(wrist.get("enabled", False))
            else None
        )
        self.supervisor = HardwareProcessSupervisor({"ur5": self.ur5, "wujihand": self.wuji})
        self.linear_limit = linear
        self.angular_limit = angular
        self._lock = threading.Lock()
        self._started = False
        self._leases = 0
        self._station_lock = None

    @contextmanager
    def lease(self, *, arm: bool = True):
        with self._lock:
            if not self._started:
                station_lock = exclusive_station_control()
                station_lock.__enter__()
                try:
                    self.supervisor.start()
                    if arm:
                        self.supervisor.arm()
                    if self.wrist is not None:
                        self.wrist.start(home=True)
                except BaseException:
                    if self.wrist is not None:
                        with suppress(Exception):
                            self.wrist.close()
                    self.supervisor.stop()
                    station_lock.__exit__(None, None, None)
                    raise
                self._station_lock = station_lock
                self._started = True
            elif arm and not self.supervisor.armed:
                self.supervisor.arm()
            self._leases += 1
        try:
            yield self
        finally:
            with self._lock:
                self._leases -= 1
                if self._leases == 0 and self._started:
                    try:
                        try:
                            if self.wrist is not None:
                                self.wrist.close()
                        finally:
                            self.supervisor.stop()
                    finally:
                        self._started = False
                        if self._station_lock is not None:
                            self._station_lock.__exit__(None, None, None)
                            self._station_lock = None

    def check(self) -> None:
        self.supervisor.check()

    def arm(self) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("cannot arm a session before it is started")
            if not self.supervisor.armed:
                self.supervisor.arm()

    def write_ur5_twist(self, twist: Any, *, acceleration: float, duration_s: float) -> None:
        limits = np.asarray(
            [self.linear_command_limit] * 3 + [self.angular_command_limit] * 3
        )
        command = _clamp_command(twist, -limits, limits)
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_twist(
                command, acceleration=acceleration, duration_s=duration_s
            ),
            check_after=False,
        )

    def read_ur5_state(self) -> dict[str, Any]:
        try:
            return self.ur5.read_state()
        except BaseException as exc:
            self.supervisor.fail_closed(exc)
            raise

    def read_wuji_positions(self) -> tuple[float, ...]:
        try:
            return self.wuji.read_positions()
        except BaseException as exc:
            self.supervisor.fail_closed(exc)
            raise

    def read_wuji_temperature(self) -> dict[str, object]:
        try:
            return self.wuji.read_temperature()
        except BaseException as exc:
            self.supervisor.fail_closed(exc)
            raise

    def prepare_ur5_control(self) -> None:
        """Finish the potentially slow RTDE setup before arming motion."""
        self.supervisor.call_with_peer_heartbeats("ur5", self.ur5.prepare_control)

    def write_ur5_joint_velocity(
        self, velocity: Any, *, acceleration: float, duration_s: float
    ) -> None:
        limits = np.full(6, self.angular_command_limit)
        command = _clamp_command(velocity, -limits, limits)
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_joint_velocity(
                command, acceleration=acceleration, duration_s=duration_s
            ),
            check_after=False,
        )

    def stop_ur5_motion(self) -> None:
        self.supervisor.call_with_peer_heartbeats("ur5", self.ur5.stop_motion)

    def write_wuji_positions(self, positions: Any) -> None:
        self.supervisor.call_with_peer_heartbeats(
            "wujihand", lambda: self.wuji.write_positions(positions)
        )


class UR5OnlySession:
    """Own the UR5 driver for groups that intentionally exclude WujiHand."""

    def __init__(self, hardware: dict[str, Any], task: dict[str, Any]) -> None:
        ur5 = _section(hardware, "ur5")
        profile = _control_profile(task)
        motion = _section(profile, "motion")
        linear = min(float(ur5.get("max_linear_m_s", MAX_LINEAR_M_S)), MAX_LINEAR_M_S)
        angular = min(float(ur5.get("max_angular_rad_s", MAX_ANGULAR_RAD_S)), MAX_ANGULAR_RAD_S)
        (
            self.speed_settings,
            self.linear_command_limit,
            self.angular_command_limit,
        ) = _bounded_speed_settings(motion, linear, angular)
        self.control_period_s = 1.0 / float(motion["control_hz"])
        self.ur5_acceleration = float(motion["acceleration"])
        self.wrist_3_jog_speed = float(motion["wrist_3_jog_speed"])
        self.home_joint_speed = float(motion["home_joint_speed"])
        self.max_offset_m = float(motion["max_offset_mm"]) / 1000.0
        self.max_rotation_rad = math.radians(float(motion["max_rotation_deg"]))
        self.spacemouse_deadzone = float(motion["deadzone"])
        self.ur5_home_joints = _task_home_joints(task)[:6].copy()
        # Clamp auxiliary joint jogging to the same UR5 angular ceiling as
        # the worker; exceeding it must not fail-closed the collection.
        self.wrist_3_jog_speed = min(
            self.wrist_3_jog_speed, self.angular_command_limit
        )
        self.home_joint_speed = min(self.home_joint_speed, self.angular_command_limit)
        self.ur5 = UR5DriverProcess(
            python=Path(ur5["driver_python"]),
            host=str(ur5["host"]),
            watchdog_s=float(ur5.get("driver_watchdog_s", 0.25)),
            max_linear_m_s=linear,
            max_angular_rad_s=angular,
            min_tcp_z_m=float(ur5["min_tcp_z_m"]),
        )
        self.supervisor = HardwareProcessSupervisor({"ur5": self.ur5})

    @contextmanager
    def lease(self, *, arm: bool = True):
        with exclusive_station_control():
            self.supervisor.start()
            try:
                if arm:
                    self.supervisor.arm()
                yield self
            finally:
                self.supervisor.stop()

    def read_ur5_state(self) -> dict[str, Any]:
        try:
            return self.ur5.read_state()
        except BaseException as exc:
            self.supervisor.fail_closed(exc)
            raise

    def write_ur5_twist(self, twist: Any, *, acceleration: float, duration_s: float) -> None:
        limits = np.asarray(
            [self.linear_command_limit] * 3 + [self.angular_command_limit] * 3
        )
        command = _clamp_command(twist, -limits, limits)
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_twist(
                command, acceleration=acceleration, duration_s=duration_s
            ),
            check_after=False,
        )

    def prepare_ur5_control(self) -> None:
        """Finish UR5 RTDE setup before arming a real command loop."""
        self.supervisor.call_with_peer_heartbeats("ur5", self.ur5.prepare_control)

    def arm(self) -> None:
        self.supervisor.arm()

    def check(self) -> None:
        self.supervisor.check()

    def write_ur5_joint_velocity(
        self, velocity: Any, *, acceleration: float, duration_s: float
    ) -> None:
        limits = np.full(6, self.angular_command_limit)
        command = _clamp_command(velocity, -limits, limits)
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_joint_velocity(
                command, acceleration=acceleration, duration_s=duration_s
            ),
            check_after=False,
        )

    def stop_ur5_motion(self) -> None:
        self.supervisor.call_with_peer_heartbeats("ur5", self.ur5.stop_motion)


class Wrist2WristTeleopLoop:
    """Run the verified ESP32-to-OpenRB program under the shared stop condition."""

    def __init__(self, hardware: dict[str, Any]) -> None:
        wrist = _section(hardware, "wrist_sensor")
        script = (
            PROJECT_ROOT
            / "third_party/02_Python_Client_CLI/closed_loop_record/record_wrist_output_v2_teleop.py"
        )
        self.command = [
            sys.executable,
            str(script),
            "--teleop-port",
            str(wrist.get("teleop_port", "auto")),
            "--openrb-port",
            str(wrist.get("openrb_port", "auto")),
            "--baud",
            str(int(wrist.get("baud", 115200))),
            "--config",
            str(PROJECT_ROOT / str(wrist.get("config"))),
            "--data-root",
            str(PROJECT_ROOT / "data/wrist-teleop"),
        ]
        self.process: subprocess.Popen[str] | None = None

    def run(self, stop_event: threading.Event) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.process = subprocess.Popen(self.command, cwd=PROJECT_ROOT, env=environment, text=True)
        try:
            while self.process.poll() is None and not stop_event.wait(0.1):
                pass
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
            try:
                return_code = self.process.wait(timeout=35.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                return_code = self.process.wait(timeout=5.0)
            if return_code != 0 and not stop_event.is_set():
                raise RuntimeError(f"Wrist2Wrist teleop exited with status {return_code}")
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5.0)


class NativeWristTeleopLoop:
    """Run the same automatic-zero wrist adapter used by formal collection."""

    def __init__(self, hardware: dict[str, Any], mouse: Any) -> None:
        wrist = _section(hardware, "wrist_sensor")
        self.mouse = mouse
        self.controller = WristMasterSlaveController(
            PROJECT_ROOT / str(wrist["config"]),
            teleop_port=str(wrist.get("teleop_port", "auto")),
            openrb_port=str(wrist.get("openrb_port", "auto")),
            baud=int(wrist.get("baud", 115200)),
        )

    def run(self, stop_event: threading.Event) -> None:
        previous: dict[int, bool] = {}
        with self.controller:
            while not stop_event.wait(0.01):
                _motion, buttons = self.mouse.state()
                action, previous = _wrist_gate_action(buttons, previous)
                if action == "home":
                    self.controller.request_soft_home()
                elif action == "park":
                    self.controller.request_park()
                elif action == "resume":
                    self.controller.request_resume()
                self.controller.check()


SPACE_MOUSE_WRIST_MAX_SPEED_DEG_S = 45.0
SPACE_MOUSE_WRIST_DEADZONE = 0.18
SPACE_MOUSE_WRIST_RESPONSE_EXPONENT = 1.00
SPACE_MOUSE_WRIST_FE_SIGN = -1.0
SPACE_MOUSE_WRIST_RU_SIGN = 1.0


def spacemouse_wrist_velocity(motion: np.ndarray) -> tuple[float, float]:
    """Convert normalized SpaceMouse motion to a bounded FE/RU jog velocity."""
    values = np.asarray(motion, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("SpaceMouse motion must contain six finite values")
    magnitude = np.abs(values[:2])
    # Remap continuously outside the deadzone so center noise cannot accumulate.
    scaled = np.clip(
        (magnitude - SPACE_MOUSE_WRIST_DEADZONE)
        / (1.0 - SPACE_MOUSE_WRIST_DEADZONE),
        0.0,
        1.0,
    )
    response = np.sign(values[:2]) * scaled**SPACE_MOUSE_WRIST_RESPONSE_EXPONENT
    velocity = response * SPACE_MOUSE_WRIST_MAX_SPEED_DEG_S
    velocity[0] *= SPACE_MOUSE_WRIST_FE_SIGN
    velocity[1] *= SPACE_MOUSE_WRIST_RU_SIGN
    return float(velocity[0]), float(velocity[1])


def _wrist_gate_action(
    buttons: dict[int, bool], previous: dict[int, bool]
) -> tuple[str | None, dict[int, bool]]:
    tracked = (int(Button.T), int(Button.R), int(Button.FIT), int(Button.MENU))
    current = {button: bool(buttons.get(button, False)) for button in tracked}
    rising = {button: current[button] and not previous.get(button, False) for button in tracked}
    if current[int(Button.FIT)] and current[int(Button.MENU)]:
        return None, current
    if rising[int(Button.R)]:
        return "home", current
    if rising[int(Button.T)] or rising[int(Button.FIT)]:
        return "park", current
    if rising[int(Button.MENU)]:
        return "resume", current
    return None, current


class RealPolicyBridge:
    """Route one schema-declared policy action through the existing safety supervisor."""

    def __init__(self, session: StationSession, input_schema: str | Path | None) -> None:
        self.session = session
        self.schema = load_input_schema(input_schema)
        self.start_pose: np.ndarray | None = None

    def apply(self, action: Any) -> None:
        components = split_capture_vector(self.schema, "action", action)
        try:
            twist = np.asarray(components["ur5"]["target_tcp_speed"], dtype=float).copy()
            hand = components["wuji"]["command_q"]
        except KeyError as exc:
            raise ValueError(
                f"real policy schema lacks required hardware component: {exc}"
            ) from exc
        for values, limit in (
            (twist[:3], self.session.speed_settings.translation),
            (twist[3:], self.session.speed_settings.rotation),
        ):
            magnitude = float(np.linalg.norm(values))
            if magnitude > limit:
                values *= limit / magnitude
        state = self.session.read_ur5_state()
        current_pose = np.asarray(state["tcp_pose"], dtype=float)
        if self.start_pose is None:
            self.start_pose = current_pose.copy()
        twist, _blocked = apply_relative_workspace_guard(
            twist,
            current_pose,
            self.start_pose,
            self.session.max_offset_m,
            self.session.max_rotation_rad,
            0.25,
        )
        self.session.write_ur5_twist(
            twist,
            acceleration=getattr(self.session, "ur5_acceleration", 0.5),
            duration_s=getattr(self.session, "control_period_s", CONTROL_PERIOD_S),
        )
        self.session.write_wuji_positions(hand)
        self.session.check()


def _apply_legacy_ur5_command(
    session: StationSession,
    motion: np.ndarray,
    buttons: dict[int, bool],
    start_pose: np.ndarray | None,
    *,
    home_requested: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wrist_pressed = bool(buttons.get(int(Button.ONE), False) or buttons.get(int(Button.TWO), False))
    home_pressed = bool(buttons.get(int(Button.T), False) or home_requested)
    cap_active = bool(np.any(np.abs(motion) > 0.0))
    # Reading UR feedback is an IPC + RTDE round trip.  It is required for
    # workspace/home safety, but not for ordinary unconstrained SpaceMouse
    # velocity streaming.  Reuse the last sample on the fast path.
    needs_feedback = (
        start_pose is None
        or wrist_pressed
        or (home_pressed and not cap_active)
        or float(getattr(session, "max_offset_m", 0.0)) > 0.0
        or float(getattr(session, "max_rotation_rad", 0.0)) > 0.0
    )
    if needs_feedback:
        state = session.read_ur5_state()
        session._last_ur5_state = state
    else:
        state = getattr(
            session,
            "_last_ur5_state",
            {"tcp_pose": np.zeros(6), "joints": np.zeros(6)},
        )
    current_pose = np.asarray(state["tcp_pose"], dtype=float)
    current_joints = np.asarray(state["joints"], dtype=float)
    if start_pose is None:
        start_pose = current_pose.copy()
    twist = np.zeros(6, dtype=np.float64)
    target_qd = np.zeros(6, dtype=np.float64)
    if wrist_pressed:
        direction = wrist_3_jog_direction(buttons)
        if direction != 0 and not cap_active:
            target_qd[5] = direction * session.wrist_3_jog_speed
        session.write_ur5_joint_velocity(
            target_qd,
            acceleration=session.ur5_acceleration,
            duration_s=session.control_period_s,
        )
        start_pose[3:] = current_pose[3:]
    elif home_pressed and not cap_active:
        target_qd, _reached = joint_home_velocity(
            current_joints,
            session.ur5_home_joints,
            session.home_joint_speed,
        )
        session.write_ur5_joint_velocity(
            target_qd,
            acceleration=session.ur5_acceleration,
            duration_s=session.control_period_s,
        )
    else:
        limits = select_speed_limits(buttons, session.speed_settings)
        command = build_hardware_twist(motion, buttons, limits)
        twist = command.twist
        twist, _boundary = apply_relative_workspace_guard(
            twist,
            current_pose,
            start_pose,
            session.max_offset_m,
            session.max_rotation_rad,
            max(0.25, 2.0 * session.control_period_s),
        )
        session.write_ur5_twist(
            twist,
            acceleration=session.ur5_acceleration,
            duration_s=session.control_period_s,
        )
    return start_pose, np.asarray(twist), np.asarray(target_qd), current_joints


class UR5TeleopLoop:
    def __init__(self, session: StationSession, mouse: Any) -> None:
        self.session, self.mouse = session, mouse

    def run(self, stop_event: threading.Event) -> None:
        start_pose: np.ndarray | None = None
        with self.session.lease():
            try:
                while not stop_event.is_set():
                    started = time.monotonic()
                    motion, buttons = self.mouse.state()
                    start_pose, _twist, _target_qd, _joints = _apply_legacy_ur5_command(
                        self.session, motion, buttons, start_pose
                    )
                    elapsed = time.monotonic() - started
                    stop_event.wait(max(0.0, self.session.control_period_s - elapsed))
            finally:
                self.session.stop_ur5_motion()


class WujiSupervisionLoop:
    def __init__(self, session: StationSession, hardware: dict[str, Any], mouse: Any) -> None:
        self.session, self.hardware, self.mouse = session, hardware, mouse
        self.ready = threading.Event()
        self.camera_serial: str | None = None

    def run(self, stop_event: threading.Event) -> None:
        provider = WujiRetargetTargetProvider(self.hardware, external_frames=False)
        with self.session.lease():
            controller = WujiTargetController(self.session, provider)
            self.camera_serial = provider.camera_serial
            self.ready.set()
            try:
                while not stop_event.wait(WUJI_PERIOD_S):
                    _motion, buttons = self.mouse.state()
                    controller.update(buttons)
            finally:
                provider.close()


class WujiRetargetTargetProvider:
    """Pinned MediaPipe retarget process with fail-closed JSON RPC."""

    def __init__(self, hardware: dict[str, Any], *, external_frames: bool) -> None:
        wuji = _section(hardware, "wujihand")
        python = Path(str(wuji.get("retarget_python") or ""))
        if not python.is_file():
            raise FileNotFoundError(f"Wuji retarget Python is missing: {python}")
        self._memory = None
        request: dict[str, Any] = {
            "op": "init",
            "hardware": hardware,
            "external_frames": external_frames,
        }
        if external_frames:
            from slai_mi.input_schema import load_input_schema

            shape = tuple(load_input_schema(hardware.get("input_schema"))["capture"]["image_shape"])
            self._memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
            self._shared_frame = np.ndarray(shape, dtype=np.uint8, buffer=self._memory.buf)
            request.update({"shared_memory": self._memory.name, "shared_shape": shape})
        environment = os.environ.copy()
        source = str(PROJECT_ROOT / "src")
        environment["PYTHONPATH"] = source + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        self._process = subprocess.Popen(
            [str(python), "-m", "slai_mi.devices.wujihand.retarget_worker"],
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            result = self._rpc(request)
            self.joint_limits = np.asarray(result["joint_limits"], dtype=float)
            self.camera_serial = str(result["camera_serial"])
        except BaseException:
            self.close()
            raise

    def _rpc(self, request: dict[str, Any]) -> Any:
        if (
            self._process.poll() is not None
            or self._process.stdin is None
            or self._process.stdout is None
        ):
            raise RuntimeError("Wuji retarget worker is not running")
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        while line := self._process.stdout.readline():
            if not line.startswith("SLAI_RETARGET_RPC "):
                continue
            response = json.loads(line.removeprefix("SLAI_RETARGET_RPC "))
            if not response.get("ok"):
                raise RuntimeError(f"Wuji retarget worker failed: {response.get('error')}")
            return response.get("result")
        raise RuntimeError("Wuji retarget worker closed its RPC stream")

    def process_frame(self, frame: np.ndarray) -> None:
        if self._memory is None:
            raise RuntimeError("retarget provider does not accept external frames")
        value = np.asarray(frame, dtype=np.uint8)
        if value.shape != self._shared_frame.shape:
            raise ValueError(
                f"retarget frame shape {value.shape} differs from schema {self._shared_frame.shape}"
            )
        self._shared_frame[:] = value
        self._rpc({"op": "process_frame"})

    def target(self, now: float) -> np.ndarray | None:
        result = self._rpc({"op": "target", "now": now})
        return None if result is None else np.asarray(result, dtype=float)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                self._rpc({"op": "close"})
            except (OSError, RuntimeError, TypeError, ValueError):
                process.terminate()
            process.wait(timeout=5.0)
        memory = getattr(self, "_memory", None)
        if memory is not None:
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass


class WujiTargetController:
    def __init__(self, session: StationSession, provider: WujiRetargetTargetProvider) -> None:
        self.session, self.provider = session, provider
        limits = np.asarray(provider.joint_limits, dtype=float)
        now = time.monotonic()
        self.trajectory = AccelerationLimitedTrajectory(
            session.wuji.read_positions(), limits[:, 0], limits[:, 1], now
        )
        min_cutoff, beta = session.retarget_filter_settings
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.last_target_at: float | None = None
        self._previous_target: np.ndarray | None = None
        self._stable_frames = 0
        self._armed = False
        self.manual = ManualWujiController(
            session,
            open_target=session.open_hand_target,
            grasp_target=session.grasp_hand_target,
            home_target=session.wuji_home_joints,
            lower=limits[:, 0],
            upper=limits[:, 1],
            settings=session.hand_settings,
            timestamp=now,
            auxiliary_open_target=session.auxiliary_open_hand_target,
            auxiliary_grasp_target=session.auxiliary_grasp_hand_target,
        )
        self.manual.trajectory = self.trajectory

    def _stable_target(self, target: np.ndarray) -> np.ndarray | None:
        if self._armed:
            return target
        if (
            self._previous_target is not None
            and float(np.max(np.abs(target - self._previous_target))) <= 0.35
        ):
            self._stable_frames += 1
        else:
            self._stable_frames = 1
        self._previous_target = target.copy()
        if self._stable_frames < 6:
            return None
        self._armed = True
        return target

    def update(self, buttons: dict[int, bool] | None = None) -> None:
        now = time.monotonic()
        if buttons is not None and self.manual.update(buttons, now, hold_when_inactive=False):
            self.session.check()
            return
        target = self.provider.target(now)
        if target is not None:
            stable = self._stable_target(np.asarray(target, dtype=float))
            if stable is not None:
                filtered = self.filter.filter(stable, now)
                command = self.trajectory.step(
                    filtered,
                    now,
                    max_speed=self.session.retarget_speed,
                    max_acceleration=self.session.retarget_acceleration,
                )
                self.session.write_wuji_positions(command)
                self.last_target_at = now
        elif (
            self.last_target_at is not None
            and now - self.last_target_at >= self.session.lost_hand_return_delay
        ):
            command = self.trajectory.step(
                self.session.open_hand_target,
                now,
                max_speed=self.session.hand_settings.release_speed,
                max_acceleration=self.session.hand_settings.release_acceleration,
            )
            self.session.write_wuji_positions(command)
        else:
            self.session.write_wuji_positions(self.trajectory.hold(now))
        self.session.check()


class ControlledSpaceMouse:
    """SpaceMouse input that also drives the supervised UR5 during collection."""

    def __init__(self, mouse: SpaceMouseProcess, session: StationSession) -> None:
        self.mouse, self.session = mouse, session
        self._lease = None
        self._hand_controller = None
        self.latest = (np.zeros(6, dtype=np.float32), {})
        self._latest_lock = threading.Lock()
        self._control_stop = threading.Event()
        self._control_threads: list[threading.Thread] = []
        self._control_failure: BaseException | None = None
        self._first_ur_command = threading.Event()
        self.latest_twist = np.zeros(6, dtype=np.float64)
        self.latest_target_qd = np.zeros(6, dtype=np.float64)
        self.latest_ur5_joints: np.ndarray | None = None
        self.latest_wuji_positions: np.ndarray | None = None
        self._home_requested = threading.Event()
        self._home_stable_since: float | None = None
        self._home_button_last = False

    def __enter__(self):
        self.mouse.start()
        # Collection opens cameras and the dataset before the first motion call.
        # Keep workers unarmed through that potentially long setup interval.
        self._lease = self.session.lease(arm=False)
        self._lease.__enter__()
        lower, upper = self.session.wuji.read_limits()
        self._hand_controller = ManualWujiController(
            self.session,
            open_target=self.session.open_hand_target,
            grasp_target=self.session.grasp_hand_target,
            home_target=self.session.wuji_home_joints,
            lower=lower,
            upper=upper,
            settings=self.session.hand_settings,
            timestamp=time.monotonic(),
            auxiliary_open_target=self.session.auxiliary_open_hand_target,
            auxiliary_grasp_target=self.session.auxiliary_grasp_hand_target,
        )
        return self

    def state(self):
        if not self.session.supervisor.armed:
            self.session.prepare_ur5_control()
            self.session.arm()
        if not self._control_threads:
            self._start_control_threads()
            if not self._first_ur_command.wait(timeout=1.0):
                self._raise_control_failure("UR5 control loop did not produce its first command")
        self._raise_control_failure()
        with self._latest_lock:
            motion, buttons = self.latest
            return motion.copy(), buttons.copy()

    def _start_control_threads(self) -> None:
        self._control_stop.clear()

        def guarded(target) -> None:
            try:
                target()
            except BaseException as exc:  # noqa: BLE001 - transfer control failure
                self._control_failure = exc
                self._control_stop.set()

        self._control_threads = [
            threading.Thread(
                target=guarded,
                args=(self._run_ur5_control,),
                name="collection-ur5-control",
            ),
            threading.Thread(
                target=guarded,
                args=(self._run_wuji_control,),
                name="collection-wuji-control",
            ),
        ]
        for thread in self._control_threads:
            thread.start()

    def _run_ur5_control(self) -> None:
        start_pose: np.ndarray | None = None
        while not self._control_stop.is_set():
            started = time.monotonic()
            motion, buttons = self.mouse.state()
            home_button = bool(buttons.get(int(Button.T), False))
            wrist = getattr(self.session, "wrist", None)
            if home_button and not self._home_button_last and wrist is not None:
                wrist.request_home()
            self._home_button_last = home_button
            with self._latest_lock:
                self.latest = (motion.copy(), buttons.copy())
            start_pose, twist, target_qd, current_joints = _apply_legacy_ur5_command(
                self.session,
                motion,
                buttons,
                start_pose,
                home_requested=self._home_requested.is_set(),
            )
            with self._latest_lock:
                self.latest_twist = np.asarray(twist, dtype=np.float64)
                self.latest_target_qd = np.asarray(target_qd, dtype=np.float64)
                self.latest_ur5_joints = np.asarray(current_joints, dtype=np.float64)
            self._first_ur_command.set()
            elapsed = time.monotonic() - started
            self._control_stop.wait(max(0.0, self.session.control_period_s - elapsed))

    def _run_wuji_control(self) -> None:
        assert self._hand_controller is not None
        period = 1.0 / self.session.hand_settings.command_hz
        while not self._control_stop.is_set():
            started = time.monotonic()
            with self._latest_lock:
                buttons = self.latest[1].copy()
            if self._home_requested.is_set():
                buttons[int(Button.T)] = True
            self._hand_controller.update(buttons, started)
            actual = np.asarray(self.session.read_wuji_positions(), dtype=np.float64)
            with self._latest_lock:
                self.latest_wuji_positions = actual
            elapsed = time.monotonic() - started
            self._control_stop.wait(max(0.0, period - elapsed))

    def _raise_control_failure(self, fallback: str | None = None) -> None:
        if self._control_failure is not None:
            raise RuntimeError(f"collection control loop failed: {self._control_failure}") from (
                self._control_failure
            )
        if fallback is not None:
            raise RuntimeError(fallback)

    @property
    def latest_hand_command(self) -> np.ndarray:
        assert self._hand_controller is not None
        return self._hand_controller.trajectory.command.copy()

    def request_home(self) -> None:
        self._home_stable_since = None
        self._home_requested.set()
        if self.session.wrist is not None:
            self.session.wrist.request_home()

    def clear_home(self) -> None:
        self._home_requested.clear()

    def home_status(self) -> dict[str, object]:
        with self._latest_lock:
            ur5 = None if self.latest_ur5_joints is None else self.latest_ur5_joints.copy()
            wuji = None if self.latest_wuji_positions is None else self.latest_wuji_positions.copy()
        if ur5 is None or wuji is None or self._hand_controller is None:
            self._home_stable_since = None
            return {"at_home": False, "detail": "等待零位遥测"}
        ur5_error = float(np.max(np.abs(ur5 - self.session.ur5_home_joints)))
        wuji_error = float(np.max(np.abs(wuji - self.session.wuji_home_joints)))
        command_error = float(
            np.max(np.abs(self._hand_controller.trajectory.command - self.session.wuji_home_joints))
        )
        wrist_status = (
            self.session.wrist.home_status()
            if self.session.wrist is not None
            else {"at_home": True, "detail": "未启用双轴手腕"}
        )
        within = bool(
            ur5_error <= 0.010
            and wuji_error <= 0.100
            and command_error <= 0.010
            and bool(wrist_status.get("at_home"))
        )
        now = time.monotonic()
        if not within:
            self._home_stable_since = None
        elif self._home_stable_since is None:
            self._home_stable_since = now
        at_home = bool(
            within and self._home_stable_since is not None and now - self._home_stable_since >= 0.30
        )
        return {
            "at_home": at_home,
            "detail": (
                f"UR5误差 {ur5_error:.3f}rad，Wuji误差 {wuji_error:.3f}rad；"
                f"{wrist_status.get('detail', '')}"
            ),
            "ur5_error_rad": ur5_error,
            "wuji_error_rad": wuji_error,
            "wuji_command_error_rad": command_error,
            "wrist": wrist_status,
        }

    def __exit__(self, *_args: object) -> None:
        self._control_stop.set()
        for thread in self._control_threads:
            thread.join(timeout=2.0)
        alive = [thread.name for thread in self._control_threads if thread.is_alive()]
        try:
            self.session.stop_ur5_motion()
        finally:
            if self._lease is not None:
                self._lease.__exit__(*_args)
            self.mouse.stop()
        context_failed = bool(_args and _args[0] is not None)
        if alive and not context_failed:
            raise RuntimeError(f"collection control threads did not stop: {', '.join(alive)}")
        if self._control_failure is not None and not context_failed:
            self._raise_control_failure()


class ControlledUR5SpaceMouse:
    """Drive UR5 while supervising the independently sampled two-axis wrist."""

    def __init__(
        self,
        mouse: SpaceMouseProcess,
        session: UR5OnlySession,
        wrist: WristMasterSlaveController,
        gripper: GripperDevice | None = None,
        *,
        continuous_gripper: bool = False,
        gripper_rate_per_s: float = 1.0,
        gripper_command_hz: float = 30.0,
        gripper_home_tolerance: float = 0.02,
        home_socket: Path | str | None = None,
        flight_recorder: FlightRecorder | NullFlightRecorder | None = None,
    ) -> None:
        self.mouse, self.session, self.wrist = mouse, session, wrist
        self.gripper = gripper
        if gripper is None:
            self.gripper_control = None
        elif continuous_gripper:
            self.gripper_control = ContinuousSpaceMouseGripperControl(
                gripper,
                rate_per_s=gripper_rate_per_s,
                command_hz=gripper_command_hz,
            )
        else:
            self.gripper_control = SpaceMouseGripperControl(gripper)
        if not 0.0 <= gripper_home_tolerance <= 1.0:
            raise ValueError("gripper_home_tolerance must be in [0, 1]")
        self.continuous_gripper = bool(continuous_gripper)
        self.gripper_home_tolerance = float(gripper_home_tolerance)
        self.flight_recorder = (
            flight_recorder if flight_recorder is not None else NullFlightRecorder()
        )
        self.latest = (np.zeros(6, dtype=np.float32), {})
        self.latest_twist = np.zeros(6, dtype=np.float64)
        self.latest_target_qd = np.zeros(6, dtype=np.float64)
        self.latest_ur5_joints: np.ndarray | None = None
        self._sample_count = 0
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._first_command = threading.Event()
        self._home_requested = threading.Event()
        self._external_home_requested = threading.Event()
        self._home_gripper_target: float | None = 0.0
        self._home_stable_since: float | None = None
        self._wrist_buttons: dict[int, bool] = {}
        self._failure: BaseException | None = None
        self._flight_dumped = False
        self._thread: threading.Thread | None = None
        self._home_receiver = (
            HomeCommandReceiver(home_socket) if home_socket is not None else None
        )

    def __enter__(self):
        self.mouse.start()
        try:
            if self.gripper is not None:
                self.gripper.open()
                synchronize = getattr(self.gripper_control, "synchronize", None)
                if callable(synchronize):
                    synchronize()
            # WristMasterSlaveController starts parked after HOME. Resume it
            # explicitly for TASK2 collection; otherwise the wrist remains still.
            self.wrist.request_resume()
            if self._home_receiver is not None:
                self._home_receiver.start()
        except BaseException:
            if self._home_receiver is not None:
                self._home_receiver.close()
            if self.gripper is not None:
                self.gripper.close()
            self.mouse.stop()
            raise
        return self

    def commanded_twist(self) -> np.ndarray:
        """Return the latest safety-filtered UR5 command for lifecycle gates."""
        with self._latest_lock:
            return self.latest_twist.copy()

    def state(self):
        if not self.session.supervisor.armed:
            self.session.supervisor.call_with_peer_heartbeats(
                "ur5", self.session.ur5.prepare_control
            )
            self.session.supervisor.arm()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_guarded, name="collection-ur5-control")
            self._thread.start()
            if not self._first_command.wait(timeout=1.0):
                self._raise_failure("UR5 control loop did not produce its first command")
        self._raise_failure()
        self.wrist.check()
        with self._latest_lock:
            return self.latest[0].copy(), self.latest[1].copy()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 - transfer control failure
            self._failure = exc
            self._stop.set()
            self._dump_flight_recorder(f"{type(exc).__name__}: {exc}")
            if self._home_receiver is not None:
                self._home_receiver.close()
            with suppress(OSError, RuntimeError):
                self.session.stop_ur5_motion()

    def _run(self) -> None:
        start_pose: np.ndarray | None = None
        while not self._stop.is_set():
            started = time.monotonic()
            self._poll_home_command()
            self.wrist.check()
            motion, buttons = self.mouse.state()
            if self.gripper_control is not None and not self._home_requested.is_set():
                # HOME owns the gripper target; ignore 3/4 button commands until
                # UR5, wrist, and gripper have all reached their start state.
                gripper_command = self.gripper_control.update(buttons)
                if gripper_command is not None and self.gripper is not None:
                    state = self.gripper.read_state()
                    print(
                        f"gripper target_normalized={float(gripper_command):.4f} "
                        f"target={state.target_position} position={state.position} "
                        f"actual_normalized={state.normalized_position:.4f} "
                        f"voltage={state.voltage_v:.2f}V "
                        f"temperature={state.temperature_c}C load={int(state.load_enabled)}",
                        flush=True,
                    )
            wrist_action, self._wrist_buttons = _wrist_gate_action(
                buttons, self._wrist_buttons
            )
            if wrist_action == "home":
                # R owns only the two-axis OpenRB wrist. Ramp to runtime zero
                # at the same 45 deg/s maximum as SpaceMouse wrist jogging;
                # coordinated T/Fit HOME retains mechanical HOME_ALL calibration.
                self.wrist.request_soft_home()
            elif wrist_action == "park":
                self.wrist.request_park()
            elif wrist_action == "resume":
                self.wrist.request_resume()
            wrist_home_only = bool(buttons.get(int(Button.R), False))
            ctrl_wrist = bool(buttons.get(int(Button.CTRL), False))
            if wrist_home_only:
                # R is an exclusive wrist-only HOME control. Suppress UR5 cap
                # motion for the entire press and leave the gripper untouched.
                self.wrist.clear_spacemouse_velocity()
                ur5_motion = np.zeros(6, dtype=np.float32)
            elif ctrl_wrist:
                # Ctrl turns the SpaceMouse cap into a two-axis wrist jog:
                # cap X -> FE and cap Y -> RU. UR5 receives zero twist while
                # the modifier is held, preventing simultaneous arm motion.
                # Keep the wrist in a deliberately slower, damped jog mode.
                # The SpaceMouse values are normalized to [-1, 1]; sending the
                # old 20 deg/s full-scale command made small cap noise turn into
                # visibly large FE/RU motion on the output loop.
                self.wrist.set_spacemouse_velocity(*spacemouse_wrist_velocity(motion))
                ur5_motion = np.zeros(6, dtype=np.float32)
            else:
                self.wrist.clear_spacemouse_velocity()
                ur5_motion = motion
            with self._latest_lock:
                self.latest = (motion.copy(), buttons.copy())
                self._sample_count += 1
            start_pose, twist, target_qd, joints = _apply_legacy_ur5_command(
                self.session,
                ur5_motion,
                buttons,
                start_pose,
                home_requested=self._home_requested.is_set(),
            )
            with self._latest_lock:
                self.latest_twist = np.asarray(twist, dtype=np.float64)
                self.latest_target_qd = np.asarray(target_qd, dtype=np.float64)
                self.latest_ur5_joints = np.asarray(joints, dtype=np.float64)
            self._record_cycle(motion, buttons, twist)
            self._first_command.set()
            self._stop.wait(max(0.0, self.session.control_period_s - (time.monotonic() - started)))

    def _dump_flight_recorder(self, reason: str) -> None:
        if self._flight_dumped or len(self.flight_recorder) == 0:
            return
        with suppress(OSError):
            dump_path = self.flight_recorder.dump(reason=reason)
            self._flight_dumped = True
            if dump_path is not None:
                print(f"Flight recorder dumped to {dump_path}", flush=True)

    def _record_cycle(
        self, motion: np.ndarray, buttons: dict[int, bool], twist: np.ndarray
    ) -> None:
        diagnostics = getattr(self.mouse, "diagnostics", None)
        sample_age: float | None = None
        if callable(diagnostics):
            info = diagnostics()
            generated = info.get("last_generated_time")
            if isinstance(generated, (int, float)):
                sample_age = time.monotonic() - float(generated)
        ctrl_wrist = bool(buttons.get(int(Button.CTRL), False))
        wrist_home_only = bool(buttons.get(int(Button.R), False))
        self.flight_recorder.record(
            ControlCycleRecord(
                monotonic_s=time.monotonic(),
                spacemouse_motion=tuple(round(float(value), 6) for value in motion),
                spacemouse_buttons=tuple(
                    sorted(code for code, pressed in buttons.items() if pressed)
                ),
                spacemouse_sample_age_s=sample_age,
                ur5_commanded_twist=tuple(round(float(value), 6) for value in twist),
                ur5_tcp_z_m=None,
                floor_guard_active=bool(
                    not ctrl_wrist
                    and not wrist_home_only
                    and not self._home_requested.is_set()
                    and twist[2] == 0.0
                    and float(motion[2]) < 0.0
                ),
                phase="homing" if self._home_requested.is_set() else "recording",
            )
        )

    def _raise_failure(self, fallback: str | None = None) -> None:
        if self._failure is not None:
            raise RuntimeError(
                f"collection control loop failed: {self._failure}"
            ) from self._failure
        if fallback is not None:
            raise RuntimeError(fallback)

    def read_gripper_state(self):
        if self.gripper is None:
            return None
        return self.gripper.read_state()

    def diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.mouse, "diagnostics", None)
        info = dict(diagnostics()) if callable(diagnostics) else {}
        with self._latest_lock:
            info["sample_count"] = self._sample_count
        return info

    def _request_home_with_gripper(self, target: float | None) -> None:
        self._home_stable_since = None
        self._home_gripper_target = None if target is None else float(target)
        self._home_requested.set()
        self.wrist.clear_spacemouse_velocity()
        self.wrist.request_home()
        if self.gripper is not None and self._home_gripper_target is not None:
            command_position = getattr(self.gripper, "command_position", None)
            if callable(command_position):
                command_position(self._home_gripper_target)
            else:
                self.gripper.command(int(self._home_gripper_target >= 0.5))
            synchronize = getattr(self.gripper_control, "synchronize", None)
            if callable(synchronize):
                synchronize()

    def request_home(self) -> None:
        """HOME using the task-pose gripper value: open (0.0)."""
        self._request_home_with_gripper(0.0)

    def request_home_preserve_gripper(self) -> None:
        """Post-Fit HOME without changing or gating on the gripper target."""
        self._request_home_with_gripper(None)

    def _poll_home_command(self) -> None:
        if self._home_receiver is not None and self._home_receiver.poll():
            # This path is deliberately inside the hardware loop. It does not
            # depend on the dashboard, recorder, or episode state.
            self._external_home_requested.set()
            self.request_home()

    def consume_external_home_request(self) -> bool:
        """Return one edge for workflow bookkeeping after low-level preemption."""
        requested = self._external_home_requested.is_set()
        if requested:
            self._external_home_requested.clear()
        return requested

    def clear_home(self) -> None:
        self._home_requested.clear()
        # HOME leaves the wrist parked; resume only after coordinated home.
        self.wrist.request_resume()

    def home_status(self) -> dict[str, object]:
        with self._latest_lock:
            joints = None if self.latest_ur5_joints is None else self.latest_ur5_joints.copy()
        if joints is None:
            self._home_stable_since = None
            return {"at_home": False, "detail": "waiting for UR5 telemetry"}
        ur5_error = float(np.max(np.abs(joints - self.session.ur5_home_joints)))
        wrist_status = self.wrist.home_status()
        gripper_status: dict[str, object] = {
            "at_home": True,
            "detail": "gripper disabled",
        }
        if self.gripper is not None:
            state = self.gripper.read_state()
            target = self._home_gripper_target
            if target is None:
                gripper_at_home = True
                target_label = "preserve"
            else:
                feedback_error = abs(float(state.normalized_position) - target)
                target_error = abs(float(state.normalized_target_position) - target)
                gripper_at_home = bool(
                    feedback_error <= self.gripper_home_tolerance
                    and target_error <= 1e-6
                )
                target_label = f"{target:.1f}"
            gripper_status = {
                "at_home": gripper_at_home,
                "detail": (
                    f"gripper normalized={state.normalized_position:.4f} "
                    f"home_target={target_label} position={state.position} "
                    f"target={state.target_position}"
                ),
                "position": int(state.position),
                "target_position": int(state.target_position),
            }
        within = (
            ur5_error <= 0.010
            and bool(wrist_status["at_home"])
            and bool(gripper_status["at_home"])
        )
        now = time.monotonic()
        if not within:
            self._home_stable_since = None
        elif self._home_stable_since is None:
            self._home_stable_since = now
        at_home = bool(
            within and self._home_stable_since is not None and now - self._home_stable_since >= 0.30
        )
        return {
            "at_home": at_home,
            "detail": (
                f"UR5 error {ur5_error:.3f} rad; {wrist_status['detail']}; "
                f"{gripper_status['detail']}"
            ),
            "ur5_error_rad": ur5_error,
            "wrist": wrist_status,
            "gripper": gripper_status,
        }

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        reason = (
            f"{type(self._failure).__name__}: {self._failure}"
            if self._failure is not None
            else "control session stopped"
        )
        self._dump_flight_recorder(reason)
        if self._home_receiver is not None:
            self._home_receiver.close()
        alive = self._thread is not None and self._thread.is_alive()
        try:
            self.session.stop_ur5_motion()
        finally:
            try:
                if self.gripper is not None:
                    self.gripper.close()
            finally:
                self.mouse.stop()
        context_failed = bool(_args and _args[0] is not None)
        if alive and not context_failed:
            raise RuntimeError("collection UR5 control thread did not stop")
        if self._failure is not None and not context_failed:
            self._raise_failure()


class CachedSpaceMouse:
    """Poll one physical device once and fan out identical state to teleop loops."""

    def __init__(self, mouse: SpaceMouseProcess, period_s: float) -> None:
        self.mouse = mouse
        self.period_s = period_s
        self.stale_timeout_s = max(0.10, 4.0 * period_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._latest = (np.zeros(6, dtype=np.float32), {})
        self._sample_count = 0
        self._last_sample_time: float | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.mouse.start()
        self._thread = threading.Thread(target=self._run, name="spacemouse-input")
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            self.__exit__(None, None, None)
            raise RuntimeError("SpaceMouse did not produce input within one second")
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                motion, buttons = self.mouse.state()
                with self._lock:
                    self._latest = (motion.copy(), buttons.copy())
                    self._sample_count += 1
                    self._last_sample_time = time.monotonic()
                self._ready.set()
                self._stop.wait(max(0.0, self.period_s - (time.monotonic() - started)))
        except BaseException as exc:  # noqa: BLE001 - transfer input failure
            self._failure = exc
            self._ready.set()
            self._stop.set()

    def state(self):
        if self._failure is not None:
            raise RuntimeError(f"SpaceMouse input failed: {self._failure}") from self._failure
        now = time.monotonic()
        with self._lock:
            if self._last_sample_time is None:
                self._latest = (np.zeros(6, dtype=np.float32), {})
                raise RuntimeError("SpaceMouse cache has no sample; motion cleared")
            age = now - self._last_sample_time
            if age > self.stale_timeout_s:
                self._latest = (np.zeros(6, dtype=np.float32), {})
                if age > 1.0:
                    raise RuntimeError(
                        f"SpaceMouse cache remained stale for {age:.3f}s; motion cleared"
                    )
            return self._latest[0].copy(), self._latest[1].copy()

    def diagnostics(self) -> dict[str, object]:
        """Expose the cache boundary so the collection loop can prove receipt."""
        diagnostics = self.mouse.diagnostics()
        with self._lock:
            diagnostics.update(
                sample_count=self._sample_count,
                last_sample_time=self._last_sample_time,
            )
        return diagnostics

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.mouse.stop()


@dataclass
class StationSynchronizer:
    sources: dict[str, Any]
    sequence: int = 0

    def read(self, timeout_s: float = 1.0) -> SynchronizedInputs:
        now = time.monotonic()
        frames = self.sources["cameras"].read(timeout_s)
        state = self.sources["ur5"].read_ur5_state()
        wuji_source = self.sources.get("wuji")
        hand = wuji_source.read_wuji_positions() if wuji_source is not None else None
        temperature = wuji_source.read_wuji_temperature() if wuji_source is not None else None
        wrist_source = self.sources.get("wrist")
        wrist_state = wrist_source.state() if wrist_source is not None else None
        controlled_mouse = self.sources["spacemouse"]
        read_gripper = getattr(controlled_mouse, "read_gripper_state", None)
        gripper_state = read_gripper() if callable(read_gripper) else None
        motion, buttons = controlled_mouse.latest
        self.sequence += 1
        ur = SimpleNamespace(
            actual_q=np.asarray(state["joints"], dtype=np.float32),
            actual_tcp_pose=np.asarray(state["tcp_pose"], dtype=np.float32),
            actual_tcp_speed=np.asarray(state["tcp_speed"], dtype=np.float32),
            target_qd=np.asarray(controlled_mouse.latest_target_qd, dtype=np.float32),
            target_tcp_speed=np.asarray(controlled_mouse.latest_twist, dtype=np.float32),
        )
        wuji = (
            SimpleNamespace(
                actual_q=np.asarray(hand, dtype=np.float32),
                command_q=np.asarray(controlled_mouse.latest_hand_command, dtype=np.float32),
                temperature=temperature,
            )
            if hand is not None
            else None
        )
        if wrist_state is None:
            wrist = None
        elif hasattr(wrist_state, "actual_q"):
            wrist = SimpleNamespace(
                actual_q=np.asarray(wrist_state.actual_q, dtype=np.float32),
                target_q=np.asarray(wrist_state.target_q, dtype=np.float32),
                motor_position_raw=np.asarray(
                    wrist_state.motor_position_raw, dtype=np.int64
                ),
                motor_goal_raw=np.asarray(wrist_state.motor_goal_raw, dtype=np.int64),
                motor_rate_raw_s=np.asarray(
                    wrist_state.motor_rate_raw_s, dtype=np.int64
                ),
            )
        else:
            wrist = SimpleNamespace(
                actual_q=np.deg2rad(
                    np.asarray([wrist_state.fe_deg, wrist_state.ru_deg], dtype=np.float32)
                ),
                target_q=np.deg2rad(
                    np.asarray(
                        [wrist_state.target_fe_deg, wrist_state.target_ru_deg],
                        dtype=np.float32,
                    )
                ),
                motor_position_raw=np.asarray(
                    [wrist_state.j1_present_raw, wrist_state.j2_present_raw], dtype=np.int64
                ),
                motor_goal_raw=np.asarray(
                    [wrist_state.j1_goal_raw, wrist_state.j2_goal_raw], dtype=np.int64
                ),
                motor_rate_raw_s=np.asarray(
                    [wrist_state.j1_rate_raw_s, wrist_state.j2_rate_raw_s], dtype=np.int64
                ),
            )
        mouse = SimpleNamespace(
            axes=np.asarray(motion, dtype=np.float32),
            buttons=np.asarray([bool(buttons.get(int(code), False)) for code in RECORDED_BUTTONS]),
        )

        def sample(
            value: Any,
            device_timestamp: float = now,
            host_timestamp: float = now,
            sequence: int | None = None,
        ) -> SourceSample:
            return SourceSample(
                value,
                device_timestamp,
                host_timestamp,
                self.sequence if sequence is None else sequence,
            )

        schema = load_input_schema(self.sources.get("input_schema"))
        cameras = {
            str(camera["role"]): sample(
                frames[str(camera["role"])].color,
                frames[str(camera["role"])].device_timestamp_s,
                frames[str(camera["role"])].host_timestamp_s,
                frames[str(camera["role"])].sequence,
            )
            for camera in enabled_cameras(schema)
        }
        command_name = str(schema["synchronization"]["command_channel"]["name"])
        channels = {"ur5": sample(ur), command_name: sample(mouse)}
        if wuji is not None:
            channels["wuji"] = sample(wuji)
        if wrist is not None:
            wrist_timestamp = float(getattr(wrist_state, "host_timestamp_s", now))
            wrist_sequence = int(getattr(wrist_state, "sequence", self.sequence))
            channels["wrist"] = sample(
                wrist,
                wrist_timestamp,
                wrist_timestamp,
                wrist_sequence,
            )
        if gripper_state is not None:
            channels["gripper"] = sample(
                gripper_state,
                gripper_state.host_timestamp_s,
                gripper_state.host_timestamp_s,
                gripper_state.sequence,
            )
        return SynchronizedInputs(cameras=cameras, channels=channels)


def _driver_python_preflight(hardware: dict[str, Any], *sections: str) -> None:
    for section_name in sections:
        python = Path(str(_section(hardware, section_name).get("driver_python", "")))
        if not python.is_file():
            raise FileNotFoundError(f"{section_name} driver Python is missing: {python}")


def _preflight(hardware: dict[str, Any]) -> None:
    _driver_python_preflight(hardware, "ur5", "wujihand")


def _wrist_teleop_preflight(hardware: dict[str, Any]) -> None:
    _driver_python_preflight(hardware, "ur5")
    wrist = _section(hardware, "wrist_sensor")
    config = PROJECT_ROOT / str(wrist.get("config", ""))
    if not config.is_file():
        raise FileNotFoundError(f"wrist_sensor.config is missing: {config}")


def _collection_preflight(hardware: dict[str, Any]) -> None:
    _preflight(hardware)
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("LeRobot v3 is not installed in the collection Python environment")


def _wrist_collection_preflight(hardware: dict[str, Any]) -> None:
    _wrist_teleop_preflight(hardware)
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("LeRobot v3 is not installed in the collection Python environment")


def _mouse(_hardware: dict[str, Any], session: StationSession) -> SpaceMouseProcess:
    # Keep every production entry point on the spacenavd-backed input path.
    return SpaceMouseProcess(
        deadzone=session.spacemouse_deadzone,
        stale_timeout=0.05,
        rate_hz=max(250.0, 1.0 / session.control_period_s),
        backend="spnav",
    )


def make_teleop(hardware: dict[str, Any], task: dict[str, Any]) -> TeleopDependencies:
    wuji_enabled = bool(_section(hardware, "wujihand").get("enabled", False))
    wrist_enabled = bool(_section(hardware, "wrist_sensor").get("enabled", False))
    if wrist_enabled and not wuji_enabled:
        session = UR5OnlySession(hardware, task)
        unavailable = lambda *_args: None
        return TeleopDependencies(
            ur5_factory=lambda _config, mouse, _stop: UR5TeleopLoop(session, mouse),
            wuji_factory=unavailable,
            spacemouse_factory=lambda config: CachedSpaceMouse(
                _mouse(config, session), session.control_period_s
            ),
            preflight=_wrist_teleop_preflight,
            runtime_factories={
                "ur5-teleop": lambda _config, mouse, _stop: UR5TeleopLoop(session, mouse),
                "wrist-teleop": lambda config, mouse, _stop: NativeWristTeleopLoop(config, mouse),
            },
            required_devices=("ur5", "wrist_sensor", "spacemouse"),
        )
    if not wuji_enabled:
        raise ValueError("real teleop requires either wujihand or wrist_sensor")
    session = StationSession(hardware, task)
    return TeleopDependencies(
        ur5_factory=lambda _config, mouse, _stop: UR5TeleopLoop(session, mouse),
        wuji_factory=lambda _config, mouse, _stop: WujiSupervisionLoop(session, hardware, mouse),
        spacemouse_factory=lambda config: CachedSpaceMouse(
            _mouse(config, session), session.control_period_s
        ),
        preflight=_preflight,
    )


def _cameras(hardware: dict[str, Any]):
    devices = _section(hardware, "cameras")["devices"]
    schema = load_input_schema(hardware.get("input_schema"))
    image_height, image_width, _channels = schema["capture"]["image_shape"]
    configs = validate_camera_set(
        (
            CameraConfig(
                str(item["role"]),
                str(item["serial"]),
                int(image_width),
                int(image_height),
                int(schema["capture"]["fps"]),
            )
            for item in devices
        ),
        expected_count=len(enabled_cameras(schema)),
    )
    capture = RealSenseCapture(
        configs,
        max_skew_ms=float(schema["synchronization"]["max_camera_skew_ms"]),
    )

    @contextmanager
    def opened():
        capture.start()
        try:
            capture.read(5.0)
            yield capture
        finally:
            capture.stop()

    return opened()


def _dataset(config: dict[str, Any], task: dict[str, Any], input_schema: str | Path | None = None):
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("LeRobot v3 is not installed in the collection Python environment")
    root = Path(str(config.get("root", "data/lerobot")))
    task_id = str(task.get("task", {}).get("id", "real"))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output = root / f"{task_id}-{stamp}"
    schema = load_input_schema(input_schema)
    return create_dataset(
        repo_id=f"local/{task_id}-{stamp}", root=output, fps=int(schema["capture"]["fps"])
    )


def _configured_dataset(
    config: dict[str, Any],
    task: dict[str, Any],
    contract: ConfiguredDatasetContract,
    input_schema: str | Path | None = None,
):
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("LeRobot v3 is not installed in the collection Python environment")
    configured_root = Path(str(config.get("root", "data/lerobot"))).expanduser()
    root = configured_root if configured_root.is_absolute() else PROJECT_ROOT / configured_root
    task_id = str(task.get("task", {}).get("id", "real"))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dof_label = {
        "ur5_wrist_gripper_9dof": "9dof",
        "ur5_wrist_gripper_continuous_9dof": "9dof-continuous",
    }.get(contract.robot_type, "8dof")
    output = root / f"{task_id}-{dof_label}-{stamp}"
    repo_id = f"local/{task_id}-{dof_label}-{stamp}"
    schema_path = (
        Path(input_schema).expanduser()
        if input_schema is not None
        else PROJECT_ROOT / "configs" / "input_schemas" / "ur5e_wrist_gripper_9dof.yaml"
    )
    if not schema_path.is_absolute():
        schema_path = PROJECT_ROOT / schema_path
    sidecar = IncrementalV21Sidecar.from_config(
        config,
        source=output,
        repo_id=f"{repo_id}-v21",
        schema_path=schema_path,
        project_root=PROJECT_ROOT,
    )
    active_aliases: dict[Path, Path] = {}
    if contract.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
        configured_aliases = config.get("active_aliases", {})
        if not isinstance(configured_aliases, dict):
            raise ValueError("dataset.active_aliases must be a mapping")
        continuous_aliases = configured_aliases.get("task2_9dof_continuous", {})
        if not isinstance(continuous_aliases, dict):
            raise ValueError("dataset.active_aliases.task2_9dof_continuous must be a mapping")
        for name, target in (("v3", output), ("v21", None if sidecar is None else sidecar.target)):
            value = continuous_aliases.get(name)
            if value is not None and target is not None:
                alias = Path(str(value)).expanduser()
                active_aliases[alias if alias.is_absolute() else PROJECT_ROOT / alias] = target
    try:
        return create_configured_dataset(
            repo_id=repo_id,
            root=output,
            contract=contract,
            pi05_v21_sidecar=sidecar,
            active_aliases=active_aliases,
            auto_merge_config=config,
            input_schema=schema_path,
        )
    except BaseException as exc:
        if sidecar is not None:
            try:
                sidecar.finalize(expected_episodes=0)
            except BaseException as cleanup_exc:  # noqa: BLE001 - preserve creation failure
                exc.add_note(f"incremental v2.1 cleanup failed: {cleanup_exc}")
        raise


def make_collection(
    hardware: dict[str, Any], dataset: dict[str, Any], task: dict[str, Any]
) -> CollectionDependencies:
    wuji_enabled = bool(_section(hardware, "wujihand").get("enabled", False))
    wrist_enabled = bool(_section(hardware, "wrist_sensor").get("enabled", False))
    if wrist_enabled and not wuji_enabled:
        session = UR5OnlySession(hardware, task)
        wrist_config = _section(hardware, "wrist_sensor")
        gripper_config = hardware.get("gripper", {})
        schema = load_input_schema(hardware.get("input_schema"))
        state_channels = {
            str(item["name"]) for item in schema["synchronization"]["state_channels"]
        }
        gripper_enabled = bool(
            "gripper" in state_channels
            and isinstance(gripper_config, dict)
            and gripper_config.get("enabled", False)
        )
        continuous_gripper = (
            gripper_value_contract(schema) == CONTINUOUS_GRIPPER_CONTRACT
        )
        wrist = WristMasterSlaveController(
            PROJECT_ROOT / str(wrist_config["config"]),
            teleop_port=str(wrist_config.get("teleop_port", "auto")),
            openrb_port=str(wrist_config.get("openrb_port", "auto")),
            baud=int(wrist_config.get("baud", 115200)),
        )
        gripper_calibration = None
        if gripper_enabled and continuous_gripper:
            resolved_gripper = resolve_gripper_config(gripper_config)
            gripper_calibration = {
                "driver": resolved_gripper["driver"],
                "open_position_raw": resolved_gripper["open_position"],
                "closed_position_raw": resolved_gripper["closed_position"],
                "feedback_position_bias_raw": resolved_gripper.get(
                    "feedback_position_bias", 0
                ),
            }
        contract = ConfiguredDatasetContract(
            schema,
            gripper_calibration=gripper_calibration,
        )
        assembler = partial(
            assemble_configured_frame,
            schema=schema,
            validator=contract.validate_frame,
        )
        unavailable = lambda *_args: None
        return CollectionDependencies(
            ur5_factory=unavailable,
            wuji_factory=unavailable,
            spacemouse_factory=unavailable,
            cameras_factory=unavailable,
            dataset_factory=lambda config, selected_task: _configured_dataset(
                config,
                selected_task,
                contract,
                str(hardware.get("input_schema")),
            ),
            synchronizer_factory=lambda sources, _config: StationSynchronizer(
                {**sources, "input_schema": hardware.get("input_schema")}
            ),
            recorder_factory=lambda data, sync, prompt: EpisodeRecorder(
                data, sync, prompt, assembler=assembler
            ),
            preflight=_wrist_collection_preflight,
            resource_factories={
                "ur5": lambda _config: session.lease(arm=False),
                "wrist": lambda _config: wrist,
                "spacemouse": lambda config: ControlledUR5SpaceMouse(
                    _mouse(config, session),
                    session,
                    wrist,
                    create_gripper(gripper_config)
                    if gripper_enabled
                    else None,
                    continuous_gripper=continuous_gripper,
                    gripper_rate_per_s=float(
                        gripper_config.get("teleop_rate_per_s", 1.0)
                    ),
                    gripper_command_hz=float(
                        gripper_config.get("teleop_command_hz", 30.0)
                    ),
                    gripper_home_tolerance=float(
                        gripper_config.get("home_tolerance_normalized", 0.02)
                    ),
                    home_socket=default_task2_home_socket(),
                    flight_recorder=FlightRecorder(),
                ),
                "cameras": _cameras,
            },
            required_devices=(
                "ur5",
                "wrist_sensor",
                "spacemouse",
                "cameras",
                *(("gripper",) if gripper_enabled else ()),
            ),
        )
    session = StationSession(hardware, task)
    return CollectionDependencies(
        ur5_factory=lambda _config: session.lease(arm=False),
        wuji_factory=lambda _config: session.lease(arm=False),
        spacemouse_factory=lambda _config: ControlledSpaceMouse(_mouse(_config, session), session),
        cameras_factory=_cameras,
        dataset_factory=lambda config, task: _dataset(config, task, hardware.get("input_schema")),
        synchronizer_factory=lambda sources, _config: StationSynchronizer(
            {**sources, "input_schema": hardware.get("input_schema")}
        ),
        recorder_factory=lambda data, sync, prompt: EpisodeRecorder(data, sync, prompt),
        preflight=_collection_preflight,
    )


def make_dependencies(*args: dict[str, Any]):
    if len(args) == 2:
        return make_teleop(args[0], args[1])
    if len(args) == 3:
        return make_collection(args[0], args[1], args[2])
    raise TypeError("site adapter expects hardware/task or hardware/dataset/task")
