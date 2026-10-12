"""Default adapter for the local UR5/WujiHand/three-RealSense station."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from slai_mi.collection.vla_recorder import EpisodeRecorder, SourceSample, SynchronizedInputs
from slai_mi.datasets.lerobot_v3.schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES
from slai_mi.datasets.lerobot_v3.writer import create_dataset
from slai_mi.devices.cameras import CameraConfig, RealSenseCapture, validate_camera_set
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
from slai_mi.devices.wujihand.filters import OneEuroFilter
from slai_mi.devices.wujihand.manual_control import (
    AccelerationLimitedTrajectory,
    ManualHandSettings,
    ManualWujiController,
)
from slai_mi.devices.wujihand.process import WujiHandDriverProcess
from slai_mi.input_schema import enabled_cameras, load_input_schema, split_capture_vector
from slai_mi.runtime import CollectionDependencies, TeleopDependencies
from slai_mi.runtime.hardware_supervisor import HardwareProcessSupervisor

MAX_LINEAR_M_S = 0.25
MAX_ANGULAR_RAD_S = 0.60
CONTROL_PERIOD_S = 1.0 / 125.0
WUJI_PERIOD_S = 1.0 / 30.0
WUJI_MAX_VELOCITY_RAD_S = 3.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        if expected_schema != "real_v1" or not schema_ref.endswith("#schemas.real_v1"):
            raise ValueError(f"task pose {label} must use the real_v1 26 DoF schema")
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
        raise ValueError("task.start_pose_ref is required for Button 4 home")
    return _configured_pose(
        task,
        reference,
        kind="task_start",
        joint_names=(*UR5_JOINT_NAMES, *WUJI_JOINT_NAMES),
    )


class StationSession:
    """Own both isolated drivers so one failure always disables the pair."""

    def __init__(self, hardware: dict[str, Any], task: dict[str, Any]) -> None:
        ur5, wuji = _section(hardware, "ur5"), _section(hardware, "wujihand")
        profile = _control_profile(task)
        motion = _section(profile, "motion")
        hand = _section(profile, "hand")
        linear = min(float(ur5.get("max_linear_m_s", MAX_LINEAR_M_S)), MAX_LINEAR_M_S)
        angular = min(float(ur5.get("max_angular_rad_s", MAX_ANGULAR_RAD_S)), MAX_ANGULAR_RAD_S)
        self.speed_settings = SpeedSettings(
            translation=float(motion["translation_speed"]),
            rotation=float(motion["rotation_speed"]),
            boost_translation=float(motion["precision_translation_speed"]),
            boost_rotation=float(motion["precision_rotation_speed"]),
        )
        if (
            max(self.speed_settings.translation, self.speed_settings.boost_translation) > linear
            or max(self.speed_settings.rotation, self.speed_settings.boost_rotation) > angular
        ):
            raise ValueError("task SpaceMouse speed exceeds the hardware safety ceiling")
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
        self.open_hand_target = _hand_preset(task, "open")
        self.grasp_hand_target = _hand_preset(task, "grasp")
        self.ur5 = UR5DriverProcess(
            python=Path(ur5["driver_python"]),
            host=str(ur5["host"]),
            watchdog_s=float(ur5.get("driver_watchdog_s", 0.25)),
            max_linear_m_s=linear,
            max_angular_rad_s=angular,
        )
        self.wuji = WujiHandDriverProcess(
            python=Path(wuji["driver_python"]),
            usb_serial=str(wuji["usb_serial"]),
            product_serial=str(wuji.get("product_serial", "")),
            watchdog_s=float(wuji.get("driver_watchdog_s", 0.5)),
            max_temperature_c=min(float(wuji.get("max_temperature_c", 80.0)), 80.0),
            thermal_warning_temperature_c=float(
                wuji.get("thermal_warning_temperature_c", 70.0)
            ),
            thermal_critical_temperature_c=float(
                wuji.get("thermal_critical_temperature_c", 75.0)
            ),
            limit_margin_rad=max(float(wuji.get("limit_margin_rad", 0.03)), 0.03),
            max_effort_fraction=min(float(wuji.get("max_effort_fraction", 0.65)), 0.65),
            max_velocity_rad_s=min(
                float(wuji.get("max_velocity_rad_s", WUJI_MAX_VELOCITY_RAD_S)),
                WUJI_MAX_VELOCITY_RAD_S,
            ),
        )
        self.supervisor = HardwareProcessSupervisor({"ur5": self.ur5, "wujihand": self.wuji})
        self.linear_limit = linear
        self.angular_limit = angular
        self._lock = threading.Lock()
        self._started = False
        self._leases = 0

    @contextmanager
    def lease(self, *, arm: bool = True):
        with self._lock:
            if not self._started:
                self.supervisor.start()
                if arm:
                    self.supervisor.arm()
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
                    self.supervisor.stop()
                    self._started = False

    def check(self) -> None:
        self.supervisor.check()

    def arm(self) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("cannot arm a session before it is started")
            if not self.supervisor.armed:
                self.supervisor.arm()

    def write_ur5_twist(
        self, twist: Any, *, acceleration: float, duration_s: float
    ) -> None:
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_twist(
                twist, acceleration=acceleration, duration_s=duration_s
            ),
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
        self.supervisor.call_with_peer_heartbeats(
            "ur5", self.ur5.prepare_control
        )

    def write_ur5_joint_velocity(
        self, velocity: Any, *, acceleration: float, duration_s: float
    ) -> None:
        self.supervisor.call_with_peer_heartbeats(
            "ur5",
            lambda: self.ur5.write_joint_velocity(
                velocity, acceleration=acceleration, duration_s=duration_s
            ),
        )

    def stop_ur5_motion(self) -> None:
        self.supervisor.call_with_peer_heartbeats("ur5", self.ur5.stop_motion)

    def write_wuji_positions(self, positions: Any) -> None:
        self.supervisor.call_with_peer_heartbeats(
            "wujihand", lambda: self.wuji.write_positions(positions)
        )


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
            raise ValueError(f"real policy schema lacks required hardware component: {exc}") from exc
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
    state = session.read_ur5_state()
    current_pose = np.asarray(state["tcp_pose"], dtype=float)
    current_joints = np.asarray(state["joints"], dtype=float)
    if start_pose is None:
        start_pose = current_pose.copy()
    cap_active = bool(np.any(np.abs(motion) > 0.0))
    wrist_pressed = bool(
        buttons.get(int(Button.ONE), False)
        or buttons.get(int(Button.TWO), False)
    )
    home_pressed = bool(buttons.get(int(Button.HOME), False) or home_requested)
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

    def run(self, stop_event: threading.Event) -> None:
        provider = WujiRetargetTargetProvider(self.hardware, external_frames=False)
        with self.session.lease():
            controller = WujiTargetController(self.session, provider)
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
            self.camera_role = str(result["camera_role"])
        except BaseException:
            self.close()
            raise

    def _rpc(self, request: dict[str, Any]) -> Any:
        if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
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
        )
        self.manual.trajectory = self.trajectory

    def _stable_target(self, target: np.ndarray) -> np.ndarray | None:
        if self._armed:
            return target
        if self._previous_target is not None and float(
            np.max(np.abs(target - self._previous_target))
        ) <= 0.35:
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
        if buttons is not None and self.manual.update(
            buttons, now, hold_when_inactive=False
        ):
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
            self._control_stop.wait(
                max(0.0, self.session.control_period_s - elapsed)
            )

    def _run_wuji_control(self) -> None:
        assert self._hand_controller is not None
        period = 1.0 / self.session.hand_settings.command_hz
        while not self._control_stop.is_set():
            started = time.monotonic()
            with self._latest_lock:
                buttons = self.latest[1].copy()
            if self._home_requested.is_set():
                buttons[int(Button.HOME)] = True
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

    def clear_home(self) -> None:
        self._home_requested.clear()

    def home_status(self) -> dict[str, object]:
        with self._latest_lock:
            ur5 = None if self.latest_ur5_joints is None else self.latest_ur5_joints.copy()
            wuji = (
                None
                if self.latest_wuji_positions is None
                else self.latest_wuji_positions.copy()
            )
        if ur5 is None or wuji is None or self._hand_controller is None:
            self._home_stable_since = None
            return {"at_home": False, "detail": "等待零位遥测"}
        ur5_error = float(np.max(np.abs(ur5 - self.session.ur5_home_joints)))
        wuji_error = float(np.max(np.abs(wuji - self.session.wuji_home_joints)))
        command_error = float(
            np.max(
                np.abs(
                    self._hand_controller.trajectory.command
                    - self.session.wuji_home_joints
                )
            )
        )
        within = bool(
            ur5_error <= 0.010
            and wuji_error <= 0.100
            and command_error <= 0.010
        )
        now = time.monotonic()
        if not within:
            self._home_stable_since = None
        elif self._home_stable_since is None:
            self._home_stable_since = now
        at_home = bool(
            within
            and self._home_stable_since is not None
            and now - self._home_stable_since >= 0.30
        )
        return {
            "at_home": at_home,
            "detail": f"UR5误差 {ur5_error:.3f}rad，Wuji误差 {wuji_error:.3f}rad",
            "ur5_error_rad": ur5_error,
            "wuji_error_rad": wuji_error,
            "wuji_command_error_rad": command_error,
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


class CachedSpaceMouse:
    """Poll one physical device once and fan out identical state to teleop loops."""

    def __init__(self, mouse: SpaceMouseProcess, period_s: float) -> None:
        self.mouse = mouse
        self.period_s = period_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._latest = (np.zeros(6, dtype=np.float32), {})
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
                self._ready.set()
                self._stop.wait(max(0.0, self.period_s - (time.monotonic() - started)))
        except BaseException as exc:  # noqa: BLE001 - transfer input failure
            self._failure = exc
            self._ready.set()
            self._stop.set()

    def state(self):
        if self._failure is not None:
            raise RuntimeError(f"SpaceMouse input failed: {self._failure}") from self._failure
        with self._lock:
            return self._latest[0].copy(), self._latest[1].copy()

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
        hand = self.sources["wuji"].read_wuji_positions()
        temperature = self.sources["wuji"].read_wuji_temperature()
        controlled_mouse = self.sources["spacemouse"]
        motion, buttons = controlled_mouse.latest
        self.sequence += 1
        ur = SimpleNamespace(
            actual_q=np.asarray(state["joints"], dtype=np.float32),
            actual_tcp_pose=np.asarray(state["tcp_pose"], dtype=np.float32),
            actual_tcp_speed=np.asarray(state["tcp_speed"], dtype=np.float32),
            target_qd=np.asarray(controlled_mouse.latest_target_qd, dtype=np.float32),
            target_tcp_speed=np.asarray(controlled_mouse.latest_twist, dtype=np.float32),
        )
        wuji = SimpleNamespace(
            actual_q=np.asarray(hand, dtype=np.float32),
            command_q=np.asarray(controlled_mouse.latest_hand_command, dtype=np.float32),
            temperature=temperature,
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
        return SynchronizedInputs(
            cameras=cameras,
            channels={"ur5": sample(ur), "wuji": sample(wuji), command_name: sample(mouse)},
        )


def _preflight(hardware: dict[str, Any]) -> None:
    for section_name in ("ur5", "wujihand"):
        python = Path(str(_section(hardware, section_name).get("driver_python", "")))
        if not python.is_file():
            raise FileNotFoundError(f"{section_name} driver Python is missing: {python}")


def _mouse(_hardware: dict[str, Any], session: StationSession) -> SpaceMouseProcess:
    return SpaceMouseProcess(
        deadzone=session.spacemouse_deadzone,
        stale_timeout=0.1,
        rate_hz=1.0 / session.control_period_s,
    )


def make_teleop(hardware: dict[str, Any], task: dict[str, Any]) -> TeleopDependencies:
    session = StationSession(hardware, task)
    return TeleopDependencies(
        ur5_factory=lambda _config, mouse, _stop: UR5TeleopLoop(session, mouse),
        wuji_factory=lambda _config, mouse, _stop: WujiSupervisionLoop(
            session, hardware, mouse
        ),
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
    capture = RealSenseCapture(configs)

    @contextmanager
    def opened():
        capture.start()
        try:
            capture.read(5.0)
            yield capture
        finally:
            capture.stop()

    return opened()


def _dataset(
    config: dict[str, Any], task: dict[str, Any], input_schema: str | Path | None = None
):
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


def make_collection(
    hardware: dict[str, Any], dataset: dict[str, Any], task: dict[str, Any]
) -> CollectionDependencies:
    session = StationSession(hardware, task)
    return CollectionDependencies(
        ur5_factory=lambda _config: session.lease(arm=False),
        wuji_factory=lambda _config: session.lease(arm=False),
        spacemouse_factory=lambda _config: ControlledSpaceMouse(
            _mouse(_config, session), session
        ),
        cameras_factory=_cameras,
        dataset_factory=lambda config, task: _dataset(
            config, task, hardware.get("input_schema")
        ),
        synchronizer_factory=lambda sources, _config: StationSynchronizer(
            {**sources, "input_schema": hardware.get("input_schema")}
        ),
        recorder_factory=lambda data, sync, prompt: EpisodeRecorder(data, sync, prompt),
        preflight=_preflight,
    )


def make_dependencies(*args: dict[str, Any]):
    if len(args) == 2:
        return make_teleop(args[0], args[1])
    if len(args) == 3:
        return make_collection(args[0], args[1], args[2])
    raise TypeError("site adapter expects hardware/task or hardware/dataset/task")
