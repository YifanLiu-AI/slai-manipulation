from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.mapping import SpeedSettings
from slai_mi.devices.wujihand.manual_control import ManualHandSettings
from slai_mi.observability.flight_recorder import FlightRecorder
from slai_mi.site_adapter import (
    CachedSpaceMouse,
    ControlledSpaceMouse,
    ControlledUR5SpaceMouse,
    NativeWristTeleopLoop,
    RealPolicyBridge,
    StationSession,
    Wrist2WristTeleopLoop,
    _bounded_speed_settings,
    _cameras,
    _clamp_command,
    _configured_dataset,
    _mouse,
    _task_home_joints,
    _wrist_gate_action,
    make_teleop,
    spacemouse_wrist_velocity,
)


class FakeSession:
    linear_limit = 0.25
    angular_limit = 0.60
    speed_settings = SpeedSettings()
    control_period_s = 1.0 / 125.0
    ur5_acceleration = 0.5
    hand_settings = ManualHandSettings()
    open_hand_target = np.zeros(20)
    grasp_hand_target = np.ones(20)
    auxiliary_open_hand_target = np.full(20, -0.5)
    auxiliary_grasp_hand_target = np.full(20, 0.5)
    wuji_home_joints = np.zeros(20)
    wrist_3_jog_speed = 0.2
    home_joint_speed = 0.6
    ur5_home_joints = np.zeros(6)
    max_offset_m = 0.0
    max_rotation_rad = np.deg2rad(5.0)

    def __init__(self):
        self.supervisor = type("Supervisor", (), {"armed": False})()
        self.lease_calls = []
        self.arm_calls = 0
        self.twists = []
        self.hands = []
        self.checks = 0
        self.prepare_calls = 0
        self.wuji = self
        self.ur5 = self

    @contextmanager
    def lease(self, *, arm):
        self.lease_calls.append(arm)
        yield self

    def arm(self):
        self.arm_calls += 1
        self.supervisor.armed = True

    def prepare_ur5_control(self):
        self.prepare_calls += 1

    def prepare_control(self):
        self.prepare_calls += 1

    def write_ur5_twist(self, twist, *, acceleration, duration_s):
        self.twists.append((tuple(twist), acceleration, duration_s))

    def stop_ur5_motion(self):
        pass

    def write_wuji_positions(self, positions):
        self.hands.append(tuple(positions))

    def read_positions(self):
        return (0.0,) * 20

    def read_wuji_positions(self):
        return self.read_positions()

    def read_limits(self):
        return (-2.0,) * 20, (2.0,) * 20

    def read_state(self):
        return {"tcp_pose": [0.0] * 6, "joints": [0.0] * 6}

    def read_ur5_state(self):
        return self.read_state()

    def write_ur5_joint_velocity(self, velocity, *, acceleration, duration_s):
        self.twists.append((tuple(velocity), acceleration, duration_s))

    def check(self):
        self.checks += 1


class FakeMouse:
    def __init__(self, motion=None, buttons=None):
        self.started = False
        self.stopped = False
        self.motion = np.zeros(6, dtype=np.float32) if motion is None else motion
        self.buttons = {} if buttons is None else buttons

    def start(self):
        self.started = True

    def state(self):
        return self.motion, self.buttons

    def stop(self):
        self.stopped = True


def test_production_spacemouse_uses_spnav_backend() -> None:
    session = type(
        "Session",
        (),
        {"spacemouse_deadzone": 0.12, "control_period_s": 1.0 / 125.0},
    )()

    mouse = _mouse({}, session)

    assert mouse.backend == "spnav"
    assert mouse.event_device is None
    assert mouse.rate_hz == 250.0


def test_cached_spacemouse_zeros_transient_stale_descent_without_exiting() -> None:
    mouse = FakeMouse(np.asarray([0.0, 0.0, -0.7, 0.0, 0.0, 0.0], dtype=np.float32))
    cached = CachedSpaceMouse(mouse, period_s=0.008)
    cached._latest = (mouse.motion.copy(), {})
    cached._last_sample_time = time.monotonic() - 0.2

    motion, buttons = cached.state()

    np.testing.assert_array_equal(motion, np.zeros(6, dtype=np.float32))
    assert buttons == {}


def test_cached_spacemouse_fails_after_one_second_without_samples() -> None:
    cached = CachedSpaceMouse(FakeMouse(), period_s=0.008)
    cached._last_sample_time = time.monotonic() - 1.1

    with np.testing.assert_raises_regex(RuntimeError, "remained stale"):
        cached.state()
    np.testing.assert_array_equal(cached._latest[0], np.zeros(6, dtype=np.float32))


def test_collection_mouse_arms_only_on_first_control_state():
    session = FakeSession()
    mouse = FakeMouse()
    controller = ControlledSpaceMouse(mouse, session)

    with controller:
        assert mouse.started
        assert session.lease_calls == [False]
        assert session.arm_calls == 0
        controller.state()
        assert session.prepare_calls == 1
        assert session.arm_calls == 1
        assert session.supervisor.armed
        assert len(session.twists) == 1

    assert mouse.stopped


def test_collection_waits_for_slow_rtde_prepare_before_first_command_timeout():
    session = FakeSession()

    def slow_prepare():
        session.prepare_calls += 1
        time.sleep(1.05)

    session.prepare_ur5_control = slow_prepare
    started = time.monotonic()
    with ControlledSpaceMouse(FakeMouse(), session) as controller:
        controller.state()
    assert time.monotonic() - started >= 1.0
    assert session.prepare_calls == 1
    assert session.arm_calls == 1


def test_collection_mouse_ignores_ctrl_and_uses_control_period():
    session = FakeSession()
    mouse = FakeMouse(
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {int(Button.CTRL): True},
    )
    with ControlledSpaceMouse(mouse, session) as controller:
        controller.state()
    assert session.twists[0] == ((0.08, 0.0, 0.0, 0.0, 0.0, 0.0), 0.5, 0.008)


def test_ur5_control_loop_dumps_flight_recorder_on_failure(monkeypatch, tmp_path) -> None:
    class BrokenWrist:
        def check(self):
            return None

        def clear_spacemouse_velocity(self):
            return None

        def set_spacemouse_velocity(self, *_args):
            return None

        def request_resume(self):
            return None

    class BrokenMouse:
        def __init__(self):
            self.calls = 0

        def start(self):
            return None

        def stop(self):
            return None

        def state(self):
            self.calls += 1
            if self.calls == 1:
                return np.asarray([0.0, 0.0, -0.35, 0.0, 0.0, 0.0], dtype=np.float32), {}
            raise RuntimeError("SpaceMouse worker remained stale for 1.200s")

        def diagnostics(self):
            return {"last_generated_time": time.monotonic()}

    session = FakeSession()
    session.supervisor = SimpleNamespace(
        armed=False,
        arm=lambda: None,
        call_with_peer_heartbeats=lambda _device_id, operation, **_kwargs: operation(),
    )
    recorder = FlightRecorder(capacity=16)
    original_dump = recorder.dump
    dumped: list[Path] = []

    def dump_to_test_directory(*, reason, directory=None, clock=time.time):
        path = original_dump(reason=reason, directory=tmp_path, clock=clock)
        dumped.append(path)
        return path

    monkeypatch.setattr(recorder, "dump", dump_to_test_directory)
    controller = ControlledUR5SpaceMouse(
        BrokenMouse(), session, BrokenWrist(), flight_recorder=recorder
    )

    with pytest.raises(RuntimeError, match="remained stale"), controller:
        for _ in range(50):
            controller.state()
            time.sleep(0.01)

    assert dumped and dumped[0].is_file()
    assert len(recorder) >= 1
    snapshot = recorder.snapshot()
    assert snapshot[0].spacemouse_motion[2] == pytest.approx(-0.35, abs=1e-6)
    assert snapshot[0].ur5_commanded_twist[2] == pytest.approx(-0.028, abs=1e-6)


def test_ur5_control_loop_dumps_flight_recorder_on_normal_stop(monkeypatch, tmp_path) -> None:
    class Wrist:
        def check(self):
            return None

        def clear_spacemouse_velocity(self):
            return None

        def set_spacemouse_velocity(self, *_args):
            return None

        def request_resume(self):
            return None

    mouse = FakeMouse(
        np.asarray([0.0, 0.0, -0.25, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    mouse.diagnostics = lambda: {"last_generated_time": time.monotonic()}
    session = FakeSession()
    session.supervisor = SimpleNamespace(
        armed=False,
        arm=lambda: None,
        call_with_peer_heartbeats=lambda _device_id, operation, **_kwargs: operation(),
    )
    recorder = FlightRecorder(capacity=16)
    original_dump = recorder.dump
    dumped: list[Path] = []

    def dump_to_test_directory(*, reason, directory=None, clock=time.time):
        path = original_dump(reason=reason, directory=tmp_path, clock=clock)
        dumped.append(path)
        return path

    monkeypatch.setattr(recorder, "dump", dump_to_test_directory)

    with ControlledUR5SpaceMouse(
        mouse, session, Wrist(), flight_recorder=recorder
    ) as controller:
        controller.state()
        time.sleep(0.02)

    assert dumped and dumped[0].is_file()
    payload = dumped[0].read_text(encoding="utf-8")
    assert "control session stopped" in payload
    assert '"spacemouse_sample_age_s"' in payload
    assert '"ur5_commanded_twist"' in payload


def test_task2_coordinated_home_opens_and_verifies_gripper(monkeypatch) -> None:
    events: list[object] = []

    class Wrist:
        def clear_spacemouse_velocity(self):
            events.append("wrist-clear")

        def request_home(self):
            events.append("wrist-home")

        def request_resume(self):
            events.append("wrist-resume")

        def home_status(self):
            return {"at_home": True, "detail": "wrist zero"}

    class Gripper:
        def command(self, value):
            events.append(("gripper", value))

        def read_state(self):
            return type(
                "State",
                (),
                {
                    "actual_q": (0.0,),
                    "target_q": (0.0,),
                    "position": 0,
                    "target_position": 0,
                },
            )()

    controller = ControlledUR5SpaceMouse.__new__(ControlledUR5SpaceMouse)
    controller.wrist = Wrist()
    controller.gripper = Gripper()
    controller.session = type("Session", (), {"ur5_home_joints": np.zeros(6)})()
    controller._home_requested = threading.Event()
    controller._home_stable_since = None
    controller._latest_lock = threading.Lock()
    controller.latest_ur5_joints = np.zeros(6)

    controller.request_home()
    assert events == ["wrist-clear", "wrist-home", ("gripper", 0)]
    assert controller._home_requested.is_set()

    clock = iter((10.0, 10.31))
    monkeypatch.setattr("slai_mi.site_adapter.time.monotonic", lambda: next(clock))
    assert not controller.home_status()["at_home"]
    status = controller.home_status()
    assert status["at_home"]
    assert status["gripper"]["at_home"]
    assert "gripper position=0" in status["detail"]

    controller.clear_home()
    assert events[-1] == "wrist-resume"


def test_task2_low_level_home_preempts_without_workflow_dependency() -> None:
    controller = ControlledUR5SpaceMouse.__new__(ControlledUR5SpaceMouse)
    controller._home_receiver = type("Receiver", (), {"poll": lambda _self: True})()
    controller._external_home_requested = threading.Event()
    requests: list[str] = []
    controller.request_home = lambda: requests.append("home")

    controller._poll_home_command()

    assert requests == ["home"]
    assert controller.consume_external_home_request()
    assert not controller.consume_external_home_request()


def test_spacemouse_wrist_velocity_reverses_both_axes_and_suppresses_center_noise() -> None:
    assert spacemouse_wrist_velocity(np.zeros(6)) == (0.0, 0.0)
    # Ctrl wrist mapping reverses both of the previous directions:
    # cap +X -> -FE and cap +Y -> +RU.
    assert spacemouse_wrist_velocity(np.ones(6)) == (-45.0, 45.0)
    assert spacemouse_wrist_velocity(-np.ones(6)) == (45.0, -45.0)
    assert spacemouse_wrist_velocity(
        np.asarray([0.17, -0.17, 0, 0, 0, 0])
    ) == (0.0, 0.0)
    small = spacemouse_wrist_velocity(np.asarray([0.30, -0.30, 0, 0, 0, 0]))
    assert -7.0 < small[0] < -6.0
    assert -7.0 < small[1] < -6.0


def test_collection_mouse_uses_legacy_wrist_and_t_home_button():
    wrist_session = FakeSession()
    wrist_mouse = FakeMouse(buttons={int(Button.ONE): True})
    with ControlledSpaceMouse(wrist_mouse, wrist_session) as controller:
        controller.state()
    assert wrist_session.twists[0][0] == (0.0, 0.0, 0.0, 0.0, 0.0, -0.2)

    home_session = FakeSession()
    home_session.ur5_home_joints = np.ones(6)
    home_mouse = FakeMouse(buttons={int(Button.T): True})
    with ControlledSpaceMouse(home_mouse, home_session) as controller:
        controller.state()
    assert home_session.twists[0][0] == (0.6,) * 6


def test_ur5_operational_margin_replays_boundary_rounding_failure() -> None:
    motion = {
        "translation_speed": 0.107,
        "rotation_speed": 0.54,
        "precision_translation_speed": 0.25,
        "precision_rotation_speed": 0.60,
    }
    speeds, linear_limit, angular_limit = _bounded_speed_settings(motion, 0.25, 0.60)
    rounded_overflow = np.asarray([0.0, 0.0, 0.6000000000000001, 0.0, 0.0, 0.0])
    limits = np.full(6, angular_limit)
    command = _clamp_command(rounded_overflow, -limits, limits)

    assert linear_limit == 0.225
    assert angular_limit == 0.54
    assert speeds.boost_translation == 0.225
    assert speeds.boost_rotation == 0.54
    assert np.max(np.abs(command)) == 0.54
    assert np.max(np.abs(command)) < 0.60


def test_task2_dataset_name_matches_nine_dof_contract(monkeypatch, tmp_path) -> None:
    captured = {}
    contract = type("Contract", (), {"robot_type": "ur5_wrist_gripper_9dof"})()
    monkeypatch.setattr("slai_mi.site_adapter.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("slai_mi.site_adapter.time.strftime", lambda _format: "20260825T190000")
    monkeypatch.setattr(
        "slai_mi.site_adapter.create_configured_dataset",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )

    _configured_dataset(
        {"root": str(tmp_path)}, {"task": {"id": "task2"}}, contract
    )

    assert captured["root"].name == "task2-9dof-20260825T190000"
    assert captured["repo_id"] == "local/task2-9dof-20260825T190000"


def test_station_session_loads_legacy_profile_without_watchdog_changes(monkeypatch):
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    task = yaml.safe_load(Path("configs/tasks/block_into_box.yaml").read_text(encoding="utf-8"))
    calls = {}

    class Driver:
        def __init__(self, **kwargs):
            calls["wuji" if "usb_serial" in kwargs else "ur5"] = kwargs

    monkeypatch.setattr("slai_mi.site_adapter.UR5DriverProcess", Driver)
    monkeypatch.setattr("slai_mi.site_adapter.WujiHandDriverProcess", Driver)
    session = StationSession(hardware, task)
    assert session.control_period_s == 1.0 / 125.0
    assert session.speed_settings.translation == 0.107
    assert session.speed_settings.boost_translation == 0.225
    assert session.speed_settings.boost_rotation == 0.54
    assert session.linear_command_limit == 0.225
    assert session.angular_command_limit == 0.54
    assert session.home_joint_speed == 0.50
    assert session.spacemouse_deadzone == 0.12
    assert session.max_offset_m == 0.0
    assert session.max_rotation_rad == 0.0
    expected = np.asarray(
        yaml.safe_load(
            Path("configs/poses/tasks/block_into_box_start.yaml").read_text(encoding="utf-8")
        )["joint_positions"],
        dtype=float,
    )
    np.testing.assert_array_equal(session.ur5_home_joints, expected[:6])
    np.testing.assert_array_equal(session.wuji_home_joints, expected[6:])
    assert calls["ur5"]["watchdog_s"] == 0.25
    assert calls["wuji"]["watchdog_s"] == 0.5
    assert calls["wuji"]["max_velocity_rad_s"] == 3.0


def test_wrist_strategy_builds_ur5_and_wrist_workers_without_wuji(monkeypatch) -> None:
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    task = yaml.safe_load(Path("configs/tasks/block_into_box.yaml").read_text(encoding="utf-8"))
    hardware["wujihand"]["enabled"] = False
    hardware["cameras"]["enabled"] = False

    class Session:
        control_period_s = 0.008
        spacemouse_deadzone = 0.12

    monkeypatch.setattr("slai_mi.site_adapter.UR5OnlySession", lambda *_args: Session())
    dependencies = make_teleop(hardware, task)

    assert dependencies.required_devices == ("ur5", "wrist_sensor", "spacemouse")
    assert set(dependencies.runtime_factories or {}) == {"ur5-teleop", "wrist-teleop"}
    monkeypatch.setattr(
        "slai_mi.site_adapter.WristMasterSlaveController",
        lambda *_args, **_kwargs: object(),
    )
    wrist_factory = (dependencies.runtime_factories or {})["wrist-teleop"]
    wrist_loop = wrist_factory(hardware, "shared-mouse", None)
    assert isinstance(wrist_loop, NativeWristTeleopLoop)
    assert wrist_loop.mouse == "shared-mouse"


def test_r_button_requests_wrist_only_home_on_one_edge() -> None:
    action, previous = _wrist_gate_action({int(Button.R): True}, {})
    assert action == "home"
    action, previous = _wrist_gate_action({int(Button.R): True}, previous)
    assert action is None
    action, previous = _wrist_gate_action({}, previous)
    assert action is None
    action, _ = _wrist_gate_action({int(Button.R): True}, previous)
    assert action == "home"


def test_wrist_teleop_parks_on_home_or_fit_and_resumes_on_menu(monkeypatch) -> None:
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    stop = threading.Event()

    class Controller:
        def __init__(self, *_args, **_kwargs):
            self.home_requests = 0
            self.park_requests = 0
            self.resume_requests = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def request_home(self):
            self.home_requests += 1

        def request_park(self):
            self.park_requests += 1

        def request_resume(self):
            self.resume_requests += 1

        def check(self):
            return None

    class Mouse:
        def __init__(self):
            self.states = [
                {int(Button.R): True},
                {int(Button.R): True},
                {},
                {int(Button.T): True},
                {int(Button.T): True},
                {},
                {int(Button.FIT): True},
                {},
                {int(Button.FIT): True, int(Button.MENU): True},
                {},
                {int(Button.MENU): True},
            ]

        def state(self):
            buttons = self.states.pop(0)
            if not self.states:
                stop.set()
            return np.zeros(6), buttons

    monkeypatch.setattr("slai_mi.site_adapter.WristMasterSlaveController", Controller)
    loop = NativeWristTeleopLoop(hardware, Mouse())
    loop.run(stop)
    assert loop.controller.home_requests == 1
    assert loop.controller.park_requests == 2
    assert loop.controller.resume_requests == 1


def test_wrist_teleop_writes_data_outside_vendor_symlink() -> None:
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    loop = Wrist2WristTeleopLoop(hardware)
    data_root_index = loop.command.index("--data-root") + 1

    assert loop.command[data_root_index].endswith("slai-manipulation/data/wrist-teleop")
    assert "third_party/02_Python_Client_CLI" not in loop.command[data_root_index]


def test_uncommissioned_task_home_is_rejected(monkeypatch) -> None:
    from slai_mi import site_adapter

    monkeypatch.setattr(
        site_adapter,
        "_load_task_ref",
        lambda _reference: {"name": "test_uncommissioned", "configured": False},
    )
    task = {"start_pose_ref": "test_uncommissioned.yaml", "state_schema": "real_v1"}
    with np.testing.assert_raises_regex(ValueError, "configured must be true"):
        _task_home_joints(task)


def test_collection_camera_path_does_not_construct_retarget_provider(monkeypatch):
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    events = []

    class Capture:
        def __init__(self, _configs, **_kwargs):
            pass

        def start(self):
            events.append("start")

        def read(self, _timeout):
            events.append("read")
            return {}

        def stop(self):
            events.append("stop")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("collection must not construct MediaPipe retargeting")

    monkeypatch.setattr("slai_mi.site_adapter.RealSenseCapture", Capture)
    monkeypatch.setattr(
        "slai_mi.site_adapter.validate_camera_set", lambda configs, **_kwargs: tuple(configs)
    )
    monkeypatch.setattr("slai_mi.site_adapter.WujiRetargetTargetProvider", forbidden)
    with _cameras(hardware):
        pass
    assert events == ["start", "read", "stop"]


def test_real_policy_bridge_uses_schema_and_supervised_writes():
    session = FakeSession()
    bridge = RealPolicyBridge(session, "configs/input_schema.yaml")
    action = np.concatenate((np.full(6, 0.01), np.arange(20) / 100))
    bridge.apply(action)
    np.testing.assert_allclose(session.twists[0][0], np.full(6, 0.01))
    assert session.twists[0][1:] == (0.5, 0.008)
    np.testing.assert_allclose(session.hands[0], np.arange(20) / 100)
    assert session.checks == 1


def test_real_policy_bridge_limits_policy_speed():
    session = FakeSession()
    session.max_rotation_rad = 0.0
    bridge = RealPolicyBridge(session, "configs/input_schema.yaml")
    bridge.apply(np.concatenate((np.ones(6), np.zeros(20))))
    linear, angular = np.split(np.asarray(session.twists[0][0]), 2)
    np.testing.assert_allclose(np.linalg.norm(linear), session.speed_settings.translation)
    np.testing.assert_allclose(np.linalg.norm(angular), session.speed_settings.rotation)
