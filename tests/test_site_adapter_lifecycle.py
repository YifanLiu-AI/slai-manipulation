from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.mapping import SpeedSettings
from slai_mi.devices.wujihand.manual_control import ManualHandSettings
from slai_mi.site_adapter import (
    ControlledSpaceMouse,
    RealPolicyBridge,
    StationSession,
    _cameras,
    _task_home_joints,
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


def test_collection_mouse_uses_legacy_ctrl_speed_and_control_period():
    session = FakeSession()
    mouse = FakeMouse(
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {int(Button.CTRL): True},
    )
    with ControlledSpaceMouse(mouse, session) as controller:
        controller.state()
    assert session.twists[0] == ((0.25, 0.0, 0.0, 0.0, 0.0, 0.0), 0.5, 0.008)


def test_collection_mouse_uses_legacy_wrist_and_home_buttons():
    wrist_session = FakeSession()
    wrist_mouse = FakeMouse(buttons={int(Button.ONE): True})
    with ControlledSpaceMouse(wrist_mouse, wrist_session) as controller:
        controller.state()
    assert wrist_session.twists[0][0] == (0.0, 0.0, 0.0, 0.0, 0.0, -0.2)

    home_session = FakeSession()
    home_session.ur5_home_joints = np.ones(6)
    home_mouse = FakeMouse(buttons={int(Button.HOME): True})
    with ControlledSpaceMouse(home_mouse, home_session) as controller:
        controller.state()
    assert home_session.twists[0][0] == (0.6,) * 6


def test_station_session_loads_legacy_profile_without_watchdog_changes(monkeypatch):
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    task = yaml.safe_load(
        Path("configs/tasks/block_into_box.yaml").read_text(encoding="utf-8")
    )
    calls = {}

    class Driver:
        def __init__(self, **kwargs):
            calls["wuji" if "usb_serial" in kwargs else "ur5"] = kwargs

    monkeypatch.setattr("slai_mi.site_adapter.UR5DriverProcess", Driver)
    monkeypatch.setattr("slai_mi.site_adapter.WujiHandDriverProcess", Driver)
    session = StationSession(hardware, task)
    assert session.control_period_s == 1.0 / 125.0
    assert session.speed_settings.translation == 0.08
    assert session.speed_settings.boost_translation == 0.25
    assert session.spacemouse_deadzone == 0.12
    assert session.max_offset_m == 0.0
    assert session.max_rotation_rad == 0.0
    expected = np.asarray(
        yaml.safe_load(
            Path("configs/poses/tasks/block_into_box_start.yaml").read_text(
                encoding="utf-8"
            )
        )["joint_positions"],
        dtype=float,
    )
    np.testing.assert_array_equal(session.ur5_home_joints, expected[:6])
    np.testing.assert_array_equal(session.wuji_home_joints, expected[6:])
    assert calls["ur5"]["watchdog_s"] == 0.25
    assert calls["wuji"]["watchdog_s"] == 0.5
    assert calls["wuji"]["max_velocity_rad_s"] == 3.0


def test_uncommissioned_task_home_is_rejected() -> None:
    task = yaml.safe_load(
        Path("configs/tasks/remove_objects_from_box.yaml").read_text(encoding="utf-8")
    )
    with np.testing.assert_raises_regex(ValueError, "configured must be true"):
        _task_home_joints(task)


def test_collection_camera_path_does_not_construct_retarget_provider(monkeypatch):
    hardware = yaml.safe_load(Path("configs/hardware.yaml").read_text(encoding="utf-8"))
    events = []

    class Capture:
        def __init__(self, _configs):
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
