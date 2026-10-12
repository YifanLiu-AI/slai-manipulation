from __future__ import annotations

import copy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slai_mi.collection.vla_recorder import (
    SourceSample,
    SynchronizedInputs,
    assemble_configured_frame,
)
from slai_mi.datasets.lerobot_v3.configured import ConfiguredDatasetContract
from slai_mi.devices.wrist_sensor.teleop import (
    COLLECTION_HOME_PROFILE_ACCELERATION_RAW,
    COLLECTION_HOME_PROFILE_VELOCITY_RAW,
    SPACEMOUSE_SOFT_HOME_SPEED_DEG_S,
    WristMasterSlaveController,
    _accept_stopped_home_residual,
    _slew_velocity,
    _wait_home_settled,
    prepare_dynamic_home,
)
from slai_mi.input_schema import load_input_schema, select_transformed_vector, select_vector
from slai_mi.ui.collection_dashboard import CollectionDashboardProvider


def _sample(value, sequence: int = 1) -> SourceSample:
    return SourceSample(value, 10.0, 10.0, sequence)


def test_wrist_collection_frame_is_exactly_8d_and_pi05_view_is_11d() -> None:
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_8dof.yaml")
    contract = ConfiguredDatasetContract(schema)
    ur5 = SimpleNamespace(
        actual_q=np.arange(6, dtype=np.float32),
        actual_tcp_pose=np.zeros(6, dtype=np.float32),
        actual_tcp_speed=np.zeros(6, dtype=np.float32),
        target_qd=np.zeros(6, dtype=np.float32),
        target_tcp_speed=np.arange(6, dtype=np.float32) / 10.0,
    )
    wrist = SimpleNamespace(
        actual_q=np.asarray([0.1, -0.2], dtype=np.float32),
        target_q=np.asarray([0.3, -0.4], dtype=np.float32),
    )
    mouse = SimpleNamespace(
        axes=np.zeros(6, dtype=np.float32), buttons=np.zeros(12, dtype=np.int64)
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = SynchronizedInputs(
        cameras={role: _sample(image) for role in ("primary", "wrist", "secondary")},
        channels={"ur5": _sample(ur5), "wrist": _sample(wrist), "spacemouse": _sample(mouse)},
    )

    frame = assemble_configured_frame(
        inputs,
        "put the block in the box",
        schema=schema,
        validator=contract.validate_frame,
        now=10.0,
    )

    np.testing.assert_allclose(frame["observation.state"], [0, 1, 2, 3, 4, 5, 0.1, -0.2])
    np.testing.assert_allclose(frame["action"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3, -0.4])
    policy_parts = [
        select_transformed_vector(frame[item["key"]], item, f"state[{index}]")
        for index, item in enumerate(schema["pi05"]["state"]["sources"])
    ]
    assert len(np.concatenate(policy_parts)) == 11
    assert len(select_vector(frame["action"], schema["pi05"]["action"], "action")) == 8


def test_wrist_motor_telemetry_is_optional_and_does_not_change_training_vectors() -> None:
    schema = copy.deepcopy(load_input_schema("configs/input_schemas/ur5e_wrist_8dof.yaml"))
    schema["telemetry"] = {"wrist_motor": {"enabled": True}}
    contract = ConfiguredDatasetContract(schema)
    ur5 = SimpleNamespace(
        actual_q=np.arange(6, dtype=np.float32),
        actual_tcp_pose=np.zeros(6, dtype=np.float32),
        actual_tcp_speed=np.zeros(6, dtype=np.float32),
        target_qd=np.zeros(6, dtype=np.float32),
        target_tcp_speed=np.zeros(6, dtype=np.float32),
    )
    wrist = SimpleNamespace(
        actual_q=np.asarray([0.1, -0.2], dtype=np.float32),
        target_q=np.asarray([0.3, -0.4], dtype=np.float32),
        motor_position_raw=np.asarray([3210, 2565], dtype=np.int64),
        motor_goal_raw=np.asarray([3220, 2550], dtype=np.int64),
        motor_rate_raw_s=np.asarray([12, -9], dtype=np.int64),
    )
    mouse = SimpleNamespace(
        axes=np.zeros(6, dtype=np.float32), buttons=np.zeros(12, dtype=np.int64)
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = SynchronizedInputs(
        cameras={role: _sample(image) for role in ("primary", "wrist", "secondary")},
        channels={"ur5": _sample(ur5), "wrist": _sample(wrist), "spacemouse": _sample(mouse)},
    )

    frame = assemble_configured_frame(
        inputs, "test", schema=schema, validator=contract.validate_frame, now=10.0
    )

    assert frame["observation.state"].shape == (8,)
    assert frame["action"].shape == (8,)
    np.testing.assert_array_equal(frame["telemetry.wrist_motor_position_raw"], [3210, 2565])
    np.testing.assert_array_equal(frame["telemetry.wrist_motor_goal_raw"], [3220, 2550])
    np.testing.assert_array_equal(frame["telemetry.wrist_motor_rate_raw_s"], [12, -9])


def test_wrist_controller_can_park_then_rezero_master_and_resume(monkeypatch) -> None:
    commands: list[str] = []
    targets: list[tuple[float, float]] = []

    class Response:
        def __init__(self):
            self.ok = True
            self.fields = {
                "zero_valid": "1",
                "enc0_deg": "1000",
                "enc1_deg": "2000",
            }

    class Teleop:
        def __init__(self, *_args):
            self.sent_frame = False
            self.stop_pending = False

        def command(self, command, _expected, timeout_s=2.0):
            commands.append(command)
            return Response()

        def read_line(self, _timeout):
            if self.stop_pending:
                self.stop_pending = False
                return "OK TELE enc0_deg=1100 enc1_deg=1800 st=0"
            if not self.sent_frame:
                self.sent_frame = True
                return "OK TELE enc0_deg=1100 enc1_deg=1800 st=1"
            time.sleep(0.001)
            return None

        def write_command(self, command):
            commands.append(command)
            if command == "STOP":
                self.stop_pending = True

        def close(self):
            return None

    class Unwrapper:
        def reset(self, value):
            self.value = value

        def update(self, value):
            return value - self.value

    class Filter:
        def __init__(self, **_kwargs):
            pass

        def reset(self, _value):
            pass

        def filter(self, value, _dt):
            return value

    class Shaper:
        def __init__(self, *_args):
            pass

        def reset(self, *_args):
            pass

        def update(self, fe, ru, _dt):
            return fe, ru

    state = SimpleNamespace(
        fe_deg=0.0,
        ru_deg=0.0,
        target_fe_deg=0.0,
        target_ru_deg=0.0,
        j1_present_raw=3210,
        j2_present_raw=2565,
        j1_goal_raw=3210,
        j2_goal_raw=2565,
        j1_rate_raw_s=0,
        j2_rate_raw_s=0,
    )

    prepare_calls = []

    class Controller:
        bounds = object()

        def __init__(self, *_args):
            pass

        def prepare(self, **kwargs):
            prepare_calls.append(kwargs)
            return state

        def stream_target_deg(self, fe, ru):
            targets.append((fe, ru))

        def read_state(self):
            return state

        def shutdown(self, **_kwargs):
            return None

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def connect(self):
            return self

        def close(self):
            return None

    config = SimpleNamespace(
        stream=SimpleNamespace(master_period_ms=10, command_hz=50.0, state_hz=20.0),
        filter=SimpleNamespace(one_euro_min_cutoff=8.0, one_euro_beta=0.15, one_euro_d_cutoff=1.0),
        settling=SimpleNamespace(tolerance_deg=0.3),
    )

    monkeypatch.setattr(
        "slai_mi.devices.wrist_sensor.teleop._vendor_imports",
        lambda: (
            Unwrapper,
            Teleop,
            lambda fields, key: int(fields[key]) / 100.0,
            lambda line: SimpleNamespace(
                ok=True,
                command="TELE",
                fields={"enc0_deg": "1100", "enc1_deg": "1800", "st": "0"},
            ),
            Filter,
            Shaper,
            Controller,
            lambda _path: config,
            lambda enc0, enc1, _config: (-enc1, enc0),
            Client,
            lambda value: value,
            lambda value: value,
        ),
    )
    controller = WristMasterSlaveController(Path("unused.yaml"))
    with controller:
        assert targets == []
        assert controller.state().actual_q.shape == (2,)

        controller.request_resume()
        deadline = time.monotonic() + 1.0
        while not targets and time.monotonic() < deadline:
            time.sleep(0.005)
        assert targets == [(2.0, 1.0)]

        # The ESP32 test stream emits only one TELE line. OpenRB feedback must
        # continue at state_hz instead of freezing the observation timestamp.
        sequence = controller.state().sequence
        deadline = time.monotonic() + 0.25
        while controller.state().sequence == sequence and time.monotonic() < deadline:
            time.sleep(0.005)
        assert controller.state().sequence > sequence

        controller.request_park()
        deadline = time.monotonic() + 1.0
        while commands.count("STOP") < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert commands.count("STOP") == 2

        controller.request_resume()
        deadline = time.monotonic() + 1.0
        while commands.count("START") < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert commands.count("START") == 2

    assert commands[:2] == ["STOP", "SET_PERIOD 10"]
    assert prepare_calls
    assert all(call == {"auto_home_zero": True, "force_home_zero": True} for call in prepare_calls)


def test_spacemouse_wrist_takeover_uses_cached_target_without_serial_read() -> None:
    controller = WristMasterSlaveController.__new__(WristMasterSlaveController)
    controller._lock = threading.Lock()
    controller._spacemouse_velocity = np.zeros(2)
    controller._spacemouse_applied_velocity = np.asarray([3.0, -2.0])
    controller._spacemouse_override = False
    controller._spacemouse_target = np.zeros(2)
    controller._spacemouse_last_update_s = 0.0
    controller._latest = SimpleNamespace(target_q=np.deg2rad(np.asarray([12.5, -4.25])))
    controller._controller = SimpleNamespace(
        read_state=lambda: (_ for _ in ()).throw(AssertionError("serial read raced worker"))
    )
    controller.check = lambda: None

    controller.set_spacemouse_velocity(8.0, -6.0)

    np.testing.assert_array_equal(controller._spacemouse_velocity, [8.0, -6.0])
    np.testing.assert_allclose(controller._spacemouse_target, [12.5, -4.25])
    np.testing.assert_array_equal(controller._spacemouse_applied_velocity, [0.0, 0.0])
    assert controller._spacemouse_override


def test_releasing_ctrl_holds_last_spacemouse_wrist_target() -> None:
    controller = WristMasterSlaveController.__new__(WristMasterSlaveController)
    controller._lock = threading.Lock()
    controller._spacemouse_velocity = np.asarray([8.0, -6.0])
    controller._spacemouse_applied_velocity = np.asarray([7.0, -5.0])
    controller._spacemouse_override = True
    controller._spacemouse_target = np.asarray([23.5, -11.25])

    controller.clear_spacemouse_velocity()

    np.testing.assert_array_equal(controller._spacemouse_velocity, [0.0, 0.0])
    np.testing.assert_array_equal(controller._spacemouse_target, [23.5, -11.25])
    np.testing.assert_array_equal(controller._spacemouse_applied_velocity, [0.0, 0.0])
    assert controller._spacemouse_override


def test_spacemouse_wrist_velocity_slew_has_no_instant_step_or_overshoot() -> None:
    current = np.zeros(2)
    desired = np.asarray([45.0, -45.0])

    first = _slew_velocity(current, desired, 0.02)
    second = _slew_velocity(first, desired, 0.02)
    reversed_velocity = _slew_velocity(second, -desired, 0.02)

    np.testing.assert_allclose(first, [6.0, -6.0])
    np.testing.assert_allclose(second, [12.0, -12.0])
    np.testing.assert_allclose(reversed_velocity, [2.0, -2.0])


def _settling_config() -> SimpleNamespace:
    return SimpleNamespace(
        stream=SimpleNamespace(max_state_age_s=0.20),
        settling=SimpleNamespace(
            tolerance_deg=0.30,
            home_acceptance_tolerance_deg=0.60,
        ),
    )


def _home_state(**overrides) -> SimpleNamespace:
    values = {
        "active": True,
        "fault": False,
        "zero_valid": True,
        "sensor_age_s": 0.01,
        "target_fe_deg": 0.0,
        "target_ru_deg": 0.0,
        "j1_rate_raw_s": 0,
        "j2_rate_raw_s": 0,
        "fe_deg": -0.02,
        "ru_deg": -0.48,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_home_accepts_small_stopped_mechanical_residual_after_precise_timeout() -> None:
    state = _home_state()
    controller = SimpleNamespace(
        config=_settling_config(),
        wait_settled=lambda *_args: (_ for _ in ()).throw(TimeoutError("precise timeout")),
        read_state=lambda: state,
    )

    assert _accept_stopped_home_residual(state, controller.config)
    assert _wait_home_settled(controller) is state


@pytest.mark.parametrize(
    "state",
    [
        _home_state(ru_deg=-0.80),
        _home_state(j2_rate_raw_s=1),
        _home_state(fault=True),
        _home_state(sensor_age_s=0.50),
        _home_state(target_ru_deg=1.0),
    ],
)
def test_home_rejects_large_moving_faulted_stale_or_nonzero_target(state) -> None:
    controller = SimpleNamespace(
        config=_settling_config(),
        wait_settled=lambda *_args: (_ for _ in ()).throw(TimeoutError("precise timeout")),
        read_state=lambda: state,
    )

    assert not _accept_stopped_home_residual(state, controller.config)
    with pytest.raises(TimeoutError, match="precise timeout"):
        _wait_home_settled(controller)


def test_soft_home_is_feedback_anchored_and_acceleration_limited(monkeypatch) -> None:
    commands: list[tuple[float, float]] = []
    feedback = iter([(2.0, -1.0), (1.9, -0.9), (0.1, -0.1), (0.0, 0.0)])

    class Controller:
        config = SimpleNamespace(settling=SimpleNamespace(timeout_s=2.0))

        def read_state(self):
            fe, ru = next(feedback)
            return SimpleNamespace(
                fe_deg=fe,
                ru_deg=ru,
                target_fe_deg=fe,
                target_ru_deg=ru,
            )

        def stream_target_deg(self, fe, ru):
            commands.append((fe, ru))

    wrist = WristMasterSlaveController.__new__(WristMasterSlaveController)
    wrist._stop = threading.Event()
    wrist._lock = threading.Lock()
    wrist._spacemouse_target = np.asarray([2.0, -1.0])
    wrist._spacemouse_velocity = np.ones(2)
    wrist._spacemouse_applied_velocity = np.ones(2)
    wrist._spacemouse_override = True
    wrist._store_state = lambda _state: None
    monkeypatch.setattr(
        "slai_mi.devices.wrist_sensor.teleop._wait_home_settled",
        lambda controller, callback: controller.read_state(),
    )

    result = wrist._run_soft_home(Controller(), command_period=0.02)

    # At 20 deg/s² and 20 ms, the first velocity is only 0.4 deg/s and
    # the target leads measured feedback by 0.008 degree, not by a distant
    # accumulated position ramp. The second step accelerates to 0.8 deg/s.
    np.testing.assert_allclose(commands[0], [1.992, -0.992])
    np.testing.assert_allclose(commands[1], [1.884, -0.884])
    np.testing.assert_allclose(commands[-1], [0.0, 0.0])
    assert result.fe_deg == 0.0 and result.ru_deg == 0.0
    np.testing.assert_array_equal(wrist._spacemouse_target, [0.0, 0.0])
    np.testing.assert_array_equal(wrist._spacemouse_velocity, [0.0, 0.0])
    assert not wrist._spacemouse_override


def test_dynamic_home_uses_post_home_encoder_reading_as_runtime_zero() -> None:
    calls = []

    class Response:
        ok = True

        def __init__(self, fields=None):
            self.fields = fields or {}
            self.raw_lines = ["ok"]

    class Client:
        def set_output_closed_loop_enable(self, enabled):
            calls.append(("closed_loop", enabled))
            return Response()

        def set_output_direct_config(self, *_args):
            calls.append(("direct_config",))
            return Response()

        def home_all(self):
            calls.append(("home_all",))
            return Response()

        def read_wrist_state(self):
            calls.append(("read_wrist_state",))
            return Response({"enc0_deg": "6493", "enc1_deg": "16488"})

        def set_output_zero_absolute_cdeg(self, enc0, enc1):
            calls.append(("set_zero", enc0, enc1))
            return Response()

        def set_arm_profile(self, velocity, acceleration):
            calls.append(("arm_profile", velocity, acceleration))
            return Response()

        def set_output_target_cdeg(self, fe, ru):
            calls.append(("target", fe, ru))
            return Response()

    gains = SimpleNamespace(
        fe_kp_milli=1,
        fe_ki_milli=2,
        ru_kp_milli=3,
        ru_ki_milli=4,
        max_motor_rate_raw_s=5,
        deadband_cdeg=6,
        fe_velocity_ff_raw_per_deg=7,
        ru_velocity_ff_raw_per_deg=8,
        fe_moving_feedback_limit_raw=9,
        ru_moving_feedback_limit_raw=10,
        motor_profile_velocity_raw=11,
        motor_profile_acceleration_raw=12,
    )

    class Controller:
        client = Client()
        config = SimpleNamespace(controller=gains)

        def inspect(self):
            return Response({"active": "1", "vmode": "0"})

        def _wait_motion_idle(self):
            calls.append(("wait_idle",))

        def wait_settled(self, fe, ru):
            calls.append(("wait_settled", fe, ru))
            return SimpleNamespace(fe_deg=fe, ru_deg=ru)

    state, baseline = prepare_dynamic_home(Controller())
    assert baseline == (64.93, 164.88)
    assert state.fe_deg == 0.0 and state.ru_deg == 0.0
    gentle_profile = (
        "arm_profile",
        COLLECTION_HOME_PROFILE_VELOCITY_RAW,
        COLLECTION_HOME_PROFILE_ACCELERATION_RAW,
    )
    regular_profile = ("arm_profile", 11, 12)
    assert calls.index(gentle_profile) < calls.index(("home_all",))
    assert calls.index(("home_all",)) < calls.index(("read_wrist_state",))
    assert calls.index(("set_zero", 6493, 16488)) < calls.index(regular_profile)
    assert calls.index(gentle_profile) < calls.index(regular_profile)
    assert calls[-1] == ("wait_settled", 0.0, 0.0)


def test_collection_dashboard_accepts_wrist_group_without_wuji() -> None:
    schema_path = "configs/input_schemas/ur5e_wrist_8dof.yaml"
    provider = CollectionDashboardProvider(
        {
            "input_schema": schema_path,
            "ur5": {"enabled": True},
            "wujihand": {"enabled": False},
            "wrist_sensor": {"enabled": True},
            "cameras": {
                "enabled": True,
                "devices": [
                    {"role": "primary", "serial": "1"},
                    {"role": "wrist", "serial": "2"},
                    {"role": "secondary", "serial": "3"},
                ],
            },
        },
        "task",
    )
    now = time.monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = SynchronizedInputs(
        cameras={
            role: SourceSample(image, now, now, index)
            for index, role in enumerate(("primary", "wrist", "secondary"))
        },
        channels={
            "ur5": SourceSample(
                SimpleNamespace(actual_q=np.zeros(6, dtype=np.float32)), now, now, 1
            ),
            "wrist": SourceSample(
                SimpleNamespace(actual_q=np.zeros(2, dtype=np.float32)), now, now, 1
            ),
            "spacemouse": SourceSample(
                SimpleNamespace(
                    axes=np.zeros(6, dtype=np.float32),
                    buttons=np.zeros(12, dtype=np.int64),
                ),
                now,
                now,
                1,
            ),
        },
    )

    provider.observe_inputs(inputs)

    status = provider.status()
    assert status["devices"]["ur5"]["state"] == "active"
    assert status["devices"]["wrist"]["state"] == "active"
    assert status["devices"]["wuji"]["state"] == "inactive"
    assert len(status["dof"]["values"]) == 8
