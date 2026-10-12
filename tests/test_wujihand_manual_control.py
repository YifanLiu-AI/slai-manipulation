from __future__ import annotations

import numpy as np

from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.wujihand.manual_control import (
    AccelerationLimitedTrajectory,
    Button3GraspControl,
    ManualHandSettings,
    ManualWujiController,
    combine_grasp_modes,
    fr_grasp_mode,
    thumb_group_mode,
)


def test_button3_close_then_double_press_open_matches_legacy() -> None:
    control = Button3GraspControl()
    assert control.update(True, 1.0) == "close"
    assert control.update(False, 1.1) is None
    assert control.update(True, 1.2) == "open"
    assert control.update(False, 1.3) is None
    assert control.update(True, 2.0) == "close"


def test_independent_fr_and_thumb_modes_match_legacy() -> None:
    assert fr_grasp_mode({int(Button.F): True}) == "close"
    assert fr_grasp_mode({int(Button.R): True}) == "open"
    assert fr_grasp_mode({int(Button.F): True, int(Button.R): True}) is None
    assert combine_grasp_modes("close", "open") == (None, "conflict")
    assert thumb_group_mode({int(Button.ROLL_CW): True}) == "toward_state_1"
    assert thumb_group_mode({int(Button.T): True}) == "restore_state_0"


def test_trajectory_applies_acceleration_and_stopping_limits() -> None:
    trajectory = AccelerationLimitedTrajectory(
        np.zeros(20), np.full(20, -2.0), np.full(20, 2.0), 1.0
    )
    first = trajectory.step(
        np.ones(20), 1.01, max_speed=1.5, max_acceleration=0.75
    )
    assert np.allclose(first, 0.000075)
    assert np.allclose(trajectory.command_velocity, 0.0075)
    second = trajectory.step(
        np.ones(20), 1.02, max_speed=1.5, max_acceleration=0.75
    )
    assert np.all(second > first)
    assert np.all(trajectory.command_velocity <= 0.0150001)


class _Session:
    def __init__(self) -> None:
        self.wuji = self
        self.commands: list[np.ndarray] = []

    def read_positions(self):
        return np.zeros(20)

    def write_wuji_positions(self, value) -> None:
        self.commands.append(np.asarray(value))


def test_manual_controller_uses_30hz_and_priority_close_speed() -> None:
    session = _Session()
    controller = ManualWujiController(
        session,
        open_target=np.zeros(20),
        grasp_target=np.ones(20),
        home_target=np.zeros(20),
        lower=np.full(20, -2.0),
        upper=np.full(20, 2.0),
        settings=ManualHandSettings(),
        timestamp=1.0,
    )
    buttons = {int(Button.THREE): True}
    assert controller.update(buttons, 1.001)
    assert controller.update(buttons, 1.02)
    assert len(session.commands) == 1
    assert controller.update(buttons, 1.04)
    assert len(session.commands) == 2
    assert session.commands[-1][8] > session.commands[-1][0]


def test_home_uses_task_zero_independently_of_open_preset() -> None:
    session = _Session()
    controller = ManualWujiController(
        session,
        open_target=np.zeros(20),
        grasp_target=np.ones(20),
        home_target=np.full(20, 0.5),
        lower=np.full(20, -2.0),
        upper=np.full(20, 2.0),
        settings=ManualHandSettings(release_speed=3.0, release_acceleration=20.0),
        timestamp=1.0,
    )
    assert controller.update({int(Button.HOME): True}, 1.04)
    assert np.all(session.commands[-1] > 0.0)
    np.testing.assert_array_equal(controller.home_target, np.full(20, 0.5))
