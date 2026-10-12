import numpy as np

from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.device import deadzone_vector, normalize_motion
from slai_mi.devices.spacemouse.mapping import (
    MotionMode,
    SpeedProfile,
    SpeedSettings,
    build_hardware_twist,
    select_speed_limits,
    wrist_3_jog_direction,
)


def test_normalize_motion_applies_deadzone_and_z_up_transform() -> None:
    motion = normalize_motion(
        [500.0, 250.0, 25.0, -500.0, 0.0, 250.0],
        500.0,
        deadzone_vector(0.1),
    )

    np.testing.assert_allclose(motion, [0.0, 1.0, 0.5, -0.5, -1.0, 0.0])


def test_shift_isolates_base_rotation_and_ctrl_selects_boost() -> None:
    settings = SpeedSettings()
    buttons = {int(Button.SHIFT): True, int(Button.CTRL): True}
    speed = select_speed_limits(buttons, settings)
    command = build_hardware_twist(np.ones(6), buttons, speed)

    assert speed.profile is SpeedProfile.BOOST
    assert command.mode is MotionMode.ROTATION_BASE
    np.testing.assert_array_equal(command.twist[:3], np.zeros(3))
    np.testing.assert_allclose(command.twist[3:], settings.boost_rotation)


def test_shift_preserves_diffusion_policy_transformed_rotation_axes() -> None:
    settings = SpeedSettings(rotation=1.0)
    command = build_hardware_twist(
        np.array([0.0, 0.0, 0.0, -0.4, 0.6, 0.2]),
        {int(Button.SHIFT): True},
        select_speed_limits({}, settings),
    )

    # Input has already passed the Diffusion Policy Z-up transform. Shift sends
    # those base-frame axes directly, without a second TCP-frame rotation.
    assert command.mode is MotionMode.ROTATION_BASE
    np.testing.assert_allclose(command.twist, [0.0, 0.0, 0.0, -0.4, 0.6, 0.2])


def test_conflicting_wrist_buttons_cancel_jog() -> None:
    assert wrist_3_jog_direction({int(Button.ONE): True}) == -1
    assert wrist_3_jog_direction({int(Button.TWO): True}) == 1
    assert wrist_3_jog_direction({int(Button.ONE): True, int(Button.TWO): True}) == 0
