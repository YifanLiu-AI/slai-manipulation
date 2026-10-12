"""SpaceMouse Pro buttons emitted by the installed Debian spacenavd 0.7.1."""

from enum import IntEnum


class Button(IntEnum):
    """Physical controls used by teleoperation after evdev code minus BTN_0."""

    MENU = 0
    FIT = 1
    T = 2
    TOP = T  # Backward-compatible alias for the previously incorrect label.
    R = 4
    F = 5
    ROLL_CW = 8
    ONE = 12
    TWO = 13
    THREE = 14
    FOUR = 15
    HOME = FOUR  # Backward-compatible name for the physical 4 key.
    ESC = 22
    ALT = 23
    SHIFT = 24
    CTRL = 25
    ROTATION_LOCK = 26


BUTTON_NAME_BY_CODE = {
    int(Button.MENU): "menu",
    int(Button.FIT): "fit",
    int(Button.T): "t",
    int(Button.R): "rear",
    int(Button.F): "front",
    int(Button.ROLL_CW): "roll_cw",
    int(Button.ONE): "one",
    int(Button.TWO): "two",
    int(Button.THREE): "three",
    int(Button.HOME): "home",
    int(Button.ESC): "esc",
    int(Button.ALT): "alt",
    int(Button.SHIFT): "shift",
    int(Button.CTRL): "ctrl",
    int(Button.ROTATION_LOCK): "rotation_lock",
}

# Preserve the established 12-value LeRobot feature contract. Extra physical
# buttons remain observable without changing existing dataset dimensions.
RECORDED_BUTTONS = (
    Button.MENU,
    Button.FIT,
    Button.R,
    Button.F,
    Button.ONE,
    Button.TWO,
    Button.THREE,
    Button.HOME,
    Button.ESC,
    Button.ALT,
    Button.SHIFT,
    Button.CTRL,
)
RECORDED_BUTTON_NAMES = (
    "menu",
    "fit",
    "r",
    "f",
    "one",
    "two",
    "three",
    "home",
    "esc",
    "alt",
    "shift",
    "ctrl",
)


SPEED_SHIFT_BUTTON = int(Button.SHIFT)
DIFFUSION_SPEED_CTRL_BUTTON = int(Button.CTRL)
