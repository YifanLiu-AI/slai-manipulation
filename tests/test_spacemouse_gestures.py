from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.gestures import RepeatedChordGesture

CHORD = {int(Button.MENU): True, int(Button.FIT): True}


def test_three_menu_fit_press_cycles_trigger_once() -> None:
    gesture = RepeatedChordGesture(CHORD, repeats=2, timeout_s=5.0)

    assert not gesture.update(CHORD, 0.0)
    assert not gesture.update(CHORD, 0.1)
    assert not gesture.update({}, 0.2)
    assert gesture.update(CHORD, 0.3)
    assert not gesture.update(CHORD, 0.4)


def test_menu_fit_gesture_resets_after_timeout_or_other_button() -> None:
    gesture = RepeatedChordGesture(CHORD, repeats=2, timeout_s=1.0)

    assert not gesture.update(CHORD, 0.0)
    assert not gesture.update({}, 0.1)
    assert not gesture.update(CHORD, 1.2)
    assert not gesture.update({}, 1.3)
    assert not gesture.update({**CHORD, int(Button.HOME): True}, 1.4)
    assert not gesture.update({}, 1.5)
    assert not gesture.update(CHORD, 1.6)
    assert not gesture.update({}, 1.7)
    assert gesture.update(CHORD, 1.8)


def test_both_buttons_must_be_released_between_counts() -> None:
    gesture = RepeatedChordGesture(CHORD, repeats=2, timeout_s=5.0)

    assert not gesture.update(CHORD, 0.0)
    assert not gesture.update({int(Button.MENU): True}, 0.1)
    assert not gesture.update(CHORD, 0.2)
    assert not gesture.update({}, 0.3)
    assert gesture.update(CHORD, 0.4)


def test_initially_held_chord_is_not_counted_when_release_is_required() -> None:
    gesture = RepeatedChordGesture(
        CHORD,
        repeats=2,
        timeout_s=5.0,
        require_initial_release=True,
    )

    assert not gesture.update(CHORD, 0.0)
    assert not gesture.update({}, 0.1)
    assert not gesture.update(CHORD, 0.2)
    assert not gesture.update({}, 0.3)
    assert gesture.update(CHORD, 0.4)
