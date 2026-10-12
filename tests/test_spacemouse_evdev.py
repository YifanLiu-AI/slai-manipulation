from __future__ import annotations

import struct

from slai_mi.devices.spacemouse.evdev import EV_KEY, EvdevReader, INPUT_EVENT


def event(event_type: int, code: int, value: int) -> bytes:
    return INPUT_EVENT.pack(0, 0, event_type, code, value)


def test_evdev_maps_linux_button_codes_to_logical_bnums() -> None:
    reader = EvdevReader(__import__("pathlib").Path("/dev/null"))
    reader.feed(event(EV_KEY, 256 + 0, 1))
    reader.feed(event(EV_KEY, 256 + 25, 1))

    first = reader.poll_event()
    second = reader.poll_event()
    assert first is not None and first.bnum == 0 and first.press
    assert second is not None and second.bnum == 25 and second.press


def test_evdev_keeps_logical_codes_unchanged() -> None:
    reader = EvdevReader(__import__("pathlib").Path("/dev/null"))
    reader.feed(event(EV_KEY, 14, 1))
    item = reader.poll_event()
    assert item is not None and item.bnum == 14
