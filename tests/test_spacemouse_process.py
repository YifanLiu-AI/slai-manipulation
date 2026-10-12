from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest

from slai_mi.devices.spacemouse import spnav
from slai_mi.devices.spacemouse.client import SpaceMouseProcess
from slai_mi.devices.spacemouse.device import SpaceMouse


def test_spacemouse_counts_input_events_and_clears_state_on_reconnect() -> None:
    mouse = SpaceMouse(backend="spnav")
    mouse._handle_event(
        spnav.SpnavMotionEvent((100, 0, 0), (0, 0, 0), 0), time.monotonic()
    )
    mouse._handle_event(spnav.SpnavButtonEvent(0, True), time.monotonic())

    diagnostics = mouse.diagnostics()
    assert diagnostics["event_count"] == 2
    assert diagnostics["last_event_time"] is not None

    with mouse._lock:
        mouse._raw_motion.fill(0.0)
        mouse._buttons.clear()
        mouse._pending_button_states.clear()
        mouse._last_motion_time = 0.0
        mouse._last_button_time = 0.0
        mouse._connected = False
        mouse._reconnect_count += 1
    motion, buttons = mouse.state()
    assert np.array_equal(motion, np.zeros(6, dtype=np.float32))
    assert buttons == {}
    assert mouse.diagnostics()["reconnect_count"] == 1


def test_process_zeros_old_generated_motion_without_exiting_during_transient() -> None:
    process = SpaceMouseProcess(rate_hz=250.0)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    parent.setblocking(False)
    process._sock = parent
    child.send(
        json.dumps(
            {
                "type": "state",
                "generated_monotonic_s": time.monotonic() - 0.2,
                "motion": [0.0, 0.0, -0.7, 0.0, 0.0, 0.0],
                "buttons": {},
            }
        ).encode()
    )
    try:
        motion, buttons = process.state()
        np.testing.assert_array_equal(motion, np.zeros(6, dtype=np.float32))
        assert buttons == {}
        assert process._stale_since is not None
    finally:
        process._sock = None
        parent.close()
        child.close()


def test_process_fails_only_after_stale_input_persists_for_one_second(monkeypatch) -> None:
    process = SpaceMouseProcess()
    process._stale_since = time.monotonic() - 1.1
    monkeypatch.setattr(process, "_drain_packets", lambda: None)

    with pytest.raises(RuntimeError, match="remained stale"):
        process.state()


def test_process_diagnostics_does_not_consume_button_edge(monkeypatch) -> None:
    process = SpaceMouseProcess()
    process._buttons = {0: True}
    process._pending_button_states.append({0: True})
    process._packet_count = 3
    process._last_packet_time = time.monotonic()

    monkeypatch.setattr(process, "_drain_packets", lambda: None)
    diagnostics = process.diagnostics()

    assert diagnostics["packet_count"] == 3
    assert process.state()[1] == {0: True}
