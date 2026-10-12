import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from slai_mi.devices.spacemouse import spnav


def test_import_does_not_load_native_library():
    code = """
import ctypes
def unavailable(*args, **kwargs):
    raise OSError("native library deliberately unavailable")
ctypes.CDLL = unavailable
from slai_mi.devices.spacemouse import spnav
assert spnav.SpnavButtonEvent(1, True).press
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_backend_loads_once_and_keeps_its_call_contract(monkeypatch):
    lib = SimpleNamespace(
        spnav_open=Mock(return_value=0),
        spnav_close=Mock(),
        spnav_poll_event=Mock(return_value=0),
    )
    loader = Mock(return_value=lib)
    spnav._load_library.cache_clear()
    monkeypatch.setattr(spnav, "CDLL", loader)
    try:
        spnav.open_connection()
        assert spnav.poll_event() is None
        spnav.close_connection()
        loader.assert_called_once_with("libspnav.so")
        lib.spnav_open.assert_called_once()
        lib.spnav_close.assert_called_once()
    finally:
        spnav._load_library.cache_clear()


def test_missing_native_backend_has_actionable_error(monkeypatch):
    spnav._load_library.cache_clear()
    monkeypatch.setattr(spnav, "CDLL", Mock(side_effect=OSError("missing")))
    with pytest.raises(RuntimeError, match="install libspnav or select the evdev"):
        spnav.open_connection()
