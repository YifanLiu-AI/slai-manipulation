"""Small Python 3 ctypes adapter for the system libspnav."""

from __future__ import annotations

from ctypes import CDLL, POINTER, Structure, Union, c_int, c_uint, c_void_p, pointer
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2


@dataclass(frozen=True)
class SpnavMotionEvent:
    translation: tuple[int, int, int]
    rotation: tuple[int, int, int]
    period: int


@dataclass(frozen=True)
class SpnavButtonEvent:
    bnum: int
    press: bool


class _MotionEvent(Structure):
    _fields_: ClassVar = [
        ("type", c_int),
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("rz", c_int),
        ("period", c_uint),
        ("data", c_void_p),
    ]


class _ButtonEvent(Structure):
    _fields_: ClassVar = [("type", c_int), ("press", c_int), ("bnum", c_int)]


class _Event(Union):
    _fields_: ClassVar = [("type", c_int), ("motion", _MotionEvent), ("button", _ButtonEvent)]


@lru_cache(maxsize=1)
def _load_library():
    """Load the optional native backend only when it is used."""
    try:
        lib = CDLL("libspnav.so")
    except OSError as exc:
        raise RuntimeError(
            "libspnav is unavailable; install libspnav or select the evdev backend"
        ) from exc
    lib.spnav_open.argtypes = []
    lib.spnav_open.restype = c_int
    lib.spnav_close.argtypes = []
    lib.spnav_close.restype = None
    lib.spnav_poll_event.argtypes = [POINTER(_Event)]
    lib.spnav_poll_event.restype = c_int
    return lib


def open_connection() -> None:
    if _load_library().spnav_open() == -1:
        raise RuntimeError("failed to connect to spacenavd")


def close_connection() -> None:
    _load_library().spnav_close()


def poll_event() -> SpnavMotionEvent | SpnavButtonEvent | None:
    event = _Event()
    if _load_library().spnav_poll_event(pointer(event)) == 0:
        return None
    if event.type == SPNAV_EVENT_MOTION:
        motion = event.motion
        return SpnavMotionEvent(
            translation=(motion.x, motion.y, motion.z),
            rotation=(motion.rx, motion.ry, motion.rz),
            period=int(motion.period),
        )
    if event.type == SPNAV_EVENT_BUTTON:
        button = event.button
        return SpnavButtonEvent(bnum=int(button.bnum), press=bool(button.press))
    raise RuntimeError(f"unknown spnav event type: {event.type}")
