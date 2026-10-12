"""Pure SpaceMouse button gating and velocity mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .buttons import Button


class MotionMode(str, Enum):
    TRANSLATION_XYZ = "translation_xyz"
    ROTATION_BASE = "rotation_base"
    WRIST_3_JOINT = "wrist_3_joint"


class SpeedProfile(str, Enum):
    TRAINING = "training"
    BOOST = "boost"


@dataclass(frozen=True)
class SpeedLimits:
    translation: float
    rotation: float
    profile: SpeedProfile
    limit_vector_norm: bool


@dataclass(frozen=True)
class SpeedSettings:
    translation: float = 0.080
    rotation: float = 0.45
    boost_translation: float = 0.25
    boost_rotation: float = 0.60


@dataclass(frozen=True)
class MotionCommand:
    twist: np.ndarray
    mode: MotionMode


def _vector6(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid six-axis SpaceMouse motion: {vector}")
    return vector


def _pressed(buttons: Mapping[int, bool], button: Button) -> bool:
    return bool(buttons.get(int(button), False))


def _limit_unit_norm(vector: np.ndarray, enabled: bool) -> np.ndarray:
    result = vector.copy()
    norm = float(np.linalg.norm(result))
    if enabled and norm > 1.0:
        result /= norm
    return result


def select_speed_limits(
    buttons: Mapping[int, bool],
    settings: SpeedSettings,
) -> SpeedLimits:
    """Ctrl selects the high-speed profile without changing axis gating."""
    if _pressed(buttons, Button.CTRL):
        return SpeedLimits(
            settings.boost_translation,
            settings.boost_rotation,
            SpeedProfile.BOOST,
            False,
        )
    return SpeedLimits(
        settings.translation,
        settings.rotation,
        SpeedProfile.TRAINING,
        True,
    )


def build_hardware_twist(
    motion: np.ndarray,
    buttons: Mapping[int, bool],
    speed: SpeedLimits,
) -> MotionCommand:
    """Map the cap to isolated base-frame XYZ translation or rotation."""
    cap = _vector6(motion)
    twist = np.zeros(6, dtype=np.float64)
    if _pressed(buttons, Button.SHIFT):
        angular = _limit_unit_norm(cap[3:], speed.limit_vector_norm)
        twist[3:] = angular * speed.rotation
        return MotionCommand(twist, MotionMode.ROTATION_BASE)

    linear = _limit_unit_norm(cap[:3], speed.limit_vector_norm)
    twist[:3] = linear * speed.translation
    return MotionCommand(twist, MotionMode.TRANSLATION_XYZ)


def wrist_3_jog_direction(buttons: Mapping[int, bool]) -> int:
    """Return -1 for clockwise, +1 for counterclockwise, or 0 for neither/conflict."""
    clockwise = _pressed(buttons, Button.ONE)
    counterclockwise = _pressed(buttons, Button.TWO)
    return int(counterclockwise) - int(clockwise)


def required_released_buttons() -> tuple[int, ...]:
    return tuple(int(button) for button in Button)
