"""Legacy-compatible SpaceMouse control for the WujiHand."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from slai_mi.devices.spacemouse.buttons import Button

MIDDLE_RING_LITTLE_JOINTS = slice(8, 20)
PRIORITY_CLOSE_SPEED_MULTIPLIER = 2.0
FR_PRIORITY_CLOSE_SPEED_MULTIPLIER = 2.5


def _joint_vector(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError(f"expected 20 finite joint values, got {result.shape}")
    return result


def _joint_limit(value: float | Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(20, float(result), dtype=float)
    if result.shape != (20,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must be a positive scalar or 20-vector")
    return result


def _stopping_speed_limit(
    distance: np.ndarray,
    max_speed: np.ndarray,
    max_acceleration: np.ndarray,
    elapsed: float,
) -> np.ndarray:
    """Legacy discrete stopping-distance bound."""
    velocity_step = max_acceleration * elapsed
    lower = np.zeros_like(distance)
    upper = max_speed.copy()
    for _ in range(24):
        velocity = (lower + upper) * 0.5
        full_steps = np.floor(velocity / velocity_step)
        remainder = velocity - full_steps * velocity_step
        stopping_distance = elapsed * (
            (full_steps + 1.0) * remainder
            + velocity_step * full_steps * (full_steps + 1.0) * 0.5
        )
        can_stop = stopping_distance <= distance
        lower = np.where(can_stop, velocity, lower)
        upper = np.where(can_stop, upper, velocity)
    return lower


class AccelerationLimitedTrajectory:
    """Port of the verified legacy Wuji command trajectory."""

    def __init__(
        self,
        position: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        timestamp: float,
    ) -> None:
        self.lower = _joint_vector(lower)
        self.upper = _joint_vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Wuji lower limits must be below upper limits")
        self.command = np.clip(_joint_vector(position), self.lower, self.upper)
        self.command_velocity = np.zeros(20, dtype=float)
        self._last_send = float(timestamp)

    def step(
        self,
        requested: Sequence[float],
        timestamp: float,
        *,
        max_speed: float | Sequence[float],
        max_acceleration: float | Sequence[float],
    ) -> np.ndarray:
        target = np.clip(_joint_vector(requested), self.lower, self.upper)
        speed = _joint_limit(max_speed, "Wuji speed limit")
        acceleration = _joint_limit(max_acceleration, "Wuji acceleration limit")
        elapsed = min(max(float(timestamp) - self._last_send, 0.001), 0.1)
        error = target - self.command
        desired_speed = _stopping_speed_limit(np.abs(error), speed, acceleration, elapsed)
        desired_velocity = np.sign(error) * desired_speed
        max_velocity_change = acceleration * elapsed
        next_velocity = self.command_velocity + np.clip(
            desired_velocity - self.command_velocity,
            -max_velocity_change,
            max_velocity_change,
        )
        self.command = np.clip(
            self.command + next_velocity * elapsed,
            self.lower,
            self.upper,
        )
        self.command_velocity = next_velocity
        self._last_send = float(timestamp)
        return self.command.copy()

    def hold(self, timestamp: float) -> np.ndarray:
        self.command_velocity.fill(0.0)
        self._last_send = float(timestamp)
        return self.command.copy()


@dataclass
class Button3GraspControl:
    """Hold Button 3 to close; quick-click then hold it to open."""

    double_click_interval: float = 0.65
    mode: str | None = None
    _pressed_last: bool = False
    _press_started: float | None = None
    _open_armed_at: float | None = None

    def update(self, pressed: bool, now: float, *, home_active: bool = False) -> str | None:
        if home_active:
            pressed = False
        if pressed and not self._pressed_last:
            second_press = (
                self._open_armed_at is not None
                and now - self._open_armed_at <= self.double_click_interval
            )
            self.mode = "open" if second_press else "close"
            self._open_armed_at = None
            self._press_started = now
        elif not pressed and self._pressed_last:
            held_for = (
                now - self._press_started
                if self._press_started is not None
                else float("inf")
            )
            if self.mode == "close" and held_for <= self.double_click_interval:
                self._open_armed_at = now
            else:
                self._open_armed_at = None
            self.mode = None
            self._press_started = None
        elif (
            not pressed
            and self._open_armed_at is not None
            and now - self._open_armed_at > self.double_click_interval
        ):
            self._open_armed_at = None
        self._pressed_last = pressed
        return self.mode


def fr_grasp_mode(buttons: Mapping[int, bool], *, home_active: bool = False) -> str | None:
    close_pressed = bool(buttons.get(int(Button.F), False))
    open_pressed = bool(buttons.get(int(Button.R), False))
    if home_active or close_pressed == open_pressed:
        return None
    return "close" if close_pressed else "open"


def thumb_group_mode(
    buttons: Mapping[int, bool], *, home_active: bool = False
) -> str | None:
    toward_pressed = bool(buttons.get(int(Button.ROLL_CW), False))
    restore_pressed = bool(buttons.get(int(Button.T), False))
    if home_active or toward_pressed == restore_pressed:
        return None
    return "toward_state_1" if toward_pressed else "restore_state_0"


def combine_grasp_modes(
    button3_mode: str | None, fr_mode: str | None
) -> tuple[str | None, str | None]:
    if button3_mode is None:
        return fr_mode, "fr" if fr_mode is not None else None
    if fr_mode is None:
        return button3_mode, "button3"
    if button3_mode == fr_mode:
        return button3_mode, "button3+fr"
    return None, "conflict"


@dataclass(frozen=True)
class ManualHandSettings:
    command_hz: float = 30.0
    grasp_speed: float = 0.25
    grasp_acceleration: float = 0.75
    release_speed: float = 3.0
    release_acceleration: float = 20.0


class ManualWujiController:
    """Apply the legacy SpaceMouse hand mapping at the legacy command cadence."""

    def __init__(
        self,
        session,
        *,
        open_target: Sequence[float],
        grasp_target: Sequence[float],
        home_target: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        settings: ManualHandSettings,
        timestamp: float,
    ) -> None:
        self.session = session
        self.open_target = _joint_vector(open_target)
        self.grasp_target = _joint_vector(grasp_target)
        self.home_target = _joint_vector(home_target)
        self.settings = settings
        self.trajectory = AccelerationLimitedTrajectory(
            session.wuji.read_positions(), lower, upper, timestamp
        )
        self.grasp_control = Button3GraspControl()
        self.next_command_at = float(timestamp)

    def update(
        self,
        buttons: Mapping[int, bool],
        now: float,
        *,
        hold_when_inactive: bool = True,
    ) -> bool:
        home_active = bool(buttons.get(int(Button.HOME), False))
        button3 = self.grasp_control.update(
            bool(buttons.get(int(Button.THREE), False)), now, home_active=home_active
        )
        fr_mode = fr_grasp_mode(buttons, home_active=home_active)
        mode, source = combine_grasp_modes(button3, fr_mode)
        thumb_mode = thumb_group_mode(buttons, home_active=home_active)
        if thumb_mode is not None and (mode is not None or source == "conflict"):
            thumb_mode = None
        manual_active = bool(
            home_active
            or thumb_mode is not None
            or mode is not None
            or source == "conflict"
        )
        if not manual_active and not hold_when_inactive:
            return False
        if now < self.next_command_at:
            return manual_active or hold_when_inactive

        speed: float | np.ndarray = self.settings.grasp_speed
        acceleration: float | np.ndarray = self.settings.grasp_acceleration
        if home_active:
            target = self.home_target
            speed = self.settings.release_speed
            acceleration = self.settings.release_acceleration
        elif thumb_mode == "toward_state_1":
            target = self.grasp_target
        elif thumb_mode == "restore_state_0":
            target = self.open_target
        elif mode == "close":
            target = self.grasp_target
            speed = np.full(20, self.settings.grasp_speed)
            multiplier = (
                FR_PRIORITY_CLOSE_SPEED_MULTIPLIER
                if source in {"fr", "button3+fr"}
                else PRIORITY_CLOSE_SPEED_MULTIPLIER
            )
            speed[MIDDLE_RING_LITTLE_JOINTS] = min(
                2.0, self.settings.grasp_speed * multiplier
            )
            acceleration = np.full(20, self.settings.grasp_acceleration)
            acceleration[MIDDLE_RING_LITTLE_JOINTS] = min(
                10.0,
                self.settings.grasp_acceleration * PRIORITY_CLOSE_SPEED_MULTIPLIER,
            )
        elif mode == "open":
            target = self.open_target
        else:
            command = self.trajectory.hold(now)
            self.session.write_wuji_positions(command)
            self.next_command_at = now + 1.0 / self.settings.command_hz
            return True

        command = self.trajectory.step(
            target,
            now,
            max_speed=speed,
            max_acceleration=acceleration,
        )
        self.session.write_wuji_positions(command)
        self.next_command_at = now + 1.0 / self.settings.command_hz
        return True
