"""SpaceMouse controls for binary and continuous serial grippers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from slai_mi.devices.spacemouse.buttons import Button


class SpaceMouseGripperControl:
    """Physical button 3 (bnum 14) closes; button 4 (bnum 15) opens."""

    def __init__(self, gripper: Any) -> None:
        self.gripper = gripper
        self._previous = {int(Button.THREE): False, int(Button.FOUR): False}

    def update(self, buttons: Mapping[int, bool]) -> int | None:
        three = bool(buttons.get(int(Button.THREE), False))
        four = bool(buttons.get(int(Button.FOUR), False))
        command: int | None = None
        if three and not self._previous[int(Button.THREE)]:
            command = 1
        elif four and not self._previous[int(Button.FOUR)]:
            command = 0
        self._previous = {int(Button.THREE): three, int(Button.FOUR): four}
        if command is not None:
            self.gripper.command(command)
        return command


class ContinuousSpaceMouseGripperControl:
    """Hold button 3 to close gradually and button 4 to open gradually."""

    def __init__(
        self,
        gripper: Any,
        *,
        rate_per_s: float = 1.0,
        command_hz: float = 30.0,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not math.isfinite(rate_per_s) or rate_per_s <= 0.0:
            raise ValueError("continuous gripper rate_per_s must be positive and finite")
        if not math.isfinite(command_hz) or command_hz <= 0.0:
            raise ValueError("continuous gripper command_hz must be positive and finite")
        self.gripper = gripper
        self.rate_per_s = float(rate_per_s)
        self.command_period_s = 1.0 / float(command_hz)
        self._monotonic = monotonic
        self._target: float | None = None
        self._sent_target: float | None = None
        self._last_update_s: float | None = None
        self._last_command_s: float | None = None
        self._direction = 0

    @property
    def target(self) -> float | None:
        return self._target

    def synchronize(self) -> float:
        """Initialize the ramp from the device target after the serial link opens."""
        state = self.gripper.read_state()
        now = float(self._monotonic())
        self._target = float(state.normalized_target_position)
        self._sent_target = self._target
        self._last_update_s = now
        self._last_command_s = now
        self._direction = 0
        return self._target

    def update(self, buttons: Mapping[int, bool]) -> float | None:
        now = float(self._monotonic())
        if self._target is None or self._last_update_s is None:
            self.synchronize()
            return None

        close_pressed = bool(buttons.get(int(Button.THREE), False))
        open_pressed = bool(buttons.get(int(Button.FOUR), False))
        direction = int(close_pressed) - int(open_pressed)
        # Bound one integration interval so a paused control loop cannot cause a jump.
        elapsed_s = min(0.1, max(0.0, now - self._last_update_s))
        self._last_update_s = now
        previous_direction = self._direction
        self._direction = direction
        if direction:
            self._target = min(
                1.0,
                max(0.0, self._target + direction * self.rate_per_s * elapsed_s),
            )

        pending = self._sent_target is None or not math.isclose(
            self._target,
            self._sent_target,
            abs_tol=1e-9,
        )
        interval_elapsed = (
            self._last_command_s is None
            or now - self._last_command_s >= self.command_period_s
        )
        released = direction == 0 and previous_direction != 0
        endpoint = self._target in (0.0, 1.0)
        if not pending or not (interval_elapsed or released or endpoint):
            return None
        self.gripper.command_position(self._target)
        self._sent_target = self._target
        self._last_command_s = now
        return self._target
