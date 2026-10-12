"""SpaceMouse button-edge control for the serial gripper."""

from __future__ import annotations

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
