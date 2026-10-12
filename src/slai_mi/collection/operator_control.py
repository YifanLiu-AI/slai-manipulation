"""SpaceMouse episode controls for real demonstration collection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

MENU_BUTTON = 0
FIT_BUTTON = 1
ESC_BUTTON = 22


class EpisodeAction(str, Enum):
    START = "start"
    SAVE = "save"
    DISCARD = "discard"


@dataclass
class SpaceMouseEpisodeControls:
    """Convert Menu/Fit/Esc rising edges into an episode state machine."""

    menu_button: int = MENU_BUTTON
    fit_button: int = FIT_BUTTON
    esc_button: int = ESC_BUTTON
    recording: bool = False
    _previous: dict[int, bool] = field(default_factory=dict, init=False)

    def update(self, buttons: Mapping[int, bool]) -> EpisodeAction | None:
        rising = {
            button: bool(buttons.get(button, False)) and not self._previous.get(button, False)
            for button in (self.menu_button, self.fit_button, self.esc_button)
        }
        self._previous = {
            button: bool(buttons.get(button, False))
            for button in (self.menu_button, self.fit_button, self.esc_button)
        }
        if self.recording:
            if rising[self.esc_button]:
                self.recording = False
                return EpisodeAction.DISCARD
            if rising[self.fit_button]:
                self.recording = False
                return EpisodeAction.SAVE
            return None
        if rising[self.menu_button]:
            self.recording = True
            return EpisodeAction.START
        return None

    def abort(self) -> None:
        self.recording = False

    def synchronize(self, buttons: Mapping[int, bool]) -> None:
        """Consume button levels while lifecycle gates intentionally block actions."""
        self._previous = {
            button: bool(buttons.get(button, False))
            for button in (self.menu_button, self.fit_button, self.esc_button)
        }
