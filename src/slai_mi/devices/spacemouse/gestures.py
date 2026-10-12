"""Stateful physical-button gestures used outside normal teleoperation mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class RepeatedChordGesture:
    """Recognize repeated press/release cycles of one exact button chord."""

    def __init__(
        self,
        buttons: Sequence[int],
        *,
        repeats: int,
        timeout_s: float,
        require_initial_release: bool = False,
    ) -> None:
        if not buttons or repeats < 1 or timeout_s <= 0.0:
            raise ValueError("gesture buttons, repeats, and timeout must be positive")
        self.buttons = frozenset(int(button) for button in buttons)
        self.repeats = int(repeats)
        self.timeout_s = float(timeout_s)
        self._require_initial_release = bool(require_initial_release)
        self._ready_for_press = not self._require_initial_release
        self._count = 0
        self._last_press_at: float | None = None

    def reset(self) -> None:
        self._ready_for_press = not self._require_initial_release
        self._count = 0
        self._last_press_at = None

    def update(self, buttons: Mapping[int, bool], now: float) -> bool:
        pressed = {int(code) for code, value in buttons.items() if value}
        chord_active = self.buttons.issubset(pressed)
        unexpected_button = bool(pressed - self.buttons)

        if self._last_press_at is not None and now - self._last_press_at > self.timeout_s:
            self._count = 0
            self._last_press_at = None

        triggered = False
        if not pressed.intersection(self.buttons):
            self._ready_for_press = True

        if chord_active and self._ready_for_press:
            self._ready_for_press = False
            if unexpected_button:
                self._count = 0
                self._last_press_at = None
            else:
                self._count += 1
                self._last_press_at = now
                if self._count >= self.repeats:
                    triggered = True
                    self._count = 0
                    self._last_press_at = None
        elif unexpected_button:
            self._count = 0
            self._last_press_at = None
            self._ready_for_press = False
        return triggered
