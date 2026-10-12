"""Small Tk operator display for live SpaceMouse axes and mode state."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Self

import numpy as np

from .buttons import Button


class SpaceMouseMonitor:
    """Render normalized six-axis input without owning the input device."""

    _AXES = ("X", "Y", "Z", "Rx", "Ry", "Rz")
    _COLORS = ("#d1495b", "#2a9d6f", "#2878b5", "#9b5de5", "#e07a2d", "#257f8f")

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._root = tk.Tk()
        self._root.title("SpaceMouse Live Input")
        self._root.geometry("680x390")
        self._root.minsize(680, 390)
        self._root.protocol("WM_DELETE_WINDOW", self._request_close)

        self._status = tk.StringVar(value="WAITING FOR INPUT")
        tk.Label(self._root, textvariable=self._status, font=("Sans", 15, "bold")).pack(
            pady=(14, 4)
        )
        self._mode = tk.StringVar(value="mode=translation_xyz  speed=training")
        tk.Label(self._root, textvariable=self._mode, font=("Monospace", 11)).pack(pady=(0, 8))

        self._canvas = tk.Canvas(
            self._root,
            width=640,
            height=252,
            bg="#f5f6f7",
            highlightthickness=1,
            highlightbackground="#c7cbd1",
        )
        self._canvas.pack(padx=20)
        self._bars: list[int] = []
        for index, (label, color) in enumerate(zip(self._AXES, self._COLORS, strict=True)):
            y = 22 + index * 38
            self._canvas.create_text(30, y, text=label, font=("Monospace", 11, "bold"))
            self._canvas.create_line(86, y, 604, y, fill="#b8bec6", width=2)
            self._canvas.create_line(345, y - 10, 345, y + 10, fill="#626a73", width=1)
            bar = self._canvas.create_rectangle(345, y - 7, 345, y + 7, fill=color, width=0)
            self._bars.append(bar)
            self._canvas.create_text(626, y, text="0.00", tags=(f"value-{index}",))

        self._buttons = tk.StringVar(
            value=(
                "1/CW: up    2/CCW: up    T: up    "
                "3/4: gripper placeholder    Shift: up"
            )
        )
        tk.Label(self._root, textvariable=self._buttons, font=("Monospace", 11)).pack(pady=10)
        self._root.update_idletasks()
        self._root.update()

    def _request_close(self) -> None:
        self._closed = True

    def update(
        self,
        motion: np.ndarray,
        buttons: Mapping[int, bool],
        *,
        mode: str = "translation_xyz",
        speed: str = "training",
    ) -> bool:
        values = np.asarray(motion, dtype=np.float64).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("SpaceMouse monitor requires six finite axis values")
        if self._closed:
            return False

        active = bool(np.any(np.abs(values) > 0.0))
        self._status.set("INPUT DETECTED" if active else "CENTERED")
        self._mode.set(f"mode={mode}  speed={speed}")
        for index, value in enumerate(np.clip(values, -1.0, 1.0)):
            y = 22 + index * 38
            end = 345 + float(value) * 250
            self._canvas.coords(self._bars[index], min(345, end), y - 7, max(345, end), y + 7)
            self._canvas.itemconfigure(f"value-{index}", text=f"{float(value):+.2f}")

        one = bool(buttons.get(int(Button.ONE), False))
        two = bool(buttons.get(int(Button.TWO), False))
        home = bool(buttons.get(int(Button.T), False))
        shift = bool(buttons.get(int(Button.SHIFT), False))
        three = bool(buttons.get(int(Button.THREE), False))
        four = bool(buttons.get(int(Button.FOUR), False))
        self._buttons.set(
            f"1/CW: {'DOWN' if one else 'up'}    2/CCW: {'DOWN' if two else 'up'}    "
            f"T: {'DOWN' if home else 'up'}    "
            f"3: {'DOWN' if three else 'up'}  4: {'DOWN' if four else 'up'}    "
            f"Shift: {'DOWN' if shift else 'up'}"
        )
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
        return not self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
        with suppress(self._tk.TclError):
            self._root.destroy()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
