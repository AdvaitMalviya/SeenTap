"""Executing a bound command.

Two executors behind one interface. The simulated one is the default: during a
live demo a stray real click closes the application or hits the wrong window,
which is a liability rather than a feature. OS injection is switched on only
for controlled evaluation runs, with the corner failsafe armed throughout.
"""
from __future__ import annotations

from seentap import config


class ActionError(ValueError):
    """A verb that cannot be executed from the current state."""


class _Base:
    def __init__(self):
        self.dragging = False

    def execute(self, verb: str, x: float, y: float) -> dict:
        if verb not in config.VOCAB:
            raise ValueError(f"unknown verb {verb!r}")
        if verb == "drop" and not self.dragging:
            raise ActionError("drop without a preceding drag")
        self._do(verb, x, y)
        if verb == "drag":
            self.dragging = True
        elif verb in ("drop", "cancel"):
            self.dragging = False
        return {"verb": verb, "x": x, "y": y}

    def _do(self, verb: str, x: float, y: float) -> None:
        raise NotImplementedError


class SimExecutor(_Base):
    """Records what would have happened. Drives the dashboard's fake desktop."""

    def __init__(self):
        super().__init__()
        self.events: list[dict] = []

    def _do(self, verb: str, x: float, y: float) -> None:
        self.events.append({"verb": verb, "x": x, "y": y})


class RealExecutor(_Base):
    """Injects into the host OS. Used for logged evaluation runs only."""

    def __init__(self):
        super().__init__()
        import pyautogui

        pyautogui.FAILSAFE = True      # second line of defence: slam to a corner
        pyautogui.PAUSE = 0.0
        self.gui = pyautogui

    def _do(self, verb: str, x: float, y: float) -> None:
        g = self.gui
        if verb in ("click", "select"):
            g.click(x, y)
        elif verb == "double click":
            g.doubleClick(x, y)
        elif verb == "right click":
            g.rightClick(x, y)
        elif verb == "scroll up":
            g.moveTo(x, y); g.scroll(5)
        elif verb == "scroll down":
            g.moveTo(x, y); g.scroll(-5)
        elif verb == "drag":
            g.moveTo(x, y); g.mouseDown()
        elif verb == "drop":
            g.moveTo(x, y); g.mouseUp()
        elif verb == "cancel":
            if self.dragging:
                g.mouseUp()
        # 'recalibrate' is handled by the application, not the OS.
