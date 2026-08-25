"""A click-through gaze cursor drawn over every other window.

This is what makes the system usable outside its own dashboard, and it does
more than show off: it closes the loop. A webcam gaze estimate carries a
standing offset that no amount of fitting removes, and without feedback the
user is aiming blind at a target they cannot see themselves missing. Give them
a dot and they simply look slightly off until it lands where they want, which
turns an accuracy problem into a much smaller steadiness one. Every practical
gaze interface leans on that.

It runs in its own process for the same reason speech does: AppKit wants the
main thread and its own run loop, and the server already owns one.
"""
from __future__ import annotations

import os

from seentap import config

# Colours as (r, g, b) in 0..1. Blue while tracking, amber when the gate is
# refusing, green for a moment after a command lands.
IDLE = (0.35, 0.65, 1.0)
BLOCKED = (0.85, 0.62, 0.15)
FIRED = (0.25, 0.75, 0.35)


def overlay_worker(queue, stop_event) -> None:  # pragma: no cover - needs a display
    """Process target: drain the queue, move the dot, never take a click."""
    import time

    from AppKit import (NSApplication, NSBackingStoreBuffered, NSColor,
                        NSMakeRect, NSScreen, NSScreenSaverWindowLevel, NSView,
                        NSWindow, NSWindowCollectionBehaviorCanJoinAllSpaces,
                        NSWindowCollectionBehaviorFullScreenAuxiliary,
                        NSWindowCollectionBehaviorStationary,
                        NSWindowStyleMaskBorderless)
    from Foundation import NSTimer

    app = NSApplication.sharedApplication()
    frame = NSScreen.mainScreen().frame()
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    # The whole point: the overlay must never intercept a click, or it would
    # swallow the very events the system is trying to inject.
    win.setIgnoresMouseEvents_(True)
    win.setLevel_(NSScreenSaverWindowLevel)
    # Above full-screen apps and on every Space, so it does not vanish the
    # moment the user switches to the window they actually want to drive.
    win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                               | NSWindowCollectionBehaviorStationary
                               | NSWindowCollectionBehaviorFullScreenAuxiliary)

    root = NSView.alloc().initWithFrame_(frame)
    win.setContentView_(root)

    r = config.OVERLAY_DOT_PX
    dot = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, r * 2, r * 2))
    dot.setWantsLayer_(True)
    dot.layer().setCornerRadius_(float(r))
    dot.layer().setBorderWidth_(2.0)
    dot.layer().setBorderColor_(NSColor.whiteColor().CGColor())
    root.addSubview_(dot)

    ring = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, r * 5, r * 5))
    ring.setWantsLayer_(True)
    ring.layer().setCornerRadius_(float(r) * 2.5)
    ring.layer().setBorderWidth_(3.0)
    ring.layer().setBackgroundColor_(NSColor.clearColor().CGColor())
    ring.setHidden_(True)
    root.addSubview_(ring)
    win.orderFrontRegardless()

    def paint(view, rgb, alpha):
        view.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb, alpha).CGColor())

    paint(dot, IDLE, 0.55)
    state = {"fired_until": 0.0, "seen": 0}

    def tick(_timer):
        if stop_event is not None and stop_event.is_set():
            # A daemon process with a Cocoa run loop: stop_() only lands on the
            # next event, and there may not be one. Leave the blunt way.
            os._exit(0)
        latest = None
        while True:
            try:
                latest = queue.get_nowait()
            except Exception:
                break
        now = time.monotonic()
        if latest is not None:
            state["seen"] += 1
            if latest.get("fired"):
                state["fired_until"] = now + config.OVERLAY_FLASH_S
                ring.setFrameOrigin_(_flip(latest, frame, r * 2.5))
                ring.layer().setBorderColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        *FIRED, 0.9).CGColor())
                ring.setHidden_(False)
            if latest.get("x") is not None:
                dot.setFrameOrigin_(_flip(latest, frame, r))
                dot.setHidden_(False)
                paint(dot, IDLE if latest.get("armed") else BLOCKED,
                      0.55 if latest.get("armed") else 0.4)
            else:
                dot.setHidden_(True)       # no face: show nothing, not a stale dot
        if now > state["fired_until"]:
            ring.setHidden_(True)

    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        1.0 / config.OVERLAY_FPS, True, tick)
    app.run()


def _flip(msg, frame, radius):
    """Screen points are measured from the top, Cocoa from the bottom."""
    return (msg["x"] - radius, frame.size.height - msg["y"] - radius)
