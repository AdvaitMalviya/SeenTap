"""The gaze cursor drawn over every other window.

The drawing itself needs a display and is not tested here. What is testable is
the arithmetic that puts the dot in the right place and the contract with the
server: never block the pump, never show a stale position.
"""
import asyncio

import pytest

from seentap import config, eventlog, gaze, overlay, server


class Frame:
    class size:
        width, height = 1512.0, 982.0


def test_screen_points_are_flipped_into_cocoa_coordinates():
    """Screen coordinates count down from the top, Cocoa up from the bottom.
    Get this wrong and the dot mirrors vertically -- it still tracks, which is
    exactly why it would survive a casual look."""
    top = overlay._flip({"x": 756.0, "y": 0.0}, Frame, 0)
    bottom = overlay._flip({"x": 756.0, "y": 982.0}, Frame, 0)
    assert top == (756.0, 982.0)
    assert bottom == (756.0, 0.0)


def test_the_dot_is_centred_on_the_gaze_not_hung_off_it():
    """AppKit positions a view by its corner; the gaze point is its middle."""
    r = 16
    assert overlay._flip({"x": 100.0, "y": 200.0}, Frame, r) == (100.0 - r,
                                                                 982.0 - 200.0 - r)


def test_a_full_queue_drops_the_frame_rather_than_stalling_the_pump(tmp_path):
    """The overlay is cosmetic; gaze and speech are not. If it ever falls
    behind, the pump must not wait for it."""
    class Full:
        def put_nowait(self, _msg):
            raise RuntimeError("queue is full")

    rt = server.Runtime(log_path=tmp_path / "s.jsonl", overlay=Full())
    try:
        rt._to_overlay(x=1.0, y=2.0)          # must not raise
    finally:
        rt.close()


def test_no_overlay_configured_is_a_no_op(tmp_path):
    rt = server.Runtime(log_path=tmp_path / "s.jsonl")
    try:
        rt._to_overlay(x=1.0, y=2.0)
    finally:
        rt.close()


def test_gaze_reaches_the_overlay_with_the_gate_state(tmp_path):
    """Colour tells the user whether speaking would do anything, which is the
    other half of the feedback: a dot on target that the gate is refusing looks
    identical to one that would fire."""
    sent = []

    class Spy:
        def put_nowait(self, msg):
            sent.append(msg)

    async def go():
        rt = server.Runtime(log_path=tmp_path / "s.jsonl", overlay=Spy())
        t0 = eventlog.now()
        try:
            for i in range(20):
                await rt.on_gaze(gaze.GazeSample(t=t0 + i / 30.0, x=700.0,
                                                 y=500.0, conf=0.9, zone=5))
        finally:
            rt.close()

    asyncio.run(go())
    assert len(sent) == 20
    assert sent[-1]["x"] == 700.0 and sent[-1]["y"] == 500.0
    assert sent[-1]["armed"] is True, "a steady on-screen fixation is armed"


def test_a_refused_fixation_is_marked_unarmed(tmp_path):
    sent = []

    class Spy:
        def put_nowait(self, msg):
            sent.append(msg)

    async def go():
        rt = server.Runtime(log_path=tmp_path / "s.jsonl", overlay=Spy())
        t0 = eventlog.now()
        try:
            for i in range(20):                       # sweeping, never fixating
                await rt.on_gaze(gaze.GazeSample(t=t0 + i / 30.0, x=i * 70.0,
                                                 y=500.0, conf=0.9, zone=0))
        finally:
            rt.close()

    asyncio.run(go())
    assert sent[-1]["armed"] is False


def test_the_overlay_never_takes_a_click():
    """It sits above every window. If it accepted mouse events it would eat
    the very clicks the system injects."""
    src = (overlay.__file__).replace(".pyc", ".py")
    text = open(src, encoding="utf-8").read()
    assert "setIgnoresMouseEvents_(True)" in text


@pytest.mark.parametrize("name", ["OVERLAY_DOT_PX", "OVERLAY_FPS", "OVERLAY_FLASH_S"])
def test_the_cursor_has_its_knobs(name):
    assert getattr(config, name) > 0
