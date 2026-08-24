"""The serve path: the pump loop, the lifespan, and shutdown actually stopping.

None of this needs a camera. It is here because it is the one path the rest of
the suite never executes, and the shutdown bug it covers -- a stop event created
inline and thrown away, so the loop ran forever -- is invisible until you try to
quit the program.
"""
import asyncio

import pytest

from seentap import eventlog, gaze, server


class FakeTracker:
    """Hands out a few samples then reports no face, counting every read."""

    def __init__(self, n=5):
        self.remaining = [
            gaze.GazeSample(t=100.0 + i / 30.0, x=700.0, y=500.0, conf=0.9,
                            zone=5, blink=False)
            for i in range(n)
        ]
        self.reads = 0

    def read(self, t):
        self.reads += 1
        return (self.remaining.pop(0), None) if self.remaining else (None, None)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_sensors():
    yield
    server._sensors.update(tracker=None, queue=None, runtime=None)
    server.runtime = None


def test_pump_logs_gaze_and_honours_the_stop_event(tmp_path):
    log = tmp_path / "s.jsonl"

    async def go():
        rt = server.Runtime(log_path=log)
        stop = asyncio.Event()
        task = asyncio.create_task(server.pump(FakeTracker(), None, rt, stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)   # fails if stop is ignored
        rt.close()

    asyncio.run(go())
    kinds = [r["kind"] for r in eventlog.read(log)]
    assert kinds.count("gaze") == 5
    assert kinds[0] == "session"


def test_lifespan_stops_the_pump_on_shutdown(tmp_path):
    """After the server exits, nothing may still be reading the camera."""
    tracker = FakeTracker(n=200)

    async def go():
        rt = server.Runtime(log_path=tmp_path / "s.jsonl")
        server.configure(tracker, None, rt)
        async with server.app.router.lifespan_context(server.app):
            await asyncio.sleep(0.05)
        after_shutdown = tracker.reads
        await asyncio.sleep(0.05)
        rt.close()
        return after_shutdown, tracker.reads

    at_exit, later = asyncio.run(go())
    assert at_exit > 0, "the pump never ran"
    assert later == at_exit, "the pump kept reading after shutdown"


def test_app_still_serves_with_no_sensors_configured():
    """`uvicorn seentap.server:app` with no hardware must not crash."""
    async def go():
        async with server.app.router.lifespan_context(server.app):
            pass

    asyncio.run(go())


def test_configure_publishes_the_runtime_for_the_health_endpoint(tmp_path):
    rt = server.Runtime(log_path=tmp_path / "s.jsonl")
    server.configure(FakeTracker(), None, rt)
    assert server.runtime is rt
    rt.close()


def test_ensure_model_is_a_no_op_once_the_weights_are_present(tmp_path):
    """Day 0 downloads once; every later run must be offline."""
    fake = tmp_path / "face_landmarker.task"
    fake.write_bytes(b"not really a model")
    assert gaze.ensure_model(str(fake)) == str(fake)
    assert fake.read_bytes() == b"not really a model"


def test_fetch_is_a_registered_command():
    from seentap import run

    with pytest.raises(SystemExit):
        run.main(["fetch", "--help"])
