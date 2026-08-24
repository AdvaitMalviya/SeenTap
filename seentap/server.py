"""FastAPI backend and the live dashboard.

The gaze stream is continuous and the transcript stream is bursty, so polling
would either waste bandwidth or add lag. One persistent WebSocket carries gaze,
transcripts and fused actions to the page as they happen.

The dashboard renders a simulated desktop by default. During a live demo a
stray real click closes the application or hits the wrong window, so OS
injection is opt-in and used for logged evaluation runs only.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from seentap import calibrate, config, eventlog
from seentap.actions import ActionError, RealExecutor, SimExecutor
from seentap.baselines import DwellSelector
from seentap.fusion import Fusion, FusionConfig

INDEX = Path(__file__).parent / "static" / "index.html"

# Set by configure() before uvicorn starts. Left empty the app still serves the
# dashboard, which is what the tests and a bare `uvicorn seentap.server:app` do.
_sensors: dict = {"tracker": None, "queue": None, "runtime": None}
_stop: asyncio.Event | None = None


def configure(tracker, queue, rt) -> None:
    global runtime
    _sensors.update(tracker=tracker, queue=queue, runtime=rt)
    runtime = rt


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Own the pump task for the life of the server.

    on_event("startup") is deprecated, and the version of this that created its
    stop event inline could never be stopped: the shutdown path set a different
    object and the loop ran on.
    """
    global _stop
    _stop = asyncio.Event()
    task = None
    if _sensors["tracker"] is not None:
        task = asyncio.create_task(pump(_sensors["tracker"], _sensors["queue"],
                                        _sensors["runtime"], _stop))
    try:
        yield
    finally:
        _stop.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="SeenTap", lifespan=lifespan)


class Hub:
    """Fan-out to every connected dashboard. Slow clients are dropped."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def join(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def leave(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def send(self, payload: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(ws)


hub = Hub()


class Runtime:
    """Owns the fusion machine, the executor and the log for one session."""

    def __init__(self, cfg: FusionConfig = None, mode: str = "B",
                 real: bool = False, condition: str = "C3",
                 log_path: str | None = None, tracker=None):
        self.fusion = Fusion(cfg or FusionConfig())
        self.tracker = tracker
        self._requalifying = False
        self._task = None
        self.mode = mode
        self.condition = condition
        self.executor = RealExecutor() if real else SimExecutor()
        self.dwell = DwellSelector()
        self.log = eventlog.EventLog(
            log_path or Path(config.LOG_DIR) / f"session-{int(eventlog.now())}.jsonl")
        self.log.write("session", mode=mode, condition=condition,
                       executor=type(self.executor).__name__,
                       config=self.fusion.cfg.__dict__,
                       screen=[config.SCREEN_W, config.SCREEN_H])

    async def on_gaze(self, sample) -> None:
        self.fusion.on_gaze(sample)
        self.log.write("gaze", t=sample.t, x=sample.x, y=sample.y,
                       conf=sample.conf, zone=sample.zone, blink=sample.blink,
                       drift=sample.drift)
        await hub.send({"kind": "gaze", "x": sample.x, "y": sample.y,
                        "zone": sample.zone, "conf": sample.conf,
                        "drift": sample.drift})
        if self.condition == "C1":                    # gaze-only baseline
            picked = self.dwell.update(sample.zone, sample.t)
            if picked is not None:
                await self._execute("select", sample.x, sample.y, picked,
                                    n=1, onset_t=sample.t)

    async def on_utterance(self, utt) -> None:
        self.log.write("utterance", t=utt.onset_t, onset_t=utt.onset_t,
                       offset_t=utt.offset_t, text=utt.text,
                       decode_ms=utt.decode_ms)
        await hub.send({"kind": "utterance", "text": utt.text,
                        "decode_ms": utt.decode_ms})
        now = eventlog.now()
        r = self.fusion.on_utterance(utt.onset_t, utt.text, now)
        await hub.send({"kind": "state", "state": self.fusion.state,
                        "gate_refusals": self.fusion.gate_refusals})
        if not r.ok:
            self.log.write("action", ok=False, reason=r.reason,
                           onset_t=utt.onset_t)
            await hub.send({"kind": "action", "ok": False, "reason": r.reason})
            return
        if r.verb in config.HELP_VOCAB:
            self.log.write("help", onset_t=utt.onset_t, said=r.verb)
            await hub.send({"kind": "help", "seconds": config.HELP_SECONDS,
                            "controls": [[v, config.VERB_HELP[v]]
                                         for v in config.VOCAB],
                            "help_words": list(config.HELP_VOCAB)})
            return
        if r.verb == "recalibrate":
            self.start_requalify()
            return
        await self._execute(r.verb, r.x, r.y, r.zone, r.n, utt.onset_t)

    def start_requalify(self) -> None:
        """Fire and forget, from the spoken verb or the dashboard hotkey alike.

        Keeping the handle matters: a bare create_task is only weakly
        referenced, and the loop is free to collect a running requalification
        halfway through.
        """
        self._task = asyncio.create_task(self.requalify())

    async def requalify(self) -> dict:
        """Five points and an affine correction, without stopping the session.

        Head drift is the likeliest way a working session stops working, and a
        full twenty-second recalibration mid-task is time the user will not
        spend. This rides the pump that is already running -- the tracker keeps
        publishing gaze and keeps its latest feature vector to hand -- so the
        only new machinery is the window that reads it.
        """
        tracker = self.tracker
        if tracker is None or getattr(tracker, "mapping", None) is None:
            return {"ok": False, "reason": "no mapping to correct"}
        if self._requalifying:
            return {"ok": False, "reason": "already requalifying"}
        self._requalifying = True

        pts = calibrate.targets(config.REQUALIFY_POINTS,
                                config.SCREEN_W, config.SCREEN_H)
        F, XY = [], []
        try:
            for i, (tx, ty) in enumerate(pts):
                for phase, seconds in (
                        ("settle", config.REQUALIFY_SETTLE_MS / 1000.0),
                        ("collect", config.REQUALIFY_COLLECT_MS / 1000.0)):
                    await hub.send({"kind": "requalify", "phase": phase,
                                    "point": i, "of": len(pts),
                                    "x": tx, "y": ty, "seconds": seconds})
                    if phase == "settle":
                        await asyncio.sleep(seconds)
                        continue
                    raw = []
                    end = eventlog.now() + seconds
                    while eventlog.now() < end:
                        f = tracker.last_features   # None while blinking or absent
                        if f is not None:
                            raw.append({"f": list(f), "conf": 1.0})
                        await asyncio.sleep(0.02)
                    med = calibrate.condense(raw)
                    if med is not None:
                        F.append(med)
                        XY.append([tx, ty])

            if len(F) < 3:
                raise ValueError(f"only {len(F)} of {len(pts)} targets gave "
                                 f"usable gaze; keeping the old mapping")
            model, before, after = calibrate.fit_correction(tracker.mapping, F, XY)
            tracker.mapping = model
            # The pose these points were collected at is the new reference, so
            # drift reads near zero again rather than carrying the old offset.
            tracker.f_ref = np.median(np.asarray(F, dtype=float), axis=0)
            tracker.filter.reset()
            out = {"ok": True, "reason": "", "n": len(F),
                   "before_px": before, "after_px": after}
        except ValueError as e:
            out = {"ok": False, "reason": str(e)}
        finally:
            self._requalifying = False

        self.log.write("requalify", **out)
        await hub.send({"kind": "requalify", "phase": "done", **out})
        await hub.send({"kind": "state", "state": self.fusion.state,
                        "gate_refusals": self.fusion.gate_refusals})
        return out

    async def _execute(self, verb, x, y, zone, n, onset_t) -> None:
        try:
            self.executor.execute(verb, x, y)
            ok, reason = True, ""
        except (ActionError, ValueError) as e:
            ok, reason = False, str(e)
        self.log.write("action", ok=ok, reason=reason, verb=verb, x=x, y=y,
                       zone=zone, n=n, onset_t=onset_t)
        await hub.send({"kind": "action", "ok": ok, "reason": reason,
                        "verb": verb, "x": x, "y": y, "zone": zone, "n": n})

    def close(self) -> None:
        self.log.close()


runtime: Runtime | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX.read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "screen": [config.SCREEN_W, config.SCREEN_H],
        "grid": [config.GRID_COLS, config.GRID_ROWS],
        "running": runtime is not None,
        "state": runtime.fusion.state if runtime else None,
        "gate_refusals": runtime.fusion.gate_refusals if runtime else 0,
    }


@app.websocket("/ws")
async def socket(ws: WebSocket) -> None:
    await hub.join(ws)
    await ws.send_text(json.dumps({
        "kind": "config", "cols": config.GRID_COLS, "rows": config.GRID_ROWS,
        "w": config.SCREEN_W, "h": config.SCREEN_H,
        "mode": runtime.mode if runtime else "B",
        "executor": type(runtime.executor).__name__ if runtime else "SimExecutor",
        "drift_warn": config.DRIFT_WARN_PX, "drift_bad": config.DRIFT_BAD_PX,
    }))
    try:
        while True:
            msg = await ws.receive_text()
            # The requalification hotkey. Same routine the spoken verb runs, so
            # the two cannot drift apart -- and it stays reachable when speech
            # is off entirely, as it is in the gaze-only baseline.
            try:
                cmd = json.loads(msg).get("cmd")
            except (ValueError, AttributeError):
                continue
            if cmd == "requalify" and runtime is not None:
                runtime.start_requalify()
    except WebSocketDisconnect:
        pass
    finally:
        hub.leave(ws)


async def pump(tracker, speech_queue, rt: Runtime, stop: asyncio.Event) -> None:
    """Drain both sensors onto one timeline. Vision in a thread, speech in a
    process; this loop is the only place they meet."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        sample, _frame = await loop.run_in_executor(
            None, tracker.read, eventlog.now())
        if sample is not None:
            await rt.on_gaze(sample)
        while speech_queue is not None and not speech_queue.empty():
            await rt.on_utterance(speech_queue.get())
        await asyncio.sleep(0)
