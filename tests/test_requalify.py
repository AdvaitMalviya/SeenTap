"""Head drift: measuring it live, and correcting it in five points.

A calibration is fitted at one head pose and silently decays as the user
settles into their chair. Nothing else in the system notices -- gaze keeps
arriving, the gate keeps arming, the clicks just land in the wrong tile.

Two halves here. ``drift_px`` puts a number on the decay without any ground
truth, and ``fit_correction`` buys the mapping back with five points instead of
the twenty seconds a full recalibration costs.
"""
import asyncio
import json

import numpy as np
import pytest

from seentap import calibrate, config, eventlog, fusion, gaze, server

CFG = fusion.FusionConfig()

# Screen point straight out of the eye ratios, plus pose terms so head movement
# actually moves the estimate. A fitted ridge with the awkward parts removed.
XG, YG, YAWG, PITCHG = 1500.0, 950.0, 900.0, 600.0


class Plane:
    def predict(self, F):
        F = np.atleast_2d(np.asarray(F, dtype=float))
        return np.column_stack([F[:, 0] * XG + F[:, 4] * YAWG,
                                F[:, 2] * YG + F[:, 5] * PITCHG])


def look_at(x, y, yaw=0.0, pitch=0.0):
    """The feature vector of someone looking at (x, y) with that head pose."""
    return np.array([x / XG, x / XG, y / YG, y / YG, yaw, pitch, 0.0, 60.0, 1.0])


# --- measuring drift -------------------------------------------------------

def test_moving_the_eyes_is_not_drift():
    """The whole point: only the pose terms count, or every saccade reads red."""
    ref = look_at(756, 491)
    assert gaze.drift_px(Plane(), look_at(120, 900), ref) == pytest.approx(0.0)


def test_moving_the_head_is_drift_and_is_reported_in_pixels():
    ref = look_at(756, 491)
    d = gaze.drift_px(Plane(), look_at(756, 491, yaw=0.1), ref)
    assert d == pytest.approx(0.1 * YAWG)          # 90 px of pure head rotation


def test_drift_grows_with_the_movement():
    ref = look_at(756, 491)
    small = gaze.drift_px(Plane(), look_at(756, 491, yaw=0.02), ref)
    large = gaze.drift_px(Plane(), look_at(756, 491, yaw=0.2), ref)
    assert 0 < small < large


def test_a_homography_cannot_see_drift_at_all():
    """Documented limitation, not an oversight: the planar fit consumes only
    the eye ratios, so it has no pose term to rewind."""
    flat = calibrate._Homography(np.eye(3))
    ref = look_at(756, 491)
    assert gaze.drift_px(flat, look_at(756, 491, yaw=0.3), ref) == pytest.approx(0.0)


def test_a_sample_carries_no_drift_until_a_reference_pose_exists():
    assert gaze.GazeSample(t=1.0, x=0.0, y=0.0, conf=1.0).drift is None


# --- correcting it ---------------------------------------------------------

def _five_points(yaw=0.0):
    pts = calibrate.targets(config.REQUALIFY_POINTS)
    return np.array([look_at(x, y, yaw=yaw) for x, y in pts]), np.array(pts)


def test_five_points_buy_back_a_drifted_mapping():
    F, XY = _five_points(yaw=0.09)               # 81 px of accumulated drift
    model, before, after = calibrate.fit_correction(Plane(), F, XY)
    assert before == pytest.approx(0.09 * YAWG, abs=1.0)
    assert after < 1.0
    assert calibrate.mean_err(model, F, XY) < 1.0


def test_correcting_twice_re_measures_rather_than_stacking():
    """Otherwise the second requalification corrects the first one's guess."""
    F, XY = _five_points(yaw=0.09)
    once, _, _ = calibrate.fit_correction(Plane(), F, XY)
    twice, _, _ = calibrate.fit_correction(once, F, XY)
    assert not isinstance(twice.base, calibrate._Corrected)
    assert calibrate.mean_err(twice, F, XY) == pytest.approx(
        calibrate.mean_err(once, F, XY), abs=1e-6)


def test_points_collected_while_the_user_looked_away_are_refused():
    """An affine fits noise as willingly as signal. Replacing a working mapping
    with that, mid-task, would leave no way back."""
    _, XY = _five_points()
    stared = np.array([look_at(756, 491) for _ in XY])   # never followed the dots
    with pytest.raises(ValueError, match="exceeds"):
        calibrate.fit_correction(Plane(), stared, XY)


def test_an_affine_needs_three_points():
    F, XY = _five_points()
    with pytest.raises(ValueError, match="three points"):
        calibrate.fit_correction(Plane(), F[:2], XY[:2])


# --- reaching the verb at all ----------------------------------------------

def steady(n=30, t0=100.0, x=700.0, y=500.0):
    return [gaze.GazeSample(t=t0 + i / 30.0, x=x, y=y, conf=0.9, zone=5)
            for i in range(n)]


def test_recalibrate_survives_the_gate_that_drift_itself_breaks():
    """The regression that matters. A badly drifted mapping puts gaze off the
    screen, the gate refuses off-screen commands, and the one verb that fixes
    the drift becomes unreachable exactly when it is needed."""
    f = fusion.Fusion(CFG)
    for s in steady(x=-400.0, y=500.0):
        f.on_gaze(s)
    assert fusion.gate(f.buffer, 100.97)[1] == "off_screen"
    r = f.on_utterance(onset_t=100.9, text="recalibrate", now=100.97)
    assert r.ok and r.verb == "recalibrate" and f.state == "recalibrating"


def test_recalibrate_works_while_the_eyes_are_sweeping():
    f = fusion.Fusion(CFG)
    for i in range(20):
        f.on_gaze(gaze.GazeSample(t=100.0 + i / 30.0, x=i * 70.0, y=500.0,
                                  conf=0.9, zone=0))
    assert fusion.gate(f.buffer, 100.63)[1] == "not_fixating"
    assert f.on_utterance(onset_t=100.6, text="recalibrate", now=100.63).ok


def test_recalibrate_still_needs_somebody_in_front_of_the_camera():
    f = fusion.Fusion(CFG)
    r = f.on_utterance(onset_t=100.0, text="recalibrate", now=100.0)
    assert not r.ok and r.reason == "no_face"
    assert f.gate_refusals == 1


def test_recalibrate_targets_nothing():
    f = fusion.Fusion(CFG)
    for s in steady():
        f.on_gaze(s)
    r = f.on_utterance(onset_t=100.9, text="recalibrate", now=101.0)
    assert (r.x, r.y, r.zone, r.n) == (None, None, None, 0)


# --- the live routine ------------------------------------------------------

class FakeFilter:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeTracker:
    def __init__(self):
        self.mapping = Plane()
        self.last_features = None
        self.f_ref = look_at(756, 491)
        self.filter = FakeFilter()


class Participant:
    """A dashboard client that looks wherever the server puts the dot.

    Driving the loop through the same broadcast the browser receives keeps the
    test honest about the protocol rather than about the internals.
    """

    def __init__(self, tracker, yaw=0.0, obedient=True):
        self.tracker, self.yaw, self.obedient = tracker, yaw, obedient
        self.sent = []

    async def send_text(self, payload):
        m = json.loads(payload)
        self.sent.append(m)
        if m.get("kind") == "requalify" and m.get("phase") == "collect":
            self.tracker.last_features = (
                look_at(m["x"], m["y"], yaw=self.yaw) if self.obedient else None)

    def phases(self, phase):
        return [m for m in self.sent
                if m.get("kind") == "requalify" and m.get("phase") == phase]


@pytest.fixture(autouse=True)
def _quick_and_clean(monkeypatch):
    monkeypatch.setattr(config, "REQUALIFY_SETTLE_MS", 10)
    monkeypatch.setattr(config, "REQUALIFY_COLLECT_MS", 60)
    yield
    server.hub.clients.clear()
    server.runtime = None


def _run(tracker, user, log):
    async def go():
        server.hub.clients.add(user)
        rt = server.Runtime(log_path=log, tracker=tracker)
        try:
            return await rt.requalify()
        finally:
            rt.close()
    return asyncio.run(go())


def test_requalifying_walks_five_targets_and_swaps_the_mapping_in(tmp_path):
    log = tmp_path / "s.jsonl"
    tracker = FakeTracker()
    user = Participant(tracker, yaw=0.09)

    out = _run(tracker, user, log)

    assert out["ok"] and out["n"] == config.REQUALIFY_POINTS
    assert out["after_px"] < out["before_px"]
    assert len(user.phases("settle")) == config.REQUALIFY_POINTS
    assert len(user.phases("collect")) == config.REQUALIFY_POINTS
    assert user.phases("done")[0]["ok"] is True

    assert isinstance(tracker.mapping, calibrate._Corrected)
    assert tracker.filter.resets == 1, "the smoother still holds pre-drift points"
    # The pose just measured becomes the new reference, so the indicator falls
    # back to zero instead of carrying the old offset forever.
    assert gaze.drift_px(tracker.mapping, look_at(756, 491, yaw=0.09),
                         tracker.f_ref) == pytest.approx(0.0, abs=1e-6)
    assert any(r["kind"] == "requalify" and r["ok"] for r in eventlog.read(log))


def test_a_failed_requalification_keeps_the_old_mapping(tmp_path):
    """Nothing usable came back -- the session must be no worse off."""
    log = tmp_path / "s.jsonl"
    tracker = FakeTracker()
    before = tracker.mapping
    user = Participant(tracker, obedient=False)

    out = _run(tracker, user, log)

    assert not out["ok"] and "usable gaze" in out["reason"]
    assert tracker.mapping is before
    assert tracker.filter.resets == 0
    assert user.phases("done")[0]["ok"] is False


def test_requalifying_without_a_tracker_says_so_rather_than_raising(tmp_path):
    async def go():
        rt = server.Runtime(log_path=tmp_path / "s.jsonl")
        try:
            return await rt.requalify()
        finally:
            rt.close()
    assert asyncio.run(go())["ok"] is False


def test_asking_twice_does_not_run_two_at_once(tmp_path):
    log = tmp_path / "s.jsonl"
    tracker = FakeTracker()
    user = Participant(tracker, yaw=0.09)

    async def go():
        server.hub.clients.add(user)
        rt = server.Runtime(log_path=log, tracker=tracker)
        try:
            return await asyncio.gather(rt.requalify(), rt.requalify())
        finally:
            rt.close()

    first, second = asyncio.run(go())
    assert first["ok"] and second["reason"] == "already requalifying"
    assert len(user.phases("collect")) == config.REQUALIFY_POINTS


# --- and out to the dashboard ----------------------------------------------

class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


def test_the_drift_number_reaches_the_dashboard_and_the_log(tmp_path):
    log = tmp_path / "s.jsonl"
    ws = FakeSocket()

    async def go():
        server.hub.clients.add(ws)
        rt = server.Runtime(log_path=log)
        await rt.on_gaze(gaze.GazeSample(t=eventlog.now(), x=700.0, y=500.0,
                                         conf=0.9, zone=5, drift=83.5))
        rt.close()

    asyncio.run(go())
    assert [m for m in ws.sent if m["kind"] == "gaze"][0]["drift"] == 83.5
    assert [r for r in eventlog.read(log) if r["kind"] == "gaze"][0]["drift"] == 83.5


def test_the_hotkey_runs_the_same_routine_the_spoken_verb_does(tmp_path):
    """Two triggers, one path, so they cannot drift apart -- and the hotkey
    still works in the gaze-only condition, which has no microphone at all."""
    from fastapi import WebSocketDisconnect

    class FakeWS(FakeSocket):
        def __init__(self, msgs):
            super().__init__()
            self.msgs = list(msgs)

        async def accept(self):
            pass

        async def receive_text(self):
            if self.msgs:
                return self.msgs.pop(0)
            raise WebSocketDisconnect()

    calls = []

    async def go():
        rt = server.Runtime(log_path=tmp_path / "s.jsonl")
        rt.start_requalify = lambda: calls.append(1)
        server.runtime = rt
        ws = FakeWS(['not json at all', '{"cmd":"nonsense"}', '{"cmd":"requalify"}'])
        try:
            await server.socket(ws)
        finally:
            rt.close()
        return ws

    ws = asyncio.run(go())
    assert calls == [1], "junk must not kill the socket, and must not fire it"
    assert ws.sent[0]["drift_warn"] == config.DRIFT_WARN_PX
    assert ws.sent[0]["drift_bad"] == config.DRIFT_BAD_PX


def test_the_dashboard_shows_targets_and_binds_the_hotkey():
    html = server.INDEX.read_text()
    assert 'id="target"' in html and "shrink" in html
    assert '"requalify"' in html and 'e.key !== "r"' in html
    assert "requestFullscreen" in html, "targets are placed against the viewport"
    assert "prefers-reduced-motion" in html
    assert "http://" not in html and "https://" not in html


def test_the_drift_thresholds_are_the_accuracy_gate_not_magic_numbers():
    assert config.DRIFT_BAD_PX == config.GATE_FRAC * config.SCREEN_W
    assert config.DRIFT_WARN_PX == config.DRIFT_BAD_PX / 2
