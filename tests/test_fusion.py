"""The contribution. bind() must be pure, and must look backwards in time."""
import copy

import pytest

from seentap import fusion
from seentap.gaze import GazeSample


def buffer(points, t0=100.0, hz=30.0, conf=0.9):
    """points: list of (x, y) at 30 Hz starting at t0."""
    return [
        GazeSample(t=t0 + i / hz, x=float(x), y=float(y), conf=conf,
                   zone=0, blink=False)
        for i, (x, y) in enumerate(points)
    ]


def held(x, y, n, **kw):
    return buffer([(x, y)] * n, **kw)


CFG = fusion.FusionConfig(lead_ms=200, window_ms=300, aggregator="median",
                          min_samples=5, conf_floor=0.5)


def test_binds_to_where_the_eyes_were_at_speech_onset_not_now():
    """The whole argument: transcription is slow, the user has moved on."""
    old = held(300.0, 300.0, 30, t0=100.0)          # 100.0 .. 100.97
    new = held(1200.0, 800.0, 30, t0=101.0)         # 101.0 .. 101.97
    buf = old + new
    onset = 101.1                                    # spoke while looking at 1200,800
    # ...but the transcript only arrives at 101.9, by which time gaze moved.
    r = fusion.bind(buf, onset, fusion.FusionConfig(lead_ms=0, window_ms=300,
                                                    aggregator="median", min_samples=3))
    assert r.ok and r.x == pytest.approx(1200.0)


def test_lead_offset_reaches_further_back():
    """People look before they speak; the offset is what exploits that."""
    buf = held(300.0, 300.0, 30, t0=100.0) + held(1200.0, 800.0, 30, t0=101.0)
    onset = 101.05
    late = fusion.bind(buf, onset, fusion.FusionConfig(lead_ms=0, window_ms=100,
                                                       aggregator="mean", min_samples=2))
    early = fusion.bind(buf, onset, fusion.FusionConfig(lead_ms=300, window_ms=100,
                                                        aggregator="mean", min_samples=2))
    assert late.x == pytest.approx(1200.0)
    assert early.x == pytest.approx(300.0)


def test_refuses_rather_than_guesses_when_too_few_samples_survive():
    r = fusion.bind(held(500.0, 500.0, 2, t0=100.0), 100.05,
                    fusion.FusionConfig(min_samples=5))
    assert not r.ok
    assert r.reason == "too_few_samples"
    assert r.x is None


def test_refuses_on_an_empty_buffer():
    r = fusion.bind([], 100.0, CFG)
    assert not r.ok and r.reason == "too_few_samples"


def test_low_confidence_samples_are_excluded():
    buf = held(500.0, 500.0, 30, t0=100.0, conf=0.1)
    assert not fusion.bind(buf, 100.5, CFG).ok


def test_blinks_are_excluded():
    buf = held(500.0, 500.0, 30, t0=100.0)
    for s in buf:
        s.blink = True
    assert not fusion.bind(buf, 100.5, CFG).ok


@pytest.mark.parametrize("agg", sorted(fusion.AGGREGATORS))
def test_every_aggregator_in_the_sweep_produces_a_binding(agg):
    buf = held(640.0, 480.0, 30, t0=100.0)
    r = fusion.bind(buf, 100.5, fusion.FusionConfig(aggregator=agg, lead_ms=0,
                                                    window_ms=1000, min_samples=3))
    assert r.ok and r.zone is not None


def test_median_survives_an_outlier_that_the_mean_does_not():
    pts = [(500.0, 500.0)] * 20
    pts[10] = (5000.0, 5000.0)
    buf = buffer(pts, t0=100.0)
    cfg = dict(lead_ms=0, window_ms=1000, min_samples=3)
    mean = fusion.bind(buf, 100.33, fusion.FusionConfig(aggregator="mean", **cfg))
    med = fusion.bind(buf, 100.33, fusion.FusionConfig(aggregator="median", **cfg))
    assert med.x == pytest.approx(500.0)
    assert mean.x > med.x


def test_zone_mode_aggregator_returns_the_modal_tile():
    buf = held(100.0, 100.0, 20, t0=100.0) + held(1400.0, 900.0, 5, t0=100.7)
    for s in buf:
        s.zone = 0 if s.x < 500 else 11
    r = fusion.bind(buf, 100.4, fusion.FusionConfig(aggregator="zone_mode",
                                                    lead_ms=0, window_ms=1000,
                                                    min_samples=3))
    assert r.zone == 0


def test_bind_is_pure():
    """Study 2 replays this function over logs. It may not touch the world."""
    buf = held(500.0, 500.0, 30, t0=100.0)
    before = copy.deepcopy(buf)
    a = fusion.bind(buf, 100.5, CFG)
    b = fusion.bind(buf, 100.5, CFG)
    assert (a.x, a.y, a.zone, a.n) == (b.x, b.y, b.zone, b.n)
    assert buf == before, "bind must not mutate the buffer"


def test_window_is_centred_so_width_controls_how_many_samples_are_seen():
    buf = held(500.0, 500.0, 90, t0=100.0)
    narrow = fusion.bind(buf, 101.5, fusion.FusionConfig(lead_ms=0, window_ms=100,
                                                         min_samples=1))
    wide = fusion.bind(buf, 101.5, fusion.FusionConfig(lead_ms=0, window_ms=1000,
                                                       min_samples=1))
    assert wide.n > narrow.n


# --- gaze gating -----------------------------------------------------------

def test_gate_opens_on_a_steady_on_screen_fixation():
    buf = held(700.0, 500.0, 20, t0=100.0)
    ok, reason = fusion.gate(buf, 100.63)
    assert ok, reason


def test_gate_closes_while_the_eyes_are_sweeping():
    buf = buffer([(x, 500.0) for x in range(0, 1400, 70)], t0=100.0)
    ok, reason = fusion.gate(buf, 100.63)
    assert not ok and reason == "not_fixating"


def test_gate_closes_with_no_face():
    ok, reason = fusion.gate([], 100.0)
    assert not ok and reason == "no_face"


def test_gate_closes_when_gaze_leaves_the_screen():
    buf = held(-400.0, 500.0, 20, t0=100.0)
    ok, reason = fusion.gate(buf, 100.63)
    assert not ok and reason == "off_screen"


# --- state machine ---------------------------------------------------------

def test_state_machine_walks_the_states_from_the_report():
    assert fusion.Fusion.STATES == (
        "tracking", "armed", "listening", "decoding",
        "binding", "executing", "cooldown", "recalibrating",
    )


def test_command_during_cooldown_is_dropped():
    f = fusion.Fusion(CFG)
    for s in held(700.0, 500.0, 30, t0=100.0):
        f.on_gaze(s)
    first = f.on_utterance(onset_t=100.9, text="click", now=101.0)
    second = f.on_utterance(onset_t=101.0, text="click", now=101.05)
    assert first.ok
    assert not second.ok and second.reason == "cooldown"


def test_unparsed_speech_never_reaches_the_executor():
    f = fusion.Fusion(CFG)
    for s in held(700.0, 500.0, 30, t0=100.0):
        f.on_gaze(s)
    r = f.on_utterance(onset_t=100.9, text="banana", now=101.0)
    assert not r.ok and r.reason == "no_verb"


def test_recalibrate_is_reachable_from_anywhere():
    f = fusion.Fusion(CFG)
    for s in held(700.0, 500.0, 30, t0=100.0):
        f.on_gaze(s)
    r = f.on_utterance(onset_t=100.9, text="recalibrate", now=101.0)
    assert r.ok and r.verb == "recalibrate"
    assert f.state == "recalibrating"


def test_every_transition_is_recorded_for_offline_reconstruction():
    f = fusion.Fusion(CFG)
    for s in held(700.0, 500.0, 30, t0=100.0):
        f.on_gaze(s)
    f.on_utterance(onset_t=100.9, text="click", now=101.0)
    assert [t[1] for t in f.transitions][-3:] == ["binding", "executing", "cooldown"]


def test_gate_refusals_are_counted_so_the_claim_is_a_number():
    f = fusion.Fusion(CFG)
    r = f.on_utterance(onset_t=100.0, text="click", now=100.0)   # no gaze at all
    assert not r.ok
    assert f.gate_refusals == 1
