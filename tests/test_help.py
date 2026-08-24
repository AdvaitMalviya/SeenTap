"""Saying "help" or "controls" puts the command list on screen briefly.

It needs no target, so it never reaches bind(). It also deliberately skips the
fixation requirement: someone who has forgotten the commands is looking around
the screen, which is precisely the state the normal gate refuses.
"""
import asyncio
import json

import pytest

from seentap import config, fusion, parse, server
from seentap.gaze import GazeSample

CFG = fusion.FusionConfig()


def steady(n=30, t0=100.0, x=700.0, y=500.0):
    return [GazeSample(t=t0 + i / 30.0, x=x, y=y, conf=0.9, zone=5, blink=False)
            for i in range(n)]


def sweeping(t0=100.0):
    """Eyes crossing the screen: a face is there, but no fixation."""
    return [GazeSample(t=t0 + i / 30.0, x=float(i * 70), y=500.0, conf=0.9,
                       zone=0, blink=False) for i in range(20)]


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("said,expect", [
    ("help", "help"),
    ("controls", "controls"),
    ("help me", "help"),
    ("show controls", "controls"),
    ("what are the controls", "controls"),
    ("hep", "help"),
])
def test_help_words_are_recognised(said, expect):
    assert parse.parse_any(said)[0] == expect


@pytest.mark.parametrize("said", ["scroll up", "scroll down", "click",
                                  "double click", "select"])
def test_help_words_do_not_steal_the_action_verbs(said):
    """'controls' and 'scroll' share most of their letters."""
    assert parse.parse_any(said)[0] == said


def test_the_action_vocabulary_is_still_exactly_ten():
    assert len(config.VOCAB) == 10
    assert not set(config.VOCAB) & set(config.HELP_VOCAB)


def test_every_action_verb_has_a_description_for_the_overlay():
    assert set(config.VERB_HELP) == set(config.VOCAB)
    assert all(config.VERB_HELP[v].strip() for v in config.VOCAB)


def test_nonsense_is_still_refused_with_the_help_words_in_the_running():
    assert parse.parse_any("banana")[0] is None
    assert parse.parse_any("")[0] is None


# --- fusion ----------------------------------------------------------------

def test_help_works_while_the_eyes_are_sweeping():
    """The normal gate would refuse this buffer. Help must not."""
    f = fusion.Fusion(CFG)
    for s in sweeping():
        f.on_gaze(s)
    assert fusion.gate(f.buffer, 100.63)[1] == "not_fixating"
    r = f.on_utterance(onset_t=100.6, text="help", now=100.63)
    assert r.ok and r.verb == "help"


def test_help_still_needs_somebody_in_front_of_the_camera():
    f = fusion.Fusion(CFG)
    r = f.on_utterance(onset_t=100.0, text="help", now=100.0)
    assert not r.ok and r.reason == "no_face"
    assert f.gate_refusals == 1


def test_help_binds_nothing_and_targets_nothing():
    f = fusion.Fusion(CFG)
    for s in steady():
        f.on_gaze(s)
    r = f.on_utterance(onset_t=100.9, text="controls", now=101.0)
    assert r.ok and r.verb == "controls"
    assert (r.x, r.y, r.zone, r.n) == (None, None, None, 0)


def test_help_is_available_immediately_after_an_action():
    """A display is not an action, so the cooldown does not apply to it."""
    f = fusion.Fusion(CFG)
    for s in steady():
        f.on_gaze(s)
    assert f.on_utterance(onset_t=100.9, text="click", now=101.0).ok
    r = f.on_utterance(onset_t=101.0, text="help", now=101.02)
    assert r.ok and r.verb == "help"


def test_help_does_not_start_a_cooldown_of_its_own():
    f = fusion.Fusion(CFG)
    for s in steady():
        f.on_gaze(s)
    assert f.on_utterance(onset_t=100.85, text="help", now=100.9).ok
    assert f.on_utterance(onset_t=100.9, text="click", now=100.95).ok


def test_help_leaves_the_machine_tracking():
    f = fusion.Fusion(CFG)
    for s in steady():
        f.on_gaze(s)
    f.on_utterance(onset_t=100.9, text="help", now=101.0)
    assert f.state == "tracking"


# --- server + dashboard ----------------------------------------------------

class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


def test_asking_for_help_broadcasts_the_control_list_and_logs_it(tmp_path):
    log = tmp_path / "s.jsonl"
    ws = FakeSocket()

    async def go():
        server.hub.clients.add(ws)
        try:
            # Runtime stamps against the real monotonic clock, so the synthetic
            # samples have to sit on that timeline rather than an invented one.
            from seentap import eventlog
            from seentap.speech import Utterance

            t0 = eventlog.now() - 1.0
            rt = server.Runtime(log_path=log)
            for s in steady(t0=t0):
                await rt.on_gaze(s)
            await rt.on_utterance(Utterance(onset_t=t0 + 0.9, offset_t=t0 + 1.0,
                                            text="help", decode_ms=100.0))
            rt.close()
        finally:
            server.hub.clients.discard(ws)

    asyncio.run(go())

    helps = [m for m in ws.sent if m["kind"] == "help"]
    assert len(helps) == 1
    assert helps[0]["seconds"] == config.HELP_SECONDS
    assert [v for v, _ in helps[0]["controls"]] == list(config.VOCAB)
    assert helps[0]["help_words"] == list(config.HELP_VOCAB)

    from seentap import eventlog
    assert any(r["kind"] == "help" for r in eventlog.read(log))
    assert not any(r["kind"] == "action" for r in eventlog.read(log))


def test_the_overlay_exists_in_the_dashboard_and_fades():
    html = server.INDEX.read_text()
    assert 'id="help"' in html
    assert "#help.show" in html and "transition:opacity" in html
    assert "prefers-reduced-motion" in html, "the fade must be skippable"
    assert "http://" not in html and "https://" not in html
