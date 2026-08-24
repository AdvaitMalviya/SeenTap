"""Study 2. If this file passes, the parameter sweep costs no participants."""

from seentap import config, eventlog, fusion, replay
from seentap.gaze import GazeSample


def write_session(path, cfg):
    """A live run: gaze streaming, one command spoken mid-fixation."""
    live = fusion.Fusion(cfg)
    with eventlog.EventLog(path) as log:
        log.write("session", t=0.0, config=cfg.__dict__, screen=[1512, 982])
        t = 100.0
        for i in range(60):
            s = GazeSample(t=t + i / 30.0, x=700.0, y=500.0, conf=0.9,
                           zone=5, blink=False)
            live.on_gaze(s)
            log.write("gaze", t=s.t, x=s.x, y=s.y, conf=s.conf, zone=s.zone,
                      blink=s.blink)
        log.write("truth", t=101.9, zone=5)
        r = live.on_utterance(onset_t=101.5, text="click", now=101.9)
        log.write("utterance", t=101.5, onset_t=101.5, text="click", decode_ms=380)
        log.write("action", t=101.9, verb=r.verb, x=r.x, y=r.y, zone=r.zone,
                  ok=r.ok, reason=r.reason)
    return r


def test_replay_reproduces_the_live_session_exactly(tmp_path):
    """The day-15 exit criterion, as an assertion."""
    p = tmp_path / "live.jsonl"
    cfg = config.FUSION_DEFAULT
    live = write_session(p, cfg)
    session = replay.load_session(p)
    out = replay.replay(session, cfg)
    assert len(out) == 1
    assert (out[0].ok, out[0].x, out[0].y, out[0].zone) == (live.ok, live.x, live.y, live.zone)


def test_replay_is_deterministic(tmp_path):
    p = tmp_path / "live.jsonl"
    write_session(p, config.FUSION_DEFAULT)
    session = replay.load_session(p)
    a = replay.replay(session, config.FUSION_DEFAULT)
    b = replay.replay(session, config.FUSION_DEFAULT)
    assert [(r.x, r.y, r.zone, r.ok) for r in a] == [(r.x, r.y, r.zone, r.ok) for r in b]


def test_replay_under_a_different_config_is_still_possible(tmp_path):
    """One recording, any configuration. That is the point of the harness."""
    p = tmp_path / "live.jsonl"
    write_session(p, config.FUSION_DEFAULT)
    session = replay.load_session(p)
    other = fusion.FusionConfig(lead_ms=0, window_ms=1000, aggregator="mean",
                                min_samples=3)
    assert replay.replay(session, other)[0].ok


def test_the_sweep_is_two_hundred_and_forty_configurations():
    """Table 7 lists four parameters; section 4.6 multiplies only three."""
    cfgs = replay.default_configs()
    assert len(cfgs) == 4 * 4 * 5 * 3 == 240
    assert len(set(cfgs)) == 240, "FusionConfig must be hashable and unique"


def test_the_headline_surface_is_the_eighty_cells_the_report_quotes():
    cfgs = replay.headline_configs()
    assert len(cfgs) == 80
    assert {c.min_samples for c in cfgs} == {config.FUSION_DEFAULT.min_samples}


def test_sweep_scores_every_configuration_against_ground_truth(tmp_path):
    p = tmp_path / "live.jsonl"
    write_session(p, config.FUSION_DEFAULT)
    session = replay.load_session(p)
    df = replay.sweep(session, replay.headline_configs())
    assert len(df) == 80
    assert {"lead_ms", "window_ms", "aggregator", "min_samples",
            "accuracy", "refused", "n_commands"} <= set(df.columns)
    assert df["accuracy"].between(0, 1).all()


def test_loading_a_session_recovers_gaze_utterances_and_truth(tmp_path):
    p = tmp_path / "live.jsonl"
    write_session(p, config.FUSION_DEFAULT)
    s = replay.load_session(p)
    assert len(s.gaze) == 60
    assert len(s.utterances) == 1 and s.utterances[0].text == "click"
    assert s.truth[0]["zone"] == 5
    assert s.screen == (1512, 982)
