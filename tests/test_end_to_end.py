"""A whole session end to end: gaze in, actions out, log, replay, sweep, CLI.

No camera and no microphone. The synthetic participant looks at the cued tile
shortly before speaking and moves on shortly after, which is the timing the
fusion window exists to exploit.
"""
import numpy as np
import pytest

from seentap import calibrate, config, eventlog, fusion, replay, run
from seentap.actions import SimExecutor
from seentap.gaze import GazeSample, zone_rect

HZ = 30.0


def _gaze_run(zone, t0, t1, out):
    x0, y0, x1, y1 = zone_rect(zone, 1512, 982)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    t = t0
    while t < t1:
        out.append(GazeSample(t=t, x=cx, y=cy, conf=0.9, zone=zone, blink=False))
        t += 1 / HZ


def synth_session(path, n_trials=8, cfg=None):
    """Drive the real Fusion machine and the real executor, and log it all."""
    cfg = cfg or fusion.FusionConfig()
    f = fusion.Fusion(cfg)
    ex = SimExecutor()
    targets = [(i * 5) % config.N_TILES for i in range(n_trials)]

    with eventlog.EventLog(path) as log:
        log.write("session", t=0.0, condition="C3", mode="B",
                  config=cfg.__dict__, screen=[1512, 982])
        t = 100.0
        for i, target in enumerate(targets):
            previous = targets[i - 1] if i else (target + 1) % config.N_TILES
            nxt = (target + 3) % config.N_TILES
            onset = t + 1.0
            samples = []
            _gaze_run(previous, t, onset - 0.25, samples)   # still on the last tile
            _gaze_run(target, onset - 0.25, onset + 0.05, samples)
            _gaze_run(nxt, onset + 0.05, onset + 0.65, samples)

            decode_ms = 400.0
            now = onset + decode_ms / 1000.0
            fired = False
            for s in samples:
                f.on_gaze(s)
                log.write("gaze", t=s.t, x=s.x, y=s.y, conf=s.conf,
                          zone=s.zone, blink=s.blink)
                if not fired and s.t >= now:
                    fired = True
                    log.write("truth", t=onset, zone=target)
                    r = f.on_utterance(onset_t=onset, text="click", now=now)
                    log.write("utterance", t=onset, onset_t=onset,
                              offset_t=onset + 0.3, text="click",
                              decode_ms=decode_ms)
                    if r.ok:
                        ex.execute(r.verb, r.x, r.y)
                    log.write("action", ok=r.ok, reason=r.reason, verb=r.verb,
                              x=r.x, y=r.y, zone=r.zone, n=r.n, onset_t=onset)
            t = samples[-1].t + 0.5
    return ex, targets


def test_a_full_session_executes_one_action_per_trial(tmp_path):
    p = tmp_path / "session.jsonl"
    ex, targets = synth_session(p)
    assert len(ex.events) == len(targets), "every trial produced an action"
    assert all(e["verb"] == "click" for e in ex.events)


def test_the_session_binds_to_the_cued_tile_every_time(tmp_path):
    p = tmp_path / "session.jsonl"
    synth_session(p)
    actions = [r for r in eventlog.read(p) if r["kind"] == "action"]
    truth = [r for r in eventlog.read(p) if r["kind"] == "truth"]
    assert len(actions) == len(truth) == 8
    assert [a["zone"] for a in actions] == [t["zone"] for t in truth]


def test_the_log_contains_everything_the_report_needs(tmp_path):
    p = tmp_path / "session.jsonl"
    synth_session(p)
    kinds = {r["kind"] for r in eventlog.read(p)}
    assert kinds == {"session", "gaze", "truth", "utterance", "action"}


def test_no_imagery_or_audio_reached_the_log(tmp_path):
    """The consent form, checked against the artefact rather than the intent."""
    p = tmp_path / "session.jsonl"
    synth_session(p)
    for row in eventlog.read(p):
        assert not (eventlog.FORBIDDEN & set(row))


def test_replay_agrees_with_the_live_run_action_for_action(tmp_path):
    p = tmp_path / "session.jsonl"
    cfg = fusion.FusionConfig()
    synth_session(p, cfg=cfg)
    session = replay.load_session(p)
    out = replay.replay(session, cfg)
    live = [r for r in eventlog.read(p) if r["kind"] == "action"]
    assert len(out) == len(live)
    for got, want in zip(out, live):
        assert got.ok == want["ok"]
        assert got.zone == want["zone"]
        assert got.x == pytest.approx(want["x"])


def test_the_full_sweep_runs_and_the_parameters_actually_matter(tmp_path):
    p = tmp_path / "session.jsonl"
    synth_session(p)
    session = replay.load_session(p)
    df = replay.sweep(session, replay.default_configs())
    assert len(df) == 240
    assert df["accuracy"].nunique() > 1, "a flat surface would mean no trade-off"

    default = config.FUSION_DEFAULT
    row = df[(df.lead_ms == default.lead_ms) & (df.window_ms == default.window_ms)
             & (df.aggregator == default.aggregator)
             & (df.min_samples == default.min_samples)]
    assert row["accuracy"].iloc[0] == 1.0, "the shipped default must bind correctly"


def test_a_window_too_narrow_for_its_minimum_refuses_rather_than_guesses(tmp_path):
    p = tmp_path / "session.jsonl"
    synth_session(p)
    df = replay.sweep(replay.load_session(p), replay.default_configs())
    tight = df[(df.window_ms == 100) & (df.min_samples == 8)]
    assert (tight["refused"] == 1.0).all()


def test_too_large_a_lead_binds_to_the_previous_target(tmp_path):
    """The failure mode the sweep is meant to expose, reproduced on purpose."""
    p = tmp_path / "session.jsonl"
    synth_session(p)
    df = replay.sweep(replay.load_session(p), replay.default_configs())
    over = df[(df.lead_ms == 300) & (df.window_ms == 100) & (df.min_samples == 3)]
    # 'last' can still catch the first target sample on the window edge; the
    # aggregators that pool over the window all land on the previous tile.
    assert over[over.aggregator == "median"]["accuracy"].iloc[0] < 1.0
    assert (over["accuracy"] == 0.0).sum() >= 4


# --- CLI -------------------------------------------------------------------

def _write_calibration(path, density, seed=0, noise=3.0):
    rng = np.random.default_rng(seed)
    pts = calibrate.targets(density, 1512, 982)
    with eventlog.EventLog(path) as log:
        log.write("calibration", density=density, screen=[1512, 982],
                  features_version=config.FEATURES_VERSION)
        for (tx, ty) in pts:
            hx = (tx / 1512 - 0.5) * 0.8
            vy = (ty / 982 - 0.5) * 0.8
            f = [hx, hx * 1.01, vy, vy * 0.99, 0.0, 0.0, 0.0, 0.12, 1.0]
            f = list(np.asarray(f) + rng.normal(0, 1e-3, 9))
            log.write("calib_point", f=f, target=[tx, ty], n_raw=30)


def test_cli_fit_prints_the_table_and_a_gate_verdict(tmp_path, capsys):
    for d in config.DENSITIES:
        _write_calibration(tmp_path / f"calib-{d}.jsonl", d, seed=d)
    held = tmp_path / "held.jsonl"
    _write_calibration(held, 5, seed=99)

    rc = run.main(["fit", "--pattern", str(tmp_path / "calib-*.jsonl"),
                   "--held", str(held), "--save", str(tmp_path / "t.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| density |" in out
    assert "day-8 gate:" in out
    assert ("PASS" in out) or ("FAIL" in out)
    assert (tmp_path / "t.json").exists()


def test_cli_sweep_writes_a_csv(tmp_path, capsys):
    p = tmp_path / "session.jsonl"
    synth_session(p)
    out_csv = tmp_path / "sweep.csv"
    rc = run.main(["sweep", str(p), "--headline", "--out", str(out_csv)])
    assert rc == 0
    assert "80 configurations" in capsys.readouterr().out
    assert out_csv.exists()


def test_cli_report_summarises_the_three_conditions(tmp_path, capsys):
    rng = np.random.default_rng(1)
    for cond, base in (("C1", 5.0), ("C2", 6.0), ("C3", 3.5)):
        with eventlog.EventLog(tmp_path / f"{cond}.jsonl") as log:
            log.write("session", condition=cond, mode="B")
            for i in range(5):
                log.write("trial", completion_s=float(base + rng.normal(0, .2)),
                          correct=True)
    rc = run.main(["report", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| condition |" in out
    assert "Friedman" in out
