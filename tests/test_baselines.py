"""C1 and C2 are interaction techniques we have to build, not just describe."""

from seentap import baselines


def test_dwell_fires_only_after_the_full_eight_hundred_milliseconds():
    d = baselines.DwellSelector(dwell_ms=800)
    assert d.update(zone=5, t=0.0) is None
    assert d.update(zone=5, t=0.5) is None
    assert d.update(zone=5, t=0.81) == 5


def test_dwell_resets_when_the_eyes_move_to_another_tile():
    d = baselines.DwellSelector(dwell_ms=800)
    d.update(zone=5, t=0.0)
    d.update(zone=6, t=0.7)          # looked away, timer restarts
    assert d.update(zone=6, t=0.9) is None
    assert d.update(zone=6, t=1.6) == 6


def test_dwell_fires_once_per_visit_not_every_frame():
    d = baselines.DwellSelector(dwell_ms=800)
    d.update(zone=5, t=0.0)
    assert d.update(zone=5, t=0.9) == 5
    assert d.update(zone=5, t=1.0) is None
    assert d.update(zone=5, t=2.0) is None


def test_dwell_can_refire_after_leaving_and_returning():
    d = baselines.DwellSelector(dwell_ms=800)
    d.update(zone=5, t=0.0)
    d.update(zone=5, t=0.9)
    d.update(zone=1, t=1.0)
    d.update(zone=5, t=1.1)
    assert d.update(zone=5, t=2.0) == 5


def test_dwell_matches_the_report_default():
    assert baselines.DwellSelector().dwell_ms == 800


def test_voice_only_baseline_selects_a_numbered_tile():
    zone, verb = baselines.voice_only_select("seven click", n_tiles=12)
    assert (zone, verb) == (6, "click")     # tile 7 is zero-indexed zone 6


def test_voice_only_refuses_an_unparseable_command():
    assert baselines.voice_only_select("banana", n_tiles=12) == (None, None)


def test_tile_labels_are_one_indexed_for_the_participant():
    labels = baselines.tile_labels(12)
    assert labels[0] == "1" and labels[-1] == "12"
