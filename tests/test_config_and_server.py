"""Constants come from the report; the app must at least assemble."""
from seentap import config


def test_grid_and_gate_come_from_the_report():
    assert (config.GRID_COLS, config.GRID_ROWS) == (4, 3)
    assert config.GATE_FRAC == 0.08
    assert config.DWELL_MS == 800
    assert config.COOLDOWN_MS == 250
    assert config.DEADZONE_PX == 15
    assert config.BLINK_HOLD_MS == 500
    assert config.DENSITIES == (5, 9, 13)


def test_screen_is_in_one_coordinate_space():
    """Retina: OpenCV pixels and PyAutoGUI points differ by 2x if we are careless."""
    assert config.SCREEN_W > 0 and config.SCREEN_H > 0
    assert isinstance(config.SCREEN_W, int)


def test_server_exposes_the_dashboard_and_the_socket():
    from seentap import server
    paths = {r.path for r in server.app.routes}
    assert {"/", "/health", "/ws"} <= paths


def test_dashboard_html_exists_and_is_self_contained():
    from seentap import server
    html = server.INDEX.read_text()
    assert "<canvas" in html
    assert "http://" not in html and "https://" not in html, "no external assets"


def test_a_mistyped_calibration_path_says_so_and_lists_what_there_is(tmp_path,
                                                                     capsys,
                                                                     monkeypatch):
    """It used to surface forty frames down inside sklearn as 'Expected 2D
    array, got 1D array instead: array=[]', which names nothing useful."""
    import pytest

    from seentap import config, eventlog, run

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    with eventlog.EventLog(tmp_path / "calib-9-1787629345.jsonl") as log:
        log.write("calibration", density=9, features_version=config.FEATURES_VERSION)
        for i in range(9):
            log.write("calib_point", f=[0.0] * 9, target=[i * 10.0, i * 10.0])

    with pytest.raises(SystemExit) as e:
        run._require_calib(str(tmp_path / "calib-9-1758100000.jsonl"))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "calib-9-1787629345.jsonl" in err, "point at the file they do have"

    F, XY = run._require_calib(str(tmp_path / "calib-9-1787629345.jsonl"))
    assert len(F) == 9


def test_a_real_but_useless_calibration_file_is_refused_too(tmp_path, capsys,
                                                           monkeypatch):
    import pytest

    from seentap import config, eventlog, run

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    path = tmp_path / "calib-9-1.jsonl"
    with eventlog.EventLog(path) as log:
        log.write("calibration", density=9, features_version=config.FEATURES_VERSION)
        log.write("calib_point", f=[0.0] * 9, target=[1.0, 1.0])

    with pytest.raises(SystemExit):
        run._require_calib(str(path))
    assert "1 calibration point(s)" in capsys.readouterr().err


def test_a_calibration_from_an_older_feature_layout_is_refused(tmp_path, capsys):
    """The file holds fitted feature vectors. Reading v1 vectors with v2
    features is silent nonsense, not a slightly worse fit."""
    import pytest

    from seentap import eventlog, run

    path = tmp_path / "calib-9-old.jsonl"
    with eventlog.EventLog(path) as log:
        log.write("calibration", density=9)          # no version: v1
        for i in range(9):
            log.write("calib_point", f=[0.0] * 9, target=[i * 10.0, i * 10.0])

    with pytest.raises(SystemExit):
        run._require_calib(str(path))
    err = capsys.readouterr().err
    assert "v1" in err and "recorded again" in err


def test_serve_picks_the_newest_usable_calibration(tmp_path, monkeypatch):
    """A run quit halfway leaves a short file behind; newest alone would take
    it. And the exact-filename requirement was a trap: the README's example
    timestamp is not yours."""
    from seentap import config, eventlog, run

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))

    def write(name, points, version=config.FEATURES_VERSION):
        with eventlog.EventLog(tmp_path / name) as log:
            log.write("calibration", density=9, features_version=version)
            for i in range(points):
                log.write("calib_point", f=[0.0] * 9, target=[i * 10.0, i * 10.0])

    write("calib-9-100.jsonl", 9)
    write("calib-9-200.jsonl", 2)               # quit halfway
    write("calib-9-300.jsonl", 9, version=1)    # older feature layout
    assert run._newest_calib().endswith("calib-9-100.jsonl")

    write("calib-9-400.jsonl", 9)
    assert run._newest_calib().endswith("calib-9-400.jsonl")


def test_no_calibration_at_all_is_a_message_not_a_crash(tmp_path, monkeypatch,
                                                        capsys):
    import pytest

    from seentap import config, run

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    assert run._newest_calib() is None
    with pytest.raises(SystemExit):
        run._require_calib(run._newest_calib())
    assert "calibrate --density 9" in capsys.readouterr().err


def test_a_damaged_run_is_filed_under_the_density_it_asked_for(tmp_path):
    """A nine-point pass that lost four targets is a damaged nine, not a five.
    Keying by surviving point count invented a density row that was never run,
    and its error landed in the table as if it were a real result."""
    from seentap import config, eventlog, run

    path = tmp_path / "calib-9-partial.jsonl"
    with eventlog.EventLog(path) as log:
        log.write("calibration", density=9,
                  features_version=config.FEATURES_VERSION)
        for i in range(5):
            log.write("calib_point", f=[0.0] * 9, target=[i * 10.0, i * 10.0])

    F, _XY, version, density = run._load_calib(str(path))
    assert len(F) == 5 and density == 9
    assert version == config.FEATURES_VERSION


def test_a_file_with_no_header_falls_back_to_its_point_count(tmp_path):
    from seentap import eventlog, run

    path = tmp_path / "calib-headerless.jsonl"
    with eventlog.EventLog(path) as log:
        for i in range(7):
            log.write("calib_point", f=[0.0] * 9, target=[float(i), float(i)])
    assert run._load_calib(str(path))[3] == 7


def test_the_newest_calibration_is_the_newest_by_clock_not_by_name(tmp_path,
                                                                   monkeypatch):
    """Sorting names put calib-9- ahead of calib-49-, because as text "9" beats
    "4". A nine-point file from any time outranked every denser calibration
    ever recorded, and two density comparisons were run against a stale one
    without either new file being read."""
    import os

    from seentap import config, eventlog, run

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))

    def write(name, points, when):
        path = tmp_path / name
        with eventlog.EventLog(path) as log:
            log.write("calibration", density=points,
                      features_version=config.FEATURES_VERSION)
            for i in range(points):
                log.write("calib_point", f=[0.0] * 9, target=[float(i), float(i)])
        os.utime(path, (when, when))

    write("calib-9-100.jsonl", 9, when=1000)
    write("calib-49-200.jsonl", 49, when=2000)      # newer, but sorts earlier
    write("calib-25-300.jsonl", 25, when=3000)      # newest of all

    assert run._newest_calib().endswith("calib-25-300.jsonl")

    order = [os.path.basename(p) for p in
             run._by_age([str(tmp_path / n) for n in
                          ("calib-25-300.jsonl", "calib-9-100.jsonl",
                           "calib-49-200.jsonl")])]
    assert order == ["calib-9-100.jsonl", "calib-49-200.jsonl",
                     "calib-25-300.jsonl"]
