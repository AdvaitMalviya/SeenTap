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
        log.write("calib_point", f=[0.0] * 9, target=[1.0, 1.0])

    with pytest.raises(SystemExit):
        run._require_calib(str(path))
    assert "1 calibration point(s)" in capsys.readouterr().err
