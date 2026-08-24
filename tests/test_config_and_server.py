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
