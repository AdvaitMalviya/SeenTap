"""Study 1: three mappings by three densities, scored on a held-out grid."""
import numpy as np
import pytest

from seentap import calibrate, config


def synth(n_points=9, noise=0.0, seed=0):
    """Screen coords generated from features by a known quadratic map."""
    rng = np.random.default_rng(seed)
    hx = rng.uniform(-0.4, 0.4, n_points)
    vy = rng.uniform(-0.4, 0.4, n_points)
    F = np.column_stack([
        hx, hx * 1.01, vy, vy * 0.99,
        np.zeros(n_points), np.zeros(n_points), np.zeros(n_points),
        np.full(n_points, 0.12), np.ones(n_points),
    ])
    X = 756 + 1400 * hx + 900 * hx ** 2
    Y = 491 + 900 * vy + 600 * vy ** 2
    XY = np.column_stack([X, Y]) + rng.normal(0, noise, (n_points, 2))
    return F, XY


@pytest.mark.parametrize("name", ["ridge", "poly", "homography"])
def test_every_mapping_fits_and_predicts_the_right_shape(name):
    F, XY = synth(13)
    model = calibrate.FITTERS[name](F, XY)
    pred = model.predict(F)
    assert pred.shape == XY.shape
    assert np.all(np.isfinite(pred))


def test_polynomial_beats_linear_on_a_quadratic_truth():
    """The report fits both precisely because the true map is not linear."""
    F, XY = synth(13)
    Fv, XYv = synth(5, seed=99)
    lin = calibrate.validate(calibrate.fit_ridge(F, XY), Fv, XYv)
    quad = calibrate.validate(calibrate.fit_poly(F, XY), Fv, XYv)
    assert quad["mean_err"] < lin["mean_err"]


def test_validate_reports_the_columns_the_report_asks_for():
    F, XY = synth(9)
    v = calibrate.validate(calibrate.fit_poly(F, XY), *synth(5, seed=3))
    assert set(v) >= {"mean_err", "std_err", "mean_dx", "mean_dy", "n"}
    assert v["n"] == 5
    assert v["mean_err"] >= 0


def test_loo_cv_picks_an_alpha_from_the_offered_grid():
    F, XY = synth(13, noise=6.0)
    alphas = [1e-4, 1e-2, 1.0, 100.0]
    a = calibrate.loo_cv_alpha(F, XY, calibrate.fit_ridge, alphas)
    assert a in alphas


def test_nine_cell_table_has_nine_cells():
    sessions = {d: synth(d, noise=4.0, seed=d) for d in config.DENSITIES}
    held = synth(5, noise=4.0, seed=77)
    table = calibrate.nine_cell(sessions, held)
    assert len(table) == len(config.DENSITIES) * len(calibrate.FITTERS)
    assert {r["density"] for r in table} == set(config.DENSITIES)
    assert {r["mapping"] for r in table} == set(calibrate.FITTERS)


def test_gate_compares_the_best_cell_against_eight_percent_of_width():
    good = [{"density": 9, "mapping": "poly", "mean_err": 80.0}]
    bad = [{"density": 9, "mapping": "poly", "mean_err": 400.0}]
    assert calibrate.gate_passed(good, 1512)[0] is True
    assert calibrate.gate_passed(bad, 1512)[0] is False
    assert calibrate.gate_passed(good, 1512)[1] == pytest.approx(0.08 * 1512)


def test_blinks_and_low_confidence_are_dropped_before_the_median():
    raw = [
        {"f": [1.0] * 9, "conf": 0.9, "blink": False},
        {"f": [9.0] * 9, "conf": 0.9, "blink": True},    # blink
        {"f": [9.0] * 9, "conf": 0.1, "blink": False},   # low confidence
        {"f": [3.0] * 9, "conf": 0.9, "blink": False},
    ]
    kept = calibrate.condense(raw)
    np.testing.assert_allclose(kept, np.full(9, 2.0))


def test_condense_returns_none_when_nothing_survives():
    assert calibrate.condense([{"f": [1.0] * 9, "conf": 0.9, "blink": True}]) is None
