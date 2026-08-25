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


def test_features_that_never_moved_are_dropped():
    """The bug that put predictions 1400 px off the screen. Told to sit still,
    the user varies head pose by half a degree, but least squares still hands
    it a coefficient to mop up residuals -- ridge priced pitch at -3474 px per
    radian. Sit down 9 degrees different the next day and that one column
    swings the prediction 638 px."""
    rng = np.random.default_rng(0)
    F = np.zeros((9, 9))
    F[:, 0] = F[:, 1] = rng.normal(0, 0.05, 9)      # eye ratios: real signal
    F[:, 2] = F[:, 3] = rng.normal(0, 0.04, 9)
    F[:, 4:7] = rng.normal(0, 0.008, (9, 3))        # pose: sitting still
    F[:, 7] = 0.079 + rng.normal(0, 0.0007, 9)      # interocular: barely moves
    F[:, 8] = 1.0

    kept = calibrate.useful_columns(F)
    assert set(kept) == {0, 1, 2, 3, 8}, "only the eye ratios and the bias"


def test_a_pose_that_did_vary_is_kept():
    """Not a blanket ban: if a calibration actually samples head movement, the
    mapping has seen the relationship and may use it."""
    rng = np.random.default_rng(1)
    F = np.zeros((9, 9))
    F[:, :4] = rng.normal(0, 0.05, (9, 4))
    F[:, 4:7] = rng.normal(0, 0.3, (9, 3))          # ~17 degrees of real range
    F[:, 7] = 0.079 + rng.normal(0, 0.02, 9)
    F[:, 8] = 1.0
    assert set(calibrate.useful_columns(F)) == set(range(9))


def test_a_fitted_mapping_ignores_the_columns_it_dropped():
    """Callers hand over the whole vector; the mapping takes its own slice."""
    rng = np.random.default_rng(2)
    F = np.zeros((9, 9))
    F[:, 0] = F[:, 1] = np.linspace(-0.1, 0.1, 9)
    F[:, 2] = F[:, 3] = np.linspace(-0.08, 0.08, 9)
    F[:, 4:7] = rng.normal(0, 0.005, (9, 3))
    F[:, 7] = 0.079
    F[:, 8] = 1.0
    XY = np.column_stack([np.linspace(100, 1400, 9), np.linspace(100, 900, 9)])

    m = calibrate.fit_ridge(F, XY)
    before = m.predict(F[4:5])[0]
    moved = F[4:5].copy()
    moved[0, 4] += np.radians(10)                   # a real head turn
    moved[0, 5] += np.radians(10)
    after = m.predict(moved)[0]
    assert np.allclose(before, after), "a dropped column cannot move the answer"


def test_bigger_densities_lay_out_without_long_jumps():
    """A full-width jump costs a large saccade and the eye has to be waited out
    again on the far side."""
    small = calibrate.targets(9, 1512, 982)
    big = calibrate.targets(49, 1512, 982)
    assert len(big) > 4 * len(small)

    def mean_jump(t):
        return np.mean([np.hypot(*(np.array(t[i + 1]) - np.array(t[i])))
                        for i in range(len(t) - 1)])

    assert mean_jump(big) < mean_jump(small) / 2


@pytest.mark.parametrize("density", config.DENSITY_CHOICES)
def test_every_offered_density_stays_on_screen(density):
    pts = calibrate.targets(density, 1512, 982)
    assert len(pts) >= 4
    assert all(0 <= x <= 1512 and 0 <= y <= 982 for x, y in pts)


@pytest.mark.parametrize("density", config.DENSITY_CHOICES)
def test_every_offered_density_collects_the_number_of_points_it_names(density):
    """Deriving rows and columns by rounding a square root did not multiply
    back to the density: 25 laid out as 4x6 and 49 as 5x10, so `--density 25`
    recorded 24 points and `--density 49` recorded 50. The log header keeps
    writing the requested number and `fit` keys its density column on it, so
    two of the three rows of the Study 1 table carried a point count that was
    never collected -- and the two real dense recordings on disk still do."""
    assert len(calibrate.targets(density, 1512, 982)) == density


def test_a_density_with_no_factor_pair_is_refused_rather_than_rounded():
    """Silently collecting a different number of points is what mislabelled the
    density table; a prime says so instead."""
    with pytest.raises(ValueError, match="factor pair"):
        calibrate.targets(23, 1512, 982)
