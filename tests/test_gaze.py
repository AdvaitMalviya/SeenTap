"""Feature extraction, smoothing and the zone grid. No camera involved."""
import numpy as np
import pytest

from seentap import config, gaze


def synth_landmarks(shift=(0.0, 0.0), scale=1.0, iris_offset=0.0):
    """478 landmarks laid out so the eye geometry is known analytically."""
    lm = np.zeros((478, 3), dtype=float)
    # left eye: outer 33, inner 133, top 159, bottom 145
    lm[33] = (0.30, 0.40, 0.0)
    lm[133] = (0.40, 0.40, 0.0)
    lm[159] = (0.35, 0.37, 0.0)
    lm[145] = (0.35, 0.43, 0.0)
    lm[160], lm[158] = (0.33, 0.37, 0.0), (0.37, 0.37, 0.0)   # upper lid
    lm[144], lm[153] = (0.33, 0.43, 0.0), (0.37, 0.43, 0.0)   # lower lid
    # right eye: inner 362, outer 263, top 386, bottom 374
    lm[362] = (0.60, 0.40, 0.0)
    lm[263] = (0.70, 0.40, 0.0)
    lm[386] = (0.65, 0.37, 0.0)
    lm[374] = (0.65, 0.43, 0.0)
    lm[385], lm[387] = (0.63, 0.37, 0.0), (0.67, 0.37, 0.0)
    lm[380], lm[373] = (0.63, 0.43, 0.0), (0.67, 0.43, 0.0)
    # iris centres 468 (left) and 473 (right), plus their rings
    lm[468] = (0.35 + iris_offset, 0.40, 0.0)
    lm[473] = (0.65 + iris_offset, 0.40, 0.0)
    for i in range(469, 473):
        lm[i] = lm[468]
    for i in range(474, 478):
        lm[i] = lm[473]
    # a few face points so head pose has something to solve on
    lm[1] = (0.50, 0.50, 0.0)
    lm[152] = (0.50, 0.75, 0.0)
    lm[61] = (0.45, 0.62, 0.0)
    lm[291] = (0.55, 0.62, 0.0)
    lm = lm * scale
    lm[:, 0] += shift[0]
    lm[:, 1] += shift[1]
    return lm


def test_feature_vector_is_nine_dimensional():
    f = gaze.features(synth_landmarks(), 640, 480)
    assert f.shape == (9,)
    assert np.all(np.isfinite(f))
    assert f[-1] == 1.0, "last element is the bias term"


def test_iris_ratios_track_iris_movement():
    centre = gaze.features(synth_landmarks(iris_offset=0.0), 640, 480)
    right = gaze.features(synth_landmarks(iris_offset=0.02), 640, 480)
    assert right[0] > centre[0]
    assert right[1] > centre[1]


def test_ratios_survive_translation():
    """A raw iris coordinate moves with the head; the ratios must not."""
    a = gaze.features(synth_landmarks(), 640, 480)
    b = gaze.features(synth_landmarks(shift=(0.05, -0.03)), 640, 480)
    np.testing.assert_allclose(a[:4], b[:4], atol=1e-9)


def test_distance_proxy_falls_as_the_face_shrinks():
    near = gaze.features(synth_landmarks(scale=1.0), 640, 480)
    far = gaze.features(synth_landmarks(scale=0.5), 640, 480)
    assert far[7] < near[7]


def test_eye_aspect_ratio_detects_a_closed_eye():
    open_lm = synth_landmarks()
    closed = synth_landmarks()
    for i in (159, 145, 160, 158, 144, 153):
        closed[i] = (closed[i][0], 0.40, 0.0)
    for i in (386, 374, 385, 387, 380, 373):
        closed[i] = (closed[i][0], 0.40, 0.0)
    assert gaze.eye_aspect_ratio(open_lm, 640, 480) > config.BLINK_EAR
    assert gaze.eye_aspect_ratio(closed, 640, 480) < config.BLINK_EAR


def test_eye_aspect_ratio_is_computed_in_pixel_space():
    """Landmarks are normalised per axis, so a 16:9 frame squashes every
    vertical distance and the ratio drifts with the camera's aspect ratio."""
    lm = synth_landmarks()
    square = gaze.eye_aspect_ratio(lm, 500, 500)
    wide = gaze.eye_aspect_ratio(lm, 640, 360)
    assert wide == pytest.approx(square * (360 / 640), rel=1e-6)


def test_blink_threshold_is_measured_per_user():
    assert gaze.blink_threshold([0.20] * 20) == pytest.approx(0.11)
    assert gaze.blink_threshold([]) == config.BLINK_EAR


# --- One Euro filter -------------------------------------------------------

def test_filter_reduces_jitter_on_a_held_fixation():
    rng = np.random.default_rng(0)
    f = gaze.OneEuro()
    raw, smoothed = [], []
    for i in range(120):
        t = i / 30.0
        x = 500.0 + rng.normal(0, 8.0)
        y = 400.0 + rng.normal(0, 8.0)
        raw.append((x, y))
        smoothed.append(f(x, y, t))
    raw_rms = np.std(np.array(raw), axis=0).mean()
    out_rms = np.std(np.array(smoothed)[30:], axis=0).mean()
    assert out_rms < raw_rms / 2, "smoothing must actually smooth"


def test_filter_still_follows_a_saccade():
    """Smooth too hard and the cursor lags behind a real eye movement."""
    f = gaze.OneEuro()
    for i in range(30):
        f(100.0, 100.0, i / 30.0)
    for i in range(30, 45):
        x, y = f(900.0, 100.0, i / 30.0)
    assert x > 0.9 * 900.0, "should reach 90% of a step within ~0.5 s"


def test_filter_reset_clears_history():
    f = gaze.OneEuro()
    f(0.0, 0.0, 0.0)
    f.reset()
    assert f(500.0, 500.0, 1.0) == (500.0, 500.0)


# --- zone grid -------------------------------------------------------------

def test_zone_grid_covers_every_tile_exactly_once():
    w, h = 1512, 982
    seen = set()
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            x = (col + 0.5) * w / config.GRID_COLS
            y = (row + 0.5) * h / config.GRID_ROWS
            seen.add(gaze.resolve_zone(x, y, w, h))
    assert seen == set(range(config.GRID_COLS * config.GRID_ROWS))


def test_zone_is_clamped_at_the_far_edge():
    """int(x/W*4) yields a fifth column at exactly x == W."""
    w, h = 1512, 982
    last = config.GRID_COLS * config.GRID_ROWS - 1
    assert gaze.resolve_zone(w, h, w, h) == last
    assert gaze.resolve_zone(w + 50, h + 50, w, h) == last
    assert gaze.resolve_zone(-10, -10, w, h) == 0


def test_zone_rect_round_trips_through_its_own_centre():
    w, h = 1512, 982
    for z in range(config.GRID_COLS * config.GRID_ROWS):
        x0, y0, x1, y1 = gaze.zone_rect(z, w, h)
        assert gaze.resolve_zone((x0 + x1) / 2, (y0 + y1) / 2, w, h) == z


def test_tile_is_larger_than_the_gate_threshold():
    """Mode B only works if a tile dwarfs the tracker error."""
    w = config.SCREEN_W
    tile_w = w / config.GRID_COLS
    assert tile_w / 2 > config.GATE_FRAC * w


def _vertical(lm, frame=(640, 480)):
    return gaze.features(lm, *frame)[2]


def test_the_vertical_feature_ignores_the_eyelids():
    """It used to divide by the eyelid gap and measure from the upper lid, so
    it tracked the lid rather than the eye. On a real frame, moving only the
    lids swung it 0.49 -- more than actually moving the eye did -- and the lid
    descends with your gaze, so during calibration the two cancelled: vertical
    correlated -0.32 with the vertical target where horizontal managed +0.96.
    """
    lm = synth_landmarks()
    before = _vertical(lm)

    lids = lm.copy()
    for i in (159, 145, 386, 374):          # both lids of both eyes descend
        lids[i, 1] += 0.01
    assert _vertical(lids) == pytest.approx(before), "eyelids are not gaze"

    squint = lm.copy()
    for i in (159, 386):                    # upper lids alone: the eye narrows
        squint[i, 1] += 0.012
    assert _vertical(squint) == pytest.approx(before), "a squint is not gaze"


def test_the_vertical_feature_does_follow_the_eye():
    lm = synth_landmarks()
    up, down = lm.copy(), lm.copy()
    for i in (468, 473):
        up[i, 1] -= 0.02
        down[i, 1] += 0.02
    assert _vertical(up) < _vertical(lm) < _vertical(down)


def test_both_axes_share_one_rigid_denominator():
    """The lid gap was four times smaller than the eye width, so whatever
    vertical signal survived came out amplified into noise."""
    lm = synth_landmarks()
    moved = lm.copy()
    for i in (468, 473):
        moved[i, 0] += 0.02                 # horizontal move
    dh = abs(gaze.features(moved, 640, 480)[0] - gaze.features(lm, 640, 480)[0])
    moved = lm.copy()
    for i in (468, 473):
        moved[i, 1] += 0.02                 # same distance, vertical
    dv = abs(_vertical(moved) - _vertical(lm))
    assert dh == pytest.approx(dv, rel=0.5), "one axis must not out-shout the other"
