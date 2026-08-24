"""The one test that exercises MediaPipe itself, on a real face.

Uses MediaPipe's own published test portrait if it has been fetched. Skipped
otherwise, so the suite still runs on a machine with no models directory.
"""
from pathlib import Path

import numpy as np
import pytest

from seentap import config, gaze

PORTRAIT = Path(config.MODEL_DIR) / "portrait.jpg"
MODEL = Path(config.MODEL_DIR) / config.FACE_MODEL
pytestmark = pytest.mark.skipif(
    not (PORTRAIT.exists() and MODEL.exists()),
    reason="run `python -m seentap.run fetch` to download the model and portrait")


@pytest.fixture(scope="module")
def detected():
    import cv2

    rgb = cv2.cvtColor(cv2.imread(str(PORTRAIT)), cv2.COLOR_BGR2RGB)
    tracker = gaze.GazeTracker(camera=None)
    lm = tracker.detect(rgb)
    tracker.close()
    h, w = rgb.shape[:2]
    return lm, w, h


def test_the_landmarker_returns_the_full_refined_mesh(detected):
    lm, _, _ = detected
    assert lm is not None, "no face found in the reference portrait"
    assert lm.shape == (478, 3), "478 points means the 10 iris points are present"


def test_the_iris_indices_sit_between_the_eye_corners(detected):
    """Indices have shifted between MediaPipe releases. Catch it here, not on
    day 8 when the calibration error looks inexplicable."""
    lm, w, h = detected
    px = lm[:, :2] * [w, h]
    for iris, outer, inner in ((gaze.L_IRIS, gaze.L_OUTER, gaze.L_INNER),
                               (gaze.R_IRIS, gaze.R_INNER, gaze.R_OUTER)):
        lo, hi = sorted([px[outer][0], px[inner][0]])
        assert lo <= px[iris][0] <= hi
        assert abs(px[iris][1] - (px[outer][1] + px[inner][1]) / 2) < 0.5 * (hi - lo)


def test_open_eyes_are_not_reported_as_a_blink(detected):
    lm, w, h = detected
    ear = gaze.eye_aspect_ratio(lm, w, h)
    assert ear > config.BLINK_EAR, f"open eyes read {ear:.3f}, threshold too high"
    assert 0.10 < ear < 0.45, "outside any plausible range for this landmark set"


def test_a_near_frontal_face_gives_a_near_zero_head_pose(detected):
    """A Y-up canonical model against Y-down image coordinates returns the
    flipped solution, and pitch comes back near 180 degrees."""
    lm, w, h = detected
    yaw, pitch, roll = gaze.head_pose(lm, w, h)
    for angle in (yaw, pitch, roll):
        assert abs(np.degrees(angle)) < 30


def test_the_feature_vector_is_finite_and_shaped(detected):
    lm, w, h = detected
    f = gaze.features(lm, w, h)
    assert f.shape == (9,) and np.all(np.isfinite(f))
    assert f[7] > 0, "interocular distance must be positive"
