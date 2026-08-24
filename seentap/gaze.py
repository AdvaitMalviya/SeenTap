"""Landmarks in, a smoothed screen point out.

The pure functions here (features, eye_aspect_ratio, OneEuro, resolve_zone) are
what the tests exercise. GazeTracker is the thin I/O shell around MediaPipe and
imports it lazily, so nothing that only needs the maths pays for the model.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

import numpy as np

from seentap import config

# MediaPipe Face Mesh indices, refine_landmarks=True (478 points).
# Verify these against the installed model version before trusting a session:
# `python -m seentap.run landmarks` draws them on a live frame.
L_OUTER, L_INNER, L_TOP, L_BOT = 33, 133, 159, 145
R_INNER, R_OUTER, R_TOP, R_BOT = 362, 263, 386, 374
L_IRIS, R_IRIS = 468, 473
NOSE_TIP, CHIN, MOUTH_L, MOUTH_R = 1, 152, 61, 291

# Six points per eye for the published Eye Aspect Ratio formula, in the order
# (outer, upper1, upper2, inner, lower2, lower1). The two-point lid gap above
# is not the same quantity and does not share the literature's 0.21 threshold.
EAR_LEFT = (33, 160, 158, 133, 153, 144)
EAR_RIGHT = (362, 385, 387, 263, 373, 380)

# Canonical 3D face model (mm), matched to the six image points below.
# Y grows downward, matching image coordinates. With the usual Y-up model the
# solver returns the flipped solution and pitch comes back near +/-180 degrees.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, 63.6, -12.5),       # chin
    (-43.3, -32.7, -26.0),    # left eye outer
    (43.3, -32.7, -26.0),     # right eye outer
    (-28.9, 28.9, -24.1),     # mouth left
    (28.9, 28.9, -24.1),      # mouth right
], dtype=float)


@dataclass
class GazeSample:
    t: float
    x: float
    y: float
    conf: float
    zone: int | None = None
    blink: bool = False
    drift_deg: float | None = None


def _xy(lm: np.ndarray, i: int) -> np.ndarray:
    return np.asarray(lm[i][:2], dtype=float)


def eye_aspect_ratio(lm: np.ndarray, frame_w: float = 1.0,
                     frame_h: float = 1.0) -> float:
    """The six-point EAR, averaged over both eyes, in pixel space.

    (|p2-p6| + |p3-p5|) / (2 |p1-p4|). The frame dimensions are not optional
    detail: landmarks arrive normalised to [0,1] on each axis independently, so
    on a 16:9 frame every vertical distance is compressed relative to every
    horizontal one and the ratio comes out too small by the aspect ratio.

    Note the absolute scale is landmark-set specific. On this mesh an open eye
    reads about 0.20, not the 0.25-0.35 that dlib's 68-point model gives, so
    the dlib threshold of 0.21 would call every frame a blink. See
    ``blink_threshold``.
    """
    scale = np.array([frame_w, frame_h], dtype=float)
    ears = []
    for p1, p2, p3, p4, p5, p6 in (EAR_LEFT, EAR_RIGHT):
        width = np.linalg.norm((_xy(lm, p1) - _xy(lm, p4)) * scale)
        if width <= 1e-9:
            continue
        ears.append((np.linalg.norm((_xy(lm, p2) - _xy(lm, p6)) * scale)
                     + np.linalg.norm((_xy(lm, p3) - _xy(lm, p5)) * scale))
                    / (2 * width))
    return float(np.mean(ears)) if ears else 0.0


def blink_threshold(open_ears) -> float:
    """Half of this user's open-eye EAR, measured during calibration.

    Eye shape varies enough between people that a fixed constant misfires at
    both ends. The knob is cheap: calibration already holds the eyes open on a
    target for a second, so the samples are there for free.
    """
    ears = [e for e in open_ears if e > 0]
    if not ears:
        return config.BLINK_EAR
    return float(np.clip(np.median(ears) * 0.55, 0.05, 0.25))


def head_pose(lm: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """Yaw, pitch, roll in radians via solvePnP on a canonical face.

    Yaw is the term that matters: a flat camera cannot absorb head rotation, so
    the mapping is given it explicitly rather than left to discover it.
    """
    import cv2

    pts = np.array([
        _xy(lm, NOSE_TIP), _xy(lm, CHIN), _xy(lm, L_OUTER),
        _xy(lm, R_OUTER), _xy(lm, MOUTH_L), _xy(lm, MOUTH_R),
    ], dtype=float) * np.array([frame_w, frame_h])
    f = float(frame_w)
    cam = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]], dtype=float)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy < 1e-6:
        return 0.0, float(math.atan2(-rmat[2, 0], sy)), 0.0
    pitch = math.atan2(rmat[2, 1], rmat[2, 2])
    yaw = math.atan2(-rmat[2, 0], sy)
    roll = math.atan2(rmat[1, 0], rmat[0, 0])
    return float(yaw), float(pitch), float(roll)


# Where head pose lives inside the feature vector.
YAW, PITCH, IOD = 4, 5, 7


def features(lm: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """The nine-dimensional feature vector from Table 6.

    [hL, hR, vL, vR, yaw, pitch, roll, interocular, 1]

    Iris position is expressed as a ratio inside its own eye, so it survives the
    head translating across the frame; head pose and the distance proxy give the
    mapping the geometry it needs to absorb the rest.
    """
    hs, vs = [], []
    for iris, inner, outer, top, bot in (
        (L_IRIS, L_INNER, L_OUTER, L_TOP, L_BOT),
        (R_IRIS, R_INNER, R_OUTER, R_TOP, R_BOT),
    ):
        ic = _xy(lm, iris)
        width = abs(_xy(lm, outer)[0] - _xy(lm, inner)[0])
        lid = abs(_xy(lm, top)[1] - _xy(lm, bot)[1])
        hs.append((ic[0] - _xy(lm, inner)[0]) / width if width > 1e-9 else 0.0)
        vs.append((ic[1] - _xy(lm, top)[1]) / lid if lid > 1e-9 else 0.0)

    yaw, pitch, roll = head_pose(lm, frame_w, frame_h)
    interocular = float(np.linalg.norm(_xy(lm, R_OUTER) - _xy(lm, L_OUTER)))
    return np.array([hs[0], hs[1], vs[0], vs[1], yaw, pitch, roll,
                     interocular, 1.0], dtype=float)


def pose_delta(f: np.ndarray, f_ref: np.ndarray) -> tuple[float, float, float]:
    """Signed head movement since calibration: yaw, pitch, and distance ratio.

    Signed and kept apart on purpose. Collapsing to a magnitude per frame and
    averaging afterwards rectifies the noise -- a magnitude is always positive,
    so it never averages out, and on a motionless head it reads 2.9 degrees at
    a pixel of landmark jitter. Average the components first and take the
    magnitude once at the end and the same head reads 0.8.

    Interocular distance stands in for depth: it grows as the user leans in, so
    the distance ratio is its inverse.
    """
    f = np.asarray(f, dtype=float)
    ref = np.asarray(f_ref, dtype=float)
    ratio = float(ref[IOD] / f[IOD]) if f[IOD] > 1e-9 else 1.0
    return float(f[YAW] - ref[YAW]), float(f[PITCH] - ref[PITCH]), ratio


def drift_degrees(dyaw: float, dpitch: float, distance_ratio: float = 1.0,
                  half_tan: float = config.SCREEN_HALF_TAN) -> float:
    """Rotation and depth as one gaze error, in degrees.

    Leaning in rotates nothing, so a rotation-only measure reads zero through a
    change worth ninety pixels. What it does instead is rescale the mapping
    about the gaze axis, which makes the error proportional to how far off
    centre you are looking -- worst at the screen edge, and that is the case
    converted to an angle here. The two sources are independent, so they
    combine in quadrature.
    """
    depth = math.atan(half_tan * abs(distance_ratio - 1.0))
    return math.degrees(math.hypot(dyaw, dpitch, depth))


def pose_drift(f: np.ndarray, f_ref: np.ndarray) -> float:
    """One frame's drift in degrees, unsmoothed.

    Never consults the mapping, deliberately. The appealing version -- predict
    twice, once with the pose terms rewound, and call the gap the drift --
    reads zero almost always: calibration is collected sitting still, so pose
    varies as noise uncorrelated with the target and the fit shrinks those
    columns away. Real drift lands in the eye ratios, where the weight is.
    Measured on a real frame that mistake reported 0 px for a turn worth 151.

    Degrees rather than pixels because the honest conversion needs the screen's
    physical width and the viewing distance, and neither is known here.
    """
    return drift_degrees(*pose_delta(f, f_ref))


class OneEuro:
    """Adaptive low-pass, 2D.

    Preferred over a fixed-gain Kalman because the cutoff rises with velocity:
    heavy smoothing while the eye is fixating, almost none during a saccade.
    """

    def __init__(self, min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
                 beta: float = config.ONE_EURO_BETA, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x = None
        self._dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=float)
        if self._t is None:
            self._t, self._x = t, p
            return float(p[0]), float(p[1])
        dt = max(t - self._t, 1e-6)
        dx = (p - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * p + (1 - a) * self._x
        self._t = t
        return float(self._x[0]), float(self._x[1])


def resolve_zone(x: float, y: float, w: int = config.SCREEN_W,
                 h: int = config.SCREEN_H, cols: int = config.GRID_COLS,
                 rows: int = config.GRID_ROWS) -> int:
    """Tile index, row-major. Clamped: at x == w the bare int() gives cols."""
    col = min(max(int(x / w * cols), 0), cols - 1)
    row = min(max(int(y / h * rows), 0), rows - 1)
    return row * cols + col


def zone_rect(zone: int, w: int = config.SCREEN_W, h: int = config.SCREEN_H,
              cols: int = config.GRID_COLS, rows: int = config.GRID_ROWS):
    col, row = zone % cols, zone // cols
    tw, th = w / cols, h / rows
    return col * tw, row * th, (col + 1) * tw, (row + 1) * th


def ensure_model(path: str | None = None) -> str:
    """Fetch face_landmarker.task once. Day 0 work, so the demo never needs
    the network."""
    import urllib.request
    from pathlib import Path

    dest = Path(path or Path(config.MODEL_DIR) / config.FACE_MODEL)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.FACE_MODEL_URL, dest)
    return str(dest)


class GazeTracker:
    """Webcam -> GazeSample, via the MediaPipe Face Landmarker task.

    ``camera=None`` builds the detector without opening a device, which is how
    the landmark path is exercised without hardware.
    """

    def __init__(self, mapping=None, camera: int | None = 0,
                 screen=(config.SCREEN_W, config.SCREEN_H),
                 model_path: str | None = None, f_ref: np.ndarray | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=ensure_model(model_path),
                # CPU explicitly: the Metal delegate aborts the process on this
                # machine ("Check failed: service_ Service is unavailable"), and
                # the report budgets for CPU inference anyway.
                delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.cap = None
        if camera is not None:
            import cv2

            self.cap = cv2.VideoCapture(camera)
        self.mapping = mapping
        self.screen = screen
        self.blink_ear = config.BLINK_EAR
        self.filter = OneEuro()
        self.last_landmarks: np.ndarray | None = None
        self.last_features: np.ndarray | None = None
        # The head pose calibration was fitted at. Drift is measured against it,
        # and requalification replaces it. None until a calibration is loaded.
        self.f_ref = None if f_ref is None else np.asarray(f_ref, dtype=float)
        # A single frame's pose is far too jittery to show anyone: solvePnP
        # swings past ten degrees on a motionless head. Report the median of
        # about a second instead.
        self._drift = deque(maxlen=config.DRIFT_MEDIAN_FRAMES)
        self.last_ear: float = 0.0
        self._last_valid: GazeSample | None = None
        self._blink_since: float | None = None
        self._frame_ms = 0

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        """One RGB frame to a (478, 3) landmark array, or None for no face.

        The task API insists on strictly increasing integer millisecond
        timestamps, which is its own clock and unrelated to the monotonic
        stamps the rest of the system runs on.
        """
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        self._frame_ms += 1
        result = self.landmarker.detect_for_video(image, self._frame_ms)
        if not result.face_landmarks:
            self.last_landmarks = None
            return None
        self.last_landmarks = np.array(
            [(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=float)
        return self.last_landmarks

    def _grab(self):
        import cv2

        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_raw(self, t: float):
        """Frame -> (feature vector, blink, frame). Calibration needs the
        features themselves rather than a mapped screen point."""
        frame, rgb = self._grab()
        if frame is None:
            return None, False, None
        lm = self.detect(rgb)
        if lm is None:
            return None, False, frame
        h, w = frame.shape[:2]
        self.last_ear = eye_aspect_ratio(lm, w, h)
        blink = self.last_ear < self.blink_ear
        return features(lm, w, h), blink, frame

    def read(self, t: float):
        """Frame -> (GazeSample, frame). Needs a fitted mapping."""
        frame, rgb = self._grab()
        if frame is None:
            return None, None
        lm = self.detect(rgb)
        if lm is None:
            self._last_valid = None
            self.last_features = None
            return None, frame

        h, w = frame.shape[:2]
        self.last_ear = eye_aspect_ratio(lm, w, h)
        if self.last_ear < self.blink_ear:
            # Freeze the last good point rather than tracking the eyelid.
            self._blink_since = self._blink_since or t
            self.last_features = None    # nothing to requalify against mid-blink
            if (self._last_valid
                    and (t - self._blink_since) * 1000 < config.BLINK_HOLD_MS):
                return GazeSample(t=t, x=self._last_valid.x, y=self._last_valid.y,
                                  conf=self._last_valid.conf,
                                  zone=self._last_valid.zone, blink=True), frame
            return GazeSample(t=t, x=0.0, y=0.0, conf=0.0, blink=True), frame
        self._blink_since = None

        if self.mapping is None:
            return None, frame
        f = features(lm, w, h)
        self.last_features = f
        x, y = self.mapping.predict(f.reshape(1, -1))[0]
        x, y = self.filter(float(x), float(y), t)
        sw, sh = self.screen
        on_screen = 0 <= x <= sw and 0 <= y <= sh
        s = GazeSample(t=t, x=x, y=y, conf=1.0 if on_screen else 0.3,
                       zone=resolve_zone(x, y, sw, sh), blink=False,
                       drift_deg=self._drift_deg(f))
        self._last_valid = s
        return s, frame

    def _drift_deg(self, f) -> float | None:
        if self.f_ref is None:
            return None
        self._drift.append(pose_delta(f, self.f_ref))
        if self._frame_ms < config.DRIFT_WARMUP_FRAMES:
            # The landmarker's tracker is still converging; its pose estimate
            # swings degrees on a face that has not moved. Say nothing rather
            # than send the dashboard a number that means nothing.
            return None
        return drift_degrees(*(statistics.median(c) for c in zip(*self._drift)))

    def rebase(self, f_ref, mapping=None) -> None:
        """Adopt a corrected mapping and the pose it was measured at.

        Everything downstream of the old pose has to go with it: the drift
        window, or the indicator carries the old offset for a second, and the
        smoother, or it drags pre-correction points into the new estimates.
        """
        if mapping is not None:
            self.mapping = mapping
        self.f_ref = np.asarray(f_ref, dtype=float)
        self._drift.clear()
        self.filter.reset()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
