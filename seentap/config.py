"""Every constant the report fixes, in one place.

Numbers here are quoted from the project report. Where the report gives a
placeholder derived from a 1920-wide screen, the value is recomputed from the
display this actually runs on.
"""
from __future__ import annotations

import os


def _screen() -> tuple[int, int]:
    """Logical points, the space PyAutoGUI drives the cursor in.

    On a Retina display the physical backing store is 2x this. Everything in
    SeenTap -- gaze estimates, tile edges, cursor targets -- lives in logical
    points, and the camera->screen conversion happens once, in the mapping.
    """
    if "SEENTAP_SCREEN" in os.environ:
        w, h = os.environ["SEENTAP_SCREEN"].split("x")
        return int(w), int(h)
    try:
        # Quartz, not pyautogui: pyautogui pulls in pyscreeze which pulls in
        # cv2, and cv2's bundled libavdevice clashes with the one PyAV ships
        # for faster-whisper ("implemented in both ... mysterious crashes").
        # The speech process must never import cv2, and it imports this module.
        from Quartz import CGDisplayBounds, CGMainDisplayID

        b = CGDisplayBounds(CGMainDisplayID())
        return int(b.size.width), int(b.size.height)
    except Exception:
        pass
    try:
        import pyautogui

        size = pyautogui.size()
        return int(size.width), int(size.height)
    except Exception:  # headless CI, no display
        return 1512, 982


SCREEN_W, SCREEN_H = _screen()

# --- Mode B: zone selection ------------------------------------------------
GRID_COLS = 4
GRID_ROWS = 3
N_TILES = GRID_COLS * GRID_ROWS

# --- the day-8 gate --------------------------------------------------------
GATE_FRAC = 0.08          # of screen width; 121 px at 1512, 154 px at 1920
DENSITIES = (5, 9, 13)    # calibration point counts compared in Study 1
HELD_OUT_POINTS = 5
STUDY1_REPETITIONS = 5

# --- gaze ------------------------------------------------------------------
# Landmark-set specific. On the MediaPipe mesh an open eye reads about 0.20,
# so dlib's published 0.21 would flag every frame as a blink. This is a
# starting point; gaze.blink_threshold() measures it per user at calibration.
BLINK_EAR = 0.12
BLINK_HOLD_MS = 500       # freeze the last valid point, then flag low confidence
DEADZONE_PX = 15          # cursor mode only; kills micro-drift
CONF_FLOOR = 0.5
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.007

# --- gaze gating -----------------------------------------------------------
GATE_WINDOW_MS = 200      # dispersion is measured over this much history
GATE_DISPERSION_PX = 120  # above this the eyes are sweeping, not fixating

# --- drift and requalification ---------------------------------------------
# Degrees of head rotation away from the calibration pose. Pixels would be
# dishonest: converting needs the screen's physical width and the viewing
# distance, neither of which is known here. The arithmetic behind these two:
# a ~300 mm wide screen at ~600 mm subtends about 28 degrees, so 2 degrees of
# uncompensated head turn is roughly the 8% gate, and 5 is well past it. Both
# assume a laptop at arm's length -- retune them for any other geometry.
DRIFT_WARN_DEG = 2.0
DRIFT_BAD_DEG = 5.0
# Half the screen width in units of viewing distance -- the same 300 mm at
# 600 mm assumed above, and the only geometry the system cannot measure for
# itself. It converts a change of distance into the gaze error it causes at the
# screen edge. Retune it for a desktop monitor or a different working distance.
SCREEN_HALF_TAN = 0.25
DRIFT_MEDIAN_FRAMES = 30  # ~1 s; a single frame's solvePnP is far too jittery
# The landmarker's own tracking filter needs time to settle. Fed identical
# frames it wanders 4.3 degrees away from its eventual steady state before
# converging, and takes about 130 frames to get within half a degree -- which
# would light the badge red at session start for no reason at all. Report
# nothing until it has calmed down. Afterwards it holds inside 0.5 degrees.
DRIFT_WARMUP_FRAMES = 150
REQUALIFY_POINTS = 5      # the smallest density; an affine wants three
REQUALIFY_SETTLE_MS = 700
REQUALIFY_COLLECT_MS = 900

# --- speech ----------------------------------------------------------------
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_ONSET_FRAMES = 3      # 90 ms of voice declares onset
VAD_OFFSET_FRAMES = 15    # 450 ms of silence declares offset
VAD_PREROLL_MS = 200      # so the first consonant survives
VAD_AGGRESSIVENESS = 2
WHISPER_MODEL = "base.en"
WHISPER_COMPUTE = "int8"

# Bump when features() changes shape or meaning: a calibration file holds
# fitted feature vectors, and reading old ones with new features is silent
# nonsense. 2 moved the vertical ratio off the eyelid onto the eye corners.
FEATURES_VERSION = 2

VOCAB = [
    "click", "double click", "right click", "select",
    "scroll up", "scroll down", "drag", "drop", "cancel", "recalibrate",
]
PARSE_THRESHOLD = 75      # below this the utterance is refused, never guessed

# Spoken to the application rather than to the screen. Kept apart from VOCAB so
# the ten action verbs stay exactly the ten the study evaluates.
HELP_VOCAB = ("help", "controls")
HELP_SECONDS = 7          # how long the overlay stays up before fading

VERB_HELP = {
    "click": "select what you are looking at",
    "double click": "open what you are looking at",
    "right click": "context menu",
    "select": "highlight the tile",
    "scroll up": "scroll up",
    "scroll down": "scroll down",
    "drag": "pick up from here",
    "drop": "release here",
    "cancel": "abandon a drag",
    "recalibrate": "five quick points, when gaze has drifted",
}

# --- fusion ----------------------------------------------------------------
COOLDOWN_MS = 250
LEAD_MS_SWEEP = (0, 100, 200, 300)
WINDOW_MS_SWEEP = (100, 300, 500, 1000)
AGGREGATOR_SWEEP = ("last", "mean", "median", "centroid", "zone_mode")
MIN_SAMPLES_SWEEP = (3, 5, 8)
BUFFER_SECONDS = 3.0

# --- baselines -------------------------------------------------------------
DWELL_MS = 800            # C1, set above a natural reading fixation

# --- study -----------------------------------------------------------------
PRACTICE_TRIALS = 5
RECORDED_TRIALS = 20
CONDITIONS = ("C1", "C2", "C3")

LOG_DIR = "logs"
MODEL_DIR = "models"
FACE_MODEL = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def __getattr__(name):
    """FUSION_DEFAULT needs fusion, which needs config. PEP 562 breaks the cycle."""
    if name == "FUSION_DEFAULT":
        from seentap.fusion import FusionConfig

        return FusionConfig(lead_ms=200, window_ms=300, aggregator="median",
                            min_samples=5, conf_floor=CONF_FLOOR)
    raise AttributeError(name)
