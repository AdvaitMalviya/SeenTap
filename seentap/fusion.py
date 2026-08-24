"""Binding a spoken verb to a gaze position. This is the contribution.

When a command arrives the layer does not ask where the eyes are now. It looks
backwards. Transcription costs several hundred milliseconds and the user has
usually moved on, so binding to the current position acts on the wrong target
-- intermittently, which is the hardest kind of bug to catch afterwards.
Binding to the gaze held at speech onset fixes it, and the enactment literature
supports it: gaze reliably leads the matching utterance.

``bind`` is a pure function of (buffer, onset, config). Study 2 replays it over
recorded logs under 240 configurations, and that only works if the offline
sweep can call the exact function the live system called.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from seentap import config, parse
from seentap.gaze import GazeSample, resolve_zone

AGGREGATORS = frozenset({"last", "mean", "median", "centroid", "zone_mode"})


@dataclass(frozen=True)
class FusionConfig:
    lead_ms: int = 200
    window_ms: int = 300
    aggregator: str = "median"
    min_samples: int = 5
    conf_floor: float = config.CONF_FLOOR

    def __post_init__(self):
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"unknown aggregator {self.aggregator!r}")


@dataclass
class BindResult:
    ok: bool
    x: float | None = None
    y: float | None = None
    zone: int | None = None
    n: int = 0
    reason: str = ""


@dataclass
class CommandResult(BindResult):
    verb: str | None = None
    onset_t: float | None = None


def dispersion(samples) -> float:
    """Spread of a run of gaze points, in pixels. Low means fixating."""
    if not samples:
        return float("inf")
    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _aggregate(samples, how: str):
    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    if how == "last":
        return xs[-1], ys[-1], samples[-1].zone
    if how == "mean":
        return sum(xs) / len(xs), sum(ys) / len(ys), None
    if how == "median":
        return statistics.median(xs), statistics.median(ys), None
    if how == "centroid":
        # Mean of the dominant fixation: drop points that stray from the median,
        # so a saccade or a blink inside the window cannot drag the binding.
        mx, my = statistics.median(xs), statistics.median(ys)
        core = [s for s in samples
                if abs(s.x - mx) <= config.GATE_DISPERSION_PX
                and abs(s.y - my) <= config.GATE_DISPERSION_PX] or list(samples)
        return (sum(s.x for s in core) / len(core),
                sum(s.y for s in core) / len(core), None)
    if how == "zone_mode":
        zones = [s.zone for s in samples if s.zone is not None]
        if not zones:
            return statistics.median(xs), statistics.median(ys), None
        modal = statistics.mode(zones)
        core = [s for s in samples if s.zone == modal]
        return (sum(s.x for s in core) / len(core),
                sum(s.y for s in core) / len(core), modal)
    raise ValueError(f"unknown aggregator {how!r}")


def bind(buffer, onset_t: float, cfg: FusionConfig = None,
         screen=None) -> BindResult:
    """Gaze held at speech onset, or a refusal. Pure: reads nothing else.

    The window is centred on ``onset_t - lead_ms``, because people look at a
    target before they name it. Too small a lead catches the saccade away; too
    large catches the previous target.
    """
    cfg = cfg or FusionConfig()
    sw, sh = screen or (config.SCREEN_W, config.SCREEN_H)
    centre = onset_t - cfg.lead_ms / 1000.0
    half = cfg.window_ms / 2000.0
    lo, hi = centre - half, centre + half

    candidates = [s for s in buffer
                  if lo <= s.t <= hi and not s.blink and s.conf >= cfg.conf_floor]
    if len(candidates) < cfg.min_samples:
        return BindResult(ok=False, n=len(candidates), reason="too_few_samples")

    x, y, zone = _aggregate(candidates, cfg.aggregator)
    if zone is None:
        zone = resolve_zone(x, y, sw, sh)
    return BindResult(ok=True, x=x, y=y, zone=zone, n=len(candidates), reason="")


def face_present(buffer, t_now: float,
                 window_ms: int = config.GATE_WINDOW_MS) -> bool:
    """Someone is in front of the camera, with no claim about where they look."""
    return any(t_now - window_ms / 1000.0 <= s.t <= t_now and not s.blink
               for s in buffer)


def gate(buffer, t_now: float, window_ms: int = config.GATE_WINDOW_MS,
         max_dispersion: float = config.GATE_DISPERSION_PX,
         screen=None) -> tuple[bool, str]:
    """Arm the recogniser only on a steady, on-screen fixation.

    This is what stops someone else in the room saying 'click' from doing
    anything. Every refusal is counted, so the claim becomes a number.
    """
    sw, sh = screen or (config.SCREEN_W, config.SCREEN_H)
    recent = [s for s in buffer
              if t_now - window_ms / 1000.0 <= s.t <= t_now and not s.blink]
    if not recent:
        return False, "no_face"
    last = recent[-1]
    if not (0 <= last.x <= sw and 0 <= last.y <= sh):
        return False, "off_screen"
    if dispersion(recent) > max_dispersion:
        return False, "not_fixating"
    return True, ""


class Fusion:
    """The interaction state machine, hand-written for want of standard tooling.

    Every transition is recorded, which makes a whole session reconstructible
    offline from the log alone.
    """

    STATES = ("tracking", "armed", "listening", "decoding",
              "binding", "executing", "cooldown", "recalibrating")

    def __init__(self, cfg: FusionConfig = None, screen=None,
                 buffer_seconds: float = config.BUFFER_SECONDS):
        self.cfg = cfg or FusionConfig()
        self.screen = screen or (config.SCREEN_W, config.SCREEN_H)
        self.buffer_seconds = buffer_seconds
        self.buffer: list[GazeSample] = []
        self.state = "tracking"
        self.transitions: list[tuple[float, str]] = []
        self.gate_refusals = 0
        self._last_action_t: float | None = None

    def _to(self, state: str, t: float) -> None:
        self.state = state
        self.transitions.append((t, state))

    def on_gaze(self, sample: GazeSample) -> None:
        self.buffer.append(sample)
        cutoff = sample.t - self.buffer_seconds
        while self.buffer and self.buffer[0].t < cutoff:
            self.buffer.pop(0)
        if self.state in ("tracking", "armed", "cooldown"):
            if self.state == "cooldown" and not self._cooled(sample.t):
                return
            ok, _ = gate(self.buffer, sample.t, screen=self.screen)
            want = "armed" if ok else "tracking"
            if want != self.state:
                self._to(want, sample.t)

    def _cooled(self, t: float) -> bool:
        return (self._last_action_t is None
                or (t - self._last_action_t) * 1000 >= config.COOLDOWN_MS)

    def on_utterance(self, onset_t: float, text: str, now: float) -> CommandResult:
        """One spoken command, from onset timestamp to executable action."""
        self._to("listening", now)
        self._to("decoding", now)
        verb, _score = parse.parse_any(text)
        if verb is None:
            self._to("tracking", now)
            return CommandResult(ok=False, reason="no_verb", onset_t=onset_t)

        if verb in config.HELP_VOCAB or verb == "recalibrate":
            # Neither needs a target, so neither binds -- and neither can be
            # made to wait on a steady fixation. A user who has forgotten the
            # commands is sweeping the screen, and a user whose calibration has
            # drifted is being mapped off it; both are states the gate refuses,
            # and refusing "recalibrate" because the calibration is broken is
            # the wrong way round. A face still has to be present, so a
            # bystander's voice alone does nothing.
            if not face_present(self.buffer, now):
                self.gate_refusals += 1
                self._to("tracking", now)
                return CommandResult(ok=False, verb=verb, reason="no_face",
                                     onset_t=onset_t)
            self._to("recalibrating" if verb == "recalibrate" else "tracking", now)
            return CommandResult(ok=True, verb=verb, reason="", onset_t=onset_t)

        if not self._cooled(now):
            return CommandResult(ok=False, reason="cooldown", onset_t=onset_t)

        ok, why = gate(self.buffer, now, screen=self.screen)
        if not ok:
            self.gate_refusals += 1
            self._to("tracking", now)
            return CommandResult(ok=False, reason=why, onset_t=onset_t)

        self._to("binding", now)
        r = bind(self.buffer, onset_t, self.cfg, screen=self.screen)
        if not r.ok:
            self._to("tracking", now)
            return CommandResult(ok=False, verb=verb, n=r.n, reason=r.reason,
                                 onset_t=onset_t)

        self._to("executing", now)
        self._last_action_t = now
        self._to("cooldown", now)
        return CommandResult(ok=True, x=r.x, y=r.y, zone=r.zone, n=r.n,
                             verb=verb, reason="", onset_t=onset_t)
