"""Study 2, offline. One recording session yields the whole parameter space.

Tuning fusion parameters live burns participant time and cannot be reproduced.
Instead every session is logged with raw gaze, onsets, transcripts and the
ground-truth target, and this module re-runs the live pipeline over that log
under any configuration, deterministically, in seconds.

It replays through ``Fusion`` rather than reimplementing it, so 'the replay
agrees with the live run' is true by construction rather than by discipline.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from seentap import config, eventlog
from seentap.fusion import CommandResult, Fusion, FusionConfig
from seentap.gaze import GazeSample
from seentap.speech import Utterance


@dataclass
class Session:
    gaze: list[GazeSample] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    truth: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    screen: tuple[int, int] = (config.SCREEN_W, config.SCREEN_H)
    live_config: dict | None = None


def load_session(path) -> Session:
    s = Session()
    for row in eventlog.read(path):
        kind = row.get("kind")
        if kind == "gaze":
            s.gaze.append(GazeSample(t=row["t"], x=row["x"], y=row["y"],
                                     conf=row.get("conf", 1.0),
                                     zone=row.get("zone"),
                                     blink=row.get("blink", False)))
        elif kind == "utterance":
            s.utterances.append(Utterance(onset_t=row["onset_t"],
                                          offset_t=row.get("offset_t", row["t"]),
                                          text=row.get("text", ""),
                                          decode_ms=row.get("decode_ms", 0.0)))
        elif kind == "truth":
            s.truth.append({"t": row["t"], "zone": row["zone"]})
        elif kind == "action":
            s.actions.append(row)
        elif kind == "session":
            if row.get("screen"):
                s.screen = tuple(row["screen"])
            s.live_config = row.get("config")
    return s


def replay(session: Session, cfg: FusionConfig) -> list[CommandResult]:
    """Re-run one session under one configuration. No I/O, no randomness."""
    f = Fusion(cfg, screen=session.screen)
    timeline = [(g.t, 0, g) for g in session.gaze]
    timeline += [(u.onset_t + u.decode_ms / 1000.0, 1, u)
                 for u in session.utterances]
    timeline.sort(key=lambda e: (e[0], e[1]))

    out: list[CommandResult] = []
    for t, kind, item in timeline:
        if kind == 0:
            f.on_gaze(item)
        else:
            out.append(f.on_utterance(onset_t=item.onset_t, text=item.text,
                                      now=t))
    return out


def default_configs() -> list[FusionConfig]:
    """All 240. Table 7 lists four parameters; section 4.6 multiplies three.

    Replay is free, so sweep the full grid and report minimum-samples as a
    separate sensitivity line rather than dropping a parameter the report
    also claims to sweep.
    """
    return [FusionConfig(lead_ms=l, window_ms=w, aggregator=a, min_samples=m)
            for l, w, a, m in itertools.product(
                config.LEAD_MS_SWEEP, config.WINDOW_MS_SWEEP,
                config.AGGREGATOR_SWEEP, config.MIN_SAMPLES_SWEEP)]


def headline_configs() -> list[FusionConfig]:
    """The 80-cell surface the report quotes, at the default minimum samples."""
    default = config.FUSION_DEFAULT.min_samples
    return [c for c in default_configs() if c.min_samples == default]


def _truth_for(session: Session, onset_t: float) -> int | None:
    if not session.truth:
        return None
    return min(session.truth, key=lambda r: abs(r["t"] - onset_t))["zone"]


def sweep(session: Session, configs=None):
    """Score every configuration against the same ground truth."""
    import pandas as pd

    configs = configs or default_configs()
    rows = []
    for cfg in configs:
        results = replay(session, cfg)
        n = len(results)
        correct = sum(1 for r in results
                      if r.ok and r.zone == _truth_for(session, r.onset_t))
        refused = sum(1 for r in results if not r.ok)
        rows.append({
            "lead_ms": cfg.lead_ms, "window_ms": cfg.window_ms,
            "aggregator": cfg.aggregator, "min_samples": cfg.min_samples,
            "accuracy": correct / n if n else 0.0,
            "refused": refused / n if n else 0.0,
            "n_commands": n,
        })
    return pd.DataFrame(rows)
