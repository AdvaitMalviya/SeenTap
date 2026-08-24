"""The two control conditions. They are code, not just table rows.

C1 is the gaze-only baseline: dwell selection, the standard answer to Midas
touch and the one that trades speed for reliability. C2 is the voice-only
baseline: every tile is numbered, so the vocabulary has to encode position in
words -- which is the weakness the fused condition is meant to remove.
"""
from __future__ import annotations

from seentap import config, parse


class DwellSelector:
    """C1. Fires once per visit, at DWELL_MS on the same tile.

    The threshold sits above a natural reading fixation (400-1000 ms in the
    gaze-typing literature) precisely so ordinary looking triggers nothing.
    Every deliberate selection then pays that tax, which is the point of
    measuring it against the fused condition.
    """

    def __init__(self, dwell_ms: int = config.DWELL_MS):
        self.dwell_ms = dwell_ms
        self.zone: int | None = None
        self.entered_t: float | None = None
        self.fired = False

    def reset(self) -> None:
        self.zone = None
        self.entered_t = None
        self.fired = False

    def update(self, zone: int | None, t: float) -> int | None:
        if zone != self.zone:
            self.zone, self.entered_t, self.fired = zone, t, False
            return None
        if self.fired or zone is None:
            return None
        if (t - self.entered_t) * 1000 >= self.dwell_ms:
            self.fired = True
            return zone
        return None


def tile_labels(n_tiles: int = config.N_TILES) -> list[str]:
    """Participants see 1..12; the code works in 0-indexed zones."""
    return [str(i + 1) for i in range(n_tiles)]


def voice_only_select(text: str, n_tiles: int = config.N_TILES):
    """C2. 'seven click' -> (zone 6, 'click'). Refuses anything incomplete."""
    tile, verb, _ = parse.parse_numbered(text, n_tiles=n_tiles)
    if tile is None or verb is None:
        return None, None
    return tile - 1, verb
