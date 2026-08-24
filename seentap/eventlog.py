"""Append-only JSON Lines. Every figure in the report regenerates from these.

Two rules are enforced here rather than remembered:

* Timestamps are ``time.monotonic()`` taken at the source. A wall clock can
  step backwards, which would corrupt the fusion argument silently.
* Raw imagery and audio can never reach disk. The consent form says derived
  landmark coordinates and transcript text only, so the writer refuses the
  fields that would break that promise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

# Fields that would put a face or a voice on disk.
FORBIDDEN = frozenset({
    "frame", "frames", "image", "img", "images", "pixels",
    "audio", "pcm", "waveform", "wav", "samples_raw",
})


class EthicsError(ValueError):
    """Raised when a write would store identifiable imagery or audio."""


def now() -> float:
    return time.monotonic()


class EventLog:
    """One session, one file. Flushed per line so a crash keeps the data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, kind: str, t: float | None = None, **fields: Any) -> dict:
        bad = FORBIDDEN & set(fields)
        if bad:
            raise EthicsError(
                f"refusing to log {sorted(bad)}: the study stores derived "
                f"coordinates and transcripts only"
            )
        row = {"kind": kind, "t": float(now() if t is None else t), **fields}
        line = json.dumps(row)          # raises TypeError before anything is written
        self._fh.write(line + "\n")
        self._fh.flush()
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
