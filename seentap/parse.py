"""Transcript to verb. A miss produces nothing, never the wrong action.

Gaze has already absorbed the spatial half of every instruction, so the
vocabulary only needs verbs and a fuzzy match over ten of them is enough.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from seentap import config

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_FILLER = {"tile", "number", "box", "cell", "please", "the"}

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def normalise(text: str) -> str:
    return _PUNCT.sub(" ", (text or "").lower()).strip()


def parse(text: str, vocab: list[str] | None = None,
          threshold: float = config.PARSE_THRESHOLD) -> tuple[str | None, float]:
    """Best verb and its score, or (None, score) if nothing clears threshold."""
    vocab = list(config.VOCAB if vocab is None else vocab)
    cleaned = normalise(text)
    if not cleaned:
        return None, 0.0
    match = process.extractOne(cleaned, vocab, scorer=fuzz.WRatio)
    if match is None:
        return None, 0.0
    verb, score, _ = match
    return (verb, float(score)) if score >= threshold else (None, float(score))


def parse_any(text: str, threshold: float = config.PARSE_THRESHOLD):
    """Match over the action verbs and the help words together.

    One ranking rather than two passes, so 'controls' cannot be claimed by a
    help-first check when the user actually said 'scroll down', and vice versa.
    """
    return parse(text, vocab=list(config.VOCAB) + list(config.HELP_VOCAB),
                 threshold=threshold)


def _take_number(tokens: list[str]) -> tuple[int | None, list[str]]:
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            return int(tok), tokens[:i] + tokens[i + 1:]
        if tok in WORD_NUMBERS:
            return WORD_NUMBERS[tok], tokens[:i] + tokens[i + 1:]
    return None, tokens


def parse_numbered(text: str, n_tiles: int = config.N_TILES,
                   threshold: float = config.PARSE_THRESHOLD):
    """C2 baseline: every tile is numbered, the user says number then verb.

    This is the voice-only control condition, and its shape is exactly why
    voice alone struggles -- the vocabulary has to encode position in words.
    """
    tokens = [t for t in normalise(text).split() if t and t not in _FILLER]
    tile, rest = _take_number(tokens)
    if tile is not None and not (1 <= tile <= n_tiles):
        tile = None
    verb, score = parse(" ".join(rest), threshold=threshold)
    return tile, verb, score
