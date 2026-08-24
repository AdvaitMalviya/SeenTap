"""VAD segmentation. The onset timestamp is the whole ballgame for fusion."""
import pytest

from seentap import speech


def feed(seg, pattern, t0=0.0, frame_ms=30):
    """pattern: string of '.' (silence) and 'v' (voiced)."""
    out = []
    for i, ch in enumerate(pattern):
        r = seg.push(ch == "v", t0 + i * frame_ms / 1000.0)
        if r is not None:
            out.append(r)
    return out


def test_onset_needs_three_voiced_frames():
    seg = speech.VadSegmenter()
    feed(seg, "vv")
    assert not seg.speaking
    feed(seg, "v", t0=0.06)
    assert seg.speaking


def test_onset_timestamp_is_the_first_voiced_frame_not_the_third():
    """Binding anchors here; a 60 ms error propagates into every fusion result."""
    seg = speech.VadSegmenter()
    feed(seg, "." * 5 + "v" * 30, t0=10.0)
    assert seg.onset_t == pytest.approx(10.0 + 5 * 0.03)


def test_preroll_is_subtracted_from_the_reported_segment_start():
    seg = speech.VadSegmenter(preroll_ms=200)
    segs = feed(seg, "." * 10 + "v" * 20 + "." * 20, t0=0.0)
    assert segs
    assert segs[0].audio_start_t == pytest.approx(segs[0].onset_t - 0.200)


def test_offset_needs_fifteen_unvoiced_frames():
    seg = speech.VadSegmenter()
    assert not feed(seg, "v" * 10 + "." * 14)
    assert feed(seg, ".", t0=24 * 0.03)


def test_a_pause_mid_command_does_not_split_the_utterance():
    """'scroll ... down' is one command, not two."""
    seg = speech.VadSegmenter()
    segs = feed(seg, "v" * 10 + "." * 10 + "v" * 10 + "." * 20)
    assert len(segs) == 1


def test_segment_carries_onset_and_offset():
    seg = speech.VadSegmenter()
    segs = feed(seg, "v" * 10 + "." * 20, t0=5.0)
    assert segs[0].offset_t > segs[0].onset_t


def test_segmenter_matches_the_reported_hangover():
    seg = speech.VadSegmenter()
    assert seg.offset_frames * seg.frame_ms == 450


def test_defaults_match_the_report():
    seg = speech.VadSegmenter()
    assert (seg.frame_ms, seg.onset_frames, seg.offset_frames, seg.preroll_ms) == (30, 3, 15, 200)
