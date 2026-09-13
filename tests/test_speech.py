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


def test_offset_needs_twelve_unvoiced_frames():
    seg = speech.VadSegmenter()
    assert not feed(seg, "v" * 10 + "." * 11)
    assert feed(seg, ".", t0=21 * 0.03)


def test_a_pause_mid_command_does_not_split_the_utterance():
    """'scroll ... down' is one command, not two."""
    seg = speech.VadSegmenter()
    segs = feed(seg, "v" * 10 + "." * 10 + "v" * 10 + "." * 20)
    assert len(segs) == 1


def test_segment_carries_onset_and_offset():
    seg = speech.VadSegmenter()
    segs = feed(seg, "v" * 10 + "." * 20, t0=5.0)
    assert segs[0].offset_t > segs[0].onset_t


def test_the_hangover_still_clears_a_mid_command_pause():
    """It is the floor on latency, so it is kept as short as the pause in
    'scroll ... down' allows -- 360 ms against the 300 ms that pause runs to."""
    seg = speech.VadSegmenter()
    assert seg.offset_frames * seg.frame_ms == 360 > 300


def test_defaults_match_the_report():
    seg = speech.VadSegmenter()
    assert (seg.frame_ms, seg.onset_frames, seg.offset_frames, seg.preroll_ms) == (30, 3, 12, 200)


def test_a_noise_blip_is_never_handed_to_whisper():
    """58 of 104 logged utterances transcribed to nothing: webrtcvad armed on
    room noise and Whisper spent seconds deciding so (4524 ms on one second of
    digital silence, measured). The worker is a serial loop, so every one of
    those was seconds with the microphone unread."""
    import numpy as np

    quiet = [np.zeros(480, dtype=np.int16) for _ in range(20)]
    assert not speech.worth_decoding(quiet)

    spoken = list(quiet)
    spoken[5] = np.random.default_rng(0).normal(0, 400, 480).astype(np.int16)
    assert speech.worth_decoding(spoken), "a single loud frame is a word"


def test_whisper_is_told_to_drop_the_silence_it_would_otherwise_loop_on():
    """Decoding one second of silence took 3.3 s and room hiss 6.3 s, because
    Whisper loops on non-speech; real one-word commands padded by the 200 ms
    preroll and hangover cost 5-7 s the same way. With vad_filter it is
    2-3 ms. Capping max_new_tokens instead truncates the loop but still returns
    it, firing a phantom command from silence."""
    seen = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return [], None

    assert speech.transcribe(FakeModel(), [0.0]) == ""
    assert seen["vad_filter"] is True
    assert seen["temperature"] == 0, "six temperature retries, on audio a retry cannot rescue"
