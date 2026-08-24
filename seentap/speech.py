"""Microphone to Utterance, in its own process.

Landmark inference and speech decoding are both CPU-bound and will fight if
left in one thread; the visible symptom is the cursor stuttering at the exact
moment a command is spoken. So this runs behind a bounded queue, and the
recogniser stays idle until voice activity is detected.

The timestamp that leaves here is the first voiced frame, not the moment
transcription finished. Everything in fusion.py depends on that distinction.
"""
from __future__ import annotations

from dataclasses import dataclass

from seentap import config


@dataclass
class Segment:
    onset_t: float
    offset_t: float
    audio_start_t: float


@dataclass
class Utterance:
    onset_t: float
    offset_t: float
    text: str
    decode_ms: float


class VadSegmenter:
    """Frame-by-frame voice activity, with hysteresis at both ends.

    Onset after 3 voiced frames so a cough does not arm the recogniser; offset
    after 15 unvoiced frames (450 ms of hangover) so a pause in the middle of
    'scroll ... down' does not truncate the command.
    """

    def __init__(self, frame_ms: int = config.VAD_FRAME_MS,
                 onset_frames: int = config.VAD_ONSET_FRAMES,
                 offset_frames: int = config.VAD_OFFSET_FRAMES,
                 preroll_ms: int = config.VAD_PREROLL_MS):
        self.frame_ms = frame_ms
        self.onset_frames = onset_frames
        self.offset_frames = offset_frames
        self.preroll_ms = preroll_ms
        self.speaking = False
        self.onset_t: float | None = None
        self._voiced = 0
        self._unvoiced = 0
        self._candidate: float | None = None

    def push(self, is_speech: bool, t: float) -> Segment | None:
        if not self.speaking:
            if is_speech:
                if self._voiced == 0:
                    self._candidate = t
                self._voiced += 1
                if self._voiced >= self.onset_frames:
                    self.speaking = True
                    self.onset_t = self._candidate
                    self._unvoiced = 0
            else:
                self._voiced = 0
                self._candidate = None
            return None

        if is_speech:
            self._unvoiced = 0
            return None
        self._unvoiced += 1
        if self._unvoiced < self.offset_frames:
            return None

        seg = Segment(onset_t=self.onset_t, offset_t=t,
                      audio_start_t=self.onset_t - self.preroll_ms / 1000.0)
        self.speaking = False
        self._voiced = 0
        self._unvoiced = 0
        self._candidate = None
        return seg


def load_model(model: str = config.WHISPER_MODEL,
               compute_type: str = config.WHISPER_COMPUTE):
    """Local decoding only: no network round trip, no per-request cost, and
    the user's voice never leaves the machine."""
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="cpu", compute_type=compute_type,
                        download_root=config.MODEL_DIR)


def transcribe(model, audio, vocab=None) -> str:
    """Greedy decode, with the command list nudging the decoder without
    constraining it."""
    prompt = ", ".join(vocab or config.VOCAB)
    segments, _ = model.transcribe(
        audio, beam_size=1, condition_on_previous_text=False,
        initial_prompt=prompt, language="en")
    return " ".join(s.text for s in segments).strip()


def speech_worker(out_queue, stop_event, device=None) -> None:  # pragma: no cover
    """Process target: capture -> VAD -> Whisper -> Utterance on the queue."""
    import time

    import numpy as np
    import sounddevice as sd
    import webrtcvad

    frame_len = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
    preroll_frames = config.VAD_PREROLL_MS // config.VAD_FRAME_MS
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    seg = VadSegmenter()
    model = load_model()
    ring: list[np.ndarray] = []
    current: list[np.ndarray] = []

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=frame_len, device=device) as stream:
        while not stop_event.is_set():
            block, _ = stream.read(frame_len)
            t = time.monotonic()
            frame = block[:, 0].copy()
            ring.append(frame)
            del ring[:-preroll_frames or None]
            if seg.speaking:
                current.append(frame)
            voiced = vad.is_speech(frame.tobytes(), config.SAMPLE_RATE)
            was = seg.speaking
            done = seg.push(voiced, t)
            if not was and seg.speaking:
                current = list(ring)          # pre-roll keeps the first consonant
            if done is None:
                continue
            audio = np.concatenate(current).astype(np.float32) / 32768.0
            current = []
            t0 = time.monotonic()
            text = transcribe(model, audio)
            out_queue.put(Utterance(onset_t=done.onset_t, offset_t=done.offset_t,
                                    text=text,
                                    decode_ms=(time.monotonic() - t0) * 1000))
