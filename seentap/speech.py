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
class Level:
    """A periodic 'the microphone is hearing this much', for the dashboard.

    Without it a silent mic and an unrecognised word look identical from the
    user's chair: you say a word and nothing happens either way.
    """
    rms: float
    voiced: bool


@dataclass
class MicError:
    """The worker is a daemon process. Anything it raises dies with it unless
    it is handed back."""
    message: str


@dataclass
class Utterance:
    onset_t: float
    offset_t: float
    text: str
    decode_ms: float


class VadSegmenter:
    """Frame-by-frame voice activity, with hysteresis at both ends.

    Onset after 3 voiced frames so a cough does not arm the recogniser; offset
    after 12 unvoiced frames (360 ms of hangover) so a pause in the middle of
    'scroll ... down' does not truncate the command. That hangover is the floor
    on end-to-end latency: every command waits it out before decoding starts.
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


def worth_decoding(frames, floor: float = config.MIC_QUIET_RMS) -> bool:
    """Whether a segment carries speech, or is a blip webrtcvad armed on.

    Aggressiveness 2 arms on room noise, and Whisper then spends seconds
    deciding the segment is empty -- 4524 ms on one second of digital silence,
    measured. This loop is serial, so that is seconds in which the microphone
    goes unread and a command spoken into the gap is lost. 58 of 104 logged
    utterances were such blips: a bare onset plus the hangover, with no loud
    frame anywhere in them.

    Peak rather than mean, because the hangover dilutes a short word away.
    """
    import numpy as np

    return any(np.sqrt(np.mean(f.astype(np.float64) ** 2)) >= floor
               for f in frames)


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
    # vad_filter drops the non-speech either side before decoding. Without it
    # Whisper loops on that silence: 3.3 s to return '' on one second of it,
    # 6.3 s on room hiss, and 5-7 s on real one-word commands padded by the
    # 200 ms preroll and the hangover. With it, 2-3 ms. Twelve of 52 logged
    # decodes ran over a second and this is what they were spending it on.
    #
    # Capping max_new_tokens looks like the same fix and is not: it truncates
    # the loop but still returns it, so silence decodes to 'click, click,
    # click...' and a phantom command fires, and a truncated 'double click,
    # double click, ...' parses as plain 'click' -- a wrong action, which this
    # vocabulary is built never to produce.
    segments, _ = model.transcribe(
        audio, beam_size=1, condition_on_previous_text=False,
        initial_prompt=prompt, language="en", vad_filter=True, temperature=0)

    # temperature=0 is one decode pass. The default is six -- [0.0 .. 1.0],
    # retried whenever the result looks repetitive (compression ratio > 2.4) or
    # unlikely (logprob < -1.0). Real commands hit that: 10 of 42 spoken
    # utterances in one session ran over a second, to 8.1 s, and the ones that
    # showed their working came back as 'drop, drop, drop, drop, drop, dr'. A
    # retry cannot help here -- a verb outside the vocabulary is refused either
    # way, so the safe outcome is reached faster by not paying for five more.
    #
    # This bounds the retries, not the loop inside a single pass, which can
    # still run to 448 tokens -- nine such loops cost ~935 ms each in one
    # session and were all refused, which is the safe outcome at a tolerable
    # price. Capping max_new_tokens is not the next lever for it, for the
    # reason above; repetition_penalty is.
    return " ".join(s.text for s in segments).strip()


def find_device(want):  # pragma: no cover - depends on the host's audio
    """Resolve --mic: an index, or a case-insensitive name fragment."""
    import sounddevice as sd

    if want is None:
        return None
    if str(want).isdigit():
        return int(want)
    matches = [i for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0 and str(want).lower() in d["name"].lower()]
    if not matches:
        raise ValueError(f"no input device matching {want!r}")
    return matches[0]


def input_levels(seconds: float = 1.0):  # pragma: no cover - needs a microphone
    """Every input device and how loud it is right now.

    A Bluetooth headset switched into its headset profile can read 20 dB below
    the built-in microphone, which is the difference between working and
    silently doing nothing.
    """
    import numpy as np
    import sounddevice as sd

    frame_len = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
    frames = max(1, int(seconds * 1000 / config.VAD_FRAME_MS))
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        try:
            with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                dtype="int16", blocksize=frame_len,
                                device=i) as stream:
                r = []
                for _ in range(frames):
                    block, _over = stream.read(frame_len)
                    r.append(float(np.sqrt(np.mean(
                        block[:, 0].astype(np.float64) ** 2))))
            out.append({"index": i, "name": d["name"], "rms": float(np.max(r)),
                        "error": None})
        except Exception as e:
            out.append({"index": i, "name": d["name"], "rms": 0.0,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def speech_worker(out_queue, stop_event, device=None) -> None:  # pragma: no cover
    """Process target: capture -> VAD -> Whisper -> Utterance on the queue."""
    try:
        _speech_worker(out_queue, stop_event, device)
    except Exception as e:
        # Daemon process: without this the traceback vanishes with it and the
        # microphone simply never works, with nothing said anywhere.
        try:
            out_queue.put(MicError(f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        raise


def _speech_worker(out_queue, stop_event, device=None) -> None:  # pragma: no cover
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
    since_level = 0

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
            since_level += 1
            if since_level >= config.MIC_LEVEL_EVERY:
                since_level = 0
                rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                if not out_queue.full():          # utterances have right of way
                    out_queue.put(Level(rms=rms, voiced=bool(voiced)))
            was = seg.speaking
            done = seg.push(voiced, t)
            if not was and seg.speaking:
                current = list(ring)          # pre-roll keeps the first consonant
            if done is None:
                continue
            frames, current = current, []
            if not worth_decoding(frames):
                continue
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            t0 = time.monotonic()
            text = transcribe(model, audio)
            out_queue.put(Utterance(onset_t=done.onset_t, offset_t=done.offset_t,
                                    text=text,
                                    decode_ms=(time.monotonic() - t0) * 1000))
