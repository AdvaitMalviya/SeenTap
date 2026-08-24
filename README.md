# SeenTap

Gaze says *where*, speech says *what*.

A hands-free interaction system for people who cannot use a keyboard or mouse.
The webcam tracks gaze and supplies the location; the microphone captures a
spoken verb and supplies the action; a fusion layer decides which gaze position
a given command belongs to. That decision, rather than either sensor, is what
the project actually contributes.

Neither channel works well alone. Gaze is fast and knows exactly where you mean
but has no button, so gaze-only systems fall back on dwell timers that are slow
and fire by accident. Voice is unambiguous about intent but hopeless at
coordinates, which is why voice-only tools end up numbering everything on
screen. Each covers for the other here.

Runs entirely on a stock laptop — no purchased hardware, no network, no
per-request cost. Speech is decoded on-device, so the user's voice never leaves
the machine.

## What it does

**Ten commands**: click, double click, right click, select, scroll up, scroll
down, drag, drop, cancel, recalibrate.

Say **"help"** or **"controls"** at any point and the full list appears on
screen, then fades. That one needs no target, so it skips binding entirely —
and skips the fixation requirement too, since someone who has forgotten the
commands is looking around the screen, which is exactly the state the normal
gate refuses.

**Two modes.** A free cursor driven by filtered gaze, or a 4×3 grid of large
tiles where gaze picks the tile and voice confirms. A tile dwarfs the tracker's
error, so the grid stays usable even at the pessimistic end of webcam accuracy.

**Binding looks backwards.** When a command arrives, the system does not ask
where the eyes are *now*. Transcription costs a few hundred milliseconds and
the user has usually moved on, so binding to the current position acts on the
wrong target — intermittently, which is the hardest kind of bug to catch after
the fact. Commands bind to the gaze held at speech onset instead.

**Two safeguards.** The recogniser only arms when a face is visible, gaze is on
screen, and the eyes are actually fixating, so someone else in the room saying
"click" does nothing. And a transcript that does not clear the match threshold
produces no action rather than the wrong one.

## Install

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
python -m seentap.run fetch --portrait     # model weights, once
```

Python 3.11 through 3.14 all work; 3.12 is a reasonable default. Two pins are
load-bearing:

* **`webrtcvad-wheels`, not `webrtcvad`.** The original builds and then dies at
  import on `pkg_resources`, which setuptools 84 removed. It fails on every
  Python version, not just recent ones.
* **`mediapipe==0.10.35`.** 1.0.1 aborts the process when the face landmarker
  graph opens on macOS arm64 (`Check failed: service_ Service is unavailable`)
  — in both IMAGE and VIDEO modes, CPU delegate included.

On macOS, grant your terminal **Camera**, **Microphone** and **Accessibility**
in System Settings → Privacy & Security. The first two prompt on use.
Accessibility never prompts, and without it real clicks silently do nothing.

## Running it

**1. Check the landmark indices.** Do this before trusting a single session.

```bash
python -m seentap.run landmarks     # q to quit
```

Confirm the green dots sit on your irises and eye corners. MediaPipe has moved
these indices between releases, and when they are wrong everything downstream
fails in a way that looks like bad calibration.

**2. Calibrate once.** Follow the shrinking dot through nine targets, about
twenty seconds. Sit still.

```bash
python -m seentap.run calibrate --density 9
python -m seentap.run fit
```

`calibrate` writes `logs/calib-9-<timestamp>.jsonl` — the feature vectors, the
target coordinates, and your measured blink threshold. **This file is reused;
you do not recalibrate every session.** `fit` then prints the accuracy table
across three calibration densities and three mapping types, with a pass/fail
verdict against 8% of screen width.

**3. Run it.**

```bash
python -m seentap.run serve --calibration logs/calib-9-1758100000.jsonl
```

Open `127.0.0.1:8000`, look at a tile, say "click". Use the real filename —
`--calibration` takes one path, so a `*` glob only works when exactly one file
matches.

Actions land on a simulated desktop by default. `--real` injects genuine OS
events and is meant for logged evaluation runs; during a demo a stray real
click closes the application or hits the wrong window. The corner failsafe
stays armed either way.

**4. Analyse.**

```bash
python -m seentap.run sweep logs/session-<timestamp>.jsonl --plot sweep.png
python -m seentap.run report logs/
```

### Commands

| Command | What it does |
| --- | --- |
| `fetch` | Download model weights. Once, then fully offline. |
| `landmarks` | Live overlay of the indices the pipeline depends on. |
| `calibrate` | One calibration pass at 5, 9 or 13 points. |
| `fit` | Accuracy table across densities and mappings, plus the gate. |
| `serve` | The live system and its dashboard. |
| `sweep` | Replay one session across the fusion parameter grid. |
| `report` | Completion time and error rate across the three conditions. |

## How it fits together

```
webcam ──> gaze.py ──────> GazeSample ──┐
          (478 landmarks,               ├──> fusion.py ──> actions.py ──> OS / sim
           9-D features, One Euro)      │    bind() + state machine
mic ─────> speech.py ────> Utterance ───┘         │
          (VAD, Whisper, own process)             └──> eventlog.py ──> replay.py
```

Landmark inference and speech decoding are both CPU-bound and will fight if
left in one thread; the visible symptom is the cursor stuttering at the exact
moment a command is spoken. Speech therefore runs in its own process behind a
bounded queue, and the recogniser stays idle until voice activity is detected.

Three invariants hold the whole thing together:

1. **One monotonic clock, stamped at the source** — at frame grab and at the
   first voiced VAD frame, never when the value is consumed. A wall clock can
   step backwards and would corrupt the binding silently rather than visibly.
2. **`bind()` is pure.** The parameter sweep is a replay over recorded logs,
   and that only works because the offline sweep calls the exact function the
   live system called.
3. **The log never holds a frame or a waveform.** `eventlog.py` refuses those
   fields outright, so "no imagery or audio is stored" is a property of the
   code rather than a promise about it.

## Evaluation

Three studies, two of which need no participants.

**Calibration accuracy** — three point densities against three mappings (ridge,
quadratic, homography), scored on a held-out grid that never touches the fit.

**The fusion sweep** — every session logs raw gaze, speech onsets, transcripts
and the cued target, and `replay.py` re-runs the binding under any
configuration deterministically. 240 configurations against identical ground
truth, at no cost to anyone. A poor parameter choice made while recording stops
being fatal.

**A three-way comparison** — the fused condition against both single-modality
baselines, which are implemented here rather than merely described: dwell
selection at 800 ms, and numbered tiles with the position spoken aloud.

At three to five participants this is a pilot and is reported as one: Friedman
with Wilcoxon signed-rank and Bonferroni correction, effect sizes and
per-participant plots beside any p-value, no population-level claim.

## Tests

```bash
python -m pytest -q
```

180 tests, none of which need a camera or a microphone.
`tests/test_end_to_end.py` drives a synthetic participant through the whole
pipeline — fusion, execution, logging, replay, the sweep and the CLI.
`tests/test_landmarks_real.py` runs MediaPipe against a reference portrait and
skips if you have not fetched one.

## Known gaps

* Saying **"recalibrate"** moves the state machine but does not yet re-collect
  points or refit the mapping.
* The dashboard's **drift indicator** is not wired up, and the five-point
  requalification hotkey that would go with it is not built. Head drift
  breaking calibration mid-session is the most likely failure in real use.
