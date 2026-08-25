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

**Drift is visible and cheap to undo.** A calibration is fitted at one head
pose and decays as the user settles into their chair — the likeliest way a
working session stops working, and one nothing else would notice, because gaze
keeps arriving and the clicks just land in the wrong tile. The dashboard shows
how far you have moved from the calibration pose — turning and leaning both,
since moving closer rotates nothing but still rescales the mapping about the
gaze axis. Say **"recalibrate"** or press **r** and five points buy the mapping
back in about eight seconds rather than the twenty a full pass costs.

It is measured from the head, in degrees, and both of those are the second
choice. The appealing version — ask the mapping how much of its own prediction
depends on head pose — reports essentially zero, because a calibration
collected sitting still gives it no pose signal to learn from: the ridge
shrinks those columns away and the real error lands in the eye ratios instead.
On a real frame that version reported 0 px for a head turn worth 151. Reporting
the honest measurement in pixels would then need the screen's physical width
and the viewing distance, and the system knows neither, so it reports the angle
it can actually defend.

**Two safeguards.** The recogniser only arms when a face is visible, gaze is on
screen, and the eyes are actually fixating, so someone else in the room saying
"click" does nothing. And a transcript that does not clear the match threshold
produces no action rather than the wrong one.

Two verbs are exempt from the fixation half of that, on purpose. "help" and
"recalibrate" name no target, and both are asked for from states the gate
refuses — a user hunting for the commands is sweeping the screen, and a user
whose calibration has drifted is being mapped off it. Refusing "recalibrate"
because the calibration is broken is the wrong way round. A face is still
required, so a bystander's voice alone does nothing either way.

## Install

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
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

On Windows, allow **Camera** and **Microphone** for desktop apps in Settings →
Privacy & security. There is no Accessibility equivalent — `--real` clicks work
with no grant at all, except into windows running elevated, which silently
ignore synthetic input unless the terminal is elevated too. Both pins above
still apply; the `mediapipe` abort the version pin avoids is macOS arm64 only,
but 0.10.35 is what the landmark indices were checked against.

## Running it

**1. Check the landmark indices.** Do this before trusting a single session.

```bash
python -m seentap.run landmarks     # q to quit
```

Confirm the green dots sit on your irises and eye corners. MediaPipe has moved
these indices between releases, and when they are wrong everything downstream
fails in a way that looks like bad calibration.

**2. Calibrate once.** Follow the shrinking dot through nine targets, about
twenty seconds. Sit still and actually look at each dot — it turns green only
once your eye has stopped moving, and a target you did not settle on is
retried rather than recorded.

```bash
python -m seentap.run calibrate --density 9
python -m seentap.run fit
```

`calibrate` writes `logs/calib-9-<timestamp>.jsonl` — the feature vectors, the
target coordinates, and your measured blink threshold. That last one is
measured rather than assumed because eye shape varies enough between people
that a fixed constant misfires at both ends, and calibration already holds your
eyes open on a target for a second, so the samples are there for free.

**This file is reused; you do not recalibrate every session.** Mid-session
drift is handled by requalification instead, which corrects the mapping in
memory and leaves this file alone.

`fit` then prints the accuracy table across three calibration densities and
three mapping types, with a pass/fail verdict against 8% of screen width. It
also reports how well each axis of your eye actually tracked the target, and
names any target recorded while your eye was still moving. A calibration can
capture nine clean points and still be useless if one carries no signal, and
the error figure alone does not say which — if an axis comes back near zero,
refitting will not help and it needs recording again.

Scored against a **held-out** recording only if you pass one as `--held`.
Without it the numbers are fitted errors, not accuracy, and `fit` says so
rather than quietly flattering itself.

**3. Check the microphone.** Optional, but it answers "I said a command and
nothing happened" in five seconds.

```bash
python -m seentap.run mic
```

It measures every input device while you speak and names the loudest, and
`serve` refuses to sit on a default input that is returning digital silence —
a Bluetooth headset in its headset profile measured 0.0 here where the built-in
managed 462. Pass a specific one as `--mic` if you want to choose.

**4. Run it.**

```bash
python -m seentap.run serve                  # add --mic 3 if `mic` named one
```

It uses the newest usable calibration and prints which one. Pass
`--calibration <path>` to pick a different file; give it one that is not there
and it says so and lists the ones that are.

The dashboard's **mic** meter shows what the microphone is hearing — green
while it has your voice. Without it a dead microphone and an unrecognised word
are the same experience: you speak, and nothing happens either way.

Open `127.0.0.1:8000`, look at a tile, say "click".

Watch the **drift** badge — how far you have moved from the calibration pose,
in degrees, amber at 2°, red at 5°. Two things about it are not decoration. It
reports the median of about a second, because a single frame's head-pose
estimate swings past 10° on a head that has not moved. And it stays blank for
the first five seconds, because MediaPipe's own tracking filter takes about
that long to settle and wanders 4° on the way — reporting during the transient
would send you off to requalify a calibration that is perfectly good.

Say **"recalibrate"** or press **r** to requalify: five targets, an affine
correction fitted on top of the existing mapping, and the old mapping kept
untouched if the new points do not clear the accuracy gate. The session keeps
running throughout — gaze never stops streaming.

Actions land on a simulated desktop by default. `--real` injects genuine OS
events and is meant for logged evaluation runs; during a demo a stray real
click closes the application or hits the wrong window. The corner failsafe
stays armed either way.

**5. Analyse.**

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
| `mic` | List input devices and how loud each one hears you. |
| `fit` | Accuracy table across densities and mappings, plus the gate. |
| `serve` | The live system and its dashboard. |
| `sweep` | Replay one session across the fusion parameter grid. |
| `report` | Completion time and error rate across the three conditions. |

## How it fits together

```
webcam ──> gaze.py ──────> GazeSample ──┐
          (478 landmarks,               ├──> fusion.py ──> actions.py ──> OS / sim
           9-D features, One Euro,      │    bind() + state machine
           drift vs calibration pose)   │
mic ─────> speech.py ────> Utterance ───┘         │
          (VAD, Whisper, own process)             └──> eventlog.py ──> replay.py

"recalibrate" / r ──> calibrate.py ──> new mapping ──> gaze.py
                      (five points, affine correction,
                       refused if it misses the gate)
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

223 tests, none of which need a camera or a microphone.
`tests/test_end_to_end.py` drives a synthetic participant through the whole
pipeline — fusion, execution, logging, replay, the sweep and the CLI.
`tests/test_requalify.py` drives a requalification through the same WebSocket
broadcast the browser receives, with an obedient synthetic participant on the
other end. `tests/test_landmarks_real.py` runs MediaPipe against a reference
portrait and skips if you have not fetched one.

## Known gaps

* Requalification targets are placed as a fraction of the **viewport**, so the
  fit is only true when the browser is fullscreen. The hotkey asks for
  fullscreen — a keypress is the only context a browser grants that from — and
  the spoken verb cannot, so it warns on screen instead.
* A correction lives **in memory for that session only**. Nothing is written
  back to the calibration file, so the next `serve` starts from the original
  mapping and any drift correction has to be earned again. Persisting it would
  mean deciding when a correction is worth keeping, which needs the second
  session's data to answer.
* Five points buy an **affine** correction: offset, scale and shear, which is
  what pose drift mostly looks like. A large change of posture deforms the
  mapping in ways an affine cannot express, and still wants a full pass.
* The calibration window is drawn at the display's **backing scale**, because
  OpenCV's macOS window draws an image one-to-one in device pixels — a canvas
  sized in points covered a quarter of the screen inside a grey backdrop. Its
  real bounds come from the window server, since `getWindowImageRect` only
  echoes the image size back. The area excludes the menu bar, so targets are
  logged at where they were actually drawn rather than where they were asked
  for.
* A target that will not settle after three attempts is **kept anyway** and
  reported, rather than dropped. Dropping it would silently change the
  calibration density the fit was scored at, which is one of the things the
  evaluation compares.
* Landmarks are detected in MediaPipe's **IMAGE mode, not VIDEO**. VIDEO
  carries a temporal tracking filter that wanders on input which never
  changes — 0.008 of horizontal eye ratio across 150 identical frames, the
  same size as the between-target noise that was destroying calibration.
  IMAGE mode measured exactly zero spread on the same input and costs half a
  millisecond a frame, so the tracking shortcut buys nothing at 30 fps.
* Calibration files carry a **feature-layout version**. Change what
  `features()` computes and every saved file becomes unreadable — the vectors
  are already extracted, so old ones would be silent nonsense rather than a
  worse fit. `serve` and `fit` refuse them and say to record again.
* Turning the depth reading into an angle assumes a **laptop at arm's length**
  — roughly a 300 mm screen at 600 mm, the `SCREEN_HALF_TAN` constant. It is
  the one piece of geometry the system cannot measure for itself, and it wants
  retuning for a desktop monitor. Rotation does not depend on it.
