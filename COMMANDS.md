# SeenTap — command reference

Every command is `python -m seentap.run <command>`. From the project root with
the venv active; otherwise use `.venv/bin/python` in place of `python`.

---

## The short version

```bash
.venv/bin/python -m seentap.run serve --mic 3 --overlay
```

Open `127.0.0.1:8000`, look at a tile, say the verb. Ctrl-C to stop.

It picks the newest usable calibration on its own and prints which one. Nothing
below is needed unless something is wrong or you are recording a study.

---

## Setup — once per machine

| Command | What it does |
| --- | --- |
| `fetch` | Download the face landmarker and Whisper weights. Offline after this. |
| `fetch --portrait` | Also fetch the reference face the landmark test needs. |
| `landmarks` | Live overlay of the indices everything depends on. `q` to quit. |

Run `landmarks` before trusting a session. MediaPipe has moved these indices
between releases, and when they are wrong every symptom downstream looks like
bad calibration instead.

On macOS grant Camera, Microphone and **Accessibility** in System Settings →
Privacy & Security. The first two prompt on use; Accessibility never does, and
without it `--real` clicks are discarded in silence.

---

## Calibration

```bash
python -m seentap.run calibrate --density 25     # ~1 minute
python -m seentap.run fit                        # score it
python -m seentap.run check                      # what it is really worth
```

| Flag | Values | Default |
| --- | --- | --- |
| `calibrate --density` | 5, 9, 13, 25, 49, 81 | 9 |
| `calibrate --out` | path | `logs/calib-<density>-<timestamp>.jsonl` |
| `fit --pattern` | glob of calibration files | `logs/calib-*.jsonl` |
| `fit --held` | a second recording to score against | none |
| `fit --save` | write the fitted mapping | none |
| `check --calibration` | path | newest usable |
| `check --density` | 5, 9, 13, 25, 49, 81 | 9 |
| `check --mapping` | ridge, poly, homography | ridge |

**Nine points is not enough** — leave-one-out error falls 183 → 93 → 69 px
going 9 → 25 → 49. Twenty-five is the usual choice.

**`fit` without `--held` reports fitted error, not accuracy.** It scores the
mapping against the points it was handed and says so. `check` is the honest
number: fresh fixations through the live read path.

**Read `check`'s error shape.** A constant offset means your head has moved
since calibrating and `recalibrate` will fix it in eight seconds. Scatter with
no pattern means the signal is not there and recalibrating will not help.

You do not recalibrate every session. The file is reused; mid-session drift is
handled by requalification.

---

## Microphone

```bash
python -m seentap.run mic              # --seconds 1.5 by default
```

Measures every input device while you speak and names the loudest. Answers "I
said a command and nothing happened" in five seconds — a Bluetooth headset in
its headset profile reads near zero where the built-in reads hundreds. Pass the
index it names to `serve --mic`.

---

## Running it

```bash
python -m seentap.run serve --mic 3 --overlay          # simulated desktop
python -m seentap.run serve --mic 3 --real             # drives real windows
```

| Flag | Values | Default |
| --- | --- | --- |
| `--calibration` | path | newest `logs/calib-*.jsonl` |
| `--mic` | device index or name fragment | OS default |
| `--mapping` | ridge, poly, homography | ridge |
| `--mode` | A, B — recorded in the log and shown on the badge only | B |
| `--condition` | C1 dwell, C2 numbered tiles, C3 fused | C3 |
| `--real` | inject real OS mouse events | off |
| `--overlay` / `--no-overlay` | gaze cursor (implied by `--real`) | off / — |
| `--port` | | 8000 |
| `--lead` | ms before speech onset to bind | 200 |
| `--window` | ms of gaze to aggregate | 300 |
| `--aggregator` | last, mean, median, centroid, zone_mode | median |
| `--min-samples` | refuse below this many | 5 |

`--real` drives whatever window is in front of you, not just the dashboard.
Slam the pointer into a screen corner to abort. Point it at a scratch window
the first time.

`C1` and `C2` are the single-modality baselines for the comparison, not ways to
run the system. `C1` needs no microphone.

`--mode` changes no behaviour in the code as it stands — the dashboard reads it
only to print the badge. It is carried into the session log so a recording can
say which condition it belonged to.

---

## Spoken vocabulary

`click` · `double click` · `right click` · `select` · `scroll up` ·
`scroll down` · `drag` · `drop` · `cancel`

Plus `recalibrate` (or press `r`) and `help` / `controls` for the on-screen
list. Those two need no target and skip the fixation gate, because both are
asked for from states the gate refuses.

A transcript that does not clear the match threshold produces no action rather
than the wrong one.

---

## Analysis

```bash
python -m seentap.run sweep logs/session-<timestamp>.jsonl --plot sweep.png
python -m seentap.run report logs/
```

| Flag | What it does |
| --- | --- |
| `sweep --headline` | the 80-cell surface instead of the full grid |
| `sweep --out` | csv path, default `sweep.csv` |
| `sweep --plot` | write the figure |
| `report [logs]` | directory of sessions, default `logs/` |

Both read cued targets from the log, so a free-play session gives `sweep` an
accuracy of zero and `report` nothing at all. They want a study run.

---

## Tests

```bash
python -m pytest -q
python -m pyflakes seentap tests
```

267 tests, none of which need a camera, a microphone or a display.
