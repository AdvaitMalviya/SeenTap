"""Command line entry points.

    python -m seentap.run landmarks              # day 1: verify indices
    python -m seentap.run calibrate --density 9  # collect one calibration pass
    python -m seentap.run fit                    # the nine-cell table + gate
    python -m seentap.run serve                  # live system + dashboard
    python -m seentap.run sweep logs/session.jsonl
    python -m seentap.run report logs/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from seentap import analyze, calibrate, config, eventlog, replay


def _tracker(mapping=None, f_ref=None):  # pragma: no cover - needs a camera
    from seentap.gaze import GazeTracker

    return GazeTracker(mapping=mapping, f_ref=f_ref)


def cmd_landmarks(args) -> int:  # pragma: no cover - needs a camera
    """Draw the indices this code depends on. They have shifted between
    MediaPipe releases, so confirm them before trusting a single session."""
    import cv2

    from seentap import gaze as G

    tr = _tracker()
    marks = {"L_IRIS": G.L_IRIS, "R_IRIS": G.R_IRIS, "L_INNER": G.L_INNER,
             "L_OUTER": G.L_OUTER, "R_INNER": G.R_INNER, "R_OUTER": G.R_OUTER}
    print("q to quit")
    try:
        while True:
            f, blink, frame = tr.read_raw(eventlog.now())
            if frame is None:
                break
            lms = tr.last_landmarks
            if lms is not None:
                h, w = frame.shape[:2]
                for name, i in marks.items():
                    x, y = int(lms[i][0] * w), int(lms[i][1] * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame, name, (x + 6, y - 6),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
            cv2.putText(frame, "BLINK" if blink else "open", (10, 24),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 200, 255), 2)
            cv2.imshow("seentap landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tr.close()
        cv2.destroyAllWindows()
    return 0


def cmd_calibrate(args) -> int:  # pragma: no cover - needs a camera
    """One calibration pass. Each target waits for the eye to stop moving.

    A fixed one-second settle was not enough after a big saccade: across three
    real sessions the targets following a full-screen jump came out three to
    thirty times less repeatable than the rest, because the point was recorded
    while the eye was still travelling. Nothing downstream can tell such a
    point from a good one -- it is a confident median of the wrong place -- so
    the check belongs here, before it is written.
    """
    import cv2

    from seentap import gaze as G

    w, h = config.SCREEN_W, config.SCREEN_H
    tr = _tracker()
    win = "seentap calibration"
    # FREERATIO, and a canvas in device pixels. OpenCV's macOS window draws an
    # image one-to-one in device pixels, so a canvas sized in points covers a
    # quarter of a Retina screen inside a grey backdrop: measured, a 1512x982
    # canvas gave a view 756x491 points, pushed off the bottom of the screen.
    scale = G.backing_scale()
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def measure():
        """The view's real bounds in screen points, once macOS has settled.

        cv2.getWindowImageRect is no use -- it echoes the image size back --
        so this asks the window server, which is also the only way to notice
        the resize that lands a second or two after the fullscreen animation.
        """
        return G.view_rect(win, (0, 0, w, h))

    blank = np.zeros((h * scale, w * scale, 3), np.uint8)
    t_settle = time.monotonic() + config.CALIB_WINDOW_SETTLE_S
    while time.monotonic() < t_settle:
        cv2.imshow(win, blank)
        cv2.waitKey(30)
    rect = measure()
    print(f"drawing area: {rect[2]}x{rect[3]} points at ({rect[0]},{rect[1]}), "
          f"canvas {scale}x for a {scale}x display")

    fracs = calibrate.targets(args.density, 1.0, 1.0)     # positions as fractions
    canvas = np.zeros((rect[3] * scale, rect[2] * scale, 3), np.uint8)

    def show(tx, ty, radius, colour, note):
        canvas[:] = 0
        cv2.circle(canvas, (int(tx * scale), int(ty * scale)),
                   radius * scale, colour, -1)
        if note:
            cv2.putText(canvas, note,
                        (canvas.shape[1] // 2 - 250 * scale,
                         int(canvas.shape[0] * 0.92)),
                        cv2.FONT_HERSHEY_PLAIN, 2.0 * scale, (110, 110, 120), 2)
        cv2.imshow(win, canvas)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def capture(tx, ty, note):
        """Settle until the eye is still, then record. None if the user quit."""
        recent, samples, ears, settled = [], [], [], None
        t0 = time.monotonic()
        while True:
            now = time.monotonic()
            if settled is None:
                # Shrink over the minimum hold, then sit at the collect size:
                # scaling it to the maximum instead meant the dot was still
                # halfway down when the eye settled and the change to green
                # arrived as a jump.
                frac = max(0.0, 1 - (now - t0) / config.CALIB_SETTLE_MIN_S)
                r = config.CALIB_DOT_PX + int(
                    (config.CALIB_DOT_MAX_PX - config.CALIB_DOT_PX) * frac)
                quit_ = show(tx, ty, r, (60, 160, 255), note)
            else:
                quit_ = show(tx, ty, config.CALIB_DOT_PX, (90, 230, 120), note)
            if quit_:
                return None, None, None
            f, blink, _ = tr.read_raw(eventlog.now())
            if f is None:
                continue
            row = {"f": f.tolist(), "conf": 0.0 if blink else 1.0,
                   "blink": bool(blink)}
            if settled is None:
                recent = (recent + [row])[-config.CALIB_STEADY_FRAMES:]
                waited = now - t0
                if ((waited >= config.CALIB_SETTLE_MIN_S
                     and calibrate.steadiness(recent) <= config.CALIB_STEADY)
                        or waited >= config.CALIB_SETTLE_MAX_S):
                    settled = now
                continue
            samples.append(row)
            if not blink:
                ears.append(tr.last_ear)
            if now - settled >= config.CALIB_COLLECT_S:
                return samples, ears, calibrate.steadiness(samples)

    out = Path(args.out or f"{config.LOG_DIR}/calib-{args.density}-{int(time.time())}.jsonl")
    kept, shaky, open_ears = 0, 0, []
    with eventlog.EventLog(out) as log:
        log.write("calibration", density=args.density, screen=[w, h],
                  window=list(rect), scale=scale,
                  features_version=config.FEATURES_VERSION)
        for (fx, fy) in fracs:
            now_rect = measure()
            if now_rect != rect:
                # The window moved under us. Points already recorded keep the
                # screen coordinates they were actually drawn at, so only the
                # remaining ones need the new geometry.
                print(f"  window changed to {now_rect}; adapting")
                rect = now_rect
                canvas = np.zeros((rect[3] * scale, rect[2] * scale, 3), np.uint8)
                log.write("calibration_window", window=list(rect))
            rx, ry, rw, rh = rect
            tx, ty = fx * rw, fy * rh
            for attempt in range(config.CALIB_ATTEMPTS):
                note = "" if attempt == 0 else "hold still and follow the dot"
                samples, ears, spread = capture(tx, ty, note)
                if samples is None:
                    tr.close()
                    cv2.destroyAllWindows()
                    return 1
                if spread <= config.CALIB_STEADY:
                    break
            med = calibrate.condense(samples)
            if med is None:
                print(f"  ({tx:.0f},{ty:.0f}): no usable samples")
                continue
            if spread > config.CALIB_STEADY:
                shaky += 1
                print(f"  ({tx:.0f},{ty:.0f}): never settled "
                      f"(spread {spread:.3f} > {config.CALIB_STEADY}), kept anyway")
            open_ears.extend(ears)
            # Logged in screen coordinates, which is the space the mapping
            # predicts in; the dot was drawn in window coordinates.
            log.write("calib_point", f=med.tolist(),
                      target=[rx + tx, ry + ty],
                      n_raw=len(samples), steadiness=spread)
            kept += 1
        # The eyes were held open on a target for a second per point, so the
        # blink threshold comes free rather than as a fixed guess.
        from seentap.gaze import blink_threshold
        thr = blink_threshold(open_ears)
        log.write("blink_threshold", value=thr, n=len(open_ears))
        print(f"blink threshold for this user: {thr:.3f} "
              f"(from {len(open_ears)} open-eye frames)")
    tr.close()
    cv2.destroyAllWindows()
    print(f"{kept}/{len(fracs)} targets captured"
          + (f", {shaky} never settled" if shaky else "") + f" -> {out}")
    return 0 if kept >= 4 else 1


def cmd_check(args) -> int:  # pragma: no cover - needs a camera
    """Look at the same targets again and measure where the mapping puts you.

    `fit` scores the calibration against itself, which says only that the
    mapping can reproduce points it was handed. This is the number that
    matters: fresh fixations, the live pipeline, and the error broken into the
    shape it actually has -- a constant offset means the head has moved since
    calibrating, a scale error means the distance has, and scatter with no
    pattern means the signal is not there.
    """
    import cv2

    from seentap import gaze as G

    path = args.calibration or _newest_calib()
    F, XY = _require_calib(path)
    mapping = calibrate.FITTERS[args.mapping](F, XY)
    print(f"checking {path} with the {args.mapping} mapping")

    w, h = config.SCREEN_W, config.SCREEN_H
    scale = G.backing_scale()
    win = "seentap check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, np.zeros((h * scale, w * scale, 3), np.uint8))
    t_end = time.monotonic() + config.CALIB_WINDOW_SETTLE_S
    while time.monotonic() < t_end:
        cv2.imshow(win, np.zeros((h * scale, w * scale, 3), np.uint8))
        cv2.waitKey(30)
    rx, ry, rw, rh = G.view_rect(win, (0, 0, w, h))
    canvas = np.zeros((rh * scale, rw * scale, 3), np.uint8)
    tr = _tracker(mapping, f_ref=np.median(F, axis=0))
    for row in eventlog.read(path):
        if row.get("kind") == "blink_threshold":
            tr.blink_ear = row["value"]

    rows = []
    try:
        for fx, fy in calibrate.targets(args.density, 1.0, 1.0):
            tx, ty = fx * rw, fy * rh
            preds, t0, settled = [], time.monotonic(), None
            while True:
                now = time.monotonic()
                canvas[:] = 0
                cv2.circle(canvas, (int(tx * scale), int(ty * scale)),
                           config.CALIB_DOT_PX * scale,
                           (90, 230, 120) if settled else (60, 160, 255), -1)
                # Draw where the mapping thinks you are, so the error is
                # visible while it is measured rather than only afterwards.
                if preds:
                    px, py = preds[-1]
                    cv2.circle(canvas, (int((px - rx) * scale), int((py - ry) * scale)),
                               14 * scale, (80, 120, 250), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 1
                s_, _f = tr.read(eventlog.now())
                if s_ is None or s_.blink:
                    continue
                if settled is None:
                    if now - t0 >= config.CALIB_SETTLE_MIN_S:
                        settled = now
                    preds.append((s_.x, s_.y))
                    continue
                preds.append((s_.x, s_.y))
                if now - settled >= config.CALIB_COLLECT_S:
                    break
            keep = preds[-int(config.CALIB_COLLECT_S * 20):]
            px = float(np.median([p[0] for p in keep]))
            py = float(np.median([p[1] for p in keep]))
            rows.append((rx + tx, ry + ty, px, py))
            print(f"  target ({rx+tx:6.0f},{ry+ty:6.0f})  predicted "
                  f"({px:7.0f},{py:6.0f})  off by "
                  f"{np.hypot(px-rx-tx, py-ry-ty):6.0f} px")
    finally:
        tr.close()
        cv2.destroyAllWindows()

    a = np.array(rows)
    d = a[:, 2:] - a[:, :2]
    err = np.linalg.norm(d, axis=1)
    print(f"\nmean error {err.mean():.0f} px  (dx {np.abs(d[:,0]).mean():.0f}, "
          f"dy {np.abs(d[:,1]).mean():.0f})   gate is "
          f"{config.GATE_FRAC*config.SCREEN_W:.0f} px")
    print(f"constant offset: ({d[:,0].mean():+.0f}, {d[:,1].mean():+.0f}) px")
    print(f"scatter about that offset: ({d[:,0].std():.0f}, {d[:,1].std():.0f}) px")
    if np.linalg.norm(d.mean(axis=0)) > 1.5 * d.std(axis=0).mean():
        print("  -> mostly a constant offset. The head has moved since "
              "calibrating; say 'recalibrate' or press r to correct it.")
    else:
        print("  -> mostly scatter, not offset. Recalibrating will not fix "
              "this; the eye signal itself is weak.")
    return 0


def _by_age(paths):
    """Oldest first, by the clock rather than by the name.

    Sorting the names put calib-9- ahead of calib-49-, because as text "9"
    beats "4". A nine-point file from any time therefore outranked every
    denser calibration ever recorded, and the newest-wins rules quietly picked
    a stale one -- two density comparisons were run against it without either
    of the new files being read.
    """
    return sorted(paths, key=lambda p: (os.path.getmtime(p), p))


def _newest_calib():
    """The newest usable calibration, which is almost always the one meant.

    Newest alone is not enough: a run quit halfway leaves a short file behind,
    and picking it would be worse than asking.
    """
    for path in reversed(_by_age(glob.glob(f"{config.LOG_DIR}/calib-*.jsonl"))):
        F, _XY, version, _d = _load_calib(path)
        if version == config.FEATURES_VERSION and len(F) >= 4:
            return path
    return None


def _load_calib(path):
    F, XY, version, density = [], [], 1, None
    for row in eventlog.read(path):
        if row.get("kind") == "calib_point":
            F.append(row["f"])
            XY.append(row["target"])
        elif row.get("kind") == "calibration":
            version = row.get("features_version", 1)
            density = row.get("density")
    F, XY = np.asarray(F, dtype=float), np.asarray(XY, dtype=float)
    return F, XY, version, density or len(F)


def _require_calib(path):
    """One named calibration file, or a message that says what to do instead.

    A mistyped path reads as zero calibration points -- eventlog.read returns
    nothing for a file that is not there -- and the empty array only fails
    forty frames down inside sklearn, where nothing names the actual mistake.
    """
    def bail(why: str, fix: str | None = None):
        found = _by_age(glob.glob(f"{config.LOG_DIR}/calib-*.jsonl"))
        tail = fix or ("\n  ".join(["calibration files here:", *found]) if found
                       else "no calibration files yet -- run:\n"
                            "  python -m seentap.run calibrate --density 9")
        print(f"{path} {why}.\n{tail}", file=sys.stderr)
        raise SystemExit(2)

    if path is None:
        bail("no calibration file given and none found")
    if not Path(path).exists():
        bail("does not exist")
    F, XY, version, _density = _load_calib(path)
    if version != config.FEATURES_VERSION:
        bail(f"was recorded with feature layout v{version}, and this build uses "
             f"v{config.FEATURES_VERSION}",
             "The saved vectors mean something different now, so it has to be "
             "recorded again:\n  python -m seentap.run calibrate --density 9")
    if len(F) < 4:
        bail(f"holds {len(F)} calibration point(s); a mapping needs at least four")
    return F, XY


def cmd_fit(args) -> int:
    """Study 1 and the day-8 gate, from saved calibration passes."""
    sessions, skipped = {}, []
    for path in _by_age(glob.glob(args.pattern)):
        F, XY, version, density = _load_calib(path)
        if version != config.FEATURES_VERSION or len(F) < 4:
            skipped.append(f"{path} (v{version}, {len(F)} points)")
            continue
        # Keyed by the density that was asked for, not the number of points
        # that survived: a nine-point run that lost four targets is a damaged
        # nine, and filing it under "5" invented a density that was never run.
        sessions[density] = (F, XY, path)
        if len(F) != density:
            # The header records the density that was asked for. It used to be
            # able to disagree with what the grid actually laid out -- 25 came
            # out as 4x6 and 49 as 5x10 -- so say so rather than printing a
            # density column that names a point count nobody collected.
            print(f"{path}: header says density {density} but holds {len(F)} "
                  f"points; the density column below is the header's",
                  file=sys.stderr)
    if skipped:
        print("skipped:\n  " + "\n  ".join(skipped), file=sys.stderr)
    if not sessions:
        print(f"no usable calibration matched {args.pattern!r}", file=sys.stderr)
        return 1

    if args.held:
        Fh, XYh, _v, _d = _load_calib(args.held)
        held, held_from = (Fh, XYh), args.held
    else:
        # Without a separate recording there is nothing held out, and scoring a
        # fit on the points it was fitted to is not an accuracy figure.
        density = max(sessions)
        held, held_from = sessions[density][:2], f"{sessions[density][2]} (ITSELF)"

    table = calibrate.nine_cell({d: v[:2] for d, v in sessions.items()}, held)
    print(analyze.markdown_table(table))
    passed, threshold, best = calibrate.gate_passed(table, config.SCREEN_W)
    print(f"\nday-8 gate: threshold {threshold:.0f} px "
          f"({config.GATE_FRAC:.0%} of {config.SCREEN_W}); "
          f"best {best['mean_err']:.1f} px "
          f"({best['mapping']}, {best['density']} points)")
    print(f"scored against: {held_from}")
    if "ITSELF" in held_from:
        print("  these are fitted errors, not accuracy -- record a second pass "
              "and pass it as --held for a real number")

    shaky = [(r["target"], r["steadiness"])
             for r in eventlog.read(sessions[max(sessions)][2])
             if r.get("kind") == "calib_point"
             and r.get("steadiness", 0) > config.CALIB_STEADY]
    if shaky:
        print("\ntargets recorded while the eye was still moving:")
        for (tx, ty), sp in shaky:
            print(f"  ({tx:.0f},{ty:.0f}) spread {sp:.3f}")
        print("  each of these is a confident median of the wrong place")

    names = ["hL", "hR", "vL", "vR", "yaw", "pitch", "roll", "iod", "1"]
    kept = calibrate.useful_columns(sessions[max(sessions)][0])
    dropped = [names[i] for i in range(len(names)) if i not in kept]
    if dropped:
        print(f"\nfeatures dropped: {', '.join(dropped)} -- they barely moved "
              f"while you were calibrating, so the fit has no measured "
              f"relationship for them and would extrapolate wildly the moment "
              f"you did move")

    Fb, XYb = sessions[max(sessions)][:2]
    sig = calibrate.signal_report(Fb, XYb)
    print(f"\neye signal: horizontal r = {sig['horizontal']:+.2f}, "
          f"vertical r = {sig['vertical']:+.2f}")
    for axis, r in sig.items():
        if abs(r) < 0.7:
            print(f"  the {axis} eye signal does not track the target. The "
                  f"calibration did not capture {axis} gaze -- refitting will "
                  f"not help, it has to be recorded again.")

    print("\nPASS - continue on MediaPipe" if passed else
          "\nFAIL - freeze MediaPipe, WebGazer.js becomes primary")
    if args.save:
        Path(args.save).write_text(json.dumps(table, indent=2))
    return 0


def cmd_fetch(args) -> int:
    """Day 0: pull the model weights so the demo never needs the network."""
    from seentap.gaze import ensure_model

    path = ensure_model()
    print("face landmarker ->", path)
    if args.portrait:
        import urllib.request

        dest = Path(config.MODEL_DIR) / "portrait.jpg"
        if not dest.exists():
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-assets/portrait.jpg", dest)
        print("reference portrait ->", dest)
    return 0


def _working_mic():  # pragma: no cover - needs a microphone
    """Swap off the default input if it is delivering nothing at all."""
    from seentap import speech

    try:
        rows = speech.input_levels(config.MIC_PROBE_S)
    except Exception:
        return None
    if not rows:
        return None
    default = rows[0]
    live = [r for r in rows if r["rms"] > config.MIC_DEAD_RMS and not r["error"]]
    if default["rms"] > config.MIC_DEAD_RMS or not live:
        return None
    best = max(live, key=lambda r: r["rms"])
    print(f"default input '{default['name']}' is silent (peak "
          f"{default['rms']:.1f}); using [{best['index']}] {best['name']} instead")
    return best["index"]


def cmd_mic(args) -> int:  # pragma: no cover - needs a microphone
    """Which input devices exist and whether any of them can hear you."""
    from seentap import speech

    print(f"speak now -- measuring each input for {args.seconds:.0f}s\n")
    rows = speech.input_levels(args.seconds)
    if not rows:
        print("no input devices at all", file=sys.stderr)
        return 1
    for r in rows:
        if r["error"]:
            print(f"  [{r['index']}] {r['name'][:38]:38} unusable: {r['error']}")
            continue
        db = 20 * np.log10(max(r["rms"], 1) / 32768)
        ok = r["rms"] >= config.MIC_QUIET_RMS
        print(f"  [{r['index']}] {r['name'][:38]:38} peak {r['rms']:7.1f} "
              f"({db:5.0f} dBFS)  {'ok' if ok else 'too quiet'}")
    best = max(rows, key=lambda r: r["rms"])
    if best["rms"] < config.MIC_QUIET_RMS:
        print("\nNothing heard you. Grant Microphone to your terminal in System "
              "Settings > Privacy & Security, and speak while this runs.")
        return 1
    print(f"\nloudest: [{best['index']}] {best['name']}\n"
          f"  python -m seentap.run serve --mic {best['index']}")
    return 0


def cmd_serve(args) -> int:  # pragma: no cover - needs a camera and a mic
    import multiprocessing as mp

    import uvicorn

    from seentap import actions, overlay, server, speech
    from seentap.fusion import FusionConfig

    cfg = FusionConfig(lead_ms=args.lead, window_ms=args.window,
                       aggregator=args.aggregator, min_samples=args.min_samples)
    path = args.calibration or _newest_calib()
    F, XY = _require_calib(path)
    print(f"calibration: {path}")
    mapping = calibrate.FITTERS[args.mapping](F, XY)
    print(f"mapping: {args.mapping} on {len(F)} points")

    # The pose the mapping was fitted at. Live drift is measured against it.
    tracker = _tracker(mapping, f_ref=np.median(F, axis=0))
    for row in eventlog.read(path):
        if row.get("kind") == "blink_threshold":
            tracker.blink_ear = row["value"]
            print(f"blink threshold: {tracker.blink_ear:.3f} (from calibration)")

    # On by default with --real: injecting clicks into other people's windows
    # while the user cannot see where the system thinks they are looking is the
    # one combination that has no defence.
    want_overlay = args.overlay or (args.real and not args.no_overlay)
    ov_queue = ov_stop = None
    if want_overlay:
        ov_queue, ov_stop = mp.Queue(maxsize=2), mp.Event()
        mp.Process(target=overlay.overlay_worker, args=(ov_queue, ov_stop),
                   daemon=True).start()
        print("gaze cursor: on (look slightly off until the dot lands, then speak)")

    try:
        rt = server.Runtime(cfg, mode=args.mode, real=args.real,
                            condition=args.condition, tracker=tracker,
                            overlay=ov_queue)
    except actions.PermissionError_ as e:
        # Up front, not at the first click that quietly does nothing.
        print(e, file=sys.stderr)
        if ov_stop is not None:
            ov_stop.set()
        tracker.close()
        return 2

    queue, stop = None, None
    if args.condition != "C1":          # the gaze-only baseline needs no mic
        queue, stop = mp.Queue(maxsize=8), mp.Event()
        mic = speech.find_device(args.mic)
        if mic is None:
            # A device returning exact digital silence is broken, not quiet: a
            # Bluetooth headset in its headset profile measured 0.0 here while
            # the built-in managed 462. Left alone it looks identical to a
            # command that was not understood.
            mic = _working_mic()
        print(f"microphone: {'system default' if mic is None else mic}"
              f"   (run `python -m seentap.run mic` if nothing is heard)")
        mp.Process(target=speech.speech_worker, args=(queue, stop, mic),
                   daemon=True).start()

    server.configure(tracker, queue, rt)
    if args.real:
        print("REAL clicks are armed -- slam the pointer into a screen corner "
              "to abort")
    print(f"dashboard: open a browser on 127.0.0.1:{args.port}")
    print("say 'recalibrate' or press r on the dashboard if gaze drifts")
    try:
        uvicorn.run(server.app, host="127.0.0.1", port=args.port,
                    log_level="warning")
    finally:
        for ev in (stop, ov_stop):
            if ev is not None:
                ev.set()
        tracker.close()
        rt.close()
    return 0


def cmd_sweep(args) -> int:
    """Study 2. One recording, the whole parameter space."""
    session = replay.load_session(args.session)
    configs = replay.headline_configs() if args.headline else replay.default_configs()
    df = replay.sweep(session, configs)
    df = df.sort_values("accuracy", ascending=False)
    print(df.head(15).to_string(index=False))
    out = args.out or "sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} configurations -> {out}")
    if args.plot:
        print("plot ->", analyze.plot_sweep(df, args.plot))
    return 0


def cmd_report(args) -> int:
    """Study 3. Completion time and error rate across the three conditions."""
    by_condition: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(f"{args.logs.rstrip('/')}/*.jsonl")):
        session = None
        trials = []
        for row in eventlog.read(path):
            if row.get("kind") == "session":
                session = row.get("condition")
            elif row.get("kind") == "trial":
                trials.append(row)
        if session:
            by_condition.setdefault(session, []).extend(trials)
    if not by_condition:
        print(f"no trial events under {args.logs!r}", file=sys.stderr)
        return 1

    print(analyze.markdown_table(analyze.condition_metrics(by_condition)))
    times = {c: [t["completion_s"] for t in v if "completion_s" in t]
             for c, v in by_condition.items()}
    lengths = {len(v) for v in times.values()}
    if len(times) >= 3 and len(lengths) == 1 and lengths != {0}:
        out = analyze.compare_conditions(times)
        print(f"\nFriedman: p = {out['omnibus']['p']:.4f}  (n = {out['n']}"
              f"{', pilot' if out['pilot'] else ''})")
        for (a, b), row in out["pairwise"].items():
            print(f"  {a} vs {b}: p = {row['p']:.4f}, "
                  f"Bonferroni p = {row['p_adj']:.4f}, r = {row['effect_size']:+.2f}")
    else:
        print("\nunbalanced or too few conditions for the omnibus test")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seentap")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("fetch", help="download model weights (day 0)")
    d.add_argument("--portrait", action="store_true",
                   help="also fetch the reference face used by the tests")
    d.set_defaults(fn=cmd_fetch)

    sub.add_parser("landmarks", help="verify MediaPipe indices on a live frame"
                   ).set_defaults(fn=cmd_landmarks)

    c = sub.add_parser("calibrate", help="run one calibration pass")
    c.add_argument("--density", type=int, default=9,
                   choices=config.DENSITY_CHOICES)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_calibrate)

    f = sub.add_parser("fit", help="nine-cell table and the day-8 gate")
    f.add_argument("--pattern", default=f"{config.LOG_DIR}/calib-*.jsonl")
    f.add_argument("--held")
    f.add_argument("--save")
    f.set_defaults(fn=cmd_fit)

    s = sub.add_parser("serve", help="live system and dashboard")
    s.add_argument("--calibration",
                   help="default: the newest logs/calib-*.jsonl")
    s.add_argument("--mic", help="input device index or name fragment; "
                                 "default is whatever the OS calls default")
    s.add_argument("--mapping", default="ridge", choices=list(calibrate.FITTERS))
    s.add_argument("--mode", default="B", choices=["A", "B"])
    s.add_argument("--condition", default="C3", choices=list(config.CONDITIONS))
    s.add_argument("--real", action="store_true",
                   help="inject real OS events instead of the simulated desktop")
    s.add_argument("--overlay", action="store_true",
                   help="draw the gaze cursor over every window (implied by --real)")
    s.add_argument("--no-overlay", action="store_true",
                   help="suppress the gaze cursor even with --real")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--lead", type=int, default=200)
    s.add_argument("--window", type=int, default=300)
    s.add_argument("--aggregator", default="median")
    s.add_argument("--min-samples", type=int, default=5, dest="min_samples")
    s.set_defaults(fn=cmd_serve)

    ck = sub.add_parser("check", help="measure real accuracy on fresh fixations")
    ck.add_argument("--calibration", help="default: the newest usable one")
    ck.add_argument("--density", type=int, default=9,
                    choices=config.DENSITY_CHOICES)
    ck.add_argument("--mapping", default="ridge", choices=list(calibrate.FITTERS))
    ck.set_defaults(fn=cmd_check)

    mi = sub.add_parser("mic", help="list input devices and how loud they are")
    mi.add_argument("--seconds", type=float, default=1.5)
    mi.set_defaults(fn=cmd_mic)

    w = sub.add_parser("sweep", help="replay one session across the grid")
    w.add_argument("session")
    w.add_argument("--headline", action="store_true", help="the 80-cell surface")
    w.add_argument("--out")
    w.add_argument("--plot")
    w.set_defaults(fn=cmd_sweep)

    r = sub.add_parser("report", help="three-condition comparison")
    r.add_argument("logs", nargs="?", default=config.LOG_DIR)
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
