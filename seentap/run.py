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
    """One calibration pass: settle for a second, then collect for a second."""
    import cv2

    w, h = config.SCREEN_W, config.SCREEN_H
    pts = calibrate.targets(args.density, w, h)
    tr = _tracker()
    win = "seentap calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    out = Path(args.out or f"{config.LOG_DIR}/calib-{args.density}-{int(time.time())}.jsonl")
    kept = 0
    open_ears = []
    with eventlog.EventLog(out) as log:
        log.write("calibration", density=args.density, screen=[w, h],
                  features_version=config.FEATURES_VERSION)
        for (tx, ty) in pts:
            for phase, seconds in (("settle", 1.0), ("collect", 1.0)):
                t_end = time.monotonic() + seconds
                samples = []
                while time.monotonic() < t_end:
                    canvas = np.zeros((h, w, 3), np.uint8)
                    frac = max(0.0, (t_end - time.monotonic()) / seconds)
                    r = int(6 + 34 * frac) if phase == "settle" else 8
                    colour = (60, 160, 255) if phase == "settle" else (90, 230, 120)
                    cv2.circle(canvas, (int(tx), int(ty)), r, colour, -1)
                    cv2.imshow(win, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        tr.close(); cv2.destroyAllWindows(); return 1
                    f, blink, _ = tr.read_raw(eventlog.now())
                    if phase == "collect" and f is not None:
                        samples.append({"f": f.tolist(), "conf": 0.0 if blink else 1.0,
                                        "blink": bool(blink)})
                        if not blink:
                            open_ears.append(tr.last_ear)
                if phase == "collect":
                    med = calibrate.condense(samples)
                    if med is None:
                        print(f"  target ({tx:.0f},{ty:.0f}): no usable samples")
                        continue
                    log.write("calib_point", f=med.tolist(), target=[tx, ty],
                              n_raw=len(samples))
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
    print(f"{kept}/{len(pts)} targets captured -> {out}")
    return 0 if kept >= 4 else 1


def _load_calib(path):
    F, XY, version = [], [], 1
    for row in eventlog.read(path):
        if row.get("kind") == "calib_point":
            F.append(row["f"])
            XY.append(row["target"])
        elif row.get("kind") == "calibration":
            version = row.get("features_version", 1)
    return np.asarray(F, dtype=float), np.asarray(XY, dtype=float), version


def _require_calib(path):
    """One named calibration file, or a message that says what to do instead.

    A mistyped path reads as zero calibration points -- eventlog.read returns
    nothing for a file that is not there -- and the empty array only fails
    forty frames down inside sklearn, where nothing names the actual mistake.
    """
    def bail(why: str, fix: str | None = None):
        found = sorted(glob.glob(f"{config.LOG_DIR}/calib-*.jsonl"))
        tail = fix or ("\n  ".join(["calibration files here:", *found]) if found
                       else "no calibration files yet -- run:\n"
                            "  python -m seentap.run calibrate --density 9")
        print(f"{path} {why}.\n{tail}", file=sys.stderr)
        raise SystemExit(2)

    if not Path(path).exists():
        bail("does not exist")
    F, XY, version = _load_calib(path)
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
    for path in sorted(glob.glob(args.pattern)):
        F, XY, version = _load_calib(path)
        if version != config.FEATURES_VERSION or len(F) < 4:
            skipped.append(f"{path} (v{version}, {len(F)} points)")
            continue
        # Keyed by density, and the newest wins: keying by point count silently
        # dropped every earlier pass at the same density, so a second nine-point
        # calibration replaced the first without saying so.
        sessions[len(F)] = (F, XY, path)
    if skipped:
        print("skipped:\n  " + "\n  ".join(skipped), file=sys.stderr)
    if not sessions:
        print(f"no usable calibration matched {args.pattern!r}", file=sys.stderr)
        return 1

    if args.held:
        Fh, XYh, _ = _load_calib(args.held)
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


def cmd_serve(args) -> int:  # pragma: no cover - needs a camera and a mic
    import multiprocessing as mp

    import uvicorn

    from seentap import server, speech
    from seentap.fusion import FusionConfig

    cfg = FusionConfig(lead_ms=args.lead, window_ms=args.window,
                       aggregator=args.aggregator, min_samples=args.min_samples)
    F, XY = _require_calib(args.calibration)
    mapping = calibrate.FITTERS[args.mapping](F, XY)
    print(f"mapping: {args.mapping} on {len(F)} points")

    # The pose the mapping was fitted at. Live drift is measured against it.
    tracker = _tracker(mapping, f_ref=np.median(F, axis=0))
    for row in eventlog.read(args.calibration):
        if row.get("kind") == "blink_threshold":
            tracker.blink_ear = row["value"]
            print(f"blink threshold: {tracker.blink_ear:.3f} (from calibration)")

    rt = server.Runtime(cfg, mode=args.mode, real=args.real,
                        condition=args.condition, tracker=tracker)

    queue, stop = None, None
    if args.condition != "C1":          # the gaze-only baseline needs no mic
        queue, stop = mp.Queue(maxsize=8), mp.Event()
        mp.Process(target=speech.speech_worker, args=(queue, stop),
                   daemon=True).start()

    server.configure(tracker, queue, rt)
    print(f"dashboard: open a browser on 127.0.0.1:{args.port}")
    print("say 'recalibrate' or press r on the dashboard if gaze drifts")
    try:
        uvicorn.run(server.app, host="127.0.0.1", port=args.port,
                    log_level="warning")
    finally:
        if stop is not None:
            stop.set()
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
    c.add_argument("--density", type=int, default=9, choices=config.DENSITIES)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_calibrate)

    f = sub.add_parser("fit", help="nine-cell table and the day-8 gate")
    f.add_argument("--pattern", default=f"{config.LOG_DIR}/calib-*.jsonl")
    f.add_argument("--held")
    f.add_argument("--save")
    f.set_defaults(fn=cmd_fit)

    s = sub.add_parser("serve", help="live system and dashboard")
    s.add_argument("--calibration", required=True)
    s.add_argument("--mapping", default="poly", choices=list(calibrate.FITTERS))
    s.add_argument("--mode", default="B", choices=["A", "B"])
    s.add_argument("--condition", default="C3", choices=list(config.CONDITIONS))
    s.add_argument("--real", action="store_true",
                   help="inject real OS events instead of the simulated desktop")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--lead", type=int, default=200)
    s.add_argument("--window", type=int, default=300)
    s.add_argument("--aggregator", default="median")
    s.add_argument("--min-samples", type=int, default=5, dest="min_samples")
    s.set_defaults(fn=cmd_serve)

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
