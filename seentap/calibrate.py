"""Study 1: three densities by three mappings, scored on a held-out grid.

Fitting reads saved samples rather than the camera, so one recording session
can be re-fitted fifty times. That is what makes the nine-cell table cheap.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from seentap import config

ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0)


class _SkMapping:
    def __init__(self, model):
        self.model = model

    def predict(self, F: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.atleast_2d(F)), dtype=float)


class _Homography:
    """The planar fit works on a 2-D point, so the eye ratios are pooled first."""

    def __init__(self, H: np.ndarray):
        self.H = H

    @staticmethod
    def to_2d(F: np.ndarray) -> np.ndarray:
        F = np.atleast_2d(F)
        return np.column_stack([F[:, :2].mean(axis=1), F[:, 2:4].mean(axis=1)])

    def predict(self, F: np.ndarray) -> np.ndarray:
        src = self.to_2d(F)
        pts = np.column_stack([src, np.ones(len(src))]) @ self.H.T
        w = np.where(np.abs(pts[:, 2]) < 1e-12, 1e-12, pts[:, 2])
        return np.column_stack([pts[:, 0] / w, pts[:, 1] / w])


def fit_ridge(F: np.ndarray, XY: np.ndarray, alpha: float | None = None) -> _SkMapping:
    if alpha is None:
        alpha = loo_cv_alpha(F, XY, fit_ridge, ALPHAS)
    return _SkMapping(Ridge(alpha=alpha).fit(F, XY))


def fit_poly(F: np.ndarray, XY: np.ndarray, alpha: float | None = None,
             degree: int = 2) -> _SkMapping:
    if alpha is None:
        alpha = loo_cv_alpha(F, XY, lambda f, xy, a=None: fit_poly(f, xy, a or 1e-6),
                             ALPHAS)
    return _SkMapping(make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=alpha),
    ).fit(F, XY))


def fit_homography(F: np.ndarray, XY: np.ndarray) -> _Homography:
    import cv2

    src = _Homography.to_2d(F).astype(np.float32)
    dst = np.asarray(XY, dtype=np.float32)
    if len(src) < 4:
        raise ValueError("a homography needs at least four calibration points")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("homography did not converge on these points")
    return _Homography(np.asarray(H, dtype=float))


FITTERS = {"ridge": fit_ridge, "poly": fit_poly, "homography": fit_homography}


class _Corrected:
    """A fitted mapping plus an affine correction from a handful of fresh points.

    Mid-session the head has moved and the mapping, fitted at one pose, is
    wrong by an amount that is mostly offset and scale. Five points cannot
    refit nine features -- the quadratic alone wants fifty-four terms -- but
    they over-determine a 2-D affine, which is the shape the error actually
    has. Correcting the output costs six parameters instead of refitting the
    input with too little data to do it honestly.
    """

    def __init__(self, base, A: np.ndarray):
        self.base = base
        self.A = A

    def predict(self, F: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(self.base.predict(F), dtype=float))
        return np.column_stack([P, np.ones(len(P))]) @ self.A.T


def mean_err(model, F: np.ndarray, XY: np.ndarray) -> float:
    """Mean Euclidean pixel error, without the breakdown ``validate`` gives."""
    pred = np.atleast_2d(np.asarray(model.predict(np.asarray(F, dtype=float))))
    return float(np.linalg.norm(pred - np.asarray(XY, dtype=float), axis=1).mean())


def fit_correction(base, F: np.ndarray, XY: np.ndarray,
                   max_err: float | None = None):
    """Requalification: correct a drifted mapping from a few fresh points.

    Returns (model, before_px, after_px). Raises when the residual is worse
    than the accuracy the project accepts at all -- five points collected while
    the user blinked or glanced away fit an affine to noise just as willingly
    as to signal, and silently replacing a working mapping with that, mid-task,
    leaves the user no way back.
    """
    F = np.asarray(F, dtype=float)
    XY = np.asarray(XY, dtype=float)
    if len(F) < 3:
        raise ValueError(f"an affine correction needs three points, got {len(F)}")
    # Correct the original mapping, never a correction: requalifying twice has
    # to re-measure the drift, not stack a second guess on top of the first.
    base = getattr(base, "base", base)
    P = np.atleast_2d(np.asarray(base.predict(F), dtype=float))
    A, *_ = np.linalg.lstsq(np.column_stack([P, np.ones(len(P))]), XY, rcond=None)
    model = _Corrected(base, np.asarray(A.T, dtype=float))

    before, after = mean_err(base, F, XY), mean_err(model, F, XY)
    limit = config.GATE_FRAC * config.SCREEN_W if max_err is None else max_err
    if after > limit:
        raise ValueError(f"requalification residual {after:.0f} px exceeds the "
                         f"{limit:.0f} px gate; keeping the old mapping")
    return model, before, after


def loo_cv_alpha(F: np.ndarray, XY: np.ndarray, fitter, alphas=ALPHAS) -> float:
    """Leave one calibration point out; keep the alpha with the lowest error."""
    F, XY = np.asarray(F, dtype=float), np.asarray(XY, dtype=float)
    n = len(F)
    best, best_err = alphas[0], np.inf
    for a in alphas:
        errs = []
        for i in range(n):
            keep = np.arange(n) != i
            try:
                m = fitter(F[keep], XY[keep], a)
            except (ValueError, TypeError):
                errs = [np.inf]
                break
            errs.append(np.linalg.norm(m.predict(F[i:i + 1])[0] - XY[i]))
        err = float(np.mean(errs))
        if err < best_err:
            best, best_err = a, err
    return best


def validate(model, F: np.ndarray, XY: np.ndarray) -> dict:
    """Mean Euclidean pixel error, with the horizontal/vertical split.

    Vertical is consistently worse than horizontal, so the two are reported
    apart rather than averaged into one flattering number.
    """
    pred = model.predict(np.asarray(F, dtype=float))
    XY = np.asarray(XY, dtype=float)
    d = pred - XY
    err = np.linalg.norm(d, axis=1)
    return {
        "mean_err": float(err.mean()),
        "std_err": float(err.std()),
        "mean_dx": float(np.abs(d[:, 0]).mean()),
        "mean_dy": float(np.abs(d[:, 1]).mean()),
        "n": int(len(XY)),
    }


def condense(raw: list[dict]) -> np.ndarray | None:
    """One calibration target's worth of samples down to one feature vector.

    Blinks and low-confidence frames go first, then the median of whatever is
    left -- a median rather than a mean so one bad frame cannot drag the point.
    """
    kept = [r["f"] for r in raw
            if not r.get("blink") and r.get("conf", 1.0) >= config.CONF_FLOOR]
    if not kept:
        return None
    return np.median(np.asarray(kept, dtype=float), axis=0)


def nine_cell(sessions: dict[int, tuple], held: tuple) -> list[dict]:
    """The project's first result: density x mapping, on points never fitted."""
    Fv, XYv = held
    rows = []
    for density, (F, XY) in sorted(sessions.items()):
        for name, fitter in FITTERS.items():
            try:
                row = validate(fitter(F, XY), Fv, XYv)
            except ValueError as e:
                row = {"mean_err": float("inf"), "std_err": float("nan"),
                       "mean_dx": float("nan"), "mean_dy": float("nan"),
                       "n": len(XYv), "error": str(e)}
            rows.append({"density": density, "mapping": name, **row})
    return rows


def signal_report(F: np.ndarray, XY: np.ndarray) -> dict:
    """Does the eye actually track the target on each axis?

    A calibration can capture nine clean points and still be useless if one
    axis carries no signal, and the fitted error alone does not say which.
    Vertical is the one that fails: it is the harder axis, and the eyelid
    covers most of the iris's travel.
    """
    F, XY = np.asarray(F, dtype=float), np.asarray(XY, dtype=float)
    out = {}
    for name, fi, xyi in (("horizontal", 0, 0), ("vertical", 2, 1)):
        col, tgt = F[:, fi], XY[:, xyi]
        out[name] = (0.0 if np.std(col) < 1e-12 or np.std(tgt) < 1e-12
                     else float(np.corrcoef(col, tgt)[0, 1]))
    return out


def gate_passed(table: list[dict], screen_w: int = config.SCREEN_W,
                frac: float = config.GATE_FRAC):
    """The day-8 decision. Best cell against 8% of screen width.

    Miss it and the MediaPipe path freezes where it stands and WebGazer becomes
    primary. Either branch leaves a working system; only an open-ended rescue
    attempt sinks the schedule.
    """
    threshold = frac * screen_w
    best = min(table, key=lambda r: r["mean_err"])
    return best["mean_err"] <= threshold, threshold, best


def targets(density: int, w: int = config.SCREEN_W, h: int = config.SCREEN_H,
            margin: float = 0.08) -> list[tuple[float, float]]:
    """Calibration target positions for 5, 9 or 13 points."""
    lo, hi, mid = margin, 1 - margin, 0.5
    five = [(mid, mid), (lo, lo), (hi, lo), (lo, hi), (hi, hi)]
    nine = five + [(mid, lo), (mid, hi), (lo, mid), (hi, mid)]
    thirteen = nine + [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    grid = {5: five, 9: nine, 13: thirteen}[density]
    return [(fx * w, fy * h) for fx, fy in grid]
