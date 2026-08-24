"""Logs to the report's tables, figures and statistics.

At three to five participants this is a pilot and it is reported as one:
non-parametric tests, effect sizes and per-participant trajectories beside any
p-value, and no population-level claim.
"""
from __future__ import annotations

import numpy as np

PILOT_MAX_N = 12


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])

    def cell(v):
        return f"{v:.1f}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def latency_summary(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _rank_biserial(a, b) -> float:
    """Matched-pairs effect size, bounded in [-1, 1] and undisturbed by ties."""
    from scipy.stats import rankdata

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d))
    plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    total = plus + minus
    return float((plus - minus) / total) if total else 0.0


def compare_conditions(data: dict[str, list]) -> dict:
    """Friedman omnibus, Wilcoxon pairwise, Bonferroni correction.

    Parametric repeated-measures analysis is not defensible at this sample
    size, so it is not offered.
    """
    from itertools import combinations

    from scipy.stats import friedmanchisquare, wilcoxon

    names = list(data)
    arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    n = len(next(iter(arrays.values())))

    omnibus = {"test": "friedman", "statistic": float("nan"), "p": float("nan")}
    if len(names) >= 3 and n >= 3:
        try:
            stat, p = friedmanchisquare(*(arrays[k] for k in names))
            omnibus = {"test": "friedman", "statistic": float(stat), "p": float(p)}
        except ValueError as e:
            omnibus["note"] = str(e)

    pairs = list(combinations(names, 2))
    pairwise = {}
    for a, b in pairs:
        try:
            _, p = wilcoxon(arrays[a], arrays[b])
            p = float(p)
        except ValueError:
            p = float("nan")
        pairwise[(a, b)] = {
            "p": p,
            "p_adj": min(1.0, p * len(pairs)) if p == p else p,
            "effect_size": _rank_biserial(arrays[a], arrays[b]),
            "median_diff": float(np.median(arrays[a] - arrays[b])),
        }

    return {"n": n, "pilot": n <= PILOT_MAX_N, "omnibus": omnibus,
            "pairwise": pairwise, "conditions": names}


def condition_metrics(sessions: dict[str, list[dict]]) -> list[dict]:
    """Per-condition completion time and error rate, ready for the report."""
    rows = []
    for cond, trials in sessions.items():
        times = [t["completion_s"] for t in trials if "completion_s" in t]
        errors = [not t.get("correct", True) for t in trials]
        rows.append({
            "condition": cond,
            "trials": len(trials),
            "median_time_s": float(np.median(times)) if times else float("nan"),
            "error_rate": float(np.mean(errors)) if errors else float("nan"),
        })
    return rows


def plot_sweep(df, path: str = "sweep.png"):  # pragma: no cover
    """The trade-off surface: accuracy against lead offset and window width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agg, sub in df.groupby("aggregator"):
        g = sub.groupby("lead_ms")["accuracy"].mean()
        ax.plot(g.index, g.values, marker="o", label=agg)
    ax.set_xlabel("lead offset (ms)")
    ax.set_ylabel("binding accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(title="aggregator", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path
