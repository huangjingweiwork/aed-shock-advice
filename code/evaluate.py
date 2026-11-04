"""Scoring against the AHA targets, and choosing a decision threshold."""
import numpy as np, pandas as pd
from config import AHA, REPORT_ONLY, SEED

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def lower_bound(hit, patient, rng, n=1000):
    """One-sided 90% lower confidence limit, by resampling whole patients (clips from one person are alike)."""
    pats = np.unique(patient)
    by = {p: hit[patient == p] for p in pats}
    stats = [np.concatenate([by[p] for p in rng.choice(pats, len(pats))]).mean() for _ in range(n)]
    return float(np.percentile(stats, 10))


def score(category, shocked, cat, metric):
    m = category == cat
    return shocked[m].mean() if metric == "sensitivity" else (~shocked[m]).mean()


def aha_table(category, patient, shocked):
    """One row per AHA category: observed score, its lower limit, and pass/fail.

    A row passes when the observed score beats the AHA goal AND its lower limit is at least the lowest limit
    the AHA accepts. Fine VF and other VT are reported without a goal, as the AHA asks.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for cat, metric, goal, min_lcl in AHA + [(c, "sensitivity", None, None) for c in REPORT_ONLY]:
        m = category == cat
        if not m.any():
            continue
        hit = shocked[m] if metric == "sensitivity" else ~shocked[m]
        s = float(hit.mean())
        lb = lower_bound(hit, patient[m], rng) if goal is not None else None
        rows.append(dict(category=cat, clips=int(m.sum()), patients=len(np.unique(patient[m])), metric=metric,
                         goal=goal, min_lower90=min_lcl, score=round(s, 4),
                         lower90=None if lb is None else round(lb, 4),
                         passes=None if goal is None else bool(s > goal and lb >= min_lcl)))
    return pd.DataFrame(rows)


def choose_threshold(category, prob, min_clips=20):
    """Threshold with the best worst-case margin over the AHA goals, judged on validation patients only.

    For each candidate t, margin = score - goal for every AHA category with at least `min_clips` clips;
    keep the t whose smallest margin is largest. Ties go to the t closest to 0.5.
    """
    rows = [(c, mt, g) for c, mt, g, _ in AHA if (category == c).sum() >= min_clips]
    worst = np.array([min(score(category, prob >= t, c, mt) - g for c, mt, g in rows) for t in GRID])
    best = GRID[worst >= worst.max() - 1e-9]
    return float(best[np.argmin(np.abs(best - 0.5))])
