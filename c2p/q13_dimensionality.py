"""Q13: in which brain area is the dimensionality of neural activity highest (stimulus -> outcome segments)?

Data: every stimulus -> outcome segment is cut into consecutive BIN-wide bins (the last partial bin dropped) and the
bins of all trials are concatenated: a [n_bins_total, n_units] matrix of spike counts per mouse.
Dimensionality: participation ratio PR = (Σλ)² / Σλ² of the eigenvalues λ of the units' correlation matrix (z-scored
counts; main) or covariance matrix (raw counts). PR = 1 if all units co-vary along one dimension, N if independent.
Also: number of principal components needed to explain 80% of the variance.
PR is bounded by the number of units, so every area uses the SAME number of units (N_UNITS, N_DRAWS random draws).
Units without spikes in the segment are skipped.

Segment selection (VARIANTS): a few stimulus -> outcome periods last minutes (up to ~23 min: pauses in the session),
and in most mice trials longer than 2 s make up >75% of all stimulus -> outcome time although the median segment is
0.3-0.6 s. Main: only trials whose segment is <= 2 s (engaged trials). Checks: the first 1 s of every segment, and all
time without any limit.
"""
import numpy as np
import pandas as pd

from .data import AREAS, RESULTS_DIR, iter_recordings
from .q11_decoding import EVENTS

OUT_CSV = RESULTS_DIR / "q13_dimensionality.csv"
BINS = (0.02, 0.05, 0.1)
N_UNITS = (5, 10)
N_DRAWS = 200
VARIANTS = {"trials <= 2 s": dict(max_dur=2.0), "first 1 s of each": dict(cap=1.0), "all time": dict()}


def segment_bins(rec, width, max_dur=None, cap=None):
    """[n_bins_total, n_units] spike counts in consecutive bins of every stim -> outcome segment.

    max_dur: skip trials whose segment is longer; cap: use only the first `cap` seconds of each segment.
    """
    tr = rec.trials[rec.trials[EVENTS].notna().all(axis=1)]
    starts = []
    for a, b in zip(tr.stim_start.to_numpy(), tr.outcome.to_numpy()):
        if max_dur is not None and b - a > max_dur:
            continue
        if cap is not None:
            b = min(b, a + cap)
        n = int((b - a) // width)
        starts.append(a + width * np.arange(n))
    starts = np.concatenate(starts)
    out = np.zeros((len(starts), len(rec.cluster_ids)))
    for j, cid in enumerate(rec.cluster_ids):
        st = np.sort(rec.spike_times(cid))
        out[:, j] = np.searchsorted(st, starts + width) - np.searchsorted(st, starts)
    return out


def participation_ratio(X, kind="corr"):
    C = np.corrcoef(X, rowvar=False) if kind == "corr" else np.cov(X, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return lam.sum() ** 2 / (lam ** 2).sum(), lam


def n_components(lam, frac=0.8):
    lam = np.sort(lam)[::-1]
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)


def compute(force=False, seed=0):
    if OUT_CSV.exists() and not force:
        return pd.read_csv(OUT_CSV)
    rng = np.random.default_rng(seed)
    rows = []
    for rec in iter_recordings():
        areas = np.asarray(rec.areas)
        for (variant, kw), width in [(v, w) for v in VARIANTS.items() for w in BINS]:
            X = segment_bins(rec, width, **kw)
            active = X.std(axis=0) > 0
            for a in AREAS:
                idx = np.flatnonzero((areas == a) & active)
                for n in N_UNITS:
                    if len(idx) < n:
                        continue
                    pr_c, pr_v, k80 = [], [], []
                    for _ in range(N_DRAWS):
                        pick = rng.choice(idx, size=n, replace=False)
                        p, lam = participation_ratio(X[:, pick], "corr")
                        pr_c.append(p); k80.append(n_components(lam))
                        pr_v.append(participation_ratio(X[:, pick], "cov")[0])
                    rows.append({"mouse": rec.mouse, "area": a, "variant": variant, "bin_s": width, "n_units": n, "n_bins": len(X),
                                 "n_available": len(idx), "pr_corr": np.mean(pr_c), "pr_cov": np.mean(pr_v),
                                 "k80": np.mean(k80)})
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df
