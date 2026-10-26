"""Q10: which brain area has the shortest intrinsic neuronal timescale during baseline activity?

Method of Murray et al. (2014, Nat Neurosci). For each unit and trial, spikes are counted in BIN = 50 ms bins over a
baseline window. The autocorrelation at lag k is the Pearson correlation, ACROSS TRIALS, between the counts in bin i and
bin i+k, averaged over all i. It decays with lag and is fitted with
        R(k) = A * (exp(-k * BIN / tau) + B)
tau is the intrinsic timescale. As in Murray et al., the fit starts at the lag where R is largest among the first
lags (the first lags can be lowered by refractoriness/adaptation).

Windows ("baseline" = no stimulus and no outcome):
  pre1s  the 1 s before stim_start, only trials where that second starts after the previous trial ended (main)
  iti    the 0.8 s before trial_start, only trials where it starts after the previous trial ended
Units with a baseline rate below MIN_RATE are left out (their autocorrelation is mostly noise).
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .data import AREAS, RESULTS_DIR, iter_recordings

ACF_NPZ = RESULTS_DIR / "q10_acf.npz"
UNITS_CSV = RESULTS_DIR / "q10_units.csv"
BIN = 0.05
WINDOWS = {"pre1s": ("stim_start", 1.0), "iti": ("trial_start", 0.8)}
MIN_RATE = 1.0


def window_starts(trials, event, length):
    """Start times of baseline windows [event - length, event) that begin after the previous trial ended."""
    end_prev = trials.trial_end.shift(1).fillna(-np.inf).to_numpy()
    start = trials[event].to_numpy() - length
    return start[(start >= end_prev) & (start >= 0)]


def acf_across_trials(counts):
    """counts [n_trials, n_bins] -> autocorrelation for lags 1..n_bins-1 (NaN where undefined)."""
    n_bins = counts.shape[1]
    c = counts - counts.mean(axis=0)
    sd = counts.std(axis=0)
    out = np.full(n_bins - 1, np.nan)
    for k in range(1, n_bins):
        r = []
        for i in range(n_bins - k):
            if sd[i] > 0 and sd[i + k] > 0:
                r.append((c[:, i] * c[:, i + k]).mean() / (sd[i] * sd[i + k]))
        if r:
            out[k - 1] = np.mean(r)
    return out


def model(lag_s, a, tau, b):
    return a * (np.exp(-lag_s / tau) + b)


def fit_tau(acf):
    """Fit the exponential from the largest of the first 3 lags onward. Returns (tau_s, A, B, R²) or NaNs."""
    lags = np.arange(1, len(acf) + 1) * BIN
    ok = np.isfinite(acf)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    k0 = int(np.nanargmax(acf[:3]))
    x, y = lags[k0:][ok[k0:]], acf[k0:][ok[k0:]]
    try:
        p, _ = curve_fit(model, x, y, p0=(max(y[0], 1e-3), 0.1, 0.0), bounds=([0, 0.005, -1], [5, 2.0, 5]), maxfev=20000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan
    r2 = 1 - np.sum((y - model(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
    return p[1], p[0], p[2], r2


def unit_acfs(rec):
    rows, acfs = [], {}
    st_all, cl_all = rec.spike_times_all, rec.spike_clusters
    for wname, (event, length) in WINDOWS.items():
        starts = window_starts(rec.trials, event, length)
        n_bins = int(round(length / BIN))
        edges = starts[:, None] + np.arange(n_bins + 1)[None, :] * BIN          # [n_trials, n_bins + 1]
        for cid, area in zip(rec.cluster_ids, rec.areas):
            st = np.sort(st_all[cl_all == cid])
            cum = np.searchsorted(st, edges)
            counts = np.diff(cum, axis=1).astype(float)
            rate = counts.sum() / (len(starts) * length)
            acf = acf_across_trials(counts)
            acfs[f"m{rec.mouse}_u{cid}_{wname}"] = acf
            tau, a, b, r2 = fit_tau(acf)
            rows.append({"mouse": rec.mouse, "cluster_id": cid, "area": area, "window": wname, "n_trials": len(starts),
                         "baseline_rate_hz": rate, "tau_ms": 1000 * tau, "A": a, "B": b, "r2": r2})
    return pd.DataFrame(rows), acfs


def compute(force=False):
    if UNITS_CSV.exists() and ACF_NPZ.exists() and not force:
        return pd.read_csv(UNITS_CSV), dict(np.load(ACF_NPZ))
    frames, acfs = [], {}
    for rec in iter_recordings():
        df, a = unit_acfs(rec)
        frames.append(df); acfs.update(a)
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(UNITS_CSV, index=False)
    np.savez_compressed(ACF_NPZ, **acfs)
    return df, acfs


def population_taus(units, acfs, window="pre1s", min_rate=MIN_RATE):
    """One fit per mouse and area on the mean autocorrelation of its units (as in Murray et al.)."""
    rows = []
    sel = units[(units.window == window) & (units.baseline_rate_hz >= min_rate)]
    for (m, a), g in sel.groupby(["mouse", "area"]):
        acf = np.nanmean([acfs[f"m{m}_u{c}_{window}"] for c in g.cluster_id], axis=0)
        tau, amp, b, r2 = fit_tau(acf)
        rows.append({"mouse": m, "area": a, "n_units": len(g), "tau_ms": 1000 * tau, "A": amp, "B": b, "r2": r2})
    return pd.DataFrame(rows)
