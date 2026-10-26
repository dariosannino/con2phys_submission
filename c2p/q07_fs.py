"""Q7: which brain area has the highest density of putative fast-spiking (FS) interneurons?

"Density" = fraction of a mouse's units in that area that are classified as putative FS (we cannot count cells per volume).

Waveform features (average waveform on the best channel, 30 kHz, cubic-spline upsampled 10x):
  trough_to_peak_ms  time from the trough to the following maximum (the main FS marker: narrow < ~0.4 ms)
  half_width_ms      width of the trough at half its depth
  peak_trough_ratio  height of the after-peak relative to the trough depth
  inverted           the positive deflection is larger than the negative one (axonal / distant / inverted units):
                     trough-to-peak is not meaningful, so these units are left out of the classification
Spike-train features: firing rate, CV of inter-spike intervals, burst fraction (fraction of ISIs < 6 ms).

Main classification: trough_to_peak below the dip of the bimodal trough-to-peak distribution of all units,
found as the minimum of a kernel density estimate between 0.25 and 0.5 ms.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

from .data import FS_WAVEFORM, RESULTS_DIR, iter_recordings

UNITS_CSV = RESULTS_DIR / "q07_units.csv"
UPSAMPLE = 10
BURST_ISI = 0.006      # s


def waveform_features(wf):
    wf = np.asarray(wf, dtype=float)
    wf = wf - np.median(wf[:20])                      # baseline from the first ~0.7 ms
    inverted = wf.max() > -wf.min()
    t = np.arange(len(wf))
    tt = np.arange(0, len(wf) - 1, 1 / UPSAMPLE)
    w = CubicSpline(t, wf)(tt)
    i_tr = int(np.argmin(w))
    i_pk = i_tr + int(np.argmax(w[i_tr:]))
    half = w[i_tr] / 2
    left = i_tr - int(np.argmax(w[:i_tr + 1][::-1] > half))
    right = i_tr + int(np.argmax(w[i_tr:] > half))
    to_ms = 1000 / FS_WAVEFORM / UPSAMPLE
    return {"trough_to_peak_ms": (i_pk - i_tr) * to_ms, "half_width_ms": (right - left) * to_ms,
            "peak_trough_ratio": w[i_pk] / -w[i_tr], "inverted": bool(inverted),
            "amplitude_uv": (w.max() - w.min()) * 1e6}


def train_features(st):
    isi = np.diff(st)
    return {"cv_isi": isi.std() / isi.mean() if len(isi) > 1 else np.nan,
            "burst_fraction": (isi < BURST_ISI).mean() if len(isi) else np.nan}


def compute(force=False):
    if UNITS_CSV.exists() and not force:
        return pd.read_csv(UNITS_CSV)
    rows = []
    for rec in iter_recordings():
        u = rec.units.set_index("cluster_id")
        for cid, area, wf in zip(rec.cluster_ids, rec.areas, rec.waveforms):
            st = rec.spike_times(cid)
            st = st[(st >= 0) & (st < rec.duration_s)]
            rows.append({"mouse": rec.mouse, "cluster_id": cid, "area": area,
                         "rate_hz": len(st) / rec.duration_s, **waveform_features(wf), **train_features(st)})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(UNITS_CSV, index=False)
    return df


def dip_threshold(tp, lo=0.25, hi=0.5):
    """Minimum of the kernel density of trough-to-peak times between lo and hi (ms)."""
    kde = gaussian_kde(tp, bw_method=0.15)
    grid = np.linspace(lo, hi, 501)
    return float(grid[np.argmin(kde(grid))])
