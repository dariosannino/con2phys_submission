"""Q3: which brain area has the highest density of ripples (as in hippocampal sharp-wave ripples)?

Constraint of this dataset: the LFP is sampled at 500 Hz, and its spectrum indicates low-pass filtering
(not documented; the dataset only says the LFP "has already undergone a preprocessing pipeline").
From 100 to 140 Hz power drops 15-100x, while the 20-80 Hz trend predicts ~2x, and above ~150 Hz
it is a flat noise floor, the same on every channel of every area. Detection therefore uses 110-180 Hz, the part of the ripple band
(~110-250 Hz in mice) where some signal is left.

Detection, per channel, over the whole recording:
  1. band-pass 110-180 Hz (4th-order Butterworth, zero phase), Hilbert envelope;
  2. z-score the envelope with median and MAD (robust to the events themselves);
  3. candidate event = stretch with z > 2 lasting 15-300 ms whose peak z > 5.
Shared-artifact rejection: high-frequency transients that appear in all areas at the same moment are not ripples
(they are locked to stimulus and outcome, i.e. movement/EMG). For each area we compute an area-wide index (median
z over its channels); an event is kept as LOCAL if, at its peak, the area-wide index of every other area is < 1.5.
"""
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings

EVENTS_CSV = RESULTS_DIR / "q03_events.csv"
CHANNELS_CSV = RESULTS_DIR / "q03_channels.csv"
VALIDATION_NPZ = RESULTS_DIR / "q03_validation.npz"

BAND = (110.0, 180.0)
Z_EDGE, Z_PEAK = 2.0, 5.0
DUR = (0.015, 0.300)
OTHERS_MAX = 1.5
TRIG_HALF = 0.2                     # s, window for event-triggered LFP average
SOS = butter(4, BAND, btype="band", fs=FS_LFP, output="sos")


def envelope_z(x):
    """Robust z-score of the band-limited Hilbert envelope of a 1D signal (float32)."""
    b = sosfiltfilt(SOS, x)
    env = np.abs(hilbert(b, N=next_fast_len(len(b))))[:len(b)]
    med = np.median(env)
    mad = 1.4826 * np.median(np.abs(env - med))
    return ((env - med) / mad).astype(np.float32)


def find_events(z):
    """Peak index, duration (s) and peak z of every stretch z > Z_EDGE that qualifies as an event."""
    d = np.diff(np.r_[0, (z > Z_EDGE).astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for s, e in zip(starts, ends):
        dur = (e - s) / FS_LFP
        if DUR[0] <= dur <= DUR[1]:
            k = s + int(np.argmax(z[s:e]))
            if z[k] > Z_PEAK:
                out.append((k, dur, float(z[k])))
    return out


def detect(rec):
    """All candidate events of one recording, with the area-wide index of every area at each event peak."""
    area_index, events = {}, []
    for a in AREAS:
        lfp = rec.lfp[a]
        z_all = np.empty(lfp.shape, dtype=np.float32)
        for c in range(lfp.shape[0]):
            z_all[c] = envelope_z(np.asarray(lfp[c], dtype=np.float64))
            for k, dur, zp in find_events(z_all[c]):
                events.append((rec.mouse, a, c, k, dur, zp))
        area_index[a] = np.median(z_all, axis=0)
        del z_all
    ev = pd.DataFrame(events, columns=["mouse", "area", "channel", "sample", "duration_s", "peak_z"])
    for a in AREAS:
        ev[f"index_area{a}"] = area_index[a][ev["sample"].to_numpy()]
    idx = ev[[f"index_area{b}" for b in AREAS]].to_numpy()
    own = ev["area"].to_numpy()[:, None] == np.array(AREAS)[None, :]
    ev["others_max"] = np.where(own, -np.inf, idx).max(axis=1)        # highest index among the other two areas
    ev["local"] = ev.others_max < OTHERS_MAX
    ev["t_s"] = ev["sample"] / FS_LFP
    return ev


def validate(rec, ev_best):
    """For the best channel of each area: event-triggered LFP and spike rate of the area's units around events."""
    out = {}
    w = int(TRIG_HALF * FS_LFP)
    for a in AREAS:
        e = ev_best[ev_best.area == a]
        if e.empty:
            continue
        ch = int(e.channel.iloc[0])
        x = np.asarray(rec.lfp[a][ch], dtype=np.float64) * 1e6
        idx = e["sample"].to_numpy()
        idx = idx[(idx > w) & (idx < len(x) - w)]
        out[f"m{rec.mouse}_a{a}_lfp"] = np.mean([x[i - w:i + w] - x[i - w:i + w].mean() for i in idx], axis=0) \
            if len(idx) else np.zeros(2 * w)
        cids = rec.units.cluster_id[rec.units.area == a].to_numpy()
        st = np.sort(rec.spike_times_all[np.isin(rec.spike_clusters, cids)])
        bins = np.arange(-TRIG_HALF, TRIG_HALF + 1e-9, 0.01)
        h = np.zeros(len(bins) - 1)
        for t in idx / FS_LFP:
            lo, hi = np.searchsorted(st, [t - TRIG_HALF, t + TRIG_HALF])
            h += np.histogram(st[lo:hi] - t, bins)[0]
        base = ((st > 0) & (st < rec.duration_s)).sum() / rec.duration_s
        out[f"m{rec.mouse}_a{a}_peth"] = h / max(len(idx), 1) / 0.01 / base      # firing relative to the mean rate
    return out


def channel_table(ev, rec):
    """Event rate (per min) of every channel: all candidates and local events."""
    rows = []
    for a in AREAS:
        n_ch = rec.lfp[a].shape[0]
        e = ev[ev.area == a]
        n_all = e.groupby("channel").size().reindex(range(n_ch), fill_value=0)
        n_loc = e[e.local].groupby("channel").size().reindex(range(n_ch), fill_value=0)
        rows.append(pd.DataFrame({"mouse": rec.mouse, "area": a, "channel": range(n_ch),
                                  "rate_all_per_min": n_all.values / rec.duration_s * 60,
                                  "rate_local_per_min": n_loc.values / rec.duration_s * 60,
                                  "duration_min": rec.duration_s / 60}))
    return pd.concat(rows, ignore_index=True)


def compute(force=False):
    if EVENTS_CSV.exists() and CHANNELS_CSV.exists() and VALIDATION_NPZ.exists() and not force:
        return pd.read_csv(EVENTS_CSV), pd.read_csv(CHANNELS_CSV), dict(np.load(VALIDATION_NPZ))
    all_ev, all_ch, val = [], [], {}
    for rec in iter_recordings():
        ev = detect(rec)
        ch = channel_table(ev, rec)
        best = ch.loc[ch.groupby("area").rate_local_per_min.idxmax(), ["area", "channel"]]
        ev_best = ev[ev.local].merge(best, on=["area", "channel"])
        val.update(validate(rec, ev_best))
        all_ev.append(ev); all_ch.append(ch)
        print(f"mouse {rec.mouse}: {len(ev)} candidates, {ev.local.sum()} local", flush=True)
    ev, ch = pd.concat(all_ev, ignore_index=True), pd.concat(all_ch, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    ev.to_csv(EVENTS_CSV, index=False)
    ch.to_csv(CHANNELS_CSV, index=False)
    np.savez_compressed(VALIDATION_NPZ, **val)
    return ev, ch, val
