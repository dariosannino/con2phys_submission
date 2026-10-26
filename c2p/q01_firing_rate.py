"""Q1: which brain area has the lowest firing rate over the entire recording?

Firing rate of a unit = number of its spikes inside the recording / recording length.
The recording spans [0, duration_s], with duration taken from the LFP length.
"""
import numpy as np
import pandas as pd

from .data import RESULTS_DIR, iter_recordings

UNIT_RATES_CSV = RESULTS_DIR / "q01_unit_rates.csv"
TIMECOURSE_CSV = RESULTS_DIR / "q01_rate_timecourse.csv"
BIN_S = 60


def unit_rates(rec):
    """One row per unit: spikes inside [0, duration] and the resulting rate (Hz)."""
    inside = (rec.spike_times_all >= 0) & (rec.spike_times_all < rec.duration_s)
    n = pd.Series(rec.spike_clusters[inside]).value_counts()
    u = rec.units[["mouse", "cluster_id", "area"]].copy()
    u["n_spikes"] = u.cluster_id.map(n).fillna(0).astype(int)
    u["n_spikes_outside"] = u.cluster_id.map(pd.Series(rec.spike_clusters[~inside]).value_counts()).fillna(0).astype(int)
    u["duration_s"] = rec.duration_s
    u["rate_hz"] = u.n_spikes / rec.duration_s
    return u


def rate_timecourse(rec, bin_s=BIN_S):
    """Mean rate per area in consecutive bins (one row per mouse x area x bin)."""
    edges = np.arange(0, rec.duration_s + bin_s, bin_s)
    rows = []
    for area, cids in rec.units.groupby("area").cluster_id:
        mask = np.isin(rec.spike_clusters, cids.to_numpy())
        counts, _ = np.histogram(rec.spike_times_all[mask], bins=edges)
        width = np.diff(np.minimum(edges, rec.duration_s))
        for i, (c, w) in enumerate(zip(counts, width)):
            if w > 0:
                rows.append({"mouse": rec.mouse, "area": area, "t_min": edges[i] / 60,
                             "rate_hz": c / w / len(cids)})
    return pd.DataFrame(rows)


def compute(force=False):
    """Compute (or read cached) unit rates and time courses for all mice."""
    if UNIT_RATES_CSV.exists() and TIMECOURSE_CSV.exists() and not force:
        return pd.read_csv(UNIT_RATES_CSV), pd.read_csv(TIMECOURSE_CSV)
    rates, tc = [], []
    for rec in iter_recordings():
        rates.append(unit_rates(rec))
        tc.append(rate_timecourse(rec))
    rates, tc = pd.concat(rates, ignore_index=True), pd.concat(tc, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    rates.to_csv(UNIT_RATES_CSV, index=False)
    tc.to_csv(TIMECOURSE_CSV, index=False)
    return rates, tc
