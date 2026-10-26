"""Q4: in which brain area are pairwise spike-train interactions strongest at the 100 ms timescale?

Main measure: spike time tiling coefficient (STTC; Cutts & Eglen 2014, J Neurosci) with dt = 100 ms, for every
pair of units recorded in the same area. For units A and B:
    T_A = fraction of the recording within ±dt of a spike of A
    P_A = fraction of A's spikes within ±dt of a spike of B
    STTC = ½ [ (P_A − T_B) / (1 − P_A T_B) + (P_B − T_A) / (1 − P_B T_A) ]
STTC is 0 for independent trains, 1 for perfect co-occurrence, and it is largely insensitive to firing rate.

Robustness: Pearson correlation of spike counts in 100-ms bins. Chance level: the same pair after circularly
shifting B by a random offset (≥ 10 s), which destroys their timing relation but keeps each train intact.
"""
import numpy as np
import pandas as pd

from .data import AREAS, RESULTS_DIR, iter_recordings

PAIRS_CSV = RESULTS_DIR / "q04_pairs.csv"
DT = 0.1            # s
BIN = 0.1           # s, for count correlations
MIN_SHIFT = 10.0    # s, minimum circular shift for the chance control


def tiled_fraction(s, dt, t_end):
    """T: fraction of [0, t_end] within ±dt of any spike in sorted s."""
    if len(s) == 0:
        return 0.0
    covered = min(s[0], dt) + np.minimum(np.diff(s), 2 * dt).sum() + min(t_end - s[-1], dt)
    return covered / t_end


def prop_close(a, b, dt):
    """P: fraction of spikes in a with a spike of b within ±dt (both sorted)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    i = np.searchsorted(b, a)
    left = np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)])
    right = np.abs(b[np.clip(i, 0, len(b) - 1)] - a)
    return float((np.minimum(left, right) <= dt).mean())


def sttc(a, b, dt, t_end, ta=None, tb=None):
    ta = tiled_fraction(a, dt, t_end) if ta is None else ta
    tb = tiled_fraction(b, dt, t_end) if tb is None else tb
    pa, pb = prop_close(a, b, dt), prop_close(b, a, dt)
    return 0.5 * ((pa - tb) / (1 - pa * tb) + (pb - ta) / (1 - pb * ta))


def pairs_of(rec, seed=0):
    rng = np.random.default_rng(seed + rec.mouse)
    t_end = rec.duration_s
    inside = (rec.spike_times_all >= 0) & (rec.spike_times_all < t_end)
    st, cl = rec.spike_times_all[inside], rec.spike_clusters[inside]
    rows = []
    for a in AREAS:
        cids = rec.units.cluster_id[rec.units.area == a].to_numpy()
        trains = [np.sort(st[cl == c]) for c in cids]
        tiles = [tiled_fraction(s, DT, t_end) for s in trains]
        edges = np.arange(0, t_end + BIN, BIN)
        counts = np.array([np.histogram(s, edges)[0] for s in trains], dtype=float)
        cc = np.corrcoef(counts)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                shift = rng.uniform(MIN_SHIFT, t_end - MIN_SHIFT)
                b_shift = np.sort((trains[j] + shift) % t_end)
                rows.append((rec.mouse, a, cids[i], cids[j], len(trains[i]) / t_end, len(trains[j]) / t_end,
                             sttc(trains[i], trains[j], DT, t_end, tiles[i], tiles[j]),
                             sttc(trains[i], b_shift, DT, t_end, tiles[i], tiled_fraction(b_shift, DT, t_end)),
                             cc[i, j]))
    return pd.DataFrame(rows, columns=["mouse", "area", "unit_a", "unit_b", "rate_a", "rate_b",
                                       "sttc", "sttc_shifted", "count_corr"])


def compute(force=False):
    if PAIRS_CSV.exists() and not force:
        return pd.read_csv(PAIRS_CSV)
    out = []
    for rec in iter_recordings():
        p = pairs_of(rec)
        out.append(p)
        print(f"mouse {rec.mouse}: {len(p)} pairs", flush=True)
    df = pd.concat(out, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(PAIRS_CSV, index=False)
    return df
