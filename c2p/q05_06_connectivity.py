"""Q5 (undirected) and Q6 (directed) functional connectivity between the three areas.

Signals. All channels share an external reference, and Q3 showed large transients that hit all areas at the same
instant (movement/EMG). Both create zero-lag "connectivity" that is not communication between areas. Therefore:
  * LFP is re-referenced locally: bipolar derivations (difference of neighbouring channels of the same area),
    N_BIPOLAR per area, spread over depth;
  * the main measures ignore zero-lag coupling.

Undirected (Q5)
  wPLI  weighted phase lag index (Vinck et al. 2011): |E[Im S_xy]| / E[|Im S_xy|]. 0 = no consistent non-zero lag.
  coh   magnitude-squared coherence (inflated by zero-lag mixing; robustness only).
Directed (Q6)
  PSI   phase slope index (Nolte et al. 2008): Im Σ_f C*(f) C(f+df) over 1-100 Hz. PSI(x, y) > 0  ⇔  x leads y.
  GC    time-domain Granger causality, x → y = ln(σ²(y | past y) / σ²(y | past y, past x)), LFP at 250 Hz, 25 lags.
  GC on population spike rates (10-ms bins, 10 lags) as a spike-based check.
All spectral quantities use 2-s Hann windows (0.5 Hz resolution) over the whole recording.
"""
import itertools

import numpy as np
import pandas as pd

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings

LFP_CSV = RESULTS_DIR / "q0506_lfp_pairs.csv"
SPECTRA_NPZ = RESULTS_DIR / "q0506_spectra.npz"
SPIKE_CSV = RESULTS_DIR / "q0506_spike_gc.csv"

N_BIPOLAR = 6
NSEG = 2 * FS_LFP
BAND = (1.0, 100.0)
GC_DECIM, GC_LAGS = 2, 25            # 250 Hz, 100 ms of history
SPK_BIN, SPK_LAGS = 0.01, 10         # 10 ms bins, 100 ms of history
AREA_PAIRS = list(itertools.combinations(AREAS, 2))


# ------------------------------------------------------------------ signals
def bipolar(lfp, n=N_BIPOLAR):
    """n bipolar derivations (channel i+1 minus channel i) spread evenly over depth: [n, time] float64."""
    idx = np.unique(np.linspace(0, lfp.shape[0] - 2, n).round().astype(int))
    return np.stack([np.asarray(lfp[i + 1], dtype=np.float64) - np.asarray(lfp[i], dtype=np.float64) for i in idx]), idx


def segment_fft(x, nseg=NSEG):
    """FFT of non-overlapping Hann-windowed segments: [n_ch, n_seg, n_freq] complex64, and the frequencies."""
    n = x.shape[1] // nseg
    seg = x[:, :n * nseg].reshape(x.shape[0], n, nseg)
    seg = seg - seg.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(seg * np.hanning(nseg), axis=-1).astype(np.complex64)
    return X, np.fft.rfftfreq(nseg, 1 / FS_LFP)


# ------------------------------------------------------------------ measures
def pair_spectra(Xa, Xb):
    """Coherence, wPLI and coherency between every channel of Xa and every channel of Xb; arrays [n_pairs, n_f]."""
    coh, wpli, cohy = [], [], []
    pa, pb = (np.abs(Xa) ** 2).mean(axis=1), (np.abs(Xb) ** 2).mean(axis=1)
    for i in range(Xa.shape[0]):
        for j in range(Xb.shape[0]):
            s = Xa[i] * np.conj(Xb[j])                    # [n_seg, n_f]
            c = s.mean(axis=0) / np.sqrt(pa[i] * pb[j])
            im = s.imag
            cohy.append(c)
            coh.append(np.abs(c) ** 2)
            wpli.append(np.abs(im.mean(axis=0)) / np.maximum(np.abs(im).mean(axis=0), 1e-30))
    return np.array(coh), np.array(wpli), np.array(cohy)


def psi(cohy, f, band=BAND):
    """Phase slope index from coherency [.., n_f]; > 0 means the first signal leads the second."""
    k = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    return np.imag(np.sum(np.conj(cohy[..., k[:-1]]) * cohy[..., k[1:]], axis=-1))


def granger(x, y, lags):
    """Granger causality x → y (natural log ratio of residual variances)."""
    n = len(y)
    Y = y[lags:]
    past_y = np.column_stack([y[lags - k:n - k] for k in range(1, lags + 1)])
    past_x = np.column_stack([x[lags - k:n - k] for k in range(1, lags + 1)])
    ones = np.ones((len(Y), 1))
    r = np.hstack([ones, past_y])
    full = np.hstack([r, past_x])
    res_r = Y - r @ np.linalg.lstsq(r, Y, rcond=None)[0]
    res_f = Y - full @ np.linalg.lstsq(full, Y, rcond=None)[0]
    return float(np.log(res_r.var() / res_f.var()))


def band_mean(v, f, band=BAND):
    keep = (f >= band[0]) & (f <= band[1])
    return v[..., keep].mean(axis=-1)


# ------------------------------------------------------------------ per recording
def lfp_connectivity(rec):
    sig, X, chans = {}, {}, {}
    for a in AREAS:
        sig[a], chans[a] = bipolar(rec.lfp[a])
        X[a], f = segment_fft(sig[a])
    rows, spec = [], {}
    for a, b in AREA_PAIRS:
        coh, wpli, cohy = pair_spectra(X[a], X[b])
        spec[f"m{rec.mouse}_{a}{b}_coh"] = coh.mean(axis=0)
        spec[f"m{rec.mouse}_{a}{b}_wpli"] = wpli.mean(axis=0)
        p = psi(cohy, f)
        k = 0
        for i in range(len(chans[a])):
            for j in range(len(chans[b])):
                xa, xb = sig[a][i, ::GC_DECIM], sig[b][j, ::GC_DECIM]
                xa, xb = (xa - xa.mean()) / xa.std(), (xb - xb.mean()) / xb.std()
                gc_ab = granger(xa, xb, GC_LAGS) if (i % 2 == 0 and j % 2 == 0) else np.nan   # GC on a subset (slow)
                gc_ba = granger(xb, xa, GC_LAGS) if (i % 2 == 0 and j % 2 == 0) else np.nan
                rows.append((rec.mouse, a, b, int(chans[a][i]), int(chans[b][j]), band_mean(coh[k], f),
                             band_mean(wpli[k], f), p[k], gc_ab, gc_ba))
                k += 1
    df = pd.DataFrame(rows, columns=["mouse", "area_a", "area_b", "ch_a", "ch_b", "coh", "wpli", "psi_ab",
                                     "gc_ab", "gc_ba"])
    return df, spec, f


def spike_gc(rec):
    """Granger causality between population spike rates (all units of an area), both directions per pair."""
    edges = np.arange(0, rec.duration_s + SPK_BIN, SPK_BIN)
    pop = {}
    for a in AREAS:
        cids = rec.units.cluster_id[rec.units.area == a].to_numpy()
        r = np.histogram(rec.spike_times_all[np.isin(rec.spike_clusters, cids)], edges)[0].astype(float)
        pop[a] = (r - r.mean()) / r.std()
    rows = []
    for a, b in itertools.permutations(AREAS, 2):
        rows.append((rec.mouse, a, b, granger(pop[a], pop[b], SPK_LAGS)))
    return pd.DataFrame(rows, columns=["mouse", "source", "target", "gc"])


def compute(force=False):
    if LFP_CSV.exists() and SPECTRA_NPZ.exists() and SPIKE_CSV.exists() and not force:
        return pd.read_csv(LFP_CSV), dict(np.load(SPECTRA_NPZ)), pd.read_csv(SPIKE_CSV)
    lfp_rows, spec_all, spk_rows = [], {}, []
    for rec in iter_recordings():
        df, spec, f = lfp_connectivity(rec)
        lfp_rows.append(df); spec_all.update(spec)
        spk_rows.append(spike_gc(rec))
        print(f"mouse {rec.mouse} done", flush=True)
    spec_all["freqs"] = f
    lfp, spk = pd.concat(lfp_rows, ignore_index=True), pd.concat(spk_rows, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    lfp.to_csv(LFP_CSV, index=False)
    spk.to_csv(SPIKE_CSV, index=False)
    np.savez_compressed(SPECTRA_NPZ, **{k: np.asarray(v, dtype=np.float32) for k, v in spec_all.items()})
    return lfp, spec_all, spk


# ------------------------------------------------------------------ spike-based undirected check (Q5)
CROSS_STTC_CSV = RESULTS_DIR / "q0506_cross_sttc.csv"


def cross_area_sttc(rec):
    """STTC (dt = 100 ms, as in Q4) for every pair of units in two different areas."""
    from .q04_pairwise import DT, sttc, tiled_fraction
    t_end = rec.duration_s
    inside = (rec.spike_times_all >= 0) & (rec.spike_times_all < t_end)
    st, cl = rec.spike_times_all[inside], rec.spike_clusters[inside]
    trains, tiles = {}, {}
    for a in AREAS:
        cids = rec.units.cluster_id[rec.units.area == a].to_numpy()
        trains[a] = [np.sort(st[cl == c]) for c in cids]
        tiles[a] = [tiled_fraction(s, DT, t_end) for s in trains[a]]
    rows = []
    for a, b in AREA_PAIRS:
        for i, sa in enumerate(trains[a]):
            for j, sb in enumerate(trains[b]):
                rows.append((rec.mouse, a, b, sttc(sa, sb, DT, t_end, tiles[a][i], tiles[b][j])))
    return pd.DataFrame(rows, columns=["mouse", "area_a", "area_b", "sttc"])


def compute_cross_sttc(force=False):
    if CROSS_STTC_CSV.exists() and not force:
        return pd.read_csv(CROSS_STTC_CSV)
    df = pd.concat([cross_area_sttc(rec) for rec in iter_recordings()], ignore_index=True)
    df.to_csv(CROSS_STTC_CSV, index=False)
    return df


# ------------------------------------------------------------------ direction per frequency band (Q6)
PSI_BANDS_CSV = RESULTS_DIR / "q0506_psi_bands.csv"
BANDS = {"theta 4-10 Hz": (4.0, 10.0), "beta 12-30 Hz": (12.0, 30.0), "gamma 30-48 Hz": (30.0, 48.0),
         "1-100 Hz": BAND}
NARROW_PEAKS = [(49.0, 51.0), (52.0, 53.5), (59.0, 61.0), (66.0, 69.0)]   # mains & equipment lines (see Q2)


def psi_bands(rec):
    """PSI per channel pair in each band of BANDS (no Granger, so this is fast)."""
    X, chans = {}, {}
    for a in AREAS:
        s, chans[a] = bipolar(rec.lfp[a])
        X[a], f = segment_fft(s)
    rows = []
    for a, b in AREA_PAIRS:
        _, wpli, cohy = pair_spectra(X[a], X[b])
        clean = np.ones_like(f, dtype=bool)
        for lo, hi in NARROW_PEAKS:
            clean &= ~((f >= lo) & (f <= hi))
        wpli_clean = wpli[:, clean & (f >= BAND[0]) & (f <= BAND[1])].mean(axis=1)
        out = {name: psi(cohy, f, band) for name, band in BANDS.items()}
        wpli_theta = band_mean(wpli, f, BANDS["theta 4-10 Hz"])
        for k in range(cohy.shape[0]):
            rows.append({"mouse": rec.mouse, "area_a": a, "area_b": b, "pair": k,
                         "wpli_no_lines": wpli_clean[k], "wpli_theta": wpli_theta[k],
                         **{f"psi {n}": out[n][k] for n in BANDS}})
    return pd.DataFrame(rows)


def compute_psi_bands(force=False):
    if PSI_BANDS_CSV.exists() and not force:
        return pd.read_csv(PSI_BANDS_CSV)
    df = pd.concat([psi_bands(rec) for rec in iter_recordings()], ignore_index=True)
    df.to_csv(PSI_BANDS_CSV, index=False)
    return df
