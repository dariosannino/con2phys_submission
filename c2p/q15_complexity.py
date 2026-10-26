"""Q15: in which brain area is the neural signal most complex (segments from stimulus presentation to trial end)?

Window: the first WIN = 1.0 s after stim_start of every trial (always inside stim_start -> trial_end; the same length
for every trial, since LZ complexity depends on length and stim -> trial end durations vary a lot).

LFP, per window and channel (6 channels per area spread over depth; monopolar main, bipolar check):
  lzc       Lempel-Ziv complexity (LZ76, Kaspar & Schuster 1987) of the signal binarised at its median,
            normalised by n / log2(n): ~0 for a regular signal, ~1 for white noise
  lzc_surr  the same for a phase-randomised surrogate (identical power spectrum, random phases)
  lzc_ratio lzc / lzc_surr: complexity beyond what the power spectrum alone implies
  pe        permutation entropy (order 4, delay 1 sample), normalised to [0, 1]
Spikes, per window: LZc of the binary population raster (N_UNITS units, 5-ms bins, read unit by unit), and the same
after shuffling each unit's bins in time (keeps rates); ratio = complexity beyond the firing rates.
"""
import math
from itertools import permutations

import numpy as np
import pandas as pd

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings
from .q05_06_connectivity import bipolar
from .q11_decoding import EVENTS

LFP_CSV = RESULTS_DIR / "q15_lfp.csv"
SPK_CSV = RESULTS_DIR / "q15_spikes.csv"
WIN = 1.0
N_CH = 6
PE_ORDER = 4
SPK_BIN, N_UNITS, N_DRAWS = 0.005, 5, 3
MAX_WINDOWS = 200          # random trials per mouse (LZ is slow in pure Python; 200 windows average well)


def lz76(s):
    """Number of LZ76 phrases of a binary sequence (Kaspar & Schuster algorithm)."""
    s = np.asarray(s, dtype=np.int8)
    n = len(s)
    if n < 2:
        return n
    c, l, i, k, k_max = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def lzc(x):
    b = (x > np.median(x)).astype(np.int8)
    n = len(b)
    return lz76(b) * math.log2(n) / n


_PATTERNS = {p: i for i, p in enumerate(permutations(range(PE_ORDER)))}


def perm_entropy(x, order=PE_ORDER):
    idx = np.lib.stride_tricks.sliding_window_view(x, order).argsort(axis=1)
    keys = idx @ (order ** np.arange(order)[::-1])
    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))


def phase_surrogate(x, rng):
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def windows(rec):
    tr = rec.trials[rec.trials[EVENTS].notna().all(axis=1)]
    ok = (tr.trial_end - tr.stim_start) >= WIN
    st = tr.stim_start[ok].to_numpy()
    rng = np.random.default_rng(rec.mouse)
    return np.sort(rng.choice(st, size=min(MAX_WINDOWS, len(st)), replace=False))


def lfp_rows(rec, rng):
    starts = (windows(rec) * FS_LFP).round().astype(int)
    n = int(WIN * FS_LFP)
    rows = []
    for a in AREAS:
        lfp = rec.lfp[a]
        chans = np.unique(np.linspace(0, lfp.shape[0] - 1, N_CH).round().astype(int))
        sigs = {"monopolar": np.stack([np.asarray(lfp[c], dtype=np.float64) for c in chans]),
                "bipolar": bipolar(lfp, N_CH)[0]}
        for ref, S in sigs.items():
            for c in range(S.shape[0]):
                v, vs, pe = [], [], []
                for s0 in starts:
                    x = S[c, s0:s0 + n]
                    if len(x) < n:
                        continue
                    v.append(lzc(x)); vs.append(lzc(phase_surrogate(x, rng))); pe.append(perm_entropy(x))
                rows.append({"mouse": rec.mouse, "area": a, "reference": ref, "channel": c, "n_windows": len(v),
                             "lzc": np.mean(v), "lzc_surr": np.mean(vs), "lzc_ratio": np.mean(v) / np.mean(vs), "pe": np.mean(pe)})
    return rows


def spike_rows(rec, rng):
    starts = windows(rec)
    nb = int(round(WIN / SPK_BIN))
    rows = []
    trains = {c: np.sort(rec.spike_times(c)) for c in rec.cluster_ids}
    for a in AREAS:
        cids = rec.cluster_ids[np.asarray(rec.areas) == a]
        if len(cids) < N_UNITS:
            continue
        edges = starts[:, None] + np.arange(nb + 1)[None, :] * SPK_BIN
        raster = {c: (np.diff(np.searchsorted(trains[c], edges), axis=1) > 0).astype(np.int8) for c in cids}
        real, shuf = [], []
        for _ in range(N_DRAWS):
            pick = rng.choice(cids, N_UNITS, replace=False)
            for t in range(len(starts)):
                M = np.stack([raster[c][t] for c in pick])          # units x bins
                if M.sum() == 0:
                    continue
                n = M.size
                real.append(lz76(M.ravel()) * math.log2(n) / n)
                Ms = np.stack([rng.permutation(r) for r in M])
                shuf.append(lz76(Ms.ravel()) * math.log2(n) / n)
        rows.append({"mouse": rec.mouse, "area": a, "lzc_spikes": np.mean(real), "lzc_spikes_shuffled": np.mean(shuf),
                     "lzc_spikes_ratio": np.mean(real) / np.mean(shuf)})
    return rows


def compute(force=False, seed=0):
    if LFP_CSV.exists() and SPK_CSV.exists() and not force:
        return pd.read_csv(LFP_CSV), pd.read_csv(SPK_CSV)
    rng = np.random.default_rng(seed)
    lfp, spk = [], []
    for rec in iter_recordings():
        lfp += lfp_rows(rec, rng)
        spk += spike_rows(rec, rng)
        print(f"mouse {rec.mouse} done", flush=True)
    lfp, spk = pd.DataFrame(lfp), pd.DataFrame(spk)
    RESULTS_DIR.mkdir(exist_ok=True)
    lfp.to_csv(LFP_CSV, index=False)
    spk.to_csv(SPK_CSV, index=False)
    return lfp, spk


# ------------------------------------------------------------------ stricter control: IAAFT surrogates
IAAFT_CSV = RESULTS_DIR / "q15_iaaft.csv"
IAAFT_ITER = 100


def iaaft(x, rng, n_iter=IAAFT_ITER):
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996).

    Keeps both the power spectrum and the distribution of values of x; only the temporal structure is scrambled.
    Alternates (1) imposing the original Fourier amplitudes and (2) imposing the original values by rank order.
    """
    x = np.asarray(x, dtype=np.float64)
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    s = rng.permutation(x)                                   # start from a random shuffle
    for _ in range(n_iter):
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=len(x))   # (1) original spectrum
        s = sorted_x[np.argsort(np.argsort(s))]                      # (2) original values, surrogate's ranks
    return s


def compute_iaaft(force=False, seed=1):
    """LZc of one IAAFT surrogate per window, same windows and channels as compute()."""
    if IAAFT_CSV.exists() and not force:
        return pd.read_csv(IAAFT_CSV)
    rng = np.random.default_rng(seed)
    rows = []
    for rec in iter_recordings():
        starts = (windows(rec) * FS_LFP).round().astype(int)
        n = int(WIN * FS_LFP)
        for a in AREAS:
            lfp = rec.lfp[a]
            chans = np.unique(np.linspace(0, lfp.shape[0] - 1, N_CH).round().astype(int))
            sigs = {"monopolar": np.stack([np.asarray(lfp[c], dtype=np.float64) for c in chans]),
                    "bipolar": bipolar(lfp, N_CH)[0]}
            for ref, S in sigs.items():
                for c in range(S.shape[0]):
                    v = [lzc(iaaft(S[c, s0:s0 + n], rng)) for s0 in starts if s0 + n <= S.shape[1]]
                    rows.append({"mouse": rec.mouse, "area": a, "reference": ref, "channel": c, "lzc_iaaft": np.mean(v)})
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(IAAFT_CSV, index=False)
    return df
