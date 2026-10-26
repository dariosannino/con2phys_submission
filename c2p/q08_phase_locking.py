"""Q8: in which brain area are spikes most strongly phase-locked to its own LFP in the 4-10 Hz range?

LFP reference per area (one channel for all units of the area, since we do not know which channel a unit is on):
  main      the channel with the highest 4-10 Hz power (from the Q2 spectra): the most reliable theta phase
  bipolar   difference between that channel and its neighbour: removes theta carried by the shared reference
  middle    the middle channel of the area (no selection on theta power)
Phase: 4-10 Hz band-pass (3rd-order Butterworth, zero phase) and Hilbert transform; spike phase = phase at the
nearest LFP sample.

Locking strength per unit:
  ppc   pairwise phase consistency (Vinck et al. 2010): the average cosine of the phase difference over all pairs of
        spikes, (|Σ e^{iθ}|² − n) / (n (n − 1)). Unlike the PLV (mean resultant length) it is not inflated by small
        spike counts. 0 = no locking.
  plv_500  PLV on 500 randomly drawn spikes (equal count for every unit), for comparison
  rayleigh_p  Rayleigh test of non-uniformity of the spike phases
  ppc_high_theta  PPC using only spikes during strong theta (4-10 Hz envelope above its median)
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.fft import next_fast_len

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings

UNITS_CSV = RESULTS_DIR / "q08_units.csv"
THETA = (4.0, 10.0)
SOS = butter(3, THETA, btype="band", fs=FS_LFP, output="sos")
N_PLV = 500
Q2_SPECTRA = RESULTS_DIR / "q02_spectra.npz"


def analytic(x):
    b = sosfiltfilt(SOS, np.asarray(x, dtype=np.float64))
    return hilbert(b, N=next_fast_len(len(b)))[:len(b)]


def ppc(theta):
    n = len(theta)
    if n < 2:
        return np.nan
    s = np.abs(np.exp(1j * theta).sum()) ** 2
    return float((s - n) / (n * (n - 1)))


def rayleigh_p(theta):
    n = len(theta)
    r = np.abs(np.exp(1j * theta).mean())
    z = n * r ** 2
    # Zar (1999) approximation
    return float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2)) - (1 + 2 * n)))


def theta_channel(mouse, area, spectra):
    p = spectra[f"m{mouse}_a{area}"]
    f = spectra["freqs"]
    return int(np.argmax(p[:, (f >= THETA[0]) & (f <= THETA[1])].sum(axis=1)))


def units_of(rec, spectra, rng):
    rows = []
    for a in AREAS:
        lfp = rec.lfp[a]
        ch = theta_channel(rec.mouse, a, spectra)
        nb = ch + 1 if ch + 1 < lfp.shape[0] else ch - 1
        mid = lfp.shape[0] // 2
        z_main = analytic(lfp[ch])
        sig = {"main": np.angle(z_main),
               "bipolar": np.angle(analytic(np.asarray(lfp[nb], dtype=np.float64) - np.asarray(lfp[ch], dtype=np.float64))),
               "middle": np.angle(analytic(lfp[mid]))}
        env = np.abs(z_main)
        high = env > np.median(env)
        n_samp = lfp.shape[1]
        for cid in rec.units.cluster_id[rec.units.area == a]:
            st = rec.spike_times(cid)
            idx = np.round(st * FS_LFP).astype(int)
            idx = idx[(idx >= 0) & (idx < n_samp)]
            th = sig["main"][idx]
            sub = rng.choice(th, size=N_PLV, replace=False) if len(th) >= N_PLV else th
            rows.append({"mouse": rec.mouse, "cluster_id": cid, "area": a, "n_spikes": len(idx),
                         "lfp_channel": ch, "ppc": ppc(th), "ppc_bipolar": ppc(sig["bipolar"][idx]),
                         "ppc_middle": ppc(sig["middle"][idx]), "ppc_high_theta": ppc(th[high[idx]]),
                         "plv_500": float(np.abs(np.exp(1j * sub).mean())) if len(th) >= N_PLV else np.nan,
                         "rayleigh_p": rayleigh_p(th), "preferred_phase": float(np.angle(np.exp(1j * th).mean()))})
    return rows


def compute(force=False):
    if UNITS_CSV.exists() and not force:
        return pd.read_csv(UNITS_CSV)
    spectra = np.load(Q2_SPECTRA)
    rng = np.random.default_rng(0)
    rows = []
    for rec in iter_recordings():
        rows += units_of(rec, spectra, rng)
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(UNITS_CSV, index=False)
    return df
