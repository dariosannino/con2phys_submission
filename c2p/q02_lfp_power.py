"""Q2: which brain area has the highest broadband LFP power (1-100 Hz) over the entire recording?

For every channel: power spectral density (PSD) over the whole recording, from 2-s Hann windows with 50% overlap
(Welch's method, 0.5 Hz resolution). Broadband power = integral of the PSD from 1 to 100 Hz, i.e. the variance
of the signal in that band, in µV² (the files appear to be in volts).

The per-window spectra are also averaged with the median instead of the mean. The median ignores rare
large artifacts, so comparing both shows whether a few bad stretches drive the result.
"""
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings

CHANNELS_CSV = RESULTS_DIR / "q02_channel_power.csv"
SPECTRA_NPZ = RESULTS_DIR / "q02_spectra.npz"
BAND = (1.0, 100.0)
LINE_NOISE = (49.0, 51.0)      # 50 Hz mains (used for the cached column power_no50_uv2)
# Narrow non-neural peaks found in the spectra: mains at 50 and (more often) 60 Hz, plus smaller ones at 52-53 and 66-69 Hz
NARROW_PEAKS = [(49.0, 51.0), (52.0, 53.5), (59.0, 61.0), (66.0, 69.0)]
NPERSEG = 2 * FS_LFP           # 2-s windows -> 0.5 Hz resolution
CHUNK = 8                      # channels read from disk at once (8 x ~3M samples x 8 bytes ≈ 200 MB)
V2_TO_UV2 = 1e12


def band_power(f, psd, band=BAND, exclude=None):
    """Integral of psd over band (last axis), optionally leaving out one inner band or a list of them."""
    keep = (f >= band[0]) & (f <= band[1])
    if exclude is not None:
        for lo, hi in ([exclude] if np.isscalar(exclude[0]) else exclude):
            keep &= ~((f >= lo) & (f <= hi))
    df = f[1] - f[0]
    return psd[..., keep].sum(axis=-1) * df


def channel_spectra(lfp):
    """Mean and median PSD over the whole recording for every channel of one area: two [n_ch, n_f] arrays."""
    means, medians = [], []
    for c0 in range(0, lfp.shape[0], CHUNK):
        x = np.asarray(lfp[c0:c0 + CHUNK], dtype=np.float64)
        f, _, s = spectrogram(x, fs=FS_LFP, window="hann", nperseg=NPERSEG, noverlap=NPERSEG // 2,
                              scaling="density", mode="psd")
        means.append(s.mean(axis=-1))
        medians.append(np.median(s, axis=-1) / np.log(2))   # median of a chi2(2) PSD estimate is biased by ln 2
    return f, np.vstack(means), np.vstack(medians)


def compute(force=False):
    if CHANNELS_CSV.exists() and SPECTRA_NPZ.exists() and not force:
        return pd.read_csv(CHANNELS_CSV), dict(np.load(SPECTRA_NPZ))
    rows, spectra = [], {}
    for rec in iter_recordings():
        for a in AREAS:
            f, mean_psd, med_psd = channel_spectra(rec.lfp[a])
            mean_psd, med_psd = mean_psd * V2_TO_UV2, med_psd * V2_TO_UV2
            spectra[f"m{rec.mouse}_a{a}"] = mean_psd.astype(np.float32)
            spectra[f"m{rec.mouse}_a{a}_median"] = med_psd.astype(np.float32)
            n_ch = mean_psd.shape[0]
            rows.append(pd.DataFrame({
                "mouse": rec.mouse, "area": a, "channel": np.arange(n_ch), "n_channels": n_ch,
                "power_uv2": band_power(f, mean_psd),
                "power_no50_uv2": band_power(f, mean_psd, exclude=LINE_NOISE),
                "power_median_uv2": band_power(f, med_psd),
            }))
            print(f"mouse {rec.mouse} area {a}: {n_ch} channels", flush=True)
    spectra["freqs"] = f
    df = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(CHANNELS_CSV, index=False)
    np.savez_compressed(SPECTRA_NPZ, **spectra)
    return df, spectra
