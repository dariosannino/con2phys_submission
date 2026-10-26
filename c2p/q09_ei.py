"""Q9: which brain area has the highest excitation-inhibition (E/I) ratio?

Proxy: the aperiodic exponent of the LFP power spectrum (Gao, Peterson & Voytek 2017, NeuroImage). Inhibitory
synaptic currents are slower than excitatory ones, so more inhibition puts relatively more power at low frequencies
and steepens the spectrum. A FLATTER spectrum (smaller exponent) therefore means a HIGHER E/I ratio.

Exponent = minus the slope of a straight-line fit of log10(PSD) against log10(frequency) over 30-50 Hz, leaving out the
mains/equipment lines (49-51 and 52-53.5 Hz).

  monopolar  per-channel spectra from Q2 (whole recording, mean or median over 2-s windows)
  bipolar    6 bipolar derivations per area (as in Q5/Q6), median over 2-s windows: cancels noise shared across
             channels (reference, movement/EMG transients), which would otherwise flatten the spectrum most in the
             lowest-power area
"""
import numpy as np
import pandas as pd

from .data import AREAS, FS_LFP, RESULTS_DIR, iter_recordings
from .q05_06_connectivity import bipolar

Q2_SPECTRA = RESULTS_DIR / "q02_spectra.npz"
BIPOLAR_CSV = RESULTS_DIR / "q09_bipolar_exponents.csv"
FIT = (30.0, 50.0)
LINES = [(49.0, 51.0), (52.0, 53.5), (59.0, 61.0), (66.0, 69.0)]
NSEG = 2 * FS_LFP


def exponent(f, psd, band=FIT):
    """Aperiodic exponent of each row of psd ([n, n_f]) over band, mains lines excluded."""
    sel = (f >= band[0]) & (f <= band[1])
    for lo, hi in LINES:
        sel &= ~((f >= lo) & (f <= hi))
    x = np.log10(f[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(A, np.log10(np.atleast_2d(psd)[:, sel]).T, rcond=None)[0][0]
    return -slope


def monopolar_table(band=FIT, which="mean"):
    s = np.load(Q2_SPECTRA)
    f = s["freqs"]
    rows = []
    mice = sorted({int(k.split("_")[0][1:]) for k in s.files if k.startswith("m")})
    for m in mice:
        for a in AREAS:
            key = f"m{m}_a{a}" + ("_median" if which == "median" else "")
            for c, e in enumerate(exponent(f, s[key], band)):
                rows.append({"mouse": m, "area": a, "channel": c, "exponent": e})
    return pd.DataFrame(rows)


def bipolar_spectra(rec):
    """Median-over-windows PSD of the bipolar derivations of every area: {area: [n, n_f]}, and the frequencies."""
    out = {}
    for a in AREAS:
        x, _ = bipolar(rec.lfp[a])
        n = x.shape[1] // NSEG
        seg = x[:, :n * NSEG].reshape(x.shape[0], n, NSEG)
        seg = seg - seg.mean(axis=-1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * np.hanning(NSEG), axis=-1)) ** 2
        out[a] = np.median(p, axis=1)
    return out, np.fft.rfftfreq(NSEG, 1 / FS_LFP)


def compute_bipolar(force=False):
    if BIPOLAR_CSV.exists() and not force:
        return pd.read_csv(BIPOLAR_CSV)
    rows = []
    for rec in iter_recordings():
        spec, f = bipolar_spectra(rec)
        for a in AREAS:
            for band in [FIT, (20.0, 40.0), (40.0, 60.0)]:
                for c, e in enumerate(exponent(f, spec[a], band)):
                    rows.append({"mouse": rec.mouse, "area": a, "channel": c, "band": f"{band[0]:.0f}-{band[1]:.0f} Hz",
                                 "exponent": e})
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(BIPOLAR_CSV, index=False)
    return df
