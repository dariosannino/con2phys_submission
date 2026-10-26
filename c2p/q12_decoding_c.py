"""Q12: during which trial segment is variable C best decoded?

From the trial table: A = 1 exactly when C "matches" B in every trial of every mouse (C = 2 with B = 1, C = 0 with
B = 0), so B is the stimulus, C the choice (0/2; the rare C = 1 has A = 0 and is treated as no choice and excluded) and A
whether the choice was correct.

Segments: S1 trial_start -> stim_start, S2 stim_start -> outcome, S3 outcome -> trial_end.
Two feature sets per segment, decoded separately:
  whole       firing rate of each unit over the whole segment (the literal question). Longer segments average more
              spikes, which favours them.
  equal 0.2s  spike counts in a 0.2-s window at the start and one at the end of the segment (2 features per unit),
              the same amount of data for every segment.
Decoder: logistic regression with cross-validation and ROC AUC, as in Q11 (c2p.q11_decoding.cv_auc), on all units of
the mouse; the three areas separately as a check. Chance: shuffled labels.
"""
import numpy as np
import pandas as pd

from .data import AREAS, RESULTS_DIR, iter_recordings
from .q11_decoding import EVENTS, cv_auc

SEG = {"S1 trial start→stim": ("trial_start", "stim_start"),
       "S2 stim→outcome": ("stim_start", "outcome"),
       "S3 outcome→trial end": ("outcome", "trial_end")}
W = 0.2
OUT_CSV = RESULTS_DIR / "q12_segments.csv"
N_SHUFFLES = 5


def counts_between(st, a, b):
    return np.searchsorted(st, b) - np.searchsorted(st, a)


def features(rec, trials):
    """{segment: {"whole": [n_trials, n_units], "equal 0.2s": [n_trials, n_units, 2]}}."""
    trains = [np.sort(rec.spike_times(c)) for c in rec.cluster_ids]
    out = {}
    for name, (s0, s1) in SEG.items():
        a, b = trials[s0].to_numpy(), trials[s1].to_numpy()
        dur = np.maximum(b - a, 1e-3)
        whole = np.column_stack([counts_between(st, a, b) / dur for st in trains])
        start = np.column_stack([counts_between(st, a, a + W) for st in trains]).astype(float)
        end = np.column_stack([counts_between(st, b - W, b) for st in trains]).astype(float)
        short = dur < W                                   # windows would leave the segment
        start[short], end[short] = np.nan, np.nan
        out[name] = {"whole": whole, "equal 0.2s": np.stack([start, end], axis=-1)}
    return out


def compute(force=False, seed=0):
    if OUT_CSV.exists() and not force:
        return pd.read_csv(OUT_CSV)
    rng = np.random.default_rng(seed)
    rows = []
    for rec in iter_recordings():
        tr = rec.trials
        keep = tr[EVENTS].notna().all(axis=1) & (tr.variable_C != 1)
        trials = tr[keep]
        y = (trials.variable_C == 2).astype(int).to_numpy()
        F = features(rec, trials)
        areas = np.asarray(rec.areas)
        for seg, sets in F.items():
            for kind, X in sets.items():
                for pop in ["all"] + list(AREAS):
                    sel = np.ones(len(areas), bool) if pop == "all" else areas == pop
                    Xp = X[:, sel].reshape(len(X), -1)
                    row = {"mouse": rec.mouse, "segment": seg, "features": kind, "population": str(pop),
                           "n_units": int(sel.sum()), "n_trials": len(y), "auc": cv_auc(Xp, y)}
                    if pop == "all":
                        row["auc_shuffled"] = np.nanmean([cv_auc(Xp, rng.permutation(y), seed=k) for k in range(N_SHUFFLES)])
                    rows.append(row)
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


WITHIN_B_CSV = RESULTS_DIR / "q12_within_stimulus.csv"


def compute_within_stimulus(force=False):
    """C decoded separately within B = 0 and B = 1 trials (removes stimulus information), all units, both feature sets."""
    if WITHIN_B_CSV.exists() and not force:
        return pd.read_csv(WITHIN_B_CSV)
    rows = []
    for rec in iter_recordings():
        tr = rec.trials
        keep = tr[EVENTS].notna().all(axis=1) & (tr.variable_C != 1)
        trials = tr[keep]
        F = features(rec, trials)
        for b in (0, 1):
            sel = (trials.variable_B == b).to_numpy()
            y = (trials.variable_C[sel] == 2).astype(int).to_numpy()
            for seg, sets in F.items():
                for kind, X in sets.items():
                    rows.append({"mouse": rec.mouse, "B": b, "segment": seg, "features": kind,
                                 "n_minority": int(min(y.sum(), len(y) - y.sum())),
                                 "auc": cv_auc(X[sel].reshape(sel.sum(), -1), y)})
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WITHIN_B_CSV, index=False)
    return df


TIME_CSV = RESULTS_DIR / "q12_time.csv"
ALIGN = {"stim_start": (-0.6, 0.6), "outcome": (-0.6, 1.0)}


def compute_time(force=False):
    """C decoded in 200-ms bins aligned to stimulus and outcome (all units; bins outside the trial excluded)."""
    from .q11_decoding import binned_counts
    if TIME_CSV.exists() and not force:
        return pd.read_csv(TIME_CSV)
    rows = []
    for rec in iter_recordings():
        tr = rec.trials
        keep = (tr[EVENTS].notna().all(axis=1) & (tr.variable_C != 1)).to_numpy()
        y = (tr.variable_C[keep] == 2).astype(int).to_numpy()
        for ev, (lo, hi) in ALIGN.items():
            B, centres = binned_counts(rec, ev, lo, hi)
            B = B[keep]
            for k, c in enumerate(centres):
                X = B[:, :, k]
                # which segment does this bin fall in for most trials?
                t_abs = tr[ev].to_numpy()[keep] + c
                seg = np.select([t_abs < tr.stim_start.to_numpy()[keep], t_abs < tr.outcome.to_numpy()[keep]], ["S1", "S2"], "S3")
                rows.append({"mouse": rec.mouse, "event": ev, "t": c, "auc": cv_auc(X, y),
                             "segment_majority": pd.Series(seg).mode()[0]})
        print(f"mouse {rec.mouse} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TIME_CSV, index=False)
    return df
