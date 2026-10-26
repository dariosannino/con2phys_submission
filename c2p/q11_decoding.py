"""Q11 (and the base for Q12): decoding trial variables from population spike counts of each area.

Features: spike counts of every unit in FIXED-LENGTH windows aligned to task events (inter-trial intervals never used):
    W1 pre-stimulus    [stim_start - 0.4 s, stim_start)
    W2 post-stimulus   [stim_start, stim_start + 0.2 s)
    W3 post-outcome    [outcome, outcome + 1.0 s)
Fixed lengths matter: trials with variable_A = 0 have a 1-s longer outcome->trial_end period (a timeout) and slightly
longer reaction times, so rates averaged over whole segments would differ between A = 0 and A = 1 just because they
average over different stretches of time. A window is used only if it lies inside its segment; the "all windows"
decoder uses the trials where all three fit. Trials with missing event times are dropped.

Decoder: L2-regularised logistic regression on z-scored features, stratified 5-fold cross-validation, scored by ROC AUC
(0.5 = chance; insensitive to the class imbalance of variable A). To compare areas fairly, every decoder uses the SAME
number of units (N_UNITS, randomly drawn, N_DRAWS times) in each area; full populations are a robustness check.
Chance: the same pipeline with shuffled labels.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import AREAS, RESULTS_DIR, iter_recordings

# window name: (event, start offset, end offset, segment that must contain it)
SEGMENTS = {"W1 pre-stim (0.4 s)": ("stim_start", -0.4, 0.0, ("trial_start", "stim_start")),
            "W2 post-stim (0.2 s)": ("stim_start", 0.0, 0.2, ("stim_start", "outcome")),
            "W3 post-outcome (1 s)": ("outcome", 0.0, 1.0, ("outcome", "trial_end"))}
EVENTS = ["trial_start", "stim_start", "outcome", "trial_end"]
N_UNITS, N_DRAWS, N_FOLDS, N_SHUFFLES = 5, 20, 5, 5
TIME_BIN = 0.2
ALIGN = {"stim_start": (-0.6, 0.6), "outcome": (-0.4, 1.0)}


def segment_rates(rec):
    """[n_trials, n_units, 3] spike counts in the fixed windows (NaN where a window does not fit), and the unit areas."""
    tr = rec.trials
    out = np.zeros((len(tr), len(rec.cluster_ids), len(SEGMENTS)))
    fits = []
    for k, (ev, lo, hi, (s0, s1)) in enumerate(SEGMENTS.values()):
        a, b = tr[ev].to_numpy() + lo, tr[ev].to_numpy() + hi
        fits.append((a >= tr[s0].to_numpy() - 1e-9) & (b <= tr[s1].to_numpy() + 1e-9))
        for j, cid in enumerate(rec.cluster_ids):
            st = np.sort(rec.spike_times(cid))
            out[:, j, k] = np.searchsorted(st, b) - np.searchsorted(st, a)
    fits = np.array(fits).T                                            # [n_trials, 3]
    out[~np.broadcast_to(fits[:, None, :], out.shape)] = np.nan
    return out, np.asarray(rec.areas)


def valid_trials(tr):
    return tr[EVENTS].notna().all(axis=1).to_numpy()


def binned_counts(rec, event, lo, hi, width=TIME_BIN):
    """[n_trials, n_units, n_bins] spike counts in bins aligned to event; bins outside the trial are NaN."""
    tr = rec.trials
    edges = np.arange(lo, hi + 1e-9, width)
    t0 = tr[event].to_numpy()[:, None] + edges[None, :]
    out = np.zeros((len(tr), len(rec.cluster_ids), len(edges) - 1))
    for j, cid in enumerate(rec.cluster_ids):
        st = np.sort(rec.spike_times(cid))
        out[:, j, :] = np.diff(np.searchsorted(st, t0), axis=1)
    inside = (t0[:, :-1] >= tr.trial_start.to_numpy()[:, None]) & (t0[:, 1:] <= tr.trial_end.to_numpy()[:, None])
    out[~np.broadcast_to(inside[:, None, :], out.shape)] = np.nan
    return out, (edges[:-1] + edges[1:]) / 2


def cv_auc(X, y, seed=0):
    """Cross-validated ROC AUC of a logistic regression (one-vs-rest macro average if y has >2 classes)."""
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < N_FOLDS:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    if len(classes) == 2:
        score = np.zeros(len(y))
        for tr_i, te_i in skf.split(X, y):
            score[te_i] = clf.fit(X[tr_i], y[tr_i]).decision_function(X[te_i])
        return roc_auc_score(y, score)
    prob = np.zeros((len(y), len(classes)))
    for tr_i, te_i in skf.split(X, y):
        prob[te_i] = clf.fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])
    return roc_auc_score(y, prob, multi_class="ovr", average="macro")


def decode_area(F, y, n_units, rng, n_draws=N_DRAWS, shuffle=False):
    """Mean AUC over random draws of n_units units. F: [n_trials, n_units_area, n_features_per_unit]."""
    aucs = []
    for d in range(n_draws if F.shape[1] > n_units else 1):
        pick = rng.choice(F.shape[1], size=min(n_units, F.shape[1]), replace=False)
        X = F[:, pick, :].reshape(len(F), -1)
        yy = rng.permutation(y) if shuffle else y
        aucs.append(cv_auc(X, yy, seed=d))
    return float(np.nanmean(aucs))


def run(variable, keep=None, seed=0):
    """Decode `variable` per mouse and area: segment-wise, all segments together, full populations, time-resolved, chance."""
    rng = np.random.default_rng(seed)
    seg_rows, time_rows = [], []
    for rec in iter_recordings():
        tr = rec.trials
        mask = valid_trials(tr) & (np.ones(len(tr), bool) if keep is None else keep(tr))
        y = tr[variable].to_numpy()[mask]
        R, areas = segment_rates(rec)
        R = R[mask]
        binned = {ev: binned_counts(rec, ev, *lim) for ev, lim in ALIGN.items()}
        for a in AREAS:
            Fa = R[:, areas == a, :]
            row = {"mouse": rec.mouse, "area": a, "n_units": int(Fa.shape[1]), "n_trials": int(mask.sum())}
            for k, name in enumerate(SEGMENTS):
                row[name] = decode_area(Fa[:, :, [k]], y, N_UNITS, rng)
            row["all segments"] = decode_area(Fa, y, N_UNITS, rng)
            row["all segments, full population"] = decode_area(Fa, y, Fa.shape[1], rng)
            row["all segments, shuffled"] = np.mean([decode_area(Fa, y, N_UNITS, rng, n_draws=4, shuffle=True)
                                                     for _ in range(N_SHUFFLES)])
            seg_rows.append(row)
            for ev, (B, centres) in binned.items():
                Ba = B[mask][:, areas == a, :]
                for b, c in enumerate(centres):
                    time_rows.append({"mouse": rec.mouse, "area": a, "event": ev, "t": c,
                                      "auc": decode_area(Ba[:, :, [b]], y, N_UNITS, rng, n_draws=8)})
        print(f"mouse {rec.mouse} done", flush=True)
    return pd.DataFrame(seg_rows), pd.DataFrame(time_rows)


def compute(variable="variable_A", tag="q11", keep=None, force=False):
    seg_csv, time_csv = RESULTS_DIR / f"{tag}_segments.csv", RESULTS_DIR / f"{tag}_time.csv"
    if seg_csv.exists() and time_csv.exists() and not force:
        return pd.read_csv(seg_csv), pd.read_csv(time_csv)
    seg, tim = run(variable, keep=keep)
    RESULTS_DIR.mkdir(exist_ok=True)
    seg.to_csv(seg_csv, index=False)
    tim.to_csv(time_csv, index=False)
    return seg, tim
