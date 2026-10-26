"""Statistics shared across questions.

Units (neurons) are nested in mice: units from the same mouse are not independent.
The hierarchical bootstrap respects this by resampling mice first, then units within each drawn mouse
(separately per area, so every drawn mouse keeps its own number of units per area).
"""
import numpy as np
import pandas as pd


def hierarchical_bootstrap(df, value, stat, group="mouse", strata="area", n_boot=10_000, seed=0):
    """Bootstrap distribution of stat(values, strata_labels) -> 1D array.

    Returns an array [n_boot, len(stat(...))]. Computing several quantities in one `stat`
    (e.g. the mean of each area and their differences) keeps them on the same resamples.
    """
    rng = np.random.default_rng(seed)
    v, s = df[value].to_numpy(), df[strata].to_numpy()
    blocks = [[idx.to_numpy() for _, idx in g.groupby(strata).groups.items()]
              for _, g in df.reset_index(drop=True).groupby(group)]
    out = []
    for _ in range(n_boot):
        picked = rng.integers(len(blocks), size=len(blocks))
        idx = np.concatenate([rng.choice(b, size=len(b)) for i in picked for b in blocks[i]])
        out.append(stat(v[idx], s[idx]))
    return np.asarray(out)


def ci(boot, level=0.95):
    """Percentile confidence interval along the first axis."""
    return np.quantile(boot, [(1 - level) / 2, (1 + level) / 2], axis=0)


def p_two_sided(boot_diff):
    """Bootstrap p-value for a difference being 0 (fraction of resamples on the other side, x2)."""
    boot_diff = np.asarray(boot_diff)
    return min(1.0, 2 * min((boot_diff <= 0).mean(), (boot_diff >= 0).mean()))


# ---------------------------------------------------------------- comparing the three areas
# Every question compares areas 1-3 recorded in the same 18 mice. These helpers give, for any per-row value
# (a unit's rate, a channel's power, ...): the mean per area with CIs, all pairwise differences, a mixed model,
# and simple per-mouse tests. `pairs` lists differences as (x, y) meaning "area x minus area y".

AREAS = (1, 2, 3)
ALL_PAIRS = [(1, 2), (1, 3), (2, 3)]


def area_bootstrap(df, value, pairs=ALL_PAIRS, n_boot=10_000, seed=0):
    """Mean of `value` per area and pairwise differences, with hierarchical-bootstrap CIs and p-values.

    Returns (table, boot). table rows: 'area a' and 'area x − area y'; boot: [n_boot, 3 + len(pairs)].
    """
    def stat(v, s):
        m = {a: v[s == a].mean() for a in AREAS}
        return [m[a] for a in AREAS] + [m[x] - m[y] for x, y in pairs]

    boot = hierarchical_bootstrap(df, value, stat, n_boot=n_boot, seed=seed)
    point = stat(df[value].to_numpy(), df["area"].to_numpy())
    lo, hi = ci(boot)
    names = [f"area {a}" for a in AREAS] + [f"area {x} − area {y}" for x, y in pairs]
    p = [np.nan] * len(AREAS) + [p_two_sided(boot[:, k]) for k in range(len(AREAS), boot.shape[1])]
    table = pd.DataFrame({"estimate": point, "ci_low": lo, "ci_high": hi, "p (diff = 0)": p}, index=names)
    return table, boot


def lmm_pairwise(df, value, pairs=ALL_PAIRS):
    """Linear mixed model value ~ area + (area | mouse); one row per pair with coefficient x − y, CI and p."""
    import warnings
    import statsmodels.formula.api as smf

    fits, rows = {}, []
    for x, y in pairs:
        if y not in fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits[y] = smf.mixedlm(f"{value} ~ C(area, Treatment(reference={y}))", df,
                                      groups=df["mouse"], re_formula="~C(area)").fit()
        fit = fits[y]
        name = next(n for n in fit.params.index if f"[T.{x}]" in n)
        lo, hi = fit.conf_int().loc[name]
        rows.append({"contrast": f"area {x} − area {y}", "coef": fit.params[name], "ci_low": lo, "ci_high": hi,
                     "p": fit.pvalues[name], "converged": fit.converged})
    table = pd.DataFrame(rows).set_index("contrast")
    table["p (Bonferroni)"] = (table["p"] * len(pairs)).clip(upper=1.0)
    return table


def per_mouse_tests(df, value, pairs=ALL_PAIRS, agg="mean"):
    """Per-mouse area values (mouse x area), Friedman test, and paired Wilcoxon tests per pair."""
    from scipy import stats as sps

    per_mouse = df.groupby(["mouse", "area"])[value].agg(agg).unstack()
    friedman = sps.friedmanchisquare(*[per_mouse[a] for a in AREAS])
    rows = [{"pair": f"area {x} vs area {y}", "mice where x > y": int((per_mouse[x] > per_mouse[y]).sum()),
             "n mice": len(per_mouse), "wilcoxon p": sps.wilcoxon(per_mouse[x], per_mouse[y]).pvalue}
            for x, y in pairs]
    table = pd.DataFrame(rows).set_index("pair")
    table["wilcoxon p (Bonferroni)"] = (table["wilcoxon p"] * len(pairs)).clip(upper=1.0)   # corrected for the pairs tested
    return per_mouse, friedman, table
