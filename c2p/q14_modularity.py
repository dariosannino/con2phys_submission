"""Q14: in which brain area is modularity the lowest (entire recording)?

Network per area: nodes = units, edge weights = pairwise STTC (dt = 100 ms, whole recording; from Q4); negative weights
are set to 0. Robustness: Pearson correlation of 100-ms spike counts (also from Q4).
Modularity: Newman's Q of the best partition found by the Louvain algorithm (best of N_SEEDS runs).
Null: the same weights randomly permuted among the edges (keeps the weight distribution, destroys structure);
Q - Q_null is the modularity beyond what a random network with the same weights would show.
Modularity depends on network size, so every area uses the SAME number of units (N_UNITS, N_DRAWS random draws).
"""
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

from .data import AREAS, RESULTS_DIR
from .q04_pairwise import PAIRS_CSV

OUT_CSV = RESULTS_DIR / "q14_modularity.csv"
N_UNITS = (10, 15)
N_DRAWS, N_SEEDS, N_NULL = 100, 5, 5


def best_q(W, seeds=N_SEEDS):
    G = nx.from_numpy_array(W)
    if G.size(weight="weight") == 0:
        return np.nan
    return max(modularity(G, louvain_communities(G, weight="weight", seed=s), weight="weight") for s in range(seeds))


def weight_matrix(pairs, units, col):
    idx = {u: i for i, u in enumerate(units)}
    W = np.zeros((len(units), len(units)))
    for a, b, w in zip(pairs.unit_a, pairs.unit_b, pairs[col]):
        if a in idx and b in idx:
            W[idx[a], idx[b]] = W[idx[b], idx[a]] = max(w, 0.0)
    return W


def permuted(W, rng):
    iu = np.triu_indices_from(W, 1)
    w = rng.permutation(W[iu])
    P = np.zeros_like(W)
    P[iu] = w
    return P + P.T


def compute(force=False, seed=0):
    if OUT_CSV.exists() and not force:
        return pd.read_csv(OUT_CSV)
    rng = np.random.default_rng(seed)
    pairs = pd.read_csv(PAIRS_CSV)
    rows = []
    for (m, a), g in pairs.groupby(["mouse", "area"]):
        units = np.unique(np.r_[g.unit_a, g.unit_b])
        for col in ["sttc", "count_corr"]:
            Wfull = weight_matrix(g, units, col)
            for n in N_UNITS:
                if len(units) < n:
                    continue
                q, qn = [], []
                for _ in range(N_DRAWS):
                    pick = rng.choice(len(units), size=n, replace=False)
                    W = Wfull[np.ix_(pick, pick)]
                    q.append(best_q(W))
                    qn.append(np.mean([best_q(permuted(W, rng), seeds=2) for _ in range(N_NULL)]))
                q, qn = np.array(q), np.array(qn)
                rows.append({"mouse": m, "area": a, "weights": col, "n_units": n, "n_available": len(units),
                             "Q": np.nanmean(q), "Q_null": np.nanmean(qn), "Q_minus_null": np.nanmean(q - qn)})
        print(f"mouse {m} area {a} done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


FULL_CSV = RESULTS_DIR / "q14_full_networks.csv"
N_NULL_FULL = 30
DENSITY = 0.2


def binary_top(W, density=DENSITY):
    """Binary network keeping the strongest `density` fraction of links."""
    iu = np.triu_indices_from(W, 1)
    k = max(1, int(round(density * len(iu[0]))))
    thr = np.sort(W[iu])[::-1][k - 1]
    B = (W >= thr).astype(float)
    np.fill_diagonal(B, 0)
    return B


def compute_full(force=False, seed=0):
    """Full networks (all units of the area). Q compared with N_NULL_FULL weight-permuted networks: z = (Q - mean) / sd."""
    if FULL_CSV.exists() and not force:
        return pd.read_csv(FULL_CSV)
    rng = np.random.default_rng(seed)
    pairs = pd.read_csv(PAIRS_CSV)
    rows = []
    for (m, a), g in pairs.groupby(["mouse", "area"]):
        units = np.unique(np.r_[g.unit_a, g.unit_b])
        for col in ["sttc", "count_corr"]:
            W = weight_matrix(g, units, col)
            for net, M in [("weighted", W), ("binary top 20%", binary_top(W))]:
                q = best_q(M)
                null = np.array([best_q(permuted(M, rng), seeds=2) for _ in range(N_NULL_FULL)])
                rows.append({"mouse": m, "area": a, "weights": col, "network": net, "n_units": len(units),
                             "Q": q, "Q_null": null.mean(), "Q_minus_null": q - null.mean(),
                             "z": (q - null.mean()) / null.std() if null.std() > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(FULL_CSV, index=False)
    return df
