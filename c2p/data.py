"""Loading one con2phys recording (one mouse) into a single object."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "py-files"          # py-files/<mouse>/ holds one recording
RESULTS_DIR = ROOT / "results"

FS_WAVEFORM = 30_000   # Hz, waveforms.npy (Data Structure page)
FS_LFP = 500           # Hz, lfp_k.npy (Data Structure page)
AREAS = (1, 2, 3)
TRIAL_EVENTS = ("trial_start", "stim_start", "outcome", "trial_end")


def mice():
    """Mouse ids available on disk, sorted numerically."""
    return sorted(int(p.name) for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())


@dataclass
class Recording:
    mouse: int
    spike_times_all: np.ndarray      # [n_spikes] time of every spike (s)
    spike_clusters: np.ndarray       # [n_spikes] cluster_id that fired each spike
    cluster_ids: np.ndarray          # [n_units]
    areas: np.ndarray                # [n_units] brain area (1-3) of each unit
    waveforms: np.ndarray            # [n_units, 128] average spike shape, same order as cluster_ids
    lfp: dict                        # {area: [n_channels, n_samples]} memory-mapped, deepest channel first
    trials: pd.DataFrame
    units: pd.DataFrame = field(init=False)

    @classmethod
    def load(cls, mouse):
        folder = DATA_DIR / str(mouse)
        ba = np.load(folder / "brain_area.npy", allow_pickle=True).item()  # a dict, not an array
        return cls(
            mouse=int(mouse),
            spike_times_all=np.load(folder / "spikes.npy"),
            spike_clusters=np.load(folder / "clusters.npy"),
            cluster_ids=ba["cluster_id"],
            areas=ba["brain_area"],
            waveforms=np.load(folder / "waveforms.npy"),
            lfp={a: np.load(folder / f"lfp_{a}.npy", mmap_mode="r") for a in AREAS},
            trials=pd.read_csv(folder / "trial_data.csv", index_col=0, encoding="utf-8-sig"),
        )

    def __post_init__(self):
        self._check()
        self._row = {cid: i for i, cid in enumerate(self.cluster_ids)}  # cluster_id -> row
        n_spikes = pd.Series(self.spike_clusters).value_counts()
        self.units = pd.DataFrame({"mouse": self.mouse, "cluster_id": self.cluster_ids, "area": self.areas})
        self.units["n_spikes"] = self.units.cluster_id.map(n_spikes).fillna(0).astype(int)
        self.units["rate_hz"] = self.units.n_spikes / self.duration_s

    def _check(self):
        assert len(self.spike_times_all) == len(self.spike_clusters), "one cluster per spike"
        assert len(self.cluster_ids) == len(self.areas) == len(self.waveforms), "one row per unit"
        assert set(np.unique(self.spike_clusters)) <= set(self.cluster_ids), "unknown cluster in clusters.npy"
        assert len({x.shape[1] for x in self.lfp.values()}) == 1, "LFP areas have different lengths"

    @property
    def duration_s(self):
        """Recording length, from the number of LFP samples."""
        return next(iter(self.lfp.values())).shape[1] / FS_LFP

    def spike_times(self, cid):
        return self.spike_times_all[self.spike_clusters == cid]

    def waveform(self, cid):
        return self.waveforms[self._row[cid]]

    def summary(self):
        per_area = self.units.groupby("area").agg(n_units=("cluster_id", "size"),
                                                  median_rate_hz=("rate_hz", "median"))
        per_area["n_lfp_channels"] = [self.lfp[a].shape[0] for a in per_area.index]
        return per_area


def iter_recordings(mouse_ids=None):
    """Yield one Recording at a time, so only one mouse is in memory."""
    for m in mouse_ids or mice():
        yield Recording.load(m)
