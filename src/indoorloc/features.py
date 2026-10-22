"""Features of a Wi-Fi scan: the RSSI values, three signal statistics and AP-affinity.

Feature sets: "rssi" (the RSSI of every WAP heard in training), "rssi+stats" (plus three
statistics) and "rssi+stats+affinity" (plus the eight AP-affinity columns).
"""

import numpy as np
import pandas as pd

from .config import FEATURE_SETS, FILL_VALUE, FLOORS_PER_BUILDING, MIN_SUPPORT, WEIGHT_EXPONENT
from .data import heard_waps

AFFINITY_COLS = [
    "aff_b0",
    "aff_b1",
    "aff_b2",
    "aff_floor",
    "aff_lon",
    "aff_lat",
    "aff_top_lon",
    "aff_top_lat",
]


def signal_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Per scan: number of WAPs heard, strongest reading and mean of the heard readings.

    A scan in which nothing is heard gets 0 / -110 / -110.
    """
    values = X.to_numpy(dtype=float)
    heard = values > FILL_VALUE
    n_heard = heard.sum(axis=1)
    total = np.where(heard, values, 0.0).sum(axis=1)
    mean = np.where(n_heard > 0, total / np.maximum(n_heard, 1), FILL_VALUE)
    stats = {"n_heard": n_heard, "max_rssi": values.max(axis=1), "mean_rssi": mean}
    return pd.DataFrame(stats, index=X.index)


class APAffinity:
    """AP-affinity: learn where each WAP "lives", then describe a scan by the WAPs it hears.

    fit() gives every WAP heard in at least 5 training scans a profile: its building shares,
    its mean floor and its location, each a weighted mean over the training scans that hear it.
    transform() averages the profiles of the WAPs a scan hears, with the same weights, and adds
    the location of the strongest of them. Scans that hear no known WAP get the training means.
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "APAffinity":
        values = X.to_numpy(dtype=float)
        # Why 1 dB below the weakest reading: that reading then still gets a small weight, not 0
        self.min_rssi_ = values[values > FILL_VALUE].min() - 1
        known = (values > FILL_VALUE).sum(axis=0) >= MIN_SUPPORT
        self.waps_ = X.columns[known].tolist()
        weights = self._weights(values[:, known])
        # Profile columns: share of the scans in building 0, 1 and 2, mean floor, then the
        # location (LONGITUDE and LATITUDE, columns 4 and 5)
        in_building = [y["BUILDINGID"] == b for b in FLOORS_PER_BUILDING]
        targets = np.column_stack([*in_building, y["FLOOR"], y["LONGITUDE"], y["LATITUDE"]])
        targets = targets.astype(float)
        self.profiles_ = (weights.T @ targets) / weights.sum(axis=0)[:, None]  # one row per WAP
        self.fallback_ = targets.mean(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        weights = self._weights(X[self.waps_].to_numpy(dtype=float))
        total = weights.sum(axis=1)
        known = total > 0
        # 8 columns: the 6 profile columns, then the location of the strongest known WAP heard
        fallback = np.append(self.fallback_, self.fallback_[4:])  # training means
        out = np.tile(fallback, (len(X), 1))
        if known.any():
            out[known, :6] = weights[known] @ self.profiles_ / total[known, None]
            strongest = weights[known].argmax(axis=1)  # the largest weight = strongest reading
            out[known, 6:] = self.profiles_[strongest, 4:]  # profile columns 4-5 = location
        return pd.DataFrame(out, columns=AFFINITY_COLS, index=X.index)

    def _weights(self, values: np.ndarray) -> np.ndarray:
        # Why: a strong reading means the scan was close to the WAP, so it counts more. The
        # "powed" weight ((rssi - min) / -min) ** e is 0 when not heard; with our min of -105 dBm
        # it is about 0.27 at -40 dBm, 0.05 at -70 dBm and almost 0 at -100 dBm.
        scaled = np.clip(values - self.min_rssi_, 0, None) / -self.min_rssi_
        return np.where(values > FILL_VALUE, scaled**WEIGHT_EXPONENT, 0.0)


class FeatureBuilder:
    """Turn scans into the feature table of one feature set.

    fit() learns everything from the training scans only: which WAPs were heard and, for the
    affinity block, the WAP profiles. transform() applies the same to any scans (e.g. the test
    set), so held-out scans never influence their own features.
    """

    def __init__(self, feature_set: str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}; use one of {FEATURE_SETS}")
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "FeatureBuilder":
        self.waps_ = heard_waps(X)
        if "affinity" in self.feature_set:
            self.affinity_ = APAffinity().fit(X[self.waps_], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rssi = X[self.waps_]
        blocks = [rssi]
        if "stats" in self.feature_set:
            blocks.append(signal_stats(rssi))
        if "affinity" in self.feature_set:
            blocks.append(self.affinity_.transform(rssi))
        return pd.concat(blocks, axis=1)
