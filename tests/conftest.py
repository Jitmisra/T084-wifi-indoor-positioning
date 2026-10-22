"""A tiny synthetic campus for the tests: 3 buildings, 13 floors, 520 WAP columns."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # draw figures without a screen

from indoorloc.config import FILL_VALUE, FLOORS_PER_BUILDING, WAP_COLS  # noqa: E402

CORNERS = {0: (-7660.0, 4864930.0), 1: (-7520.0, 4864860.0), 2: (-7400.0, 4864780.0)}
SIZE = (60.0, 40.0)  # every building is 60 m x 40 m


def make_campus(scans_per_floor: int = 15, seed: int = 0) -> pd.DataFrame:
    """Scans in the format of a loaded CSV file, made with a simple signal model.

    Every floor has 8 WAPs. A reading is -40 dBm - 20 log10(distance) - 15 dB per floor between
    scan and WAP - 30 dB for another building, plus noise; below -95 dBm a WAP is not heard.
    Users 1 and 2 share phone 50; every other user has a phone of their own.
    """
    rng = np.random.default_rng(seed)
    floors = np.array([(b, f) for b, n in FLOORS_PER_BUILDING.items() for f in range(n)])
    building, floor = np.repeat(floors, scans_per_floor, axis=0).T
    xy = np.array([CORNERS[b] for b in building]) + rng.uniform(0, 1, (len(building), 2)) * SIZE
    ap_building, ap_floor = np.repeat(floors, 8, axis=0).T
    ap_xy = np.array([CORNERS[b] for b in ap_building])
    # Why seed 0 here: every campus has the same WAPs, so another seed gives new scans in the
    # same buildings (like a test set)
    ap_xy += np.random.default_rng(0).uniform(0, 1, ap_xy.shape) * SIZE
    distance = np.linalg.norm(xy[:, None] - ap_xy[None], axis=2)
    rssi = -40 - 20 * np.log10(np.maximum(distance, 1)) + rng.normal(0, 3, distance.shape)
    rssi -= 15 * np.abs(floor[:, None] - ap_floor) + 30 * (building[:, None] != ap_building)
    rssi = np.clip(np.rint(rssi), -100, -35)
    values = np.full((len(building), len(WAP_COLS)), FILL_VALUE)
    values[:, : len(ap_building)] = np.where(rssi >= -95, rssi, FILL_VALUE)
    scans = pd.DataFrame(values, columns=WAP_COLS)
    users = rng.integers(1, 9, len(building))
    extra = {"LONGITUDE": xy[:, 0], "LATITUDE": xy[:, 1], "FLOOR": floor, "BUILDINGID": building}
    extra |= {"USERID": users, "PHONEID": np.where(users <= 2, 50, 100 + users)}
    extra["TIMESTAMP"] = 1370000000 + rng.integers(0, 10**6, len(building))
    return pd.concat([scans, pd.DataFrame(extra)], axis=1)


@pytest.fixture(scope="session")
def campus() -> pd.DataFrame:
    """195 training scans (15 per floor). Do not change it inside a test."""
    return make_campus()
