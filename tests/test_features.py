import numpy as np
import pandas as pd
import pytest

from indoorloc import data
from indoorloc.config import FILL_VALUE, TARGET_COLS, WAP_COLS
from indoorloc.features import AFFINITY_COLS, APAffinity, FeatureBuilder, signal_stats

EXTRA_COLUMNS = [("rssi", 0), ("rssi+stats", 3), ("rssi+stats+affinity", 11)]


def test_signal_stats_by_hand():
    X = pd.DataFrame([[-50, -70, FILL_VALUE], [FILL_VALUE] * 3], columns=WAP_COLS[:3])
    assert signal_stats(X).to_numpy().tolist() == [[2, -50, -60], [0, FILL_VALUE, FILL_VALUE]]


def test_affinity_by_hand():
    # Equal readings give equal weights, so a profile is the plain mean over the scans that
    # hear the WAP. WAP002 is heard in only 3 scans, too few to be trusted.
    X = pd.DataFrame({"WAP001": [-60] * 5 + [FILL_VALUE], "WAP002": [FILL_VALUE] * 3 + [-60] * 3})
    y = pd.DataFrame(
        {
            "BUILDINGID": [0, 0, 1, 1, 1, 2],
            "FLOOR": [0, 1, 2, 3, 4, 0],
            "LONGITUDE": [10, 20, 30, 40, 50, 60],
            "LATITUDE": [5, 5, 5, 5, 5, 11],
        }
    )
    affinity = APAffinity().fit(X, y)
    assert affinity.waps_ == ["WAP001"]
    np.testing.assert_allclose(affinity.profiles_[0], [0.4, 0.6, 0, 2, 30, 5])
    scans = pd.DataFrame({"WAP001": [-50, FILL_VALUE, -70], "WAP002": [-60, -60, -60]})
    out = affinity.transform(scans)
    assert list(out.columns) == AFFINITY_COLS
    np.testing.assert_allclose(out.iloc[0], [0.4, 0.6, 0, 2, 30, 5, 30, 5])
    # A scan that hears no trusted WAP gets the means of all training scans; so does a reading
    # weaker than every training reading (-70 < -60 dBm), which gets weight 0
    for row in (1, 2):
        np.testing.assert_allclose(out.iloc[row], [2 / 6, 3 / 6, 1 / 6, 10 / 6, 35, 6, 35, 6])


def test_stronger_readings_count_more():
    X = pd.DataFrame({"WAP001": [-40, -80, -80, -80, -80]})
    y = pd.DataFrame(
        {"BUILDINGID": [0, 1, 1, 1, 1], "FLOOR": 0, "LONGITUDE": [0, 10, 10, 10, 10], "LATITUDE": 0}
    )
    share_b0, _, _, _, lon, _ = APAffinity().fit(X, y).profiles_[0]
    assert share_b0 > 0.99 and lon < 0.1  # the one strong scan outweighs the four weak ones


def test_affinity_of_a_scan_that_hears_two_waps():
    # WAP001 is heard at -40 dBm at LONGITUDE 0 and WAP002 at -90 dBm at LONGITUDE 100. The
    # weakest reading is -90, so min_rssi_ = -91 and a reading r gets ((r + 91) / 91) ** e.
    X = pd.DataFrame(
        {"WAP001": [-40] * 5 + [FILL_VALUE] * 5, "WAP002": [FILL_VALUE] * 5 + [-90] * 5}
    )
    y = pd.DataFrame({"BUILDINGID": 0, "FLOOR": 0, "LONGITUDE": [0] * 5 + [100] * 5, "LATITUDE": 0})
    out = APAffinity().fit(X, y).transform(pd.DataFrame({"WAP001": [-40], "WAP002": [-90]}))
    strong, weak = (51 / 91) ** np.e, (1 / 91) ** np.e
    np.testing.assert_allclose(out["aff_lon"], 100 * weak / (strong + weak), rtol=1e-9)
    assert out["aff_top_lon"].item() == 0  # the location of the strongest WAP heard


@pytest.mark.parametrize("feature_set, extra", EXTRA_COLUMNS)
def test_feature_sets(campus, feature_set, extra):
    builder = FeatureBuilder(feature_set).fit(campus[WAP_COLS], campus[TARGET_COLS])
    F = builder.transform(campus[WAP_COLS])
    assert F.shape == (len(campus), len(data.heard_waps(campus)) + extra)
    assert not F.isna().to_numpy().any()
    test = campus[WAP_COLS].assign(WAP500=-50)  # a WAP never heard in training is ignored
    assert list(builder.transform(test).columns) == list(F.columns)


def test_unknown_feature_set():
    with pytest.raises(ValueError):
        FeatureBuilder("rssi+magic")
