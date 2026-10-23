import pandas as pd
import pytest
from conftest import make_campus

from indoorloc import data
from indoorloc.config import FILL_VALUE, NOT_HEARD, TRAIN_CSV, 


def test_load_csv_turns_the_not_heard_code_into_minus_110(tmp_path, campus):
    published = campus.head(3).copy()
    published[WAP_COLS] = published[WAP_COLS].replace(FILL_VALUE, NOT_HEARD)
    published.to_csv(tmp_path / "scans.csv", index=False)
    loaded = data.load_csv(tmp_path / "scans.csv")
    pd.testing.assert_frame_equal(loaded, campus.head(3))


def test_cleaning_rules_run_in_order_and_are_counted(campus):
    scans = campus.head(10).copy()
    scans.loc[0, WAP_COLS] = FILL_VALUE  # nothing heard
    scans.loc[3, "WAP001"] = -20  # implausibly strong
    scans = pd.concat([scans, scans.iloc[[5, 0]]])  # copies of a good and of the empty scan
    cleaned, table = data.clean_training(scans)
    # The copy of the empty scan is counted as empty, because that rule runs first
    assert table["scans_removed"].tolist() == [0, 2, 1, 1]
    assert table["scans_left"].tolist() == [12, 10, 9, 8]
    assert cleaned.index.equals(pd.RangeIndex(8))


def test_cleaning_counts_on_the_real_training_file():
    if not TRAIN_CSV.exists():
        pytest.skip("the UJIIndoorLoc files are not downloaded (run `make data`)")
    _, table = data.clean_training(data.load_csv(TRAIN_CSV))
    assert table["scans_removed"].tolist() == [0, 76, 634, 508]
    assert table["scans_left"].iloc[-1] == 18719


def test_heard_waps_skips_columns_that_are_never_heard(campus):
    waps = data.heard_waps(campus)
    assert 0 < len(waps) <= 104  # the synthetic campus only uses the first 104 WAP columns
    assert waps == [w for w in WAP_COLS[:104] if (campus[w] > FILL_VALUE).any()]


def test_users_who_shared_a_phone_are_in_one_group():
    scans = pd.DataFrame(
        {
            "USERID": [1, 9, 16, 2, 5, 6, 6, 7],
            "PHONEID": [14, 14, 14, 3, 21, 21, 20, 20],  # 5-6 share phone 21, 6-7 share 20
        }
    )
    # Phone 20 is looked at before phone 21, so user 7 joins group 5 only in a second pass
    assert data.user_device_groups(scans).tolist() == [1, 1, 1, 2, 5, 5, 5, 5]


def test_eda_summary(campus):
    summary = data.eda_summary(campus, make_campus(scans_per_floor=4, seed=1))
    assert summary["training_scans"] == len(campus)
    assert summary["building_floor_classes"] == 13
    assert summary["user_device_groups"] == campus["USERID"].nunique() - 1  # users 1 and 2
    assert 0 < summary["share_of_readings_not_heard_training"] < 1
