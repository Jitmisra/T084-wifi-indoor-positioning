import pandas as pd
import pytest

from indoorloc.metrics import (
    error_breakdown,
    floor_confusion,
    per_phone,
    per_scan_errors,
    summarize,
)

Y_TRUE = pd.DataFrame(
    {"BUILDINGID": [0, 0, 1, 2], "FLOOR": [0, 1, 1, 3], "LONGITUDE": 0.0, "LATITUDE": 0.0}
)
Y_PRED = pd.DataFrame(
    {
        "BUILDINGID": [0, 0, 1, 1],
        "FLOOR": [0, 2, 1, 3],
        "LONGITUDE": [3.0, 0.0, 1.0, 30.0],
        "LATITUDE": [4.0, 2.0, 0.0, 40.0],
    }
)


def test_per_scan_errors_and_summary():
    errors = per_scan_errors(Y_TRUE, Y_PRED)
    assert errors["err_2d"].tolist() == [5, 2, 1, 50]
    expected = {"building_acc": 0.75, "floor_acc": 0.75, "mae_x": 8.5, "mae_y": 11.5}
    expected |= {"mean_2d": 14.5, "median_2d": 3.5}
    assert summarize(errors) == pytest.approx(expected)


def test_error_breakdown_cases():
    table = error_breakdown(per_scan_errors(Y_TRUE, Y_PRED)).set_index("case")
    # cases in order: building wrong / building right, floor wrong / both right
    assert table["n_scans"].tolist() == [1, 1, 2]
    assert table["share_of_scans"].tolist() == [0.25, 0.25, 0.5]
    assert table["mean_2d"].tolist() == [50, 2, 3]
    assert table["share_of_error"].tolist() == pytest.approx([50 / 58, 2 / 58, 6 / 58])


def test_per_phone_splits_seen_and_new_phones():
    errors = per_scan_errors(Y_TRUE, Y_PRED)
    table = per_phone(errors, test_phones=[7, 7, 8, 9], train_phones=[7, 10]).set_index("phones")
    assert table.loc["seen in training"].tolist() == [1, 2, 0.5, 3.5]
    assert table.loc["new"].tolist() == [2, 2, 1.0, 25.5]


def test_floor_confusion_has_a_row_per_building_floor():
    table = floor_confusion(Y_TRUE, Y_PRED)
    assert len(table) == 13
    assert table.filter(like="pred_floor_").to_numpy().sum() == len(Y_TRUE)
    row = table[(table["building"] == 0) & (table["true_floor"] == 1)]
    assert row["pred_floor_2"].item() == 1
