"""Evaluation metrics: per-scan errors, summaries, the error breakdown and the phone split."""

import numpy as np
import pandas as pd

from .config import FLOORS_PER_BUILDING

CASES = ["building wrong", "building right, floor wrong", "both right"]


def per_scan_errors(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """For every scan: building right?, floor right?, x and y errors and the 2D error (meters).

    The 2D error is the straight-line distance between the predicted and the true position.
    """
    dx = y_pred["LONGITUDE"].to_numpy() - y_true["LONGITUDE"].to_numpy()
    dy = y_pred["LATITUDE"].to_numpy() - y_true["LATITUDE"].to_numpy()
    errors = {
        "building_ok": y_pred["BUILDINGID"].to_numpy() == y_true["BUILDINGID"].to_numpy(),
        "floor_ok": y_pred["FLOOR"].to_numpy() == y_true["FLOOR"].to_numpy(),
        "err_x": np.abs(dx),
        "err_y": np.abs(dy),
        "err_2d": np.hypot(dx, dy),
    }
    return pd.DataFrame(errors, index=y_true.index)


def summarize(errors: pd.DataFrame) -> dict:
    """Building and floor accuracy, mean absolute x and y error, mean and median 2D error."""
    return {
        "building_acc": float(errors["building_ok"].mean()),
        "floor_acc": float(errors["floor_ok"].mean()),
        "mae_x": float(errors["err_x"].mean()),
        "mae_y": float(errors["err_y"].mean()),
        "mean_2d": float(errors["err_2d"].mean()),
        "median_2d": float(errors["err_2d"].median()),
    }


def error_breakdown(errors: pd.DataFrame) -> pd.DataFrame:
    """Split the scans into three cases and show where the 2D error comes from.

    Cases: building wrong / building right but floor wrong / both right. For each case: number
    and share of scans, mean 2D error and share of the total 2D error.
    """
    wrong = [~errors["building_ok"], ~errors["floor_ok"]]
    case = np.select(wrong, CASES[:2], default=CASES[2])  # the first true condition wins
    by_case = errors["err_2d"].groupby(case)
    table = pd.DataFrame({"n_scans": by_case.size(), "mean_2d": by_case.mean()})
    table = table.reindex(CASES)
    table["n_scans"] = table["n_scans"].fillna(0).astype(int)
    table.insert(1, "share_of_scans", table["n_scans"] / len(errors))
    table["share_of_error"] = by_case.sum().reindex(CASES).fillna(0) / errors["err_2d"].sum()
    return table.rename_axis("case").reset_index()


def per_phone(errors: pd.DataFrame, test_phones, train_phones) -> pd.DataFrame:
    """Floor accuracy and mean 2D error on test scans from phones seen in training vs new ones."""
    test_phones = np.asarray(test_phones)
    seen = np.isin(test_phones, np.asarray(train_phones))
    rows = []
    for label, mask in (("seen in training", seen), ("new", ~seen)):
        rows.append(
            {
                "phones": label,
                "n_phones": len(np.unique(test_phones[mask])),
                "n_scans": int(mask.sum()),
                "floor_acc": float(errors["floor_ok"].to_numpy()[mask].mean()),
                "mean_2d": float(errors["err_2d"].to_numpy()[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def floor_confusion(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Floor confusion counts: a row per true (building, floor), a column per predicted floor."""
    counts = pd.crosstab([y_true["BUILDINGID"], y_true["FLOOR"]], y_pred["FLOOR"])
    rows = [(b, f) for b, n_floors in FLOORS_PER_BUILDING.items() for f in range(n_floors)]
    floors = range(max(FLOORS_PER_BUILDING.values()))
    counts = counts.reindex(index=pd.MultiIndex.from_tuples(rows), columns=floors, fill_value=0)
    counts.index.names = ["building", "true_floor"]
    counts.columns = [f"pred_floor_{f}" for f in floors]
    return counts.reset_index()
