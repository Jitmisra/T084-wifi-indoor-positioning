"""Load the two UJIIndoorLoc files, clean the training file and summarize the data."""

import numpy as np
import pandas as pd

from .config import FILL_VALUE, FLOORS_PER_BUILDING, HIGH_RSSI, NOT_HEARD, WAP_COLS


def load_csv(path) -> pd.DataFrame:
    """Read one CSV file and replace the "not heard" code 100 with -110 dBm."""
    df = pd.read_csv(path)
    df[WAP_COLS] = df[WAP_COLS].replace(NOT_HEARD, FILL_VALUE)
    return df


def clean_training(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove bad training scans with three rules, in this order, and count each step.

    1. no WAP heard at all (the scan says nothing about where it was taken);
    2. exact duplicate rows (all columns equal; the first copy is kept);
    3. any reading of -30 dBm or stronger (physically implausible, mostly one faulty phone).
    Only the training file is cleaned; the test file is used exactly as published.
    """
    steps = [("raw training file", len(train))]
    train = train[(train[WAP_COLS] > FILL_VALUE).any(axis=1)]
    steps.append(("no WAP heard", len(train)))
    train = train[~train.duplicated(keep="first")]
    steps.append(("exact duplicate", len(train)))
    train = train[(train[WAP_COLS] < HIGH_RSSI).all(axis=1)]
    steps.append((f"reading of {HIGH_RSSI} dBm or stronger", len(train)))
    table = pd.DataFrame(steps, columns=["step", "scans_left"])
    table.insert(1, "scans_removed", (-table["scans_left"].diff()).fillna(0).astype(int))
    return train.reset_index(drop=True), table


def heard_waps(X: pd.DataFrame) -> list[str]:
    """The WAP columns heard in at least one scan of X (a WAP never heard tells us nothing)."""
    heard = (X[WAP_COLS] > FILL_VALUE).any()
    return heard.index[heard].tolist()


def user_device_groups(scans: pd.DataFrame) -> np.ndarray:
    """Group of every scan for cross-validation: users who shared a phone form one group.

    A group is named after its smallest user id. In the training file users 1, 9 and 16 used
    the same phone (14), so the 18 users form 16 groups.
    """
    group = {user: user for user in scans["USERID"].unique()}
    pairs = scans[["USERID", "PHONEID"]].drop_duplicates()
    changed = True
    while changed:  # Why a loop: sharing can chain (A and B share one phone, B and C another)
        changed = False
        for _, users in pairs.groupby("PHONEID")["USERID"]:
            smallest = min(group[user] for user in users)
            for user in users:
                if group[user] != smallest:
                    group[user] = smallest
                    changed = True
    return scans["USERID"].map(group).to_numpy()


def eda_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Key facts about the cleaned training scans and the test scans."""
    heard_train = train[WAP_COLS] > FILL_VALUE
    heard_test = test[WAP_COLS] > FILL_VALUE
    readings = train[WAP_COLS].to_numpy()
    readings = readings[readings > FILL_VALUE]
    new_phone = ~test["PHONEID"].isin(train["PHONEID"])

    def dates(df: pd.DataFrame) -> str:
        days = pd.to_datetime(df["TIMESTAMP"], unit="s")
        return f"{days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}"

    # Why: Web-Mercator (Earth radius R = 6,378,137 m) stretches distances by 1 / cos(latitude),
    # about 1.3 here, so 1 m in the data is cos(latitude) m on the ground. arctan(sinh(y / R))
    # turns the Web-Mercator northing y back into the latitude (about 40.0 degrees here).
    latitude = np.arctan(np.sinh(train["LATITUDE"].mean() / 6378137.0))
    return {
        "training_scans": len(train),
        "test_scans": len(test),
        "waps_in_files": len(WAP_COLS),
        "waps_heard_in_training": int(heard_train.any().sum()),
        "waps_heard_in_test": int(heard_test.any().sum()),
        "waps_heard_only_in_test": int((heard_test.any() & ~heard_train.any()).sum()),
        "share_of_readings_not_heard_training": round(float(1 - heard_train.to_numpy().mean()), 4),
        "mean_waps_per_scan_training": round(float(heard_train.sum(axis=1).mean()), 2),
        "mean_waps_per_scan_test": round(float(heard_test.sum(axis=1).mean()), 2),
        "strongest_reading_training_dbm": int(readings.max()),
        "weakest_reading_training_dbm": int(readings.min()),
        "buildings": len(FLOORS_PER_BUILDING),
        "building_floor_classes": sum(FLOORS_PER_BUILDING.values()),
        "users_training": int(train["USERID"].nunique()),
        "phones_training": int(train["PHONEID"].nunique()),
        "user_device_groups": len(np.unique(user_device_groups(train))),
        "phones_test": int(test["PHONEID"].nunique()),
        "test_phones_seen_in_training": int(test.loc[~new_phone, "PHONEID"].nunique()),
        "share_of_test_scans_from_new_phones": round(float(new_phone.mean()), 4),
        "training_dates": dates(train),
        "test_dates": dates(test),
        "campus_width_m": round(float(np.ptp(train["LONGITUDE"])), 1),
        "campus_height_m": round(float(np.ptp(train["LATITUDE"])), 1),
        "ground_meters_per_data_meter": round(float(np.cos(latitude)), 4),
    }
