"""The three model families and LocalizationModel, which predicts building, floor and position."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

from .config import COORD_COLS, RF_TREES, SEED, XGB_FIXED
from .features import FeatureBuilder

TASKS = ["building", "floor", "position"]


def make_estimator(family: str, task: str, params: dict | None = None):
    """An unfitted model of one family for one task.

    Building and floor are classifiers; position is a regressor with two outputs (x and y).
    `params` are the settings being tuned; everything else stays fixed.
    """
    params = params or {}
    classify = task != "position"
    if family == "knn":
        model = KNeighborsClassifier if classify else KNeighborsRegressor
        return model(**params, n_jobs=-1)
    if family == "rf":
        model = RandomForestClassifier if classify else RandomForestRegressor
        settings = {"n_estimators": RF_TREES} | params  # a tuned value replaces the fixed one
        return model(**settings, random_state=SEED, n_jobs=-1)
    if family == "xgb":
        model = XGBClassifier if classify else XGBRegressor
        settings = XGB_FIXED | params
        return model(**settings, random_state=SEED, n_jobs=-1)
    raise ValueError(f"unknown model family {family!r}; use 'knn', 'rf' or 'xgb'")


class LocalizationModel:
    """Predict the building, floor and position of Wi-Fi scans with three separate models.

    The three models of one family use the same features. fit(X, y) takes raw scans (WAP
    columns) and their targets; predict(X) returns BUILDINGID, FLOOR, LONGITUDE and LATITUDE,
    so any other model with the same two methods (e.g. in Phase 2) is evaluated the same way.
    """

    def __init__(self, family: str, feature_set: str, params: dict | None = None):
        self.family = family
        self.feature_set = feature_set
        self.params = params or {}  # settings per task, e.g. {"floor": {"n_neighbors": 5}}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LocalizationModel":
        self.features_ = FeatureBuilder(self.feature_set).fit(X, y)
        F = self.features_.transform(X)
        # Why: XGBoost computes with 32-bit floats, which store a northing of 4,864,900 m only
        # to the nearest 0.5 m, so positions are learned as offsets from the training mean.
        self.origin_ = y[COORD_COLS].mean().to_numpy()
        targets = {
            "building": y["BUILDINGID"],
            "floor": y["FLOOR"],
            "position": y[COORD_COLS] - self.origin_,
        }
        self.models_ = {}
        for task in TASKS:
            model = make_estimator(self.family, task, self.params.get(task))
            self.models_[task] = model.fit(F, targets[task].to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F = self.features_.transform(X)
        position = self.models_["position"].predict(F) + self.origin_
        prediction = {
            "BUILDINGID": self.models_["building"].predict(F),
            "FLOOR": self.models_["floor"].predict(F),
            "LONGITUDE": position[:, 0],
            "LATITUDE": position[:, 1],
        }
        return pd.DataFrame(prediction, index=X.index)
