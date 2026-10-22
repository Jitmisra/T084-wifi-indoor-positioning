import pytest
from conftest import make_campus
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from indoorloc.config import TARGET_COLS, WAP_COLS
from indoorloc.metrics import per_scan_errors
from indoorloc.models import TASKS, LocalizationModel, make_estimator

FAMILIES = [("knn", "rssi"), ("rf", "rssi+stats+affinity"), ("xgb", "rssi+stats+affinity")]


def test_make_estimator_fixed_and_tuned_settings():
    forest = make_estimator("rf", "floor")
    assert isinstance(forest, RandomForestClassifier)
    assert (forest.n_estimators, forest.random_state) == (300, 42)
    knn = make_estimator("knn", "position", {"n_neighbors": 3})
    assert isinstance(knn, KNeighborsRegressor) and knn.n_neighbors == 3
    boosting = make_estimator("xgb", "position", {"max_depth": 4})
    assert isinstance(boosting, XGBRegressor)
    assert (boosting.max_depth, boosting.learning_rate, boosting.subsample) == (4, 0.1, 0.85)
    with pytest.raises(ValueError):
        make_estimator("svm", "floor")


@pytest.mark.parametrize("family, feature_set", FAMILIES)
def test_localization_model_fit_and_predict(campus, family, feature_set):
    params = None if family == "knn" else dict.fromkeys(TASKS, {"n_estimators": 100})
    model = LocalizationModel(family, feature_set, params)
    model.fit(campus[WAP_COLS], campus[TARGET_COLS])
    test = make_campus(scans_per_floor=3, seed=1)  # new scans in the same buildings
    prediction = model.predict(test[WAP_COLS])
    assert list(prediction.columns) == TARGET_COLS and prediction.index.equals(test.index)
    assert set(prediction["FLOOR"]) <= {0, 1, 2, 3, 4}
    errors = per_scan_errors(test, prediction)
    assert errors["building_ok"].mean() > 0.9  # the buildings of the synthetic campus are easy
    assert errors["err_2d"].mean() < 15  # guessing the middle of the true building gives 19 m
