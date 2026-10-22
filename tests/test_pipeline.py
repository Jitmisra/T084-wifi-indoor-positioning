import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_campus

from indoorloc import data, pipeline
from indoorloc.config import TARGET_COLS
from indoorloc.models import LocalizationModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"


def test_user_folds_never_split_a_group(campus):
    groups = data.user_device_groups(campus)
    for fit_rows, held_out_rows in pipeline.make_folds(campus, "user", 3):
        assert not set(groups[fit_rows]) & set(groups[held_out_rows])
        assert len(fit_rows) + len(held_out_rows) == len(campus)


def test_random_folds_hold_out_every_scan_once(campus):
    folds = pipeline.make_folds(campus, "random", 5)
    held_out = np.concatenate([rows for _, rows in folds])
    assert sorted(held_out) == list(range(len(campus)))
    assert np.any(np.diff(folds[0][1]) > 1)  # shuffled: not one block of neighboring rows
    assert pipeline.make_folds(campus, "random", 5)[0][1].tolist() == folds[0][1].tolist()


class KeepProfiles(LocalizationModel):
    """A model that remembers the AP-affinity profiles it learned in every fit."""

    def __init__(self, family, feature_set):
        super().__init__(family, feature_set)
        self.profiles = []

    def fit(self, X, y):
        super().fit(X, y)
        self.profiles.append(self.features_.affinity_.profiles_)
        return self


def test_held_out_labels_never_reach_the_features(campus):
    folds = pipeline.make_folds(campus, "user", 3)
    held_out = folds[0][1]
    shuffled = campus.copy()
    labels = campus.loc[held_out, TARGET_COLS]
    shuffled.loc[held_out, TARGET_COLS] = labels.sample(frac=1, random_state=1).set_axis(held_out)
    models = [KeepProfiles("knn", "rssi+stats+affinity") for _ in range(2)]
    for model, scans in zip(models, (campus, shuffled), strict=True):
        pipeline.cross_validate(model, scans, folds)
    # Fold 0 holds those scans out, so its profiles stay the same; fold 1 trains on them
    np.testing.assert_array_equal(models[0].profiles[0], models[1].profiles[0])
    assert not np.array_equal(models[0].profiles[1], models[1].profiles[1])


def test_grid_search_picks_the_best_setting_and_the_earlier_one_on_a_tie(campus):
    folds = pipeline.make_folds(campus, "user", 3)
    # With one neighbor, "uniform" and "distance" weights give the same predictions (a tie)
    settings = [{"n_neighbors": 1, "weights": w} for w in ("uniform", "distance")]
    settings.append({"n_neighbors": 9, "weights": "uniform"})
    tables, best, tuned = pipeline.grid_search("knn", "rssi", campus, folds, settings)
    for task, table in tables.items():
        assert table.loc[0, "cv_mean"] == table.loc[1, "cv_mean"]
        top = table["cv_mean"].min() if task == "position" else table["cv_mean"].max()
        first_best = int(table.index[table["cv_mean"] == top][0])
        assert table["chosen"].tolist() == [row == first_best for row in range(3)]
        assert best[task] == settings[first_best]
        metric = pipeline.CV_METRICS[task]
        assert tuned[metric] == table.loc[first_best, "cv_mean"]


def test_quick_run_writes_every_result(tmp_path, campus):
    run = runpy.run_path(str(SCRIPT))["run"]  # the script's run() function, without main()
    test = make_campus(scans_per_floor=4, seed=1)
    test.loc[test["USERID"] > 4, "PHONEID"] += 1000  # phones never used in training
    # Two scans the cleaning rules would remove; the test file must be used exactly as it is
    test.loc[0, "WAP001"] = -20
    test = pd.concat([test, test.iloc[[1]]], ignore_index=True)
    run(campus, test, tmp_path, quick=True)
    metrics = ["cleaning", "feature_check", "cv_check", "cv_results", "test_results"]
    metrics += ["error_breakdown", "per_phone", "floor_confusion"]
    for name in metrics:
        assert (tmp_path / "metrics" / f"{name}.csv").exists()
    for family in ("knn", "rf", "xgb"):
        for task in ("building", "floor", "position"):
            assert (tmp_path / "tuning" / f"{family}_{task}.csv").exists()
    for model in ("baseline_rf", "knn", "rf", "xgb"):
        assert len(pd.read_csv(tmp_path / "predictions" / f"{model}.csv")) == len(test)
    figures = ["eda_campus_map", "eda_waps_per_scan", "eda_scans_per_floor", "feature_check"]
    for name in [*figures, "model_comparison", "floor_confusion", "error_map"]:
        assert (tmp_path / "figures" / f"{name}.png").exists()
        assert (tmp_path / "figures" / f"{name}.pdf").exists()
    # The final model is the tuned model with the lowest CV mean 2D error
    final = json.loads((tmp_path / "metrics" / "final_model.json").read_text())
    cv = pd.read_csv(tmp_path / "metrics" / "cv_results.csv").set_index("model")
    assert final["final_model"] == cv.loc[["knn", "rf", "xgb"], "mean_2d"].idxmin()
    phones = pd.read_csv(tmp_path / "metrics" / "per_phone.csv")
    assert set(phones["phones"]) == {"seen in training", "new"}
