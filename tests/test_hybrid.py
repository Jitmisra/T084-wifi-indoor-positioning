import numpy as np
import pandas as pd
from conftest import make_campus

from indoorloc import hybrid, pipeline
from indoorloc.config import TARGET_COLS, WAP_COLS
from indoorloc.dl import NeuralLocalizationModel
from indoorloc.models import LocalizationModel

FAST = {"hidden": (32,), "batch_size": 32}
SETTING = {"n_estimators": 20}


def fitted_models(campus):
    forest = LocalizationModel("rf", "rssi+stats+affinity", dict.fromkeys(hybrid.METRIC, SETTING))
    network = NeuralLocalizationModel(params=FAST, epochs=20)
    return [model.fit(campus[WAP_COLS], campus[TARGET_COLS]) for model in (forest, network)]


def test_blend_weights_0_and_1_give_each_model_alone(campus):
    forest, network = fitted_models(campus)
    test = make_campus(scans_per_floor=3, seed=1)
    outputs = [hybrid.forest_outputs(forest, test[WAP_COLS]), network.outputs(test[WAP_COLS])]
    for task in ("building", "floor"):  # probabilities: one column per class, rows sum to 1
        for output in outputs:
            np.testing.assert_allclose(output[task].sum(axis=1), 1, rtol=1e-5)
    only_network = hybrid.blend(*outputs, dict.fromkeys(hybrid.METRIC, 0.0), test.index)
    pd.testing.assert_frame_equal(only_network, network.predict(test[WAP_COLS]), check_dtype=False)
    only_forest = hybrid.blend(*outputs, dict.fromkeys(hybrid.METRIC, 1.0), test.index)
    pd.testing.assert_frame_equal(only_forest, forest.predict(test[WAP_COLS]), check_dtype=False)
    halfway = hybrid.blend(*outputs, dict.fromkeys(hybrid.METRIC, 0.5), test.index)
    middle = (only_network[["LONGITUDE", "LATITUDE"]] + only_forest[["LONGITUDE", "LATITUDE"]]) / 2
    np.testing.assert_allclose(halfway[["LONGITUDE", "LATITUDE"]], middle)


def test_forest_probabilities_line_up_with_the_classes(campus):
    # Without floor 4 in its training scans, the forest still gets 5 floor columns (the last 0)
    no_top_floor = campus[campus["FLOOR"] < 4].reset_index(drop=True)
    forest = LocalizationModel("rf", "rssi", dict.fromkeys(hybrid.METRIC, SETTING))
    forest.fit(no_top_floor[WAP_COLS], no_top_floor[TARGET_COLS])
    floor = hybrid.forest_outputs(forest, campus[WAP_COLS])["floor"]
    assert floor.shape == (len(campus), 5) and np.all(floor[:, 4] == 0)
    np.testing.assert_array_equal(floor.argmax(axis=1), forest.predict(campus[WAP_COLS])["FLOOR"])


def test_out_of_fold_outputs_come_from_models_that_never_saw_the_scan(campus):
    folds = pipeline.make_folds(campus, "user", 3)

    class Recorder:
        def fit(self, X, y):
            self.seen = set(X.index)
            return self

    def outputs_of(model, X):
        assert not model.seen & set(X.index)
        return {task: np.tile(X.index.to_numpy()[:, None], (1, 2)) for task in hybrid.METRIC}

    stored = hybrid.out_of_fold_outputs(Recorder, outputs_of, campus, folds)
    np.testing.assert_array_equal(stored["position"][:, 0], campus.index)  # every row filled


def test_choose_weights_per_task_and_ties():
    table = pd.DataFrame(
        {
            "forest_weight": [0.0, 0.5, 1.0],
            "building_acc": [0.99, 1.0, 1.0],  # a tie: the smaller weight wins
            "floor_acc": [0.80, 0.82, 0.85],
            "mean_2d": [9.0, 8.5, 11.0],
        }
    )
    assert hybrid.choose_weights(table) == {"building": 0.5, "floor": 1.0, "position": 0.5}
