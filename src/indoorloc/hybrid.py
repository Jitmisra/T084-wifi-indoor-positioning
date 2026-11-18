"""Phase 2 hybrid: blend the Phase 1 Random Forest with the Phase 2 network.

Trees and neural networks make different kinds of mistakes (in Phase 1 no model was best at every
task), so a weighted average of what the two predict can beat both. forest_outputs gives the class
probabilities and positions of a fitted Random Forest (LocalizationModel), the network's own
outputs() gives the same for the network, and blend() averages them with one forest weight per
task: 0 = the network alone, 1 = the forest alone. The weights are chosen by user-group CV.
"""

import numpy as np
import pandas as pd

from .config import FLOORS_PER_BUILDING, TARGET_COLS, WAP_COLS
from .metrics import per_scan_errors, summarize
from .models import TASKS

N_CLASSES = {"building": len(FLOORS_PER_BUILDING), "floor": max(FLOORS_PER_BUILDING.values())}
METRIC = {"building": "building_acc", "floor": "floor_acc", "position": "mean_2d"}


def forest_outputs(model, X: pd.DataFrame) -> dict:
    """Class probabilities (building, floor) and positions (meters) of a fitted Phase 1
    LocalizationModel; a class that is missing from its training scans gets probability 0."""
    F = model.features_.transform(X)
    outputs = {}
    for task, n_classes in N_CLASSES.items():
        estimator = model.models_[task]
        probabilities = np.zeros((len(X), n_classes))
        probabilities[:, estimator.classes_.astype(int)] = estimator.predict_proba(F)
        outputs[task] = probabilities
    outputs["position"] = model.models_["position"].predict(F) + model.origin_
    return outputs


def blend(forest: dict, network: dict, weights: dict, index) -> pd.DataFrame:
    """Per task, weights[task] x the forest + (1 - weights[task]) x the network: probabilities for
    building and floor (the most likely class wins), positions for x and y."""
    mixed = {
        task: weights[task] * forest[task] + (1 - weights[task]) * network[task] for task in TASKS
    }
    prediction = {
        "BUILDINGID": mixed["building"].argmax(axis=1),
        "FLOOR": mixed["floor"].argmax(axis=1),
        "LONGITUDE": mixed["position"][:, 0],
        "LATITUDE": mixed["position"][:, 1],
    }
    return pd.DataFrame(prediction, index=index)


def out_of_fold_outputs(make_model, outputs_of, train: pd.DataFrame, folds: list) -> dict:
    """Fit a new model on the training rows of every fold and keep its outputs for the held-out
    rows: every training scan gets outputs from a model that never saw it."""
    stored = {task: None for task in TASKS}
    for fit_rows, held_out_rows in folds:
        fit_part, held_out = train.iloc[fit_rows], train.iloc[held_out_rows]
        model = make_model().fit(fit_part[WAP_COLS], fit_part[TARGET_COLS])
        for task, values in outputs_of(model, held_out[WAP_COLS]).items():
            if stored[task] is None:
                stored[task] = np.zeros((len(train), values.shape[1]))
            stored[task][held_out_rows] = values
    return stored


def weight_table(forest: dict, network: dict, train: pd.DataFrame, folds: list, weights: list):
    """CV scores (mean over the folds, as in Phase 1) of the blend for every forest weight."""
    rows = []
    for weight in weights:
        prediction = blend(forest, network, dict.fromkeys(TASKS, weight), train.index)
        errors = per_scan_errors(train, prediction)
        scores = [summarize(errors.iloc[held_out_rows]) for _, held_out_rows in folds]
        rows.append({"forest_weight": weight, **pd.DataFrame(scores)[list(METRIC.values())].mean()})
    return pd.DataFrame(rows)


def choose_weights(table: pd.DataFrame) -> dict:
    """One forest weight per task: the highest CV accuracy (building, floor) or the lowest CV mean
    2D error (position); on a tie the smaller weight (more network) wins."""
    chosen = {}
    for task, metric in METRIC.items():
        values = table[metric] if task == "position" else -table[metric]
        chosen[task] = float(table.loc[values.idxmin(), "forest_weight"])
    return chosen
