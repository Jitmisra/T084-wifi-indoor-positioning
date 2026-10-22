"""Cross-validation folds, cross-validation, the grid search and the test evaluation."""

import itertools
import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .config import N_FOLDS, SEED, TARGET_COLS, WAP_COLS
from .data import user_device_groups
from .metrics import per_scan_errors, summarize
from .models import TASKS, LocalizationModel

log = logging.getLogger(__name__)
CV_METRICS = {"building": "building_acc", "floor": "floor_acc", "position": "mean_2d"}


def make_folds(train: pd.DataFrame, kind: str = "user", n_folds: int = N_FOLDS) -> list:
    """Cross-validation splits as a list of (training rows, held-out rows).

    kind="user": each held-out part holds whole user-device groups, so its users and phones are
    never in the training part (like the test set, recorded later with mostly new phones).
    kind="random": ordinary shuffled K-fold, only used to show how optimistic it is.
    """
    if kind == "user":
        return list(GroupKFold(n_folds).split(train, groups=user_device_groups(train)))
    if kind == "random":
        return list(KFold(n_folds, shuffle=True, random_state=SEED).split(train))
    raise ValueError(f"unknown kind of folds {kind!r}; use 'user' or 'random'")


def cross_validate(model: LocalizationModel, train: pd.DataFrame, folds: list) -> pd.DataFrame:
    """Fit the model on the training rows of each fold and score its held-out rows.

    The model (features included) is fitted again in every fold, so the held-out scans never
    influence their own features. Returns one row of metrics per fold.
    """
    rows = []
    for fit_rows, held_out_rows in folds:
        fit_part, held_out = train.iloc[fit_rows], train.iloc[held_out_rows]
        model.fit(fit_part[WAP_COLS], fit_part[TARGET_COLS])
        prediction = model.predict(held_out[WAP_COLS])
        rows.append(summarize(per_scan_errors(held_out, prediction)))
    return pd.DataFrame(rows)


def cv_summary(fold_scores: pd.DataFrame) -> dict:
    """Mean and standard deviation over the folds of building accuracy, floor accuracy and
    mean 2D error."""
    summary = {}
    for metric in CV_METRICS.values():
        summary[metric] = float(fold_scores[metric].mean())
        summary[f"{metric}_std"] = float(fold_scores[metric].std())
    return summary


def grid_settings(grid: dict) -> list[dict]:
    """Every combination of a grid, in the order the values are written (first key slowest)."""
    return [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]


def grid_search(
    family: str, feature_set: str, train: pd.DataFrame, folds: list, settings: list[dict]
) -> tuple[dict, dict, dict]:
    """Cross-validate every setting on the same folds and pick the best setting for each task.

    Each task has its own model, so each task picks its own setting: the highest CV mean
    accuracy (building, floor) or the lowest CV mean 2D error (position); on a tie the earlier
    setting wins. Returns one table per task (setting, CV mean and std, chosen or not), the
    chosen setting per task and the CV summary of the tuned model (its chosen settings' scores).
    """
    summaries = []
    for number, setting in enumerate(settings, start=1):
        start = time.time()
        model = LocalizationModel(family, feature_set, dict.fromkeys(TASKS, setting))
        summaries.append(cv_summary(cross_validate(model, train, folds)))
        scores = " | ".join(f"{m} {summaries[-1][m]:.4f}" for m in CV_METRICS.values())
        seconds = time.time() - start
        log.info(f"   {family} {number}/{len(settings)} {setting}: {scores} ({seconds:.0f} s)")
    tables, best, tuned = {}, {}, {}
    for task, metric in CV_METRICS.items():
        means = [summary[metric] for summary in summaries]
        pick = int(np.argmin(means) if task == "position" else np.argmax(means))  # first best wins
        tables[task] = pd.DataFrame(settings).assign(
            cv_mean=means,
            cv_std=[summary[f"{metric}_std"] for summary in summaries],
            chosen=[row == pick for row in range(len(settings))],
        )
        best[task] = settings[pick]
        tuned[metric] = summaries[pick][metric]
        tuned[f"{metric}_std"] = summaries[pick][f"{metric}_std"]
    return tables, best, tuned


def evaluate_on_test(
    model: LocalizationModel, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Fit the model on all training scans and predict the test scans.

    This is the only function that shows the test set to a model, after every choice is made.
    """
    model.fit(train[WAP_COLS], train[TARGET_COLS])
    return model.predict(test[WAP_COLS])
