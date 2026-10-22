"""Run every experiment of the project in order and write the results to results/.

Stages: EDA -> feature check -> CV check -> tuning -> final model choice -> test evaluation ->
figures. Every choice is made with cross-validation on the training scans; the test scans are
only used in the test evaluation, at the very end.

    python scripts/run_experiments.py           # full run, writes results/
    python scripts/run_experiments.py --quick   # small, fast check run, writes results/_quick/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # draw figures without a screen
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # no install needed

import pandas as pd  # noqa: E402

from indoorloc import config, data, metrics, pipeline, plots  # noqa: E402
from indoorloc.models import TASKS, LocalizationModel  # noqa: E402

log = logging.getLogger("run_experiments")
TUNED = ["knn", "rf", "xgb"]
HEADLINE = ["building_acc", "floor_acc", "mean_2d"]


def save_csv(table: pd.DataFrame, path: Path) -> None:
    table.round(4).to_csv(path, index=False)


def save_json(content: dict, path: Path) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def one_line(scores: dict) -> str:
    """Building accuracy, floor accuracy and mean 2D error on one line (for the log)."""
    accuracy = f"building {scores['building_acc']:.4f}, floor {scores['floor_acc']:.4f}"
    return f"{accuracy}, mean 2D {scores['mean_2d']:.2f} m"


def settings_to_try(family: str, quick: bool) -> list[dict]:
    """The grid settings tried for a family; --quick uses 20 trees and keeps the first two."""
    grid = dict(config.GRIDS[family])
    if quick and family != "knn":
        grid["n_estimators"] = [config.QUICK_TREES]
    settings = pipeline.grid_settings(grid)
    return settings[: config.QUICK_SETTINGS] if quick else settings


def run(raw_train: pd.DataFrame, test: pd.DataFrame, out: Path, quick: bool = False) -> None:
    """Run all stages on the given data and write every result below `out`."""
    folder = {name: out / name for name in ("metrics", "tuning", "predictions", "figures")}
    for path in folder.values():
        path.mkdir(parents=True, exist_ok=True)

    log.info("1. EDA: clean the training file (never the test file) and summarize the data")
    train, cleaning = data.clean_training(raw_train)
    save_csv(cleaning, folder["metrics"] / "cleaning.csv")
    save_json(data.eda_summary(train, test), folder["metrics"] / "eda_summary.json")
    log.info(f"   {len(train)} cleaned training scans, {len(test)} test scans")
    full_train, n_folds = train, config.N_FOLDS
    if quick:
        n_scans = min(config.QUICK_SCANS, len(train))
        train = train.sample(n_scans, random_state=config.SEED).sort_index().reset_index(drop=True)
        n_folds = config.QUICK_FOLDS
    folds = pipeline.make_folds(train, "user", n_folds)
    baseline = {"n_estimators": config.QUICK_TREES} if quick else {}  # 300 trees, sklearn defaults
    params = {"baseline_rf": dict.fromkeys(TASKS, baseline)}

    log.info("2. Feature check: the Baseline RF settings on three feature sets (user-group CV)")
    check = {}
    for feature_set in config.FEATURE_SETS:
        model = LocalizationModel("rf", feature_set, params["baseline_rf"])
        check[feature_set] = pipeline.cv_summary(pipeline.cross_validate(model, train, folds))
        log.info(f"   {feature_set}: {one_line(check[feature_set])}")
    feature_check = pd.DataFrame([{"feature_set": name, **cv} for name, cv in check.items()])
    save_csv(feature_check, folder["metrics"] / "feature_check.csv")
    baseline_cv = check[config.MODEL_FEATURES["baseline_rf"]]  # this row is the Baseline RF

    log.info("3. CV check: the Baseline RF under random CV (the test row is added in stage 6)")
    model = LocalizationModel("rf", config.MODEL_FEATURES["baseline_rf"], params["baseline_rf"])
    random_folds = pipeline.make_folds(train, "random", n_folds)
    random_cv = pipeline.cv_summary(pipeline.cross_validate(model, train, random_folds))
    random_2d, user_2d = random_cv["mean_2d"], baseline_cv["mean_2d"]
    log.info(f"   mean 2D error: random CV {random_2d:.2f} m, user-group CV {user_2d:.2f} m")

    log.info("4. Tuning: every setting on the same user-group folds, best setting per task")
    cv_rows = [{"model": "baseline_rf", **baseline_cv}]
    for family in TUNED:
        settings = settings_to_try(family, quick)
        features = config.MODEL_FEATURES[family]
        tables, best, tuned_cv = pipeline.grid_search(family, features, train, folds, settings)
        for task, table in tables.items():
            save_csv(table, folder["tuning"] / f"{family}_{task}.csv")
        params[family] = best
        cv_rows.append({"model": family, **tuned_cv})
        log.info(f"   {family} best settings: {best}")
    save_json({family: params[family] for family in TUNED}, folder["tuning"] / "best_params.json")
    cv_results = pd.DataFrame(cv_rows)
    cv_results.insert(1, "name", cv_results["model"].map(config.MODEL_NAMES))
    save_csv(cv_results, folder["metrics"] / "cv_results.csv")

    log.info("5. Final model: the tuned model with the lowest CV mean 2D error (fixed rule)")
    tuned = cv_results.set_index("model").loc[TUNED]
    final = tuned["mean_2d"].idxmin()
    choice = {"final_model": final, "name": config.MODEL_NAMES[final]}
    choice["rule"] = "lowest CV mean 2D error among kNN, Random Forest and XGBoost"
    for metric in HEADLINE:
        choice[f"cv_{metric}"] = tuned[metric].round(4).to_dict()
    save_json(choice, folder["metrics"] / "final_model.json")
    log.info(f"   final model: {config.MODEL_NAMES[final]}")

    log.info("6. Test evaluation: fit on all training scans, predict the test scans once")
    test_rows, breakdowns, phone_rows, predicted, prediction_tables = [], [], [], {}, {}
    for key, name in config.MODEL_NAMES.items():
        model = LocalizationModel(config.MODEL_FAMILY[key], config.MODEL_FEATURES[key], params[key])
        predicted[key] = pipeline.evaluate_on_test(model, train, test)
        errors = metrics.per_scan_errors(test, predicted[key])
        true = test[config.TARGET_COLS].add_prefix("true_")
        parts = [test[["PHONEID"]], true, predicted[key].add_prefix("pred_"), errors[["err_2d"]]]
        prediction_tables[key] = pd.concat(parts, axis=1)
        save_csv(prediction_tables[key], folder["predictions"] / f"{key}.csv")
        test_rows.append({"model": key, "name": name, **metrics.summarize(errors)})
        breakdown = metrics.error_breakdown(errors)
        breakdown.insert(0, "model", key)
        breakdowns.append(breakdown)
        phones = metrics.per_phone(errors, test["PHONEID"], train["PHONEID"])
        phones.insert(0, "model", key)
        phone_rows.append(phones)
        log.info(f"   {name}: {one_line(test_rows[-1])}")
    test_results = pd.DataFrame(test_rows)
    save_csv(test_results, folder["metrics"] / "test_results.csv")
    save_csv(pd.concat(breakdowns), folder["metrics"] / "error_breakdown.csv")
    save_csv(pd.concat(phone_rows), folder["metrics"] / "per_phone.csv")
    confusion = metrics.floor_confusion(test, predicted[final])
    save_csv(confusion, folder["metrics"] / "floor_confusion.csv")
    baseline_test = test_results.set_index("model").loc["baseline_rf"].to_dict()
    cv_check = pd.DataFrame([random_cv, baseline_cv, baseline_test])[HEADLINE]
    cv_check.insert(0, "evaluation", ["random CV", "user-group CV", "test set"])
    save_csv(cv_check, folder["metrics"] / "cv_check.csv")

    log.info("7. Figures")
    figures = {
        "eda_campus_map": plots.campus_map(full_train, test),
        "eda_waps_per_scan": plots.waps_per_scan(full_train, test),
        "eda_scans_per_floor": plots.scans_per_floor(full_train, test),
        "feature_check": plots.feature_check(feature_check),
        "model_comparison": plots.model_comparison(cv_results, test_results),
        "floor_confusion": plots.floor_confusion(confusion),
        "error_map": plots.error_map(prediction_tables[final]),
    }
    for name, figure in figures.items():
        plots.save(figure, name, folder["figures"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small, fast check run")
    args = parser.parse_args()
    out = config.RESULTS_DIR / "_quick" if args.quick else config.RESULTS_DIR
    (out / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(out / "logs" / "run.log", mode="w")]
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S", handlers=handlers)
    for name in ("run_experiments", "indoorloc"):  # our messages only, not those of other packages
        logging.getLogger(name).setLevel(logging.INFO)
    start = time.time()
    run(data.load_csv(config.TRAIN_CSV), data.load_csv(config.TEST_CSV), out, quick=args.quick)
    minutes = (time.time() - start) / 60
    log.info(f"Done in {minutes:.1f} min; results are in {out.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
