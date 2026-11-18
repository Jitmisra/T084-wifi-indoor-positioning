"""Run every Phase 2 experiment (neural networks) in order and write the results to results/phase2/.

Stages: tuning -> ablation study (with a CV seed check) -> final network -> test evaluation (with
a seed check) -> hybrid blend of the Phase 1 Random Forest and the network -> comparison with
Phase 1 -> figures. As in Phase 1, every setting is chosen with
user-group cross-validation on the training scans, the final network is fixed before testing,
and the test scans are only used in the test evaluation, at the end.
Phase 1's results (results/metrics/) must exist: `make run` writes them.

    python scripts/run_dl.py           # full run, writes results/phase2/
    python scripts/run_dl.py --quick   # small, fast check run, writes results/_quick/phase2/
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

from indoorloc import config, data, dl, hybrid, metrics, pipeline, plots  # noqa: E402
from indoorloc.models import LocalizationModel  # noqa: E402

log = logging.getLogger("run_dl")
HEADLINE = ["building_acc", "floor_acc", "mean_2d"]
# Labels of the Phase 1 models under the bars of the comparison figure (two lines where long)
PHASE1_SHORT = {
    "baseline_rf": "Baseline\nRF",
    "knn": "kNN",
    "rf": "Random\nForest",
    "xgb": "XGBoost",
}


def save_csv(table: pd.DataFrame, path: Path) -> None:
    # Why 6 decimals: an accuracy of 0.94149 stored as 0.9415 would later be rounded again to
    # 94.2% instead of 94.1%
    table.round(6).to_csv(path, index=False)


def save_json(content: dict, path: Path) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def one_line(scores: dict) -> str:
    """Building accuracy, floor accuracy and mean 2D error on one line (for the log)."""
    accuracy = f"building {scores['building_acc']:.4f}, floor {scores['floor_acc']:.4f}"
    return f"{accuracy}, mean 2D {scores['mean_2d']:.2f} m"


def make_network(variant: dict, params: dict, epochs: int, seed: int = config.SEED):
    """A NeuralLocalizationModel for one variant (its input and task choices) and settings."""
    switches = {key: variant[key] for key in ("hybrid", "multitask", "indicators")}
    return dl.NeuralLocalizationModel(
        **switches, params=params | variant["params"], epochs=epochs, seed=seed
    )


def run(train: pd.DataFrame, test: pd.DataFrame, phase1: Path, out: Path, quick: bool = False):
    """Run all Phase 2 stages on cleaned training scans and the test scans; write below `out`.

    `phase1` is the results folder of Phase 1: tuning/best_params.json is read for the Random
    Forest of the blend (stage 6), metrics/cv_results.csv and metrics/test_results.csv for the
    comparison (stage 7).
    """
    folder = {name: out / name for name in ("metrics", "tuning", "predictions", "figures")}
    for path in folder.values():
        path.mkdir(parents=True, exist_ok=True)
    epochs, n_folds = config.DL_EPOCHS, config.N_FOLDS
    # Scored every DL_EVERY epochs and always after the last one (the networks of the test set)
    checkpoints = sorted({*range(config.DL_EVERY, epochs + 1, config.DL_EVERY), epochs})
    settings = pipeline.grid_settings(config.DL_GRID)
    seeds, cv_seeds = config.DL_SEEDS, config.DL_CV_SEEDS
    if quick:
        n_scans = min(config.QUICK_SCANS, len(train))
        train = train.sample(n_scans, random_state=config.SEED).sort_index().reset_index(drop=True)
        n_folds, epochs, checkpoints = config.QUICK_FOLDS, 2, [1, 2]
        settings, seeds, cv_seeds = settings[: config.QUICK_SETTINGS], seeds[:2], cv_seeds[:1]
    folds = pipeline.make_folds(train, "user", n_folds)
    variants = config.DL_VARIANTS

    log.info("1. Tuning: the hybrid multi-task network, every setting on the user-group folds")
    rows, curves = [], []
    for number, setting in enumerate(settings, start=1):
        start = time.time()
        model = make_network(variants["hybrid_mlp"], setting, epochs)
        scores = dl.cross_validate_epochs(model, train, folds, checkpoints)
        last = dl.epoch_summary(scores).iloc[-1]  # the scores after the last epoch
        rows.append(
            {
                **setting,
                **last[["building_acc", "floor_acc", "mean_2d"]].to_dict(),
                **last[["building_acc_std", "floor_acc_std", "mean_2d_std"]].to_dict(),
            }
        )
        log.info(
            f"   {number}/{len(settings)} {setting}: {one_line(rows[-1])} "
            f"({time.time() - start:.0f} s)"
        )
    tuning = pd.DataFrame(rows)
    pick = int(tuning["mean_2d"].idxmin())  # lowest CV mean 2D error; on a tie the earlier one
    tuning["chosen"] = [row == pick for row in range(len(tuning))]
    save_csv(tuning, folder["tuning"] / "hybrid_mlp.csv")
    best = settings[pick]
    fixed = {key: value for key, value in config.DL_FIXED.items() if key not in best}
    save_json(
        {"hybrid_mlp": best, "fixed": fixed | {"epochs": epochs}},
        folder["tuning"] / "best_params.json",
    )
    log.info(f"   best setting: {best}")

    log.info(
        "2. Ablation study: the tuned network, then one change at a time (new seed, same folds)"
    )
    fold_scores, summaries = {}, {}
    for key, variant in variants.items():
        start = time.time()
        model = make_network(variant, best, epochs, seed=config.DL_ABLATION_SEED)
        scores = dl.cross_validate_epochs(model, train, folds, checkpoints)
        fold_scores[key] = scores[scores["epoch"] == epochs].reset_index(drop=True)
        summary = dl.epoch_summary(scores).assign(variant=key)
        curves.append(summary)
        summaries[key] = summary.iloc[-1].to_dict()
        log.info(f"   {variant['label']}: {one_line(summaries[key])} ({time.time() - start:.0f} s)")
    curves = pd.concat(curves, ignore_index=True)
    save_csv(curves, folder["metrics"] / "learning_curves.csv")
    final_errors = fold_scores["hybrid_mlp"]["mean_2d"]
    ablation = pd.DataFrame(
        [
            {
                "variant": key,
                "label": variant["label"],
                **{m: summaries[key][m] for m in HEADLINE},
                **{f"{m}_std": summaries[key][f"{m}_std"] for m in HEADLINE},
                "change_mean_2d": summaries[key]["mean_2d"] - summaries["hybrid_mlp"]["mean_2d"],
                "folds_worse": int((fold_scores[key]["mean_2d"] > final_errors).sum()),
            }
            for key, variant in variants.items()
        ]
    )
    save_csv(ablation, folder["metrics"] / "ablation.csv")

    log.info("   CV seed check: the full network again with other seeds (same folds)")
    noise = [{"seed": config.DL_ABLATION_SEED, **{m: summaries["hybrid_mlp"][m] for m in HEADLINE}}]
    for seed in cv_seeds:
        model = make_network(variants["hybrid_mlp"], best, epochs, seed=seed)
        cv = dl.epoch_summary(dl.cross_validate_epochs(model, train, folds, [epochs])).iloc[-1]
        noise.append({"seed": seed, **{m: cv[m] for m in HEADLINE}})
        log.info(f"   seed {seed} (CV): {one_line(noise[-1])}")
    noise = pd.DataFrame(noise)
    spread = noise.drop(columns="seed").agg(["mean", "std"]).reset_index(names="seed")
    save_csv(pd.concat([noise, spread], ignore_index=True), folder["metrics"] / "cv_seeds.csv")

    log.info("3. Final network: fixed in advance, the hybrid multi-task network (our design)")
    candidates = ablation.set_index("variant").loc[config.DL_CANDIDATES]
    final = config.DL_FINAL
    choice = {"final_model": final, "name": variants[final]["name"]}
    choice["rule"] = (
        "fixed before the test set was used: the hybrid multi-task network (one shared trunk, "
        "three heads), with the setting of lowest CV mean 2D error; the other candidates are "
        "tested next to it"
    )
    for metric in HEADLINE:
        choice[f"cv_{metric}"] = candidates[metric].round(4).to_dict()
    save_json(choice, folder["metrics"] / "final_model.json")
    log.info(f"   final network: {choice['name']}")

    log.info("4. Test evaluation: fit on all training scans, predict the test scans once")
    test_rows, breakdowns, phone_rows, tables = [], [], [], {}

    def record(key: str, name: str, prediction: pd.DataFrame) -> None:
        """Score one model's test predictions and keep its tables (written after stage 6)."""
        errors = metrics.per_scan_errors(test, prediction)
        true = test[config.TARGET_COLS].add_prefix("true_")
        parts = [test[["PHONEID"]], true, prediction.add_prefix("pred_"), errors[["err_2d"]]]
        tables[key] = pd.concat(parts, axis=1)
        save_csv(tables[key], folder["predictions"] / f"{key}.csv")
        test_rows.append({"model": key, "name": name, **metrics.summarize(errors)})
        for table, rows in (
            (metrics.error_breakdown(errors), breakdowns),
            (metrics.per_phone(errors, test["PHONEID"], train["PHONEID"]), phone_rows),
        ):
            table.insert(0, "model", key)
            rows.append(table)
        log.info(f"   {name}: {one_line(test_rows[-1])}")

    for key in config.DL_CANDIDATES:
        model = make_network(variants[key], best, epochs)
        prediction = pipeline.evaluate_on_test(model, train, test)
        record(key, variants[key]["name"], prediction)
        if key == final:
            final_network = model  # fitted on all training scans; reused by the blend
            confusion = metrics.floor_confusion(test, prediction)
    save_csv(confusion, folder["metrics"] / "floor_confusion.csv")

    log.info("5. Seed check: the final network trained again with other random seeds")
    seed_rows = []
    for seed in seeds:
        model = make_network(variants[final], best, epochs, seed=seed)
        errors = metrics.per_scan_errors(test, pipeline.evaluate_on_test(model, train, test))
        seed_rows.append({"seed": seed, **metrics.summarize(errors)})
        log.info(f"   seed {seed}: {one_line(seed_rows[-1])}")
    seed_table = pd.DataFrame(seed_rows)
    spread = seed_table.drop(columns="seed").agg(["mean", "std"]).reset_index(names="seed")
    save_csv(pd.concat([seed_table, spread], ignore_index=True), folder["metrics"] / "seeds.csv")

    log.info("6. Hybrid: blend the Phase 1 Random Forest with the final network (weights by CV)")
    forest_params = json.loads((phase1 / "tuning" / "best_params.json").read_text())["rf"]
    if quick:  # a small forest, as in Phase 1's check run
        forest_params = {
            task: setting | {"n_estimators": config.QUICK_TREES}
            for task, setting in forest_params.items()
        }

    def make_forest():
        return LocalizationModel("rf", config.MODEL_FEATURES["rf"], forest_params)

    forest_cv = hybrid.out_of_fold_outputs(make_forest, hybrid.forest_outputs, train, folds)
    # Why the ablation seed: the tuning runs with the default seed won the tuning, which would
    # favor the network; with this seed, the network alone (weight 0) is the ablation's full network
    network_cv = hybrid.out_of_fold_outputs(
        lambda: make_network(variants[final], best, epochs, seed=config.DL_ABLATION_SEED),
        dl_outputs,
        train,
        folds,
    )
    weight_table = hybrid.weight_table(forest_cv, network_cv, train, folds, config.BLEND_WEIGHTS)
    weights = hybrid.choose_weights(weight_table)
    for task in hybrid.METRIC:
        weight_table[f"chosen_{task}"] = weight_table["forest_weight"] == weights[task]
    save_csv(weight_table, folder["tuning"] / "blend.csv")
    blend_cv = {
        metric: float(weight_table.loc[weight_table[f"chosen_{task}"], metric].iloc[0])
        for task, metric in hybrid.METRIC.items()
    }
    save_json({"forest_weights": weights, "cv": blend_cv}, folder["tuning"] / "blend.json")
    log.info(f"   forest weight per task: {weights}; CV: {one_line(blend_cv)}")
    forest = make_forest().fit(train[config.WAP_COLS], train[config.TARGET_COLS])
    outputs = [
        hybrid.forest_outputs(forest, test[config.WAP_COLS]),
        dl_outputs(final_network, test[config.WAP_COLS]),
    ]
    record("blend", config.BLEND["name"], hybrid.blend(*outputs, weights, test.index))
    test_results = pd.DataFrame(test_rows)
    save_csv(test_results, folder["metrics"] / "test_results.csv")
    save_csv(pd.concat(breakdowns), folder["metrics"] / "error_breakdown.csv")
    save_csv(pd.concat(phone_rows), folder["metrics"] / "per_phone.csv")

    log.info("7. Comparison with Phase 1: CV and test scores of all models")
    phase1_cv = pd.read_csv(phase1 / "metrics" / "cv_results.csv")
    phase1_test = pd.read_csv(phase1 / "metrics" / "test_results.csv")
    dl_cv = ablation.set_index("variant").loc[config.DL_CANDIDATES].reset_index(names="model")
    dl_cv.insert(1, "name", [variants[key]["name"] for key in config.DL_CANDIDATES])
    columns = ["model", "name", *HEADLINE]
    blend_row = pd.DataFrame([{"model": "blend", "name": config.BLEND["name"], **blend_cv}])
    comparison_cv = pd.concat([phase1_cv[columns], dl_cv[columns], blend_row], ignore_index=True)
    comparison_test = pd.concat([phase1_test, test_results], ignore_index=True)
    comparison = comparison_test.merge(comparison_cv, on=["model", "name"], suffixes=("", "_cv"))
    comparison.insert(2, "phase", [1] * len(phase1_test) + [2] * len(test_results))
    save_csv(comparison, folder["metrics"] / "comparison.csv")

    log.info("8. Figures")
    short = {**PHASE1_SHORT, "blend": config.BLEND["short"]}
    labels = [short.get(key) or variants[key]["short"] for key in comparison["model"]]
    figures = {
        "dl_learning_curves": plots.learning_curves(
            curves, {key: variants[key]["label"] for key in config.DL_CURVES}
        ),
        "dl_ablation": plots.ablation(ablation),
        "aml_vs_dl": plots.model_comparison(
            comparison_cv, comparison_test, names=labels, split=len(phase1_test)
        ),
        "dl_floor_confusion": plots.floor_confusion(confusion),
        "dl_error_map": plots.error_map(tables[final]),
    }
    for name, figure in figures.items():
        plots.save(figure, name, folder["figures"])


def dl_outputs(model, X: pd.DataFrame) -> dict:
    """Class probabilities and positions of a fitted network (for the blend)."""
    return model.outputs(X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small, fast check run")
    args = parser.parse_args()
    base = config.RESULTS_DIR / "_quick" if args.quick else config.RESULTS_DIR
    out = base / "phase2"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(out / "logs" / "run.log", mode="w")]
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S", handlers=handlers)
    for name in ("run_dl", "indoorloc"):  # our messages only, not those of other packages
        logging.getLogger(name).setLevel(logging.INFO)
    start = time.time()
    train, _ = data.clean_training(data.load_csv(config.TRAIN_CSV))  # the same cleaning as Phase 1
    if args.quick:
        # Why: a check run must not see the test set, so it is tested on held-out training users
        fit_rows, held_out_rows = pipeline.make_folds(train, "user")[0]
        test = train.iloc[held_out_rows].reset_index(drop=True)
        train = train.iloc[fit_rows].reset_index(drop=True)
    else:
        test = data.load_csv(config.TEST_CSV)
    run(train, test, config.RESULTS_DIR, out, quick=args.quick)
    minutes = (time.time() - start) / 60
    log.info(f"Done in {minutes:.1f} min; results are in {out.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
