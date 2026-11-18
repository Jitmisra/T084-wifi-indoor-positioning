import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conftest import make_campus

from indoorloc import config, pipeline
from indoorloc.config import FILL_VALUE, TARGET_COLS, WAP_COLS
from indoorloc.dl import (
    MultiTaskMLP,
    NetworkInputs,
    NeuralLocalizationModel,
    cross_validate_epochs,
    epoch_summary,
    hide_waps,
)

FAST = {"hidden": (32,), "batch_size": 32}  # a small network keeps the tests fast


def test_inputs_by_hand():
    X = pd.DataFrame({"WAP001": [-40, FILL_VALUE, -100], "WAP002": [FILL_VALUE] * 3})
    X = X.reindex(columns=WAP_COLS, fill_value=FILL_VALUE)
    inputs = NetworkInputs(indicators=True, hybrid=False).fit(X, None)
    assert inputs.waps_ == ["WAP001"] and inputs.min_rssi_ == -101  # 1 dB below -100 dBm
    out = inputs.transform(X)
    # Scaled RSSI (not heard = 0), then the heard indicator
    np.testing.assert_allclose(out, [[61 / 101, 1], [0, 0], [1 / 101, 1]], rtol=1e-6)
    # A test reading weaker than every training reading is clipped to 0, never negative
    weaker = X.iloc[[0]].assign(WAP001=-105)
    assert inputs.transform(weaker)[0, 0] == 0


def test_input_columns_and_training_statistics(campus):
    test = make_campus(scans_per_floor=2, seed=1)
    for indicators, hybrid, extra in ((False, False, 0), (True, False, 0), (True, True, 11)):
        inputs = NetworkInputs(indicators, hybrid).fit(campus[WAP_COLS], campus[TARGET_COLS])
        n_waps = len(inputs.waps_)
        assert inputs.transform(test).shape == (len(test), n_waps * (1 + indicators) + extra)
    # The hybrid columns are standardized with the training mean and spread, not the test's
    hybrid = inputs.transform(campus)[:, -11:]
    np.testing.assert_allclose(hybrid.mean(axis=0), 0, atol=1e-5)
    assert not np.allclose(inputs.transform(test)[:, -11:].mean(axis=0), 0, atol=1e-3)


def test_network_heads():
    net = MultiTaskMLP(20, ["building", "floor", "position"], hidden=(16, 8), dropout=0.2)
    out = net(torch.zeros(4, 20))
    assert {task: tuple(value.shape) for task, value in out.items()} == {
        "building": (4, 3),
        "floor": (4, 5),
        "position": (4, 2),
    }
    assert list(MultiTaskMLP(20, ["floor"], (16,), 0.0)(torch.zeros(1, 20))) == ["floor"]
    # All heads read one shared trunk: one trunk pass per forward, and changing the trunk
    # changes every head's output
    calls = []
    net.trunk.register_forward_hook(lambda module, args, output: calls.append(output))
    x = torch.randn(4, 20)
    before = net.eval()(x)
    assert len(calls) == 1
    with torch.no_grad():
        net.trunk[0].bias.add_(1.0)
    after = net(x)
    assert all(not torch.equal(before[task], after[task]) for task in before)


def test_hidden_waps_only_change_wap_columns(campus):
    inputs = NetworkInputs().fit(campus[WAP_COLS], campus[TARGET_COLS])
    x = torch.from_numpy(inputs.transform(campus))
    n_waps = len(inputs.waps_)
    hidden = hide_waps(x, n_waps, 1.0, torch.Generator().manual_seed(0))
    assert torch.all(hidden[:, : 2 * n_waps] == 0)  # every WAP hidden: RSSI and indicator 0
    assert torch.equal(hidden[:, 2 * n_waps :], x[:, 2 * n_waps :])  # Phase 1 columns unchanged
    assert torch.equal(hide_waps(x, n_waps, 0.0, torch.Generator()), x)
    # With rate 0.5, about half of the heard WAPs are hidden, and RSSI and indicator agree
    half = hide_waps(x, n_waps, 0.5, torch.Generator().manual_seed(0))
    heard, kept = (x[:, :n_waps] > 0).sum(), (half[:, :n_waps] > 0).sum()
    assert 0.4 < kept / heard < 0.6
    assert torch.equal(half[:, :n_waps] > 0, half[:, n_waps : 2 * n_waps] > 0)


def test_multitask_and_single_task_models_fit_and_predict(campus):
    test = make_campus(scans_per_floor=3, seed=1)
    for multitask in (True, False):
        model = NeuralLocalizationModel(multitask=multitask, params=FAST, epochs=60)
        model.fit(campus[WAP_COLS], campus[TARGET_COLS])
        assert len(model.networks_) == (1 if multitask else 3) and len(model.history_) == 60
        assert model.history_[-1] < model.history_[0]  # the training loss went down
        prediction = model.predict(test[WAP_COLS])
        assert list(prediction.columns) == TARGET_COLS and prediction.index.equals(test.index)
        assert set(prediction["FLOOR"]) <= {0, 1, 2, 3, 4}
        assert (prediction["BUILDINGID"] == test["BUILDINGID"]).mean() > 0.9
        assert (prediction["FLOOR"] == test["FLOOR"]).mean() > 0.45  # 5 floors: chance is 0.2
        errors = np.hypot(
            prediction["LONGITUDE"] - test["LONGITUDE"], prediction["LATITUDE"] - test["LATITUDE"]
        )
        assert errors.mean() < 25  # guessing the campus mean gives about 85 m here


def test_position_weight_and_cosine_schedule(campus):
    predictions = []
    for weight in (1.0, 300.0):
        model = NeuralLocalizationModel(params=FAST | {"position_weight": weight}, epochs=4)
        rates = []

        def record(epoch, fitted, rates=rates):
            rates.append(fitted.optimizers_[0].param_groups[0]["lr"])

        model.fit(campus[WAP_COLS], campus[TARGET_COLS], after_epoch=record)
        predictions.append(model.predict(campus[WAP_COLS]))
        # Cosine schedule: after epoch e of 4 the rate is 0.001 * (1 + cos(pi * e / 4)) / 2
        expected = [0.0005 * (1 + np.cos(np.pi * e / 4)) for e in range(1, 5)]
        np.testing.assert_allclose(rates, expected, atol=1e-9)
    # The position weight changes what the shared layers learn
    xy = ["LONGITUDE", "LATITUDE"]
    assert not predictions[0][xy].equals(predictions[1][xy])


def test_same_seed_gives_the_same_model(campus):
    predictions = []
    for seed in (7, 7, 8):
        model = NeuralLocalizationModel(params=FAST, epochs=3, seed=seed)
        model.fit(campus[WAP_COLS], campus[TARGET_COLS])
        predictions.append(model.predict(campus[WAP_COLS]))
    pd.testing.assert_frame_equal(predictions[0], predictions[1])
    assert not predictions[0].equals(predictions[2])


def test_predicted_positions_are_turned_back_into_meters(campus):
    model = NeuralLocalizationModel(params=FAST, epochs=1)
    model.fit(campus[WAP_COLS], campus[TARGET_COLS])
    head = model.networks_[0].heads["position"]
    with torch.no_grad():  # a head that always outputs 0 (standardized) = the training mean
        head.weight.zero_()
        head.bias.zero_()
    prediction = model.predict(campus[WAP_COLS])
    means = campus[["LONGITUDE", "LATITUDE"]].mean().to_numpy()
    np.testing.assert_allclose(
        prediction[["LONGITUDE", "LATITUDE"]].to_numpy(), [means] * len(campus)
    )
    with torch.no_grad():  # an output of 1 is one training standard deviation above the mean
        head.bias.fill_(1.0)
    shifted = model.predict(campus[WAP_COLS])[["LONGITUDE", "LATITUDE"]].to_numpy()
    spread = campus[["LONGITUDE", "LATITUDE"]].std().to_numpy()
    np.testing.assert_allclose(shifted, [means + spread] * len(campus))


def test_dropout_is_on_again_after_a_prediction_during_training(campus):
    model = NeuralLocalizationModel(params=FAST, epochs=3)
    modes = []

    def predict_and_watch(epoch, fitted):
        if epoch == 1:  # from now on, record the mode of every training forward pass
            fitted.networks_[0].register_forward_pre_hook(
                lambda net, args: modes.append(net.training) if torch.is_grad_enabled() else None
            )
        fitted.predict(campus[WAP_COLS])  # switches the network to eval mode

    model.fit(campus[WAP_COLS], campus[TARGET_COLS], after_epoch=predict_and_watch)
    assert modes and all(modes)  # epochs 2 and 3 trained with dropout on


class Oracle:
    """A stand-in model: after e epochs it predicts the true labels, with x shifted by e meters."""

    def __init__(self, truth):
        self.truth = truth

    def fit(self, X, y, after_epoch=None):
        self.trained_on = list(X.index)
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            after_epoch(epoch, self)
        return self

    def predict(self, X):
        assert not set(X.index) & set(self.trained_on)  # only rows it was not trained on
        prediction = self.truth.loc[X.index, TARGET_COLS].copy()
        prediction["LONGITUDE"] += self.epoch
        return prediction


def test_epoch_cross_validation_scores_held_out_rows_after_each_checkpoint(campus):
    folds = pipeline.make_folds(campus, "user", 3)
    scores = cross_validate_epochs(Oracle(campus), campus, folds, [1, 3])
    assert scores["building_acc"].eq(1).all() and scores["floor_acc"].eq(1).all()
    np.testing.assert_allclose(scores["mean_2d"], scores["epoch"])  # error = epochs trained


class KeepInputs(NeuralLocalizationModel):
    """A model that remembers the AP-affinity profiles of its inputs in every fit."""

    def fit(self, X, y, after_epoch=None):
        super().fit(X, y, after_epoch)
        self.profiles = getattr(self, "profiles", []) + [self.inputs_.affinity_.profiles_]
        return self


def test_epoch_cross_validation_keeps_held_out_labels_away(campus):
    folds = pipeline.make_folds(campus, "user", 3)
    held_out = folds[0][1]
    shuffled = campus.copy()
    labels = campus.loc[held_out, TARGET_COLS]
    shuffled.loc[held_out, TARGET_COLS] = labels.sample(frac=1, random_state=1).set_axis(held_out)
    models = [KeepInputs(params=FAST) for _ in range(2)]
    scans = (campus, shuffled)
    scores = [
        cross_validate_epochs(m, s, folds, [1, 2]) for m, s in zip(models, scans, strict=True)
    ]
    # Fold 0 holds those scans out, so its inputs stay the same; fold 1 trains on them
    np.testing.assert_array_equal(models[0].profiles[0], models[1].profiles[0])
    assert not np.array_equal(models[0].profiles[1], models[1].profiles[1])
    assert scores[0][["fold", "epoch"]].values.tolist() == [
        [f, e] for f in (1, 2, 3) for e in (1, 2)
    ]
    summary = epoch_summary(scores[0])
    assert summary["epoch"].tolist() == [1, 2] and "mean_2d_std" in summary


def write_forest_settings(phase1: Path) -> None:
    """Phase 1's tuned Random Forest settings, as `make run` writes them (for the blend)."""
    (phase1 / "tuning").mkdir(parents=True, exist_ok=True)
    setting = {"max_features": "sqrt", "min_samples_leaf": 1}
    tuned = {"rf": dict.fromkeys(["building", "floor", "position"], setting)}
    (phase1 / "tuning" / "best_params.json").write_text(json.dumps(tuned))


def test_quick_run_writes_every_result(tmp_path, campus):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dl.py"
    run = runpy.run_path(str(script))["run"]  # the script's run() function, without main()
    test = make_campus(scans_per_floor=4, seed=1)
    test.loc[test["USERID"] > 4, "PHONEID"] += 1000  # phones never used in training
    # Phase 1's tables, as `make run` writes them (only the columns the comparison reads)
    phase1 = tmp_path / "phase1"
    (phase1 / "metrics").mkdir(parents=True)
    names = {"baseline_rf": "Baseline RF", "knn": "kNN", "rf": "Random Forest", "xgb": "XGBoost"}
    scores = {"building_acc": 1.0, "floor_acc": 0.9, "mean_2d": 8.0}
    rows = [{"model": key, "name": name, **scores} for key, name in names.items()]
    pd.DataFrame(rows).to_csv(phase1 / "metrics" / "cv_results.csv", index=False)
    pd.DataFrame(rows).to_csv(phase1 / "metrics" / "test_results.csv", index=False)
    write_forest_settings(phase1)
    out = tmp_path / "phase2"
    run(campus, test, phase1, out, quick=True)
    for name in ["learning_curves", "ablation", "test_results", "error_breakdown", "per_phone"]:
        assert (out / "metrics" / f"{name}.csv").exists()
    for name in ["floor_confusion", "seeds", "comparison"]:
        assert (out / "metrics" / f"{name}.csv").exists()
    ablation = pd.read_csv(out / "metrics" / "ablation.csv")
    assert ablation["variant"].tolist() == list(config.DL_VARIANTS)
    for model in [*config.DL_CANDIDATES, "blend"]:
        assert len(pd.read_csv(out / "predictions" / f"{model}.csv")) == len(test)
    blend = json.loads((out / "tuning" / "blend.json").read_text())
    assert set(blend["forest_weights"]) == {"building", "floor", "position"}
    weights = pd.read_csv(out / "tuning" / "blend.csv")
    assert weights["forest_weight"].tolist() == config.BLEND_WEIGHTS
    assert all(weights[f"chosen_{task}"].sum() == 1 for task in ("building", "floor", "position"))
    figures = ["dl_learning_curves", "dl_ablation", "aml_vs_dl", "dl_floor_confusion"]
    for name in [*figures, "dl_error_map"]:
        assert (out / "figures" / f"{name}.png").exists()
        assert (out / "figures" / f"{name}.pdf").exists()
    # The final network is fixed in advance; its CV scores and those of the candidates are kept
    final = json.loads((out / "metrics" / "final_model.json").read_text())
    assert final["final_model"] == config.DL_FINAL == "hybrid_mlp"
    candidates = ablation.set_index("variant").loc[config.DL_CANDIDATES, "mean_2d"].round(4)
    assert final["cv_mean_2d"] == candidates.to_dict()
    tuning = pd.read_csv(out / "tuning" / "hybrid_mlp.csv")
    assert tuning["chosen"].sum() == 1
    assert tuning.loc[tuning["chosen"], "mean_2d"].item() == tuning["mean_2d"].min()
    comparison = pd.read_csv(out / "metrics" / "comparison.csv")
    assert comparison["phase"].tolist() == [1] * 4 + [2] * (len(config.DL_CANDIDATES) + 1)
    assert comparison["model"].tolist()[-1] == "blend"
    assert ablation["mean_2d"].nunique() > 1  # the variants really differ
    seeds = pd.read_csv(out / "metrics" / "seeds.csv")
    assert seeds["seed"].tolist()[:-2] == [str(seed) for seed in config.DL_SEEDS[:2]]
    assert seeds["seed"].tolist()[-2:] == ["mean", "std"]
    cv_seeds = pd.read_csv(out / "metrics" / "cv_seeds.csv")
    assert cv_seeds["seed"].tolist()[0] == str(config.DL_ABLATION_SEED)
    phones = pd.read_csv(out / "metrics" / "per_phone.csv")
    assert set(phones["phones"]) == {"seen in training", "new"}
    confusion = pd.read_csv(out / "metrics" / "floor_confusion.csv")
    assert confusion.filter(like="pred_floor_").to_numpy().sum() == len(test)


def test_choices_never_depend_on_the_test_set(tmp_path, campus):
    run = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_dl.py"))["run"]
    phase1 = tmp_path / "phase1"
    (phase1 / "metrics").mkdir(parents=True)
    rows = [
        {"model": k, "name": k, "building_acc": 1.0, "floor_acc": 0.9, "mean_2d": 8.0}
        for k in ("baseline_rf", "knn", "rf", "xgb")
    ]
    for name in ("cv_results", "test_results"):
        pd.DataFrame(rows).to_csv(phase1 / "metrics" / f"{name}.csv", index=False)
    write_forest_settings(phase1)
    test = make_campus(scans_per_floor=4, seed=1)
    test.loc[test["USERID"] > 4, "PHONEID"] += 1000  # phones never used in training
    # Scans the cleaning rules would remove: the test file must be used exactly as it is
    test.loc[0, "WAP001"] = -20
    test = pd.concat([test, test.iloc[[1]]], ignore_index=True)
    shuffled = test.copy()
    shuffled[TARGET_COLS] = test[TARGET_COLS].sample(frac=1, random_state=0).to_numpy()
    for scans, out in ((test, tmp_path / "a"), (shuffled, tmp_path / "b")):
        run(campus, scans, phase1, out, quick=True)
        assert len(pd.read_csv(out / "predictions" / "hybrid_mlp.csv")) == len(test)
    # Other test labels change the test scores, but no choice and no CV number
    choices = ["tuning/hybrid_mlp.csv", "tuning/best_params.json", "metrics/ablation.csv"]
    for name in [*choices, "tuning/blend.csv", "tuning/blend.json"]:
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()
    for name in ("cv_seeds.csv", "learning_curves.csv"):
        assert (tmp_path / "a" / "metrics" / name).read_text() == (
            tmp_path / "b" / "metrics" / name
        ).read_text()
    scores = [pd.read_csv(tmp_path / x / "metrics" / "test_results.csv") for x in "ab"]
    assert not scores[0]["mean_2d"].equals(scores[1]["mean_2d"])
    # The trained networks never saw the test labels, so their predictions are the same
    for model in [*config.DL_CANDIDATES, "blend"]:
        tables = [pd.read_csv(tmp_path / x / "predictions" / f"{model}.csv") for x in "ab"]
        pd.testing.assert_frame_equal(
            tables[0].filter(like="pred_"), tables[1].filter(like="pred_")
        )
