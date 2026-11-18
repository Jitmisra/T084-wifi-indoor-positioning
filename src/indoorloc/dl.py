"""Phase 2: neural networks that predict the building, floor and position of a scan.

NetworkInputs turns scans into the network's input columns, learned from the training scans only.
MultiTaskMLP is the network: shared hidden layers (the trunk), then one output layer (head) per
task. NeuralLocalizationModel trains it and has the same fit/predict methods as the Phase 1
LocalizationModel, so the networks are evaluated with exactly the same code and metrics.

Do not train XGBoost in a process that imported this module: PyTorch and XGBoost each bring their
own copy of the OpenMP library, which crashes on macOS (see models.make_estimator).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import (
    COORD_COLS,
    DL_EPOCHS,
    DL_FIXED,
    FILL_VALUE,
    FLOORS_PER_BUILDING,
    SEED,
    TARGET_COLS,
    WAP_COLS,
)
from .data import heard_waps
from .features import APAffinity, signal_stats
from .metrics import per_scan_errors, summarize
from .models import TASKS

# Output size of each head: 3 buildings, 5 floors (0-4) and 2 coordinates (x, y)
OUTPUTS = {
    "building": len(FLOORS_PER_BUILDING),
    "floor": max(FLOORS_PER_BUILDING.values()),
    "position": 2,
}
LOSSES = {
    "building": nn.CrossEntropyLoss(),
    "floor": nn.CrossEntropyLoss(),
    "position": nn.MSELoss(),
}


class NetworkInputs:
    """The network's input columns, learned from the training scans only.

    For every WAP heard in training: its scaled RSSI, (rssi - min) / -min, between 0 and 1
    (not heard = 0, -40 dBm = 0.62), and a heard indicator (1 = heard, 0 = not heard). With
    hybrid=True the Phase 1 features are added: the 3 signal statistics and the 8 AP-affinity
    columns, standardized with the mean and standard deviation of the training scans.
    """

    def __init__(self, indicators: bool = True, hybrid: bool = True):
        self.indicators = indicators
        self.hybrid = hybrid

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "NetworkInputs":
        self.waps_ = heard_waps(X)
        values = X[self.waps_].to_numpy(dtype=float)
        # Why 1 dB below the weakest reading: that reading then still counts a little, not 0
        self.min_rssi_ = values[values > FILL_VALUE].min() - 1
        if self.hybrid:
            self.affinity_ = APAffinity().fit(X[self.waps_], y)
            extra = self._phase1_features(X)
            self.mean_ = extra.mean(axis=0)
            self.std_ = np.where(extra.std(axis=0) > 0, extra.std(axis=0), 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.waps_].to_numpy(dtype=float)
        heard = values > FILL_VALUE
        scaled = np.clip((values - self.min_rssi_) / -self.min_rssi_, 0, 1)
        blocks = [np.where(heard, scaled, 0.0)]
        if self.indicators:
            blocks.append(heard.astype(float))
        if self.hybrid:
            blocks.append((self._phase1_features(X) - self.mean_) / self.std_)
        return np.hstack(blocks).astype(np.float32)

    def _phase1_features(self, X: pd.DataFrame) -> np.ndarray:
        rssi = X[self.waps_]
        return np.hstack([signal_stats(rssi).to_numpy(), self.affinity_.transform(rssi).to_numpy()])


class MultiTaskMLP(nn.Module):
    """Shared hidden layers (linear -> ReLU -> dropout), then one linear head per task."""

    def __init__(self, n_inputs: int, tasks: list[str], hidden: tuple, dropout: float):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({task: nn.Linear(width, OUTPUTS[task]) for task in tasks})

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.trunk(x)  # every head reads the same shared representation
        return {task: head(shared) for task, head in self.heads.items()}


class NeuralLocalizationModel:
    """Predict building, floor and position with neural networks, like LocalizationModel.

    multitask=True trains one network with three heads (our Phase 2 model); multitask=False
    trains three separate networks with one head each (single-task, for the comparison).
    hybrid and indicators choose the input columns (see NetworkInputs); `params` replaces values
    of DL_FIXED; `epochs` is the number of passes over the training scans.
    """

    def __init__(
        self,
        hybrid: bool = True,
        multitask: bool = True,
        indicators: bool = True,
        params: dict | None = None,
        epochs: int = DL_EPOCHS,
        seed: int = SEED,
    ):
        self.hybrid = hybrid
        self.multitask = multitask
        self.indicators = indicators
        self.params = DL_FIXED | (params or {})
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, after_epoch=None) -> "NeuralLocalizationModel":
        """Train for self.epochs epochs; after_epoch(epoch, self) is called after each epoch."""
        torch.manual_seed(self.seed)  # same seed -> same starting weights, batches and hidden WAPs
        generator = torch.Generator().manual_seed(self.seed)
        self.inputs_ = NetworkInputs(self.indicators, self.hybrid).fit(X, y)
        inputs = torch.from_numpy(self.inputs_.transform(X))
        # Why standardized coordinates: outputs near 0 with spread 1 train well; predict() turns
        # them back into meters
        self.origin_ = y[COORD_COLS].mean().to_numpy()
        self.scale_ = y[COORD_COLS].std().to_numpy()
        position = (y[COORD_COLS].to_numpy() - self.origin_) / self.scale_
        targets = {  # torch.tensor copies the values (pandas may hand out read-only arrays)
            "building": torch.tensor(y["BUILDINGID"].to_numpy(), dtype=torch.long),
            "floor": torch.tensor(y["FLOOR"].to_numpy(), dtype=torch.long),
            "position": torch.tensor(position, dtype=torch.float32),
        }
        p = self.params
        groups = [TASKS] if self.multitask else [[task] for task in TASKS]
        self.networks_ = [
            MultiTaskMLP(inputs.shape[1], g, p["hidden"], p["dropout"]) for g in groups
        ]
        # AdamW: Adam with weight decay, which pulls every weight a little towards 0 at each step
        optimizers = [
            torch.optim.AdamW(
                net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
            )
            for net in self.networks_
        ]
        # Why a cosine schedule: the learning rate falls smoothly to 0 by the last batch, so the
        # last epochs only make small changes and the result does not jump from epoch to epoch
        steps = self.epochs * -(-len(inputs) // p["batch_size"])  # batches in the whole training
        schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=steps) for o in optimizers]
        self.optimizers_ = optimizers  # kept so that the learning rate can be checked
        self.history_ = []  # mean training loss of every epoch (summed over the networks)
        for epoch in range(1, self.epochs + 1):
            loss = sum(
                self._train_one_epoch(net, optimizer, schedule, inputs, targets, generator)
                for net, optimizer, schedule in zip(
                    self.networks_, optimizers, schedules, strict=True
                )
            )
            self.history_.append(loss)
            if after_epoch is not None:
                after_epoch(epoch, self)
        return self

    def outputs(self, X: pd.DataFrame) -> dict:
        """Class probabilities (softmax) of building and floor, and positions in meters."""
        inputs = torch.from_numpy(self.inputs_.transform(X))
        raw = {}
        with torch.no_grad():
            for net in self.networks_:
                net.eval()  # no dropout when predicting
                raw |= net(inputs)
        return {
            "building": torch.softmax(raw["building"], dim=1).numpy(),
            "floor": torch.softmax(raw["floor"], dim=1).numpy(),
            "position": raw["position"].numpy() * self.scale_ + self.origin_,
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        outputs = self.outputs(X)
        prediction = {
            "BUILDINGID": outputs["building"].argmax(axis=1),  # the most likely class
            "FLOOR": outputs["floor"].argmax(axis=1),
            "LONGITUDE": outputs["position"][:, 0],
            "LATITUDE": outputs["position"][:, 1],
        }
        return pd.DataFrame(prediction, index=X.index)

    def _train_one_epoch(self, net, optimizer, schedule, inputs, targets, generator) -> float:
        """One pass over the training scans in random mini-batches; returns the mean loss."""
        net.train()  # dropout on
        p = self.params
        weights = {"building": 1.0, "floor": 1.0, "position": p["position_weight"]}
        order = torch.randperm(len(inputs), generator=generator)
        total = 0.0
        for start in range(0, len(inputs), p["batch_size"]):
            rows = order[start : start + p["batch_size"]]
            x = hide_waps(
                inputs[rows],
                len(self.inputs_.waps_),
                p["hide_wap_rate"],
                generator,
                self.indicators,
            )
            outputs = net(x)
            # The loss of a network is the weighted sum of the losses of its heads
            loss = sum(weights[t] * LOSSES[t](outputs[t], targets[t][rows]) for t in outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()  # the next batch uses a slightly smaller learning rate
            total += loss.item() * len(rows)
        return total / len(inputs)


def hide_waps(
    x: torch.Tensor, n_waps: int, rate: float, generator: torch.Generator, indicators: bool = True
) -> torch.Tensor:
    """Training only: mark a random share (`rate`) of the heard WAPs of a batch as not heard.

    The first n_waps columns are the scaled RSSI values and, with indicators, the next n_waps
    columns the heard indicators; both are set to 0 for a hidden WAP. Other columns stay as
    they are. Why: a new phone, or the same phone on another day, misses some WAPs; hiding WAPs
    at random teaches the network not to depend on any single WAP (dropout on the input).
    """
    if rate == 0:
        return x
    keep = (torch.rand(len(x), n_waps, generator=generator) >= rate).to(x.dtype)
    x = x.clone()
    x[:, :n_waps] *= keep  # scaled RSSI -> 0 (not heard)
    if indicators:
        x[:, n_waps : 2 * n_waps] *= keep  # indicator -> 0 (not heard)
    return x


def cross_validate_epochs(
    model, train: pd.DataFrame, folds: list, checkpoints: list
) -> pd.DataFrame:
    """Like pipeline.cross_validate, but one training run per fold is scored after several
    numbers of epochs (the checkpoints). Returns one row of metrics per fold and checkpoint.

    The model, inputs included, is fitted again in every fold, so held-out scans never influence
    their own inputs.
    """
    rows = []
    model.epochs = max(checkpoints)
    for fold, (fit_rows, held_out_rows) in enumerate(folds, start=1):
        fit_part, held_out = train.iloc[fit_rows], train.iloc[held_out_rows]

        def score(epoch, fitted, fold=fold, held_out=held_out):
            if epoch in checkpoints:
                prediction = fitted.predict(held_out[WAP_COLS])
                errors = summarize(per_scan_errors(held_out, prediction))
                rows.append({"fold": fold, "epoch": epoch, **errors})

        model.fit(fit_part[WAP_COLS], fit_part[TARGET_COLS], after_epoch=score)
    return pd.DataFrame(rows)


def epoch_summary(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the folds of every metric, one row per epoch count."""
    metrics = ["building_acc", "floor_acc", "mean_2d"]
    grouped = fold_scores.groupby("epoch")[metrics]
    return grouped.mean().join(grouped.std().add_suffix("_std")).reset_index()
