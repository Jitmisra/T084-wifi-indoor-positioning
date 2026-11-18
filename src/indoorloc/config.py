"""Paths and fixed settings of the project, all defined in this one file."""

import math
from pathlib import Path

# Paths (this file is ROOT/src/indoorloc/config.py)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "UJIndoorLoc"
TRAIN_CSV = RAW_DIR / "trainingData.csv"
TEST_CSV = RAW_DIR / "validationData.csv"  # the UCI "validation" file is our test set
RESULTS_DIR = ROOT / "results"

# Where the data comes from (UCI Machine Learning Repository, CC BY 4.0) and the files' SHA-256
DATA_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
SHA256 = {
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
}

# Columns. Positions are Web-Mercator meters; 1 m in the data is about 0.766 m on the ground.
WAP_COLS = [f"WAP{i:03d}" for i in range(1, 521)]
TARGET_COLS = ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]
COORD_COLS = ["LONGITUDE", "LATITUDE"]
FLOORS_PER_BUILDING = {0: 4, 1: 4, 2: 5}

# Signal strengths (RSSI, in dBm)
NOT_HEARD = 100  # code used in the CSV files for "this WAP was not heard in the scan"
FILL_VALUE = -110  # Why: weaker than every real reading (-104 dBm), so "not heard" = "very weak"
HIGH_RSSI = -30  # Why: readings this strong are physically implausible; most come from one phone

# AP-affinity features
MIN_SUPPORT = 5  # a WAP must be heard in this many training scans before we trust its profile
# Why e: the exponent Torres-Sospedra et al. (2015) suggest for their "powed" representation
WEIGHT_EXPONENT = math.e

# Cross-validation and tuning
SEED = 42
N_FOLDS = 5
FEATURE_SETS = ["rssi", "rssi+stats", "rssi+stats+affinity"]
RF_TREES = 300  # Why not tuned: more trees make a forest slower, but not worse
# Why fixed: common values; we tune only the depth and the number of trees to keep the grid small
XGB_FIXED = {
    "learning_rate": 0.1,
    "subsample": 0.85,
    "colsample_bytree": 0.5,
    "tree_method": "hist",
}
# Why small grids: every setting costs one full 5-fold cross-validation
GRIDS = {
    "knn": {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "rf": {"max_features": ["sqrt", 0.2, 0.5], "min_samples_leaf": [1, 4]},
    "xgb": {"max_depth": [4, 6, 10], "n_estimators": [300, 600]},
}

# The four models we compare: display name, model family and feature set
MODEL_NAMES = {"baseline_rf": "Baseline RF", "knn": "kNN", "rf": "Random Forest", "xgb": "XGBoost"}
MODEL_FAMILY = {"baseline_rf": "rf", "knn": "knn", "rf": "rf", "xgb": "xgb"}
MODEL_FEATURES = {
    "baseline_rf": "rssi+stats",
    "knn": "rssi",
    "rf": "rssi+stats+affinity",
    "xgb": "rssi+stats+affinity",
}

# Phase 2: the neural networks (indoorloc.dl). Settings in DL_GRID are tuned by CV; DL_FIXED
# holds the rest (a tuned value replaces the fixed one)
DL_FIXED = {
    "hidden": (512, 256, 128),  # three shared layers, each half as wide as the one before
    "dropout": 0.2,
    "learning_rate": 1e-3,  # at the start; it falls to 0 along a cosine curve (indoorloc.dl)
    "weight_decay": 0.01,  # AdamW's usual default
    "batch_size": 256,
    "position_weight": 100.0,  # weight of the position loss next to the two classification losses
    "hide_wap_rate": 0.1,  # share of the heard WAPs hidden at random in every training batch
}
# Why the position weight: it decides how much the shared layers serve position versus building
# and floor. A 10 m error is only about 0.1 in standardized units, so the position loss needs a
# large weight not to be drowned by the two classification losses.
DL_GRID = {"position_weight": [30.0, 100.0, 300.0]}
DL_EPOCHS = 200  # passes over the training scans
DL_EVERY = 10  # the held-out folds are scored after every 10 epochs (for the learning curves)
DL_SEEDS = [42, 43, 44, 45, 46]  # seed check: the final network trained five times
# Why another seed for the ablation: its reference must not be the very run that won the tuning,
# which would favor it; all ablation networks share this seed, so only the one change differs
DL_ABLATION_SEED = 43
DL_CV_SEEDS = [44, 45]  # CV seed check: the full network with two more seeds, to see CV noise
# The networks of the ablation study: the tuned hybrid multi-task network first, then one change
# at a time. "name" is used in tables, "short" under the bars of figures, "label" in the ablation.
_DL_BASE = {"hybrid": True, "multitask": True, "indicators": True, "params": {}}
DL_VARIANTS = {
    "hybrid_mlp": _DL_BASE | {"name": "Hybrid MLP", "short": "Hybrid\nMLP", "label": "Hybrid MLP"},
    "hybrid_single": _DL_BASE
    | {
        "multitask": False,
        "name": "Hybrid single-task MLPs",
        "short": "Single-\ntask",
        "label": "three single-task networks",
    },
    "mlp": _DL_BASE
    | {"hybrid": False, "name": "MLP", "short": "MLP", "label": "no Phase 1 features"},
    "no_indicators": _DL_BASE | {"indicators": False, "label": "no heard indicators"},
    "no_hidden_waps": _DL_BASE | {"params": {"hide_wap_rate": 0.0}, "label": "no hidden WAPs"},
    "no_regularization": _DL_BASE
    | {"params": {"dropout": 0.0, "weight_decay": 0.0}, "label": "no dropout, no weight decay"},
}
DL_CANDIDATES = ["mlp", "hybrid_single", "hybrid_mlp"]  # the networks tested on the test set
# Why fixed in advance: the brief asks for one shared network with three heads, so this is our
# Phase 2 model; the other two candidates are tested next to it for comparison
DL_FINAL = "hybrid_mlp"
DL_CURVES = ["hybrid_mlp", "no_regularization"]  # the lines of the learning-curve figure
# Phase 2 hybrid (indoorloc.hybrid): forest weights tried for the blend of the Phase 1 Random
# Forest and the final network, per task (0 = the network alone, 1 = the forest alone)
BLEND_WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
BLEND = {"name": "Blend: Random Forest + Hybrid MLP", "short": "Blend"}

# --quick: a small, fast run that only checks that everything works (written to results/_quick/)
QUICK_SCANS = 3000
QUICK_FOLDS = 3
QUICK_SETTINGS = 2  # the first settings of every grid
QUICK_TREES = 20
