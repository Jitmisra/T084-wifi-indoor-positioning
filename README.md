# Indoor Location Prediction Using Wi-Fi Signals

Team T084: we predict the building, the floor and the position of a phone from the Wi-Fi signals
it hears, on the UJIIndoorLoc dataset. Phase 1 uses classical machine learning; Phase 2 a hybrid
multi-task neural network.

## What we do

A Wi-Fi scan lists how strongly a phone hears each of 520 access points (WAPs). From one scan we
predict the building, the floor and the position. We train on UJIIndoorLoc and test on scans
recorded months later, mostly with new phones. In Phase 1, features, settings and the final model
are chosen with user-group cross-validation (CV) on the training data; in Phase 2 the network design
is fixed in advance (the brief's shared network) and CV chooses its position-loss weight and the
blend weights. The test set is used only at the end.
Our own feature, AP-affinity, learns where each WAP "lives" and describes a scan by the WAPs it
hears. Phase 2 trains one neural network with shared layers and three heads (building, floor,
position), compares it with Phase 1 on the same folds, test set and metrics, and blends it with
the Phase 1 Random Forest.

## Results

Test set: the UCI file `validationData.csv` (1,111 scans; 9 of its 11 phones are not in training).

### Phase 2: the hybrid multi-task network

Numbers from [comparison.csv](results/phase2/metrics/comparison.csv) (CV = user-group
cross-validation on the training scans; test = the test set, used only at the end).

| Model | CV mean 2D error | Test building | Test floor | Test mean 2D error | Test median 2D error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest (Phase 1) | 11.09 m | 100.0% | 92.2% | 7.42 m | 5.34 m |
| XGBoost (Phase 1) | 11.25 m | 100.0% | 95.2% | 7.86 m | 6.19 m |
| MLP (Wi-Fi inputs only) | 9.17 m | 100.0% | 90.9% | 8.44 m | 6.33 m |
| Hybrid single-task MLPs | 8.90 m | 100.0% | 94.1% | 8.03 m | 5.75 m |
| **Hybrid MLP** (final network) | 9.21 m | 99.9% | 90.8% | 8.32 m | 6.10 m |
| Blend: Random Forest + Hybrid MLP | 9.14 m | 100.0% | 92.2% | 8.10 m | 5.92 m |

- **The network:** one shared trunk (512, 256 and 128 units) and three heads; inputs are the scaled
  RSSI and a heard/not-heard indicator of each of the 464 WAPs heard in training, plus our Phase 1
  features (the hybrid inputs).
  It was fixed as our Phase 2 model before testing; its position-loss weight (300) was chosen by CV.
- **Five seeds:** 8.21 ± 0.06 m and 91.7 ± 0.7% of floors on the test set
  ([seeds.csv](results/phase2/metrics/seeds.csv)).
- **Ablation** ([ablation.csv](results/phase2/metrics/ablation.csv), CV; chance moves the error by
  ±0.11 m for one run and about ±0.16 m for the difference of two runs,
  [cv_seeds.csv](results/phase2/metrics/cv_seeds.csv)): without dropout and weight decay +1.08 m,
  without heard indicators +0.58 m (but more floors right), without hidden WAPs +0.16 m (at the
  level of chance), without the Phase 1 features −0.04 m (no effect); three single-task networks
  −0.30 m and 6.0 more floor points, with three times the weights.
- **CV and the test set disagree.** CV ranked the networks ahead of the Random Forest (9.21 against
  11.09 m), but on the test set the forest is better (7.42 against 8.32 m).
- **The blend** (hybrid 2, [blend.json](results/phase2/tuning/blend.json)): the Random Forest's and
  the network's probabilities and positions averaged with one forest weight per task, chosen by CV
  (building 0.8, floor 1.0, position 0.1). It beats both models in CV on position (9.14 m) and
  improves the network on the test set (8.10 m, 92.2% of floors), but the forest alone is still
  better on position (7.42 m).

### Phase 1: classical machine learning

Numbers from [test_results.csv](results/metrics/test_results.csv).

| Model | Building accuracy | Floor accuracy | Mean 2D error | Median 2D error |
| --- | ---: | ---: | ---: | ---: |
| Baseline RF | 99.8% | 91.2% | 12.52 m | 7.70 m |
| kNN | 99.8% | 87.5% | 9.22 m | 5.88 m |
| **Random Forest** (final model) | 100.0% | 92.2% | 7.42 m | 5.34 m |
| XGBoost | 100.0% | 95.2% | 7.86 m | 6.19 m |

- **Final model: Random Forest**, by a rule fixed before the test set was used: the tuned model with
  the lowest CV mean 2D error ([cv_results.csv](results/metrics/cv_results.csv): Random Forest
  11.09 m, XGBoost 11.25 m, kNN 12.75 m). No model wins every task: XGBoost predicts floors best.
- **AP-affinity helped a lot.** With the Baseline RF settings it cut the CV mean 2D error from
  15.87 m to 11.38 m and raised floor accuracy from 78.0% to 80.6%; the signal statistics alone did
  not help ([feature_check.csv](results/metrics/feature_check.csv)).
- **Random CV is far too optimistic.** For the Baseline RF it promised a mean 2D error of 3.67 m;
  the test set gave 12.52 m. User-group CV said 15.87 m, stricter than the test set, which is the
  safe side ([cv_check.csv](results/metrics/cv_check.csv)).
- **Units.** Mean 2D error = straight-line distance between the predicted and the true position, in
  the dataset's Web-Mercator meters (one such meter is about 0.77 m on the ground).

## The brief, item by item

Every item of the project brief, the code that does it and the file that shows the result.

| Brief item | Code | Result |
| --- | --- | --- |
| Treat the missing-signal code consistently | `data.load_csv` turns 100 into −110 dBm in both files; `dl.NetworkInputs` turns "not heard" into 0 | [notebook 04](notebooks/04_dl_multitask_network.ipynb), section 2 |
| Signal count and strongest-signal features | `features.signal_stats` (`n_heard`, `max_rssi`, `mean_rssi`) | [feature_check.csv](results/metrics/feature_check.csv) |
| Random Forest / XGBoost for building and floor, a regressor for the position | `models.make_estimator`, `models.LocalizationModel` | [test_results.csv](results/metrics/test_results.csv) |
| The gap: separate models share nothing | `models.LocalizationModel` (three models) vs `dl.MultiTaskMLP` (one shared trunk) | row "three single-task networks" of [ablation.csv](results/phase2/metrics/ablation.csv) |
| Shared MLP trunk, classification heads (building, floor), regression head (position) | `dl.MultiTaskMLP`, `dl.NeuralLocalizationModel` | [test_results.csv](results/phase2/metrics/test_results.csv) |
| Hybrid: sparse signal features into multi-task learning | `dl.NetworkInputs(hybrid=True)`: signal statistics and AP-affinity as network inputs; `hybrid.blend`: the Phase 1 Random Forest blended with the network | rows `mlp` and `hybrid_mlp` of [ablation.csv](results/phase2/metrics/ablation.csv); [blend.csv](results/phase2/tuning/blend.csv); [comparison.csv](results/phase2/metrics/comparison.csv) |
| Fair comparison: the dataset's train/test files, a split by user | `data.user_device_groups`, `pipeline.make_folds("user")`; `validationData.csv` is the test set | [comparison.csv](results/phase2/metrics/comparison.csv) (CV and test, both phases) |
| Do not normalize test data separately | `dl.NetworkInputs.fit` uses training scans only; test scans are only transformed | [notebook 04](notebooks/04_dl_multitask_network.ipynb), section 3; test `test_input_columns_and_training_statistics` |
| Missingness indicators | `dl.NetworkInputs(indicators=True)`: one heard/not-heard column per WAP heard in training (464) | row "no heard indicators" of [ablation.csv](results/phase2/metrics/ablation.csv) |
| Scale observed RSSI values | `dl.NetworkInputs.transform`: (rssi − r_min) / −r_min, between 0 and 1 | [notebook 04](notebooks/04_dl_multitask_network.ipynb), sections 1–2 |
| Keep user and phone ids for robustness | user-device groups for the folds; `metrics.per_phone` (seen vs new phones) | [per_phone.csv](results/phase2/metrics/per_phone.csv) |
| Building and floor accuracy, coordinate MAE | `metrics.summarize` (`building_acc`, `floor_acc`, `mae_x`, `mae_y`, mean and median 2D error) | [comparison.csv](results/phase2/metrics/comparison.csv) |
| Hierarchical error breakdown | `metrics.error_breakdown` (building wrong / floor wrong / both right) | [error_breakdown.csv](results/phase2/metrics/error_breakdown.csv); [notebook 05](notebooks/05_aml_vs_dl.ipynb), section 5 |
| Floor confusion matrices | `metrics.floor_confusion`, `plots.floor_confusion` | figure `dl_floor_confusion`; [notebook 05](notebooks/05_aml_vs_dl.ipynb), section 6 |
| Coordinate error maps | `plots.error_map` | figure `dl_error_map`; [notebook 05](notebooks/05_aml_vs_dl.ipynb), section 7 |
| A multi-task notebook with the AML vs DL comparison | [notebook 05](notebooks/05_aml_vs_dl.ipynb) (and [04](notebooks/04_dl_multitask_network.ipynb) for the network) | |
| Ablation studies | `config.DL_VARIANTS`, stage 2 of `scripts/run_dl.py` | [ablation.csv](results/phase2/metrics/ablation.csv), [cv_seeds.csv](results/phase2/metrics/cv_seeds.csv) |
| Regularization | dropout (`dl.MultiTaskMLP`, on only while training), weight decay (AdamW in `dl.NeuralLocalizationModel`), hidden WAPs (`dl.hide_waps`) | row "no dropout, no weight decay" and "no hidden WAPs" of [ablation.csv](results/phase2/metrics/ablation.csv) |

## How to run

```bash
# We used Python 3.14. On macOS, run `brew install libomp` first (XGBoost needs it).
git clone https://github.com/Jitmisra/T084-wifi-indoor-positioning.git
cd T084-wifi-indoor-positioning
make setup   # .venv with the exact versions behind results/, the package and the Jupyter kernel
make all     # data -> test -> run -> run2 -> notebooks -> report -> report2
```

| Command | What it does |
| --- | --- |
| `make data` | checks the two CSV files (SHA-256); downloads them from UCI only if they are missing |
| `make test` | runs the unit tests (under a minute) |
| `make run` | Phase 1: runs every Phase 1 experiment and writes its tables and figures to `results/`; the full run took about 26 minutes on our 11-core laptop |
| `make run QUICK=1` | a quick check that everything works (under a minute); writes only to `results/_quick/` |
| `make run2` | Phase 2: trains, tunes and tests the networks and the blend, writes `results/phase2/` (needs `make run` first); the full run took 78 minutes on our 11-core laptop |
| `make run2 QUICK=1` | a quick check of Phase 2 (under a minute); tests on held-out training users, never on the test set |
| `make notebooks` | executes the five notebooks in place |
| `make report` | builds the Phase 1 report `report/main.pdf` (needs [tectonic](https://tectonic-typesetting.github.io)) |
| `make report2` | builds the Phase 2 report `report/main_phase2.pdf` |
| `make clean` | removes caches and quick-run outputs (keeps `results/`) |

## Repository structure

```text
├── data/              dataset card (README.md) and the two CSV files (raw/UJIndoorLoc/)
├── notebooks/         01_eda_data_quality, 02_feature_engineering, 03_ml_models_evaluation (Phase 1);
│                      04_dl_multitask_network, 05_aml_vs_dl (Phase 2)
├── src/indoorloc/     the Python package: config, data, features, models, metrics, pipeline, plots;
│                      dl (Phase 2 networks), hybrid (blend of the Random Forest and the network)
├── scripts/           download_data.py, run_experiments.py (Phase 1), run_dl.py (Phase 2)
├── tests/             unit tests (pytest)
├── results/           Phase 1: metrics/, tuning/, predictions/, figures/
│   └── phase2/        Phase 2: the same four folders
├── report/            IEEE reports: main.tex + sections/ (Phase 1), main_phase2.tex + sections/phase2/
├── presentation/      slides of both phases (.pptx and .pdf)
└── docs/              literature reviews and literature matrices of both phases
```

## Method in short

### Phase 1

- **Data.** `trainingData.csv` (19,937 scans) for training, `validationData.csv` (1,111) as the
  test set. The code 100 ("not heard") becomes −110 dBm, weaker than any real reading.
- **Cleaning, training file only.** Drop scans with no WAP heard (76), exact duplicates (634) and
  scans with a reading of −30 dBm or stronger (508): 18,719 scans and 464 heard WAPs are left.
- **Fair evaluation.** 5-fold user-group CV holds out whole users with their phones (users who
  shared a phone form one group: 16 groups).
- **Features.** `rssi` (the WAP readings); signal statistics (WAPs heard, strongest reading, mean
  heard reading); AP-affinity: each WAP heard in at least 5 training scans gets a profile (building
  shares, mean floor and location), and a scan gets the weighted average of the profiles of the WAPs
  it hears (stronger readings count more) plus the location of its strongest known WAP.
- **No extra missingness columns, no scaling.** −110 already marks "not heard" for trees
  (and `n_heard` counts heard WAPs); trees ignore scale and all RSSI columns share one unit (dBm).
  Scaling, fitted on training data only, comes in Phase 2 for the neural network.
- **Models.** Three separate models per method: building, floor, position. Baseline RF: 300 trees,
  default settings, `rssi` + signal statistics. kNN: `rssi`. Random Forest and XGBoost: all
  features. Each task picks its own setting from a small grid by CV ([tuning](results/tuning/)).
- **Phase 2 gap.** Our three task models share nothing and no model is best at every task.

### Phase 2

- **Same data and protocol.** The same cleaned training scans, the same 5 user-group folds for
  every setting, and the test set used only at the end.
- **Inputs** (`dl.NetworkInputs`, learned from training scans only): for each of the 464 WAPs
  heard in training, the scaled RSSI (rssi − r_min) / −r_min with r_min = −105 dBm (not heard = 0)
  and a heard indicator, and (hybrid inputs) the 3 signal statistics and 8 AP-affinity columns,
  standardized: 464 + 464 + 11 = 939 columns.
- **Network** (`dl.MultiTaskMLP`): a shared trunk of 512, 256 and 128 units (ReLU, dropout 0.2),
  then three heads: building (3 classes) and floor (5 classes) with softmax and cross-entropy,
  position (x, y) with the mean squared error of standardized coordinates. Loss = building loss +
  floor loss + λ × position loss; λ from {30, 100, 300} by CV.
- **Training** (`dl.NeuralLocalizationModel`): AdamW (weight decay 0.01), batches of 256, 200
  epochs (`config.DL_EPOCHS`) with a learning rate that falls from 0.001 to 0 along a cosine curve,
  and in every batch 10% of the heard WAPs hidden at random (`dl.hide_waps`; the Phase 1 columns are
  left as they are).
- **Ablation and checks.** One change at a time with a new seed and the same folds; the full
  network cross-validated with three seeds and tested with five.
- **Hybrid blend** (`hybrid`): out-of-fold class probabilities and positions of the Phase 1 Random
  Forest and of the network (ablation seed) on the 5 folds; for every forest weight w = 0, 0.1, …, 1
  the blend w × forest + (1 − w) × network; each task keeps the weight with its best CV score (ties:
  the smaller w); then the blend is tested once.

## Team

Team T084 · Section D · Project 9 · Newton School of Technology, Rishihood University, Sonipat, Haryana, India.

| Name | Enrollment No. |
| --- | --- | 
| Gauri Mehrotra | 2401010170 |
| Ansh | 2401010077 | 
| Agnik Misra | 2401010044 | 
| Kartikey Gupta | 2401010214 |
| Lokendra Singh | 2401010254 |

Who did what, file by file (code, tests, notebooks, report sections, slides): [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## Data and license

UJIIndoorLoc, UCI Machine Learning Repository, [dataset 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc),
license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the [dataset card](data/README.md).
If you use the data, cite: J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and
multi-floor database for WLAN fingerprint-based indoor localization problems," in *Proc. IPIN
2014*, Busan, South Korea, pp. 261–270, doi: [10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492).

Code: MIT License, see [LICENSE](LICENSE). `report/IEEEtran.cls` keeps its own license (LPPL).
