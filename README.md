# Indoor Location Prediction Using Wi-Fi Signals

Phase 1 of Team T084: we predict the building, the floor and the position of a phone from the
Wi-Fi signals it hears, with classical machine learning on the UJIIndoorLoc dataset.

## What we do

A Wi-Fi scan lists how strongly a phone hears each of 520 access points (WAPs). From one scan we
predict the building, the floor and the position. We train on UJIIndoorLoc and test on scans
recorded months later, mostly with new phones. Features, settings and the final model are chosen
with user-group cross-validation (CV) on the training data; the test set is used once, at the end.
Our own feature, AP-affinity, learns where each WAP "lives" and describes a scan by the WAPs it
hears.

## Results

Test set: the UCI file `validationData.csv` (1,111 scans; 9 of its 11 phones are not in training).
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

## How to run

```bash
# We used Python 3.14. On macOS, run `brew install libomp` first (XGBoost needs it).
git clone https://github.com/Jitmisra/T084-wifi-indoor-positioning.git
cd T084-wifi-indoor-positioning
make setup   # .venv with the exact versions behind results/, the package and the Jupyter kernel
make all     # data -> test -> run -> notebooks -> report
```

| Command | What it does |
| --- | --- |
| `make data` | checks the two CSV files (SHA-256); downloads them from UCI only if they are missing |
| `make test` | runs the unit tests (about 10 s) |
| `make run` | runs every experiment and writes all tables and figures to `results/`; the full run took about 26 minutes on our 11-core laptop |
| `make run QUICK=1` | a quick check that everything works (under a minute); writes only to `results/_quick/` |
| `make notebooks` | executes the three notebooks in place |
| `make report` | builds `report/main.pdf` (needs [tectonic](https://tectonic-typesetting.github.io)) |
| `make clean` | removes caches and quick-run outputs (keeps `results/`) |

## Repository structure

```text
├── data/              dataset card (README.md) and the two CSV files (raw/UJIndoorLoc/)
├── notebooks/         01_eda_data_quality, 02_feature_engineering, 03_ml_models_evaluation
├── src/indoorloc/     the Python package: config, data, features, models, metrics, pipeline, plots
├── scripts/           download_data.py (download + check), run_experiments.py (all experiments)
├── tests/             unit tests (pytest)
├── results/
│   ├── metrics/       tables: cleaning, CV, test results, error breakdown, phones
│   ├── tuning/        CV score of every setting, best_params.json
│   ├── predictions/   test-set predictions, one CSV per model
│   └── figures/       all figures (PNG and PDF)
├── report/            IEEE-style report: main.tex, sections/, main.pdf
├── presentation/      slides (.pptx and .pdf)
└── docs/              literature review and literature matrix
```

## Method in short

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
- **Models.** Three separate models per method: building, floor, position. Baseline RF: 300 trees,
  default settings, `rssi` + signal statistics. kNN: `rssi`. Random Forest and XGBoost: all
  features. Each task picks its own setting from a small grid by CV ([tuning](results/tuning/)).
- **Phase 2.** Our three task models share nothing and no model is best at every task, so Phase 2
  will train one multi-task neural network (shared layers, three output heads), compared with
  Phase 1 on the same split, test set and metrics.

## Team

Team T084 · Section D · Project 9 · Newton School of Technology, Rishihood University, Sonipat, Haryana, India.

| Name | Enrollment No. | Study area |
| --- | --- | --- |
| Gauri Mehrotra | 2401010170 | Problem and literature review |
| Ansh | 2401010077 | Data quality, EDA and the fair split |
| Agnik Misra | 2401010044 | Features: signal statistics and AP-affinity |
| Kartikey Gupta | 2401010214 | Models, tuning and results |
| Lokendra Singh | 2401010254 | Error analysis and the Phase 2 gap |

## Data and license

UJIIndoorLoc, UCI Machine Learning Repository, [dataset 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc),
license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the [dataset card](data/README.md).
If you use the data, cite: J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and
multi-floor database for WLAN fingerprint-based indoor localization problems," in *Proc. IPIN
2014*, Busan, South Korea, pp. 261–270, doi: [10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492).

Code: MIT License, see [LICENSE](LICENSE). `report/IEEEtran.cls` keeps its own license (LPPL).
