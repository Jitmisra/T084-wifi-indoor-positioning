# Team Contributions

Team T084 · Section D · Project 9. Each member owns one part of the pipeline, end to end: the
code, its tests, the notebook section, the report section, the slides and the results it
produces. Everyone reviewed the whole project and can explain every part.

```text
Gauri            Ansh                 Agnik               Kartikey              Lokendra
problem +   ->   data, cleaning,  ->  features,       ->  models, tuning,   ->  error analysis,
literature       fair split           AP-affinity         test results          Phase 2 gap
```

## Gauri Mehrotra (2401010170): problem and literature review

| Part | Files |
| --- | --- |
| Literature review | [docs/literature_review.md](docs/literature_review.md), [docs/literature_matrix.csv](docs/literature_matrix.csv) |
| Report | [main.tex](report/main.tex), [abstract](report/sections/abstract.tex), [introduction](report/sections/introduction.tex), [related work](report/sections/related_work.tex), [references.bib](report/references.bib) |
| Repository docs | [README.md](README.md) |
| Slides | 1–3: title, the problem, related work and the gap |

## Ansh (2401010077): data quality, EDA and the fair split

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/data.py](src/indoorloc/data.py) (loading, −110 fill, 3 cleaning rules, user-device groups), [scripts/download_data.py](scripts/download_data.py) |
| Tests | [tests/test_data.py](tests/test_data.py) |
| Notebook | [01_eda_data_quality.ipynb](notebooks/01_eda_data_quality.ipynb) |
| Results | [cleaning.csv](results/metrics/cleaning.csv), [eda_summary.json](results/metrics/eda_summary.json), [cv_check.csv](results/metrics/cv_check.csv), figures `eda_campus_map`, `eda_waps_per_scan`, `eda_scans_per_floor` |
| Report | [Data and Exploratory Analysis](report/sections/dataset_eda.tex), Method: Fair Evaluation, Results: Random CV Is Too Optimistic |
| Docs | [data/README.md](data/README.md) (dataset card) |
| Slides | 4–5: the data, cleaning and the fair split |

## Agnik Misra (2401010044): features and AP-affinity

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/features.py](src/indoorloc/features.py) (signal statistics, `APAffinity`, `FeatureBuilder`) |
| Tests | [tests/test_features.py](tests/test_features.py) |
| Notebook | [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) |
| Results | [feature_check.csv](results/metrics/feature_check.csv), figure `feature_check` |
| Report | Method: Features (AP-affinity equations), Results: Feature Check |
| Slides | 6–7: features and the feature check |

## Kartikey Gupta (2401010214): models, tuning and results

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/models.py](src/indoorloc/models.py), [src/indoorloc/pipeline.py](src/indoorloc/pipeline.py) (folds, CV, grid search, test), [src/indoorloc/config.py](src/indoorloc/config.py), [scripts/run_experiments.py](scripts/run_experiments.py), [Makefile](Makefile) |
| Tests | [tests/test_models.py](tests/test_models.py), [tests/test_pipeline.py](tests/test_pipeline.py) |
| Notebook | [03_ml_models_evaluation.ipynb](notebooks/03_ml_models_evaluation.ipynb), sections 1–4 |
| Results | [results/tuning/](results/tuning/), [cv_results.csv](results/metrics/cv_results.csv), [test_results.csv](results/metrics/test_results.csv), [final_model.json](results/metrics/final_model.json), [results/predictions/](results/predictions/), figure `model_comparison` |
| Report | Method: Models, Tuning and the Final Model, Results: Model Comparison |
| Slides | 8–9: models, tuning and test results |

## Lokendra Singh (2401010254): error analysis and the Phase 2 gap

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/metrics.py](src/indoorloc/metrics.py) (errors, 3-case breakdown, per phone, floor confusion), [src/indoorloc/plots.py](src/indoorloc/plots.py) (all figures) |
| Tests | [tests/test_metrics.py](tests/test_metrics.py) |
| Notebook | [03_ml_models_evaluation.ipynb](notebooks/03_ml_models_evaluation.ipynb), sections 5–9 |
| Results | [error_breakdown.csv](results/metrics/error_breakdown.csv), [per_phone.csv](results/metrics/per_phone.csv), [floor_confusion.csv](results/metrics/floor_confusion.csv), figures `floor_confusion`, `error_map` |
| Report | Method: Metrics, Results: Where the Error Comes From, [Phase 2 Gap and Conclusion](report/sections/conclusion.tex) |
| Slides | 10–11: where the errors are, the Phase 2 gap |
