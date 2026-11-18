# Team Contributions

Team T084 · Section D · Project 9. Phase 1 (classical machine learning) is below; Phase 2 (deep
learning) follows it. Each member owns one part of the pipeline, end to end: the
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

# Phase 2 (deep learning)

The same five parts, now for the network. Each member owns the code, tests, notebook sections,
report sections and slides of one part.

```text
Agnik            Ansh                 Gauri               Kartikey              Lokendra
DL literature -> network inputs,  ->  two hybrids:    ->  network, loss,    ->  ablation study,
+ report         regularization       Phase 1 features    tuning, Phase 1       error analysis,
                                      in the network,     vs Phase 2            conclusion
                                      the blend
```

## Agnik Misra (2401010044): deep learning literature and the report

| Part | Files |
| --- | --- |
| Literature review | [docs/literature_review_phase2.md](docs/literature_review_phase2.md), [docs/literature_matrix_phase2.csv](docs/literature_matrix_phase2.csv) |
| Report | [main_phase2.tex](report/main_phase2.tex), [abstract](report/sections/phase2/abstract.tex), [introduction](report/sections/phase2/introduction.tex), [related work](report/sections/phase2/related_work.tex), Phase 2 part of [references.bib](report/references.bib) |
| Repository docs | [README.md](README.md) (Phase 2 sections, brief checklist) |
| Slides | Phase 2 slides 1–2: title; from Phase 1 to Phase 2, the literature and the gap |

## Ansh (2401010077): network inputs and regularization

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/dl.py](src/indoorloc/dl.py): `NetworkInputs` (scaled RSSI, heard indicators, scaling learned from training scans only) and `hide_waps` (hidden WAPs); regularization settings in `DL_FIXED` ([config.py](src/indoorloc/config.py)) |
| Tests | [tests/test_dl.py](tests/test_dl.py): `test_inputs_by_hand`, `test_input_columns_and_training_statistics`, `test_hidden_waps_only_change_wap_columns`, `test_dropout_is_on_again_after_a_prediction_during_training` |
| Notebook | [04_dl_multitask_network.ipynb](notebooks/04_dl_multitask_network.ipynb), sections 1–4 and 8 |
| Results | [learning_curves.csv](results/phase2/metrics/learning_curves.csv), figure `dl_learning_curves` |
| Report | [Data and Network Inputs](report/sections/phase2/inputs.tex), Method: Training and Regularization |
| Slides | Phase 2 slides 3–4: the inputs, training without memorizing |

## Gauri Mehrotra (2401010170): the two hybrids (Phase 1 features in the network, the blend)

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/dl.py](src/indoorloc/dl.py): the hybrid block of `NetworkInputs` (signal statistics and AP-affinity from [features.py](src/indoorloc/features.py), learned again in every fold, standardized); the `mlp` variant without Phase 1 features in `DL_VARIANTS`; [src/indoorloc/hybrid.py](src/indoorloc/hybrid.py): the blend of the Phase 1 Random Forest and the network (out-of-fold outputs, one forest weight per task by CV) and stage 6 of [run_dl.py](scripts/run_dl.py) |
| Tests | `test_epoch_cross_validation_keeps_held_out_labels_away` (AP-affinity learned without held-out labels), `test_input_columns_and_training_statistics` (hybrid columns scaled with training numbers), [tests/test_hybrid.py](tests/test_hybrid.py) (the blend) |
| Notebook | [04_dl_multitask_network.ipynb](notebooks/04_dl_multitask_network.ipynb), section 5; [05_aml_vs_dl.ipynb](notebooks/05_aml_vs_dl.ipynb), section 4 |
| Results | rows `hybrid_mlp` and `mlp` of [ablation.csv](results/phase2/metrics/ablation.csv) and [test_results.csv](results/phase2/metrics/test_results.csv); [blend.csv](results/phase2/tuning/blend.csv), [blend.json](results/phase2/tuning/blend.json), row `blend` of [test_results.csv](results/phase2/metrics/test_results.csv) |
| Report | Method: Two Hybrids of Phase 1 and Phase 2; Results: The Hybrid Blend |
| Slides | Phase 2 slides 5–6: the two hybrids, and whether they help |

## Kartikey Gupta (2401010214): the network, its training and Phase 1 vs Phase 2

| Part | Files |
| --- | --- |
| Code | [src/indoorloc/dl.py](src/indoorloc/dl.py): `MultiTaskMLP`, `NeuralLocalizationModel` (loss, AdamW, cosine schedule, single-task mode), `cross_validate_epochs`; [scripts/run_dl.py](scripts/run_dl.py); [Makefile](Makefile) (`run2`, `report2`); PyTorch in the requirements |
| Tests | `test_network_heads`, `test_multitask_and_single_task_models_fit_and_predict`, `test_same_seed_gives_the_same_model`, `test_predicted_positions_are_turned_back_into_meters`, `test_quick_run_writes_every_result` |
| Notebook | [04_dl_multitask_network.ipynb](notebooks/04_dl_multitask_network.ipynb), sections 6–7; [05_aml_vs_dl.ipynb](notebooks/05_aml_vs_dl.ipynb), sections 2–3 |
| Results | [results/phase2/tuning/](results/phase2/tuning/), [test_results.csv](results/phase2/metrics/test_results.csv), [comparison.csv](results/phase2/metrics/comparison.csv), [final_model.json](results/phase2/metrics/final_model.json), [results/phase2/predictions/](results/phase2/predictions/), figure `aml_vs_dl` |
| Report | Method: The Multi-Task Network, Loss, Tuning; Results: tuning and Phase 1 vs Phase 2 |
| Slides | Phase 2 slides 7–8: one network with three heads, the test results |

## Lokendra Singh (2401010254): ablation study, error analysis and conclusion

| Part | Files |
| --- | --- |
| Code | the ablation design `DL_VARIANTS` and the seed checks ([config.py](src/indoorloc/config.py), stages 2 and 5 of [run_dl.py](scripts/run_dl.py)); [src/indoorloc/plots.py](src/indoorloc/plots.py): `learning_curves`, `ablation`, `model_comparison` for both phases; Phase 1's error metrics reused for the networks |
| Tests | `test_choices_never_depend_on_the_test_set`, `test_quick_run_writes_every_result` (ablation table and comparison rows) |
| Notebook | [05_aml_vs_dl.ipynb](notebooks/05_aml_vs_dl.ipynb), sections 1 and 5–9 |
| Results | [ablation.csv](results/phase2/metrics/ablation.csv), [cv_seeds.csv](results/phase2/metrics/cv_seeds.csv), [seeds.csv](results/phase2/metrics/seeds.csv), [error_breakdown.csv](results/phase2/metrics/error_breakdown.csv), [per_phone.csv](results/phase2/metrics/per_phone.csv), [floor_confusion.csv](results/phase2/metrics/floor_confusion.csv), figures `dl_ablation`, `dl_floor_confusion`, `dl_error_map` |
| Report | Results: Ablation Study, Where the Errors Are; [Conclusion](report/sections/phase2/conclusion.tex) |
| Slides | Phase 2 slides 9–10: ablation study; where the errors are, and what we learned |
