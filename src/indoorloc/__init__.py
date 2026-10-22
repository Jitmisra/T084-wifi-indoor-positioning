"""indoorloc: predict the building, floor and position of a phone from Wi-Fi signals.

We use the UJIIndoorLoc dataset (520 Wi-Fi access points, three buildings). The modules are:
config (paths and settings), data (loading, cleaning, user groups), features (RSSI, signal
statistics, AP-affinity), models (kNN, Random Forest, XGBoost and LocalizationModel), metrics
(errors and summaries), pipeline (cross-validation, grid search, test evaluation) and plots.
"""

from . import config, data, features, metrics, models, pipeline, plots

__all__ = ["config", "data", "features", "metrics", "models", "pipeline", "plots"]
