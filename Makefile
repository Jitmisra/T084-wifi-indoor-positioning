# Wi-Fi indoor positioning on UJIIndoorLoc: predict the building, floor and position of a scan.
#
#   make setup         create .venv and install everything (locked versions, package, Jupyter kernel)
#   make all           data -> test -> run -> notebooks -> report
#   make run QUICK=1   a small, fast check run (writes results/_quick/ only)
#   make help          list every target

PY ?= .venv/bin/python
KERNEL ?= indoorloc
# Why: on Python 3.14, scikit-learn's worker threads can wipe Python's shared list of warning
# filters and flood the output with harmless warnings; with this option every thread has its own.
export PYTHON_CONTEXT_AWARE_WARNINGS := 1
NB_TIMEOUT ?= 1800
QUICK ?=

NOTEBOOKS := notebooks/01_eda_data_quality.ipynb \
             notebooks/02_feature_engineering.ipynb \
             notebooks/03_ml_models_evaluation.ipynb
# `python -m nbconvert` (not `jupyter nbconvert`): the `jupyter` command may belong to another
# Python that cannot see the kernel of this project's .venv.
# Why record_timing=False: otherwise every executed cell stores its start and end time, so a
# re-executed notebook would always differ from the committed one.
NBCONVERT := $(PY) -m nbconvert --to notebook --execute --inplace \
             --ExecutePreprocessor.kernel_name=$(KERNEL) \
             --ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
             --ExecutePreprocessor.record_timing=False
# Why a second pass: printed text can be split into output chunks differently from run to run;
# merging the chunks keeps a re-executed notebook unchanged when its outputs are unchanged.
NBCOALESCE := $(PY) -m nbconvert --to notebook --inplace --CoalesceStreamsPreprocessor.enabled=True

.PHONY: help setup data test run notebooks report all clean

help:  ## list the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

# Why the import check at the end: it fails at once with XGBoost's own message (e.g. on macOS
# without `brew install libomp`), not in the middle of a run.
setup:  ## create .venv (if missing), install the locked versions, the package and the kernel
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps
	$(PY) -m ipykernel install --sys-prefix --name $(KERNEL) --display-name "Python (indoorloc)" \
		--env PYTHON_CONTEXT_AWARE_WARNINGS 1
	$(PY) -c "import sklearn, xgboost"

data:  ## download and check the two UJIIndoorLoc CSV files (only if they are missing)
	$(PY) scripts/download_data.py

test:  ## run the unit tests (about 10 s)
	$(PY) -m pytest -q

run: data  ## run every experiment and write results/ (QUICK=1: fast check run, results/_quick/)
	$(PY) scripts/run_experiments.py $(if $(QUICK),--quick,)

notebooks: data  ## execute notebooks 01-03 in place with the 'indoorloc' kernel
	$(NBCONVERT) $(NOTEBOOKS)
	$(NBCOALESCE) $(NOTEBOOKS)

# Why a fixed SOURCE_DATE_EPOCH (21 October 2026, 00:00 UTC): tectonic then uses it instead of the
# current time for the PDF's dates and ID, so rebuilding unchanged sources gives an identical file.
report:  ## build the IEEE report (report/main.pdf)
	cd report && ./sync_figures.sh && SOURCE_DATE_EPOCH=1792540800 tectonic main.tex

all: data test run notebooks report  ## the whole project, in order

clean:  ## remove caches, build files and quick-run outputs (keeps results/ and report/main.pdf)
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -print -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache src/*.egg-info results/_quick
	rm -f report/*.aux report/*.log report/*.bbl report/*.blg report/*.out report/*.synctex.gz
