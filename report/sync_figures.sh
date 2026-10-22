#!/usr/bin/env bash
# Copy the figures used by main.tex from results/figures/ so the report folder stays self-contained (e.g. for Overleaf).
set -euo pipefail
cd "$(dirname "$0")"
for name in eda_campus_map feature_check model_comparison floor_confusion; do
    cp "../results/figures/${name}.pdf" figures/
done
echo "Copied 4 figures into report/figures/"
