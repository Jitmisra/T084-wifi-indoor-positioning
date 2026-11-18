#!/usr/bin/env bash
# Copy the figures used by main.tex (Phase 1) and main_phase2.tex (Phase 2) from results/ so the
# report folder stays self-contained (e.g. for Overleaf).
set -euo pipefail
cd "$(dirname "$0")"
for name in eda_campus_map feature_check model_comparison floor_confusion; do
    cp "../results/figures/${name}.pdf" figures/
done
if [ -d ../results/phase2/figures ]; then
    for name in dl_learning_curves dl_ablation aml_vs_dl dl_floor_confusion dl_error_map; do
        cp "../results/phase2/figures/${name}.pdf" figures/
    done
    echo "Copied 4 Phase 1 and 5 Phase 2 figures into report/figures/"
else
    echo "Copied 4 Phase 1 figures into report/figures/ (no Phase 2 results yet: run make run2)"
fi
