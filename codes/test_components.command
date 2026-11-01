#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/layout.sh"
[[ -x .venv/bin/python ]] || { echo "先从 codes 目录运行 bash prepare_dependencies.command。" >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q --junitxml=verification/local_component_tests.xml
