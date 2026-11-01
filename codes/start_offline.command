#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/layout.sh"
[[ -x .venv/bin/python ]] || { echo "先从 codes 目录运行 bash prepare_dependencies.command。" >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TABLELAB_DATA_ROOT="$PWD/data"
export TABLELAB_ARTIFACT_ROOT="$PWD/artifacts/runs"
.venv/bin/python -m tablelab.doctor --require-all
printf '\n浏览器打开 http://127.0.0.1:8765 。按 Control-C 退出。\n'
exec .venv/bin/python -m tablelab.web
