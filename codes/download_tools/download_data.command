#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "需要 Python 3.11+。请先安装受支持的 Python，再运行本脚本。")'
if [[ ! -x .venv-download/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv-download
fi
printf '%s\n' '准备独立下载环境；从 PyPI 安装 requirements_download.txt 中固定的 kagglehub 版本。'
.venv-download/bin/python -m pip install -r requirements_download.txt
.venv-download/bin/python -m pip freeze > downloader_environment.lock.txt
.venv-download/bin/python download_tablelab_data.py --login "$@"
