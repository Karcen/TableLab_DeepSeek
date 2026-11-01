#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/layout.sh"
PYTHON="${TABLELAB_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,11)<=sys.version_info<(3,14), "请使用 Python 3.11、3.12 或 3.13"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > runtime/python_environment.lock.txt
printf '\n依赖已安装。联网版用顶层菜单 9；离线版另需准备 Ollama。此步骤没有下载数据或语言模型。\n'
