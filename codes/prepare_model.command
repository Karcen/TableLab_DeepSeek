#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/layout.sh"
[[ -x .venv/bin/python ]] || { echo '先执行 prepare_dependencies.command。' >&2; exit 1; }
command -v ollama >/dev/null || { echo '请先在 Mac 安装 Ollama。' >&2; exit 1; }
export OLLAMA_HOST=127.0.0.1:11434
printf '准备本地模型。请保持运行 serve_model.command 的终端打开。\n'
ollama pull qwen3:8b-q4_K_M
.venv/bin/python -m tablelab.doctor --lock-model --require-all
