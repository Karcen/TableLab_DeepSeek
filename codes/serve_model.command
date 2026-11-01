#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/layout.sh"
command -v ollama >/dev/null || { echo '请先安装官方 Ollama；本脚本不下载或执行远程安装脚本。'; exit 1; }
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=16384
exec ollama serve
