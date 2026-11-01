#!/bin/bash
# TableLab 联网演示入口。请在解压后的工程目录中运行。
# 可选：只在自己的电脑上，把新密钥填在下方的单引号内。留空使用系统钥匙串。
# 填过密钥的这个文件不要提交、转发或录屏展示；不要使用已贴到聊天中的旧密钥。
set +x
set -euo pipefail
PRIVATE_API_KEY=''
# Keep files installed or created by this launcher private to the current account.
umask 077
if [[ -n "$PRIVATE_API_KEY" ]]; then chmod go-rwx "$0"; fi

stop() { printf '%s\n' "$*" >&2; exit 1; }
finish() {
  local code="$1"
  trap - EXIT
  if [[ "$code" != 0 && "$code" != 130 && "$code" != 143 && -t 0 && -t 1 ]]; then
    printf '\n启动未完成，错误原因在上方。按回车结束这个窗口：' >&2
    IFS= read -r _ || true
  fi
  exit "$code"
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

here="$(cd "$(dirname "$0")" && pwd -P)"
root="$here"
[[ -f "$root/codes/tablelab/cloud/credentials.py" && -f "$root/data/manifest.json" ]] || stop '请完整解压 ZIP，并在含 codes 和 data 的目录里运行此启动器。'
# The original project helper creates data, figures, tables and runtime links and enters codes/.
source "$root/codes/layout.sh"
printf 'TableLab · DeepSeek 联网问答\n工程：%s\n' "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py="$PWD/.venv/bin/python"
version_check='import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)'
if [[ ! -x "$py" ]]; then
  if [[ -n "${TABLELAB_PYTHON:-}" ]]; then
    candidates=("$TABLELAB_PYTHON")
  else
    # Finder and interactive Terminal may use different PATHs. Prefer the user's existing Anaconda.
    candidates=(/opt/anaconda3/bin/python "$HOME/anaconda3/bin/python" "$HOME/miniconda3/bin/python" python3.13 python3.12 python3.11 python3 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3)
  fi
  chosen=''
  for p in "${candidates[@]}"; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c "$version_check"; then chosen="$p"; break; fi
  done
  [[ -n "$chosen" ]] || stop '需要 Python 3.11、3.12 或 3.13。请先安装其中一个版本，或用 TABLELAB_PYTHON 指定解释器，然后重新运行。'
  printf '正在创建项目专用 Python 环境（%s）……\n' "$chosen"
  "$chosen" -m venv "$PWD/.venv"
fi
"$py" -c "$version_check" || stop '项目中的 .venv 版本不兼容。请另解压一份工程，并用 Python 3.11、3.12 或 3.13 启动。'
"$py" -c 'from pathlib import Path; import sys; sys.exit(0 if Path(sys.prefix).resolve() == Path(".venv").resolve() and sys.prefix != sys.base_prefix else 1)' || stop '当前 Python 并非此工程的独立环境，已停止，未向系统 Python 安装依赖。'
fingerprint="$("$py" -c 'import hashlib,pathlib,sys,platform; print(hashlib.sha256(pathlib.Path("requirements.txt").read_bytes()+str((sys.version_info[:3],sys.prefix,platform.machine(),sys.platform)).encode()).hexdigest())')"
marker="$PWD/.venv/.tablelab-deepseek-deps"
if [[ ! -f "$marker" ]] || [[ "$(cat "$marker")" != "$fingerprint" ]]; then
  printf '正在检查并安装 Python 依赖。安装完成后本次直接启动网页。\n'
  "$py" -m pip install --disable-pip-version-check -r requirements.txt
  "$py" -m pip check
  "$py" -m pip freeze > runtime/python_environment.lock.txt
  printf '%s\n' "$fingerprint" > "$marker"
fi
# Pass an optional private literal via a dedicated descriptor, not argv, stdin or the environment.
# Keep stdin attached to Terminal for getpass in Keychain mode. Do not set -x or export the key.
"$py" -c '
import os, socket, subprocess, sys
from pathlib import Path
with os.fdopen(3) as key_file:
    key = key_file.read().removesuffix("\n")
from tablelab.catalog import Catalog, SPECS
from tablelab.cloud import credentials
from tablelab.cloud.deepseek_online import DeepSeek
from tablelab.cloud.web import create_app
from tablelab.offline import install_loopback_guard
import uvicorn
host, port = "127.0.0.1", 8765
url = f"http://{host}:{port}"
with socket.socket() as probe:
    if probe.connect_ex((host, port)) == 0:
        sys.exit(f"端口 {port} 已被占用。若是已开的 TableLab，可直接在浏览器打开 {url}；要重启，请先在旧服务窗口按 Control-C。")
cat = Catalog(Path("data"))
for ds in SPECS:
    _, item = cat.load(ds)
    if item["source_kind"] not in ("original", "course_dataset"):
        sys.exit(f"数据 {ds} 未登记为课程数据。停止启动。")
if not key:
    key = credentials.acquire(credentials.native_store())
client = DeepSeek(credentials.validated(key))
del key
install_loopback_guard()
class BrowserServer(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started:
            print(f"网页已就绪：{url}\n在网页上输入问题。演示时保留此终端，按 Control-C 停止服务。", flush=True)
            if sys.platform == "darwin":
                result = subprocess.run(["/usr/bin/open", url], check=False)
                if result.returncode:
                    print(f"浏览器未自动打开，请在地址栏输入 {url}", file=sys.stderr)
server = BrowserServer(uvicorn.Config(create_app(llm=client), host=host, port=port, workers=1, access_log=False))
server.run()
if not server.started:
    sys.exit("网页未能启动。请检查上方的错误信息。")
' 3<<< "$PRIVATE_API_KEY"
