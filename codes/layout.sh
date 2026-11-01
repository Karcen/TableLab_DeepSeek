#!/bin/bash
# Called from launcher scripts. Macintosh and Linux both support directory links.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
link_dir() {
  local alias="$1" target="$2"
  [[ -d "$target" ]] || { printf '找不到 %s，请完整解压 TableLab。\n' "$target" >&2; return 1; }
  if [[ -L "$alias" ]]; then
    [[ "$alias" -ef "$target" ]] || { printf '目录链接 %s 指向其他位置；请检查。\n' "$alias" >&2; return 1; }
  elif [[ -e "$alias" ]]; then
    printf '%s 已有同名文件或目录，不会覆盖。请在新文件夹解压。\n' "$alias" >&2; return 1
  else
    ln -s "$target" "$alias"
  fi
}
link_dir data ../data
link_dir figures ../figures
link_dir tables ../tables
link_dir docs ../docs
link_dir artifacts ../models
link_dir runtime ../models/runtime
