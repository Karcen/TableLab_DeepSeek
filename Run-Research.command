#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
research_python="$PWD/codes/.venv/bin/python"
if [ ! -x "$research_python" ]; then
 echo '请先运行 TableLab-DeepSeek.command 准备项目 Python 环境。'
 exit 1
fi
"$research_python" -m pip install -r codes/requirements-research.txt
"$research_python" codes/tools/run_research.py --workers 2
"$research_python" codes/tools/verify_research.py
"$research_python" codes/tools/summarize_research.py
nature_skill="${NATURE_FIGURE_SKILL:-$HOME/Downloads/SKILLS/nature-skills-main/skills/nature-figure}"
"$research_python" codes/tools/plot_research.py --skill-root "$nature_skill"
"$research_python" codes/tools/audit_research_figures.py --skill-root "$nature_skill"
"$research_python" codes/tools/integrate_research_docs.py
bash docs/build_research_docs.sh
