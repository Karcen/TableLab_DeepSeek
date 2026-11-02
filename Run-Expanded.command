#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
expanded_python="$PWD/codes/.venv/bin/python"
if [ ! -x "$expanded_python" ]; then
 echo '请先运行 TableLab-DeepSeek.command 准备 Python 环境。'
 exit 1
fi
"$expanded_python" -m pip install -r codes/requirements-expanded.txt
"$expanded_python" codes/tools/run_expanded.py --workers 4
"$expanded_python" codes/tools/canonicalize_expanded.py
"$expanded_python" codes/tools/verify_expanded.py
"$expanded_python" codes/tools/summarize_expanded.py
"$expanded_python" codes/tools/diagnose_expanded.py
"$expanded_python" codes/tools/expanded_holdout.py --stage train
"$expanded_python" codes/tools/expanded_holdout.py --stage evaluate
"$expanded_python" codes/tools/verify_expanded_holdout.py
"$expanded_python" codes/tools/check_expanded_optimization.py
"$expanded_python" codes/tools/explain_expanded.py
"$expanded_python" codes/tools/verify_expanded_shap.py
expanded_nature_skill="${NATURE_FIGURE_SKILL:-$HOME/Downloads/SKILLS/nature-skills-main/skills/nature-figure}"
"$expanded_python" codes/tools/plot_expanded.py --skill-root "$expanded_nature_skill"
"$expanded_python" codes/tools/audit_expanded_figures.py --skill-root "$expanded_nature_skill"
"$expanded_python" codes/tools/integrate_research_docs.py
bash docs/build_docs.sh
"$expanded_python" codes/tools/review_expanded_delivery.py
