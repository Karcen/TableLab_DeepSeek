"""在项目根目录运行此文件；使用相同 data/manifest.json 与已核对的执行器。"""
from pathlib import Path
import json,hashlib,os,sys
sys.path.insert(0,str(Path.cwd()))
from tablelab import engine as module
from tablelab.engine import Engine,runtime_fingerprint
from tablelab.catalog import Catalog
from tablelab.contracts import Plan
here=Path(__file__).resolve().parent
meta=json.loads((here/"provenance.json").read_text())
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==meta["engine_sha256"], "执行器版本已改变"
current=runtime_fingerprint()
assert current["package_sha256"]==meta["runtime"]["package_sha256"], "分析模块版本已改变"
assert current["dependencies"]==meta["runtime"]["dependencies"], "依赖版本已改变"
root=Path(os.environ.get("TABLELAB_DATA_ROOT", "data"))
assert hashlib.sha256((root/"processed"/(meta["dataset"]+".csv")).read_bytes()).hexdigest()==meta["sha256"], "数据已改变"
plan=Plan.model_validate_json((here/"plan.json").read_text())
if plan.operation=="predict": raise RuntimeError("单条预测依赖会话中的已训练模型；先重放对应训练方案再预测。")
print(json.dumps(Engine(Catalog(root),Path("artifacts/replay")).execute(plan),ensure_ascii=False,indent=2))
