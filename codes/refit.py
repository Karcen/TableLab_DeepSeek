"""按论文基准或课堂例题重新训练一次。不会覆盖已交付的预测或结果表。"""
import argparse
import csv
import json
from pathlib import Path
from tablelab.catalog import Catalog
from tablelab.engine import Engine, runtime_fingerprint
from tablelab.contracts import Plan

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    with (ROOT / 'models/index.csv').open(encoding='utf-8-sig', newline='') as f:
        items = {row['case']: row for row in csv.DictReader(f)}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case', choices=sorted(items))
    entry = items[ap.parse_args().case]
    folder = ROOT / 'models' / entry['directory']
    plan = Plan.model_validate_json((folder / 'plan.json').read_text())
    meta = json.loads((folder / 'provenance.json').read_text())
    cat = Catalog(ROOT / 'data')
    _, data_meta = cat.load(plan.dataset)
    if data_meta['sha256'] != meta['sha256']:
        raise ValueError('输入数据已改变。')
    current = runtime_fingerprint()
    for field in ('package_sha256', 'dependencies'):
        if current[field] != meta['runtime'][field]:
            raise ValueError(f'{field} 与原模型不同。请使用随包的代码和依赖版本。')
    engine = Engine(cat, ROOT / 'models/runs')
    result = engine.execute(plan, meta['question'], json.loads((folder / 'evidence.json').read_text()))
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
    print('本次训练结果：', ROOT / 'models/runs' / result['run_id'])

if __name__ == '__main__':
    main()
