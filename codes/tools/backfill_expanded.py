"""Compute distant queued datasets early using the same frozen job function."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import run_job,sha
from tablelab.research.experiment import dump

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--fold',type=int);args=parser.parse_args()
    protocol=json.loads((DATA/'protocol.json').read_text())['specification']
    for p,h in protocol['source_sha256'].items():assert sha(ROOT/p)==h
    tasks=json.loads((DATA/'jobs.json').read_text())
    tasks=[t for t in tasks if t['dataset']==args.dataset and (args.fold is None or t['fold']==args.fold)]
    tasks=[t for t in tasks if not (Path(t['folder'])/(t['model']+'_'+t['method'])).exists()]
    name=args.dataset+(f'_fold{args.fold}' if args.fold is not None else '')
    dump(DATA/f'backfill_{name}.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'workers':2,'source_sha256':sha(Path(__file__)),
                                      'rule':'Distant queued jobs only; identical frozen data, code, seeds and search. Main runner later reuses completed markers.', 'tasks':tasks})
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_job,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();print(f'{i}/{len(tasks)} {r["dataset"]} f{r["fold"]} {r["variant"]} {r["model"]}_{r["method"]} {r.get("seconds",0):.1f}s',flush=True)
    print('Backfill complete.',flush=True)
