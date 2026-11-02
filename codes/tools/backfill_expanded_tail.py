"""Schedule an untouched distant variant with identical frozen fitting code."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import run_job,sha
from tablelab.research.experiment import dump
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'artifacts/expanded'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--fold',required=True,type=int);parser.add_argument('--variant',required=True);args=parser.parse_args()
    spec=json.loads((DATA/'protocol.json').read_text())['specification']
    for p,h in spec['source_sha256'].items():assert sha(ROOT/p)==h
    tasks=[t for t in json.loads((DATA/'jobs.json').read_text()) if t['dataset']==args.dataset and t['fold']==args.fold and t['variant']==args.variant]
    assert tasks and all(not (Path(t['folder'])/(t['model']+'_'+t['method'])).exists() for t in tasks),'Variant already started; do not overlap running jobs'
    name=f'{args.dataset}_fold{args.fold}_{args.variant.replace(" ","_")}'
    dump(DATA/f'backfill_{name}.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'workers':2,'source_sha256':sha(Path(__file__)),
                                      'rule':'Only execution order changes; the original queue later reuses completion markers','tasks':tasks})
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,f in enumerate(as_completed([pool.submit(run_job,t) for t in tasks]),1):
            r=f.result();print(f'{i}/{len(tasks)} {r["model"]} {r.get("seconds",0):.1f}s',flush=True)
    print('Distant variant complete.',flush=True)
