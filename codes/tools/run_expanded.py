"""Freeze and execute the expanded benchmark with per-flow resumable jobs."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import argparse
import json
import importlib.metadata
import sys
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import MODELS,SEQUENCE_MODELS,domains,sha,prepare_jobs,run_job
from tablelab.research.experiment import dump

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/expanded'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    specification = {'static_models':MODELS,'sequence_models':SEQUENCE_MODELS,
                     'outer_folds':3,'inner_folds':3,'methods':['grid','tpe'],
                     'ablation_search':'grid repeated within each reduced-feature training set',
                     'structural_controls':{'MLP':'disable activations','ResMLP':'disable residual addition','DCN':'zero the cross branch output','GatedMLP':'replace gates by one','FeatureTransformer':'bypass self-attention encoder'},
                     'primary_search_comparison':'grid results compare model families; TPE is a separately reported search comparison, not selected by outer-fold scores',
                     'domains':{str(cls):{m:domains(m,cls) for m in MODELS} for cls in [False,True]},
                     'sequence_domains':{m:domains(m,False) for m in SEQUENCE_MODELS},
                     'neural_training':{'device':'CPU','epochs':40,'batch_size':2048,'optimizer':'AdamW','gradient_clip':5.,'dropout':.1,'threads':1,'seeds':'42 + 100*outer_fold; inner adds fold number + 1','early_stopping':False},
                     'tpe':'same finite domain and same evaluation count as grid; startup min(3, domain size); duplicate deterministic candidates cached within search',
                     'preprocessing':'original six/four fields, training-only imputation/encoding/scaling; train-only target mean/SD for every regression model',
                     'comparison':'new common protocol; do not pool means with the earlier 13 flows',
                     'adult_reserved_test':'Already evaluated in prior study; keep prior outputs unchanged; any later use is an explicitly labelled extension, not a newly unseen test',
                     'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'codes/tablelab/research/expanded.py',ROOT/'codes/tablelab/research/networks.py',Path(__file__),ROOT/'codes/tablelab/research/experiment.py']},
                     'data_sha256':{ds:sha(ROOT/'data/processed'/f'{ds}.csv') for ds in ['house','adult','nba','students']},
                     'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','optuna','torch','xgboost','lightgbm','catboost']}}
    protocol = OUT/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['specification'] == specification,'Frozen expanded protocol differs; archive rather than mix incompatible fits'
    else:
        dump(protocol,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'specification':specification})
    jobs = prepare_jobs(ROOT,OUT)
    dump(OUT/'jobs.json',jobs)
    pending = [t for t in jobs if not (Path(t['folder'])/(t['model']+'_'+t['method'])/'complete.json').exists()]
    print(f'{len(jobs)} flow-fold-variant jobs; {len(pending)} pending; {args.workers} workers',flush=True)
    finished = len(jobs)-len(pending)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job,t):t for t in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                r = future.result()
                finished += 1
                print(f'{finished}/{len(jobs)} {r["dataset"]} f{r["fold"]} {r["variant"]} {r["model"]}_{r["method"]} {r.get("seconds",0):.1f}s',flush=True)
            except Exception as exc:
                failures.append({'task':task,'error':repr(exc)})
                dump(OUT/'failures.json',failures)
                print(f'FAILED {task}: {exc!r}',flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed; inspect failures.json and retain successful evidence')
    dump(OUT/'training_complete.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'jobs':len(jobs),'protocol_sha256':sha(protocol)})
    print('All expanded nested search and ablation jobs complete.',flush=True)


if __name__=='__main__':
    main()
