"""Run all datasets: python tools/run_research.py --workers 2."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from tablelab.research.experiment import run_dataset, dump, OUTER, INNER, TRIALS
import importlib.metadata
import hashlib
import json
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['house','adult','nba','students']);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
 root=Path(__file__).resolve().parents[2]
 previous=root/'artifacts/research/protocol.json'
 current_hash=hashlib.sha256((root/'codes/tablelab/research/experiment.py').read_bytes()).hexdigest()
 if previous.exists() and json.loads(previous.read_text()).get('code_sha256')!=current_hash:
  raise SystemExit('实验代码已变化。请先归档 artifacts/research，再按新协议运行，不能混用旧折结果。')
 dump(root/'artifacts/research/protocol.json',{'outer_folds':OUTER,'inner_folds':INNER,'search_evaluations':TRIALS,'seed':42,'search_models':['RF','HGB'],'methods':['grid','tpe'],'tpe_startup_trials':4,'model_selection':'predefined families; no outer-driven retuning','shap_model':'HGB_grid fixed in advance','classification_search_metric':'roc_auc','regression_search_metric':'negative_rmse','workers':args.workers,'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','numpy','pandas','optuna','shap']},'code_sha256':hashlib.sha256((root/'codes/tablelab/research/experiment.py').read_bytes()).hexdigest()})
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for f in as_completed([pool.submit(run_dataset,root,ds) for ds in args.datasets]):print('Completed',f.result(),flush=True)
