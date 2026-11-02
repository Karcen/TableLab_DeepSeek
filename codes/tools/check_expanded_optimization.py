"""Inspect selected estimators for finite coefficients and iteration limits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
from expanded_paths import local_path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'codes'))
DATA=ROOT/'artifacts/expanded'


def inspect(model,key,rows):
    if hasattr(model,'coef_'):assert np.isfinite(model.coef_).all(),key
    if hasattr(model,'calibrated_classifiers_'):
        for i,calibrator in enumerate(model.calibrated_classifiers_):inspect(calibrator.estimator,{**key,'component':f'calibration_{i}'},rows)
    if hasattr(model,'output'):inspect(model.output,{**key,'component':'basis_output'},rows)
    limit=getattr(model,'max_iter',None);iterations=getattr(model,'n_iter_',None)
    if limit is not None and iterations is not None:
        used=int(np.asarray(iterations).max())
        rows.append({**key,'iterations':used,'limit':int(limit),'limit_reached':used>=int(limit)})


if __name__=='__main__':
    rows=[]
    for t in json.loads((DATA/'jobs.json').read_text()):
        if t['model'] not in ['Linear','Ridge','Sparse','SVM','ELM','RBFNet'] or t['method']=='fixed':continue
        folder=local_path(t['folder'])/(t['model']+'_'+t['method'])
        assert (folder/'complete.json').exists()
        inspect(joblib.load(folder/'model.joblib'),{k:t[k] for k in ['dataset','fold','model','method','variant']},rows)
    for name in ['Linear','Ridge','Sparse','SVM','ELM','RBFNet']:
        path=DATA/'holdout_extension/fits'/name/'model.joblib'
        inspect(joblib.load(path),{'dataset':'adult_extension','fold':-1,'model':name,'method':'grid','variant':'Full'},rows)
    q=pd.DataFrame(rows);out=ROOT/'codes/verification/research/expanded'
    q.to_csv(out/'optimization_iterations.csv',index=False)
    report={'status':'passed','selected_solver_components_checked':len(rows),'iteration_limits_reached':int(q.limit_reached.sum()),
            'limits':q.loc[q.limit_reached].to_dict('records'),'neural_training':'Fixed 40 epochs; no claim that this establishes optimization convergence'}
    (out/'optimization_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
