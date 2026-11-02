"""Training-only interface/timing checks before freezing the expanded protocol."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,time,json
import numpy as np
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import MODELS,RECURRENT,domains,make_model,prepare,build_sequence,seq_splits,model_predict
from tablelab.research.experiment import load_data,splits,dump

if __name__=='__main__':
    threadpool_limits(1)
    root=Path(__file__).resolve().parents[2]
    out=root/'codes/verification/research/expanded_pilot';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ds in ['house','adult','nba_sequence']:
        if ds=='nba_sequence':
            df,features=build_sequence(root,out/'sequence')
            a,b=seq_splits(df)[0];target='pts';models=RECURRENT
        else:
            df,meta=load_data(root,ds);features=meta['features'];target=meta['target']
            a,b=splits(df,ds,3)[0];models=[m for m in MODELS if m!='Mean']
        train=df.iloc[a].copy()
        p=prepare(train,train,features,target,ds=='adult')
        for model in models:
            params=list(ParameterGrid(domains(model,ds=='adult')))[0]
            start=time.perf_counter()
            fit=make_model(model,params,ds=='adult',42,p['groups'],7 if ds=='nba_sequence' else 0,3)
            fit.fit(p['X'],p['y'])
            prediction=model_predict(fit,p['X'][:32],ds=='adult',model)
            assert np.isfinite(prediction).all() and prediction.shape==(32,)
            rows.append({'dataset':ds,'model':model,'n_training':len(train),'seconds':time.perf_counter()-start,
                         'params':params,'parameter_count':getattr(fit,'parameter_count',None),
                         'last_training_objective':getattr(fit,'history',[{}])[-1].get('training_objective')})
            dump(out/'training_only_checks.json',rows)
            print(ds,model,len(train),f'{time.perf_counter()-start:.2f}s',flush=True)
    print('Training-only interfaces passed; no outer-validation/test scores computed.',flush=True)
