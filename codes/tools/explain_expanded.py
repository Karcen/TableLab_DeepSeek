"""Explain the prespecified expanded HGB grid reference on every outer record."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import json,sys
import joblib,numpy as np,pandas as pd
import shap
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.experiment import dump
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'


def main():
    threadpool_limits(1)
    importance=[];audits=[]
    for ds in ['house','adult','nba','students']:
        for fold in range(3):
            base=DATA/ds/f'fold{fold}'/'Full'
            o=joblib.load(base/'prepared.joblib')['outer']
            model=joblib.load(base/'HGB_grid/model.joblib')
            explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
            ex=explainer(o['V'],check_additivity=True)
            values=np.asarray(ex.values)
            grouped=np.column_stack([values[:,list(group)].sum(axis=1) for group in o['groups']])
            assert len(o['groups'])==len(o['feature_names'])
            base_value=np.asarray(ex.base_values).reshape(-1)
            raw=model._raw_predict(o['V']).reshape(-1) if ds=='adult' else model.predict(o['V'])
            grouped=grouped*o['scale']
            base_value=base_value*o['scale']+o['centre']
            raw=raw*o['scale']+o['centre']
            error=float(np.max(np.abs(base_value+grouped.sum(axis=1)-raw)))
            assert error<1e-5,(ds,fold,error)
            frame=pd.DataFrame(grouped,columns=o['feature_names'])
            frame.insert(0,'row_id',o['validation_ids']);frame['base']=base_value;frame['raw_prediction']=raw
            frame.to_csv(base/'HGB_grid/shap.csv',index=False)
            for col,mean in zip(o['feature_names'],np.abs(grouped).mean(axis=0)):
                importance.append({'dataset':ds,'fold':fold,'field':col,'mean_absolute_shap':mean,'n':len(frame)})
            audits.append({'dataset':ds,'fold':fold,'n_explained':len(frame),'max_additivity_error':error,
                           'model':'HGB_grid fixed before results','units':'log_odds' if ds=='adult' else 'target_units'})
    pd.DataFrame(importance).to_csv(DATA/'shap_importance.csv',index=False)
    dump(DATA/'shap_audit.json',{'total_explained':sum(r['n_explained'] for r in audits),'folds':audits,
                               'background':'training tree-path counts','encoded_fields':'sum signed encoded contributions before absolute importance',
                               'record_sampling':False,'max_additivity_error':max(r['max_additivity_error'] for r in audits)})
    print(json.dumps({'total':sum(r['n_explained'] for r in audits),'max_error':max(r['max_additivity_error'] for r in audits)}))


if __name__=='__main__':main()
