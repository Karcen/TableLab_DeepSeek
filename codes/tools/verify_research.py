"""Independent reconstruction and split/explanation checks on stored evidence."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,brier_score_loss,log_loss,average_precision_score,f1_score
from tablelab.research.experiment import load_data
from tablelab.engine import predictor_fingerprints
ROOT=Path(__file__).resolve().parents[2]
checks=[]
for ds in ['house','adult','nba','students']:
 df,_=load_data(ROOT,ds);df=df.set_index('_row_id',drop=False)
 for fold in range(3):
  folder=ROOT/'artifacts/research'/ds/f'fold{fold}';assert (folder/'complete.json').exists()
  sp=pd.read_csv(folder/'split.csv');tr=sp.loc[sp.partition=='train','row_id'];va=sp.loc[sp.partition=='validation','row_id'];assert not set(tr)&set(va)
  if ds=='adult':
   cols=[c for c in df if not c.startswith('_') and c not in ('income','income_gt_50k')];finger=predictor_fingerprints(df[cols]);assert set(finger.loc[tr]).isdisjoint(finger.loc[va])
  if ds=='nba':assert df.loc[tr,'season'].max()<df.loc[va,'season'].min()
  for inner in json.loads((folder/'inner_splits.json').read_text()):
   a=inner['train_row_ids'];b=inner['validation_row_ids'];assert set(a+b)<=set(tr);assert not set(a)&set(b)
   if ds=='adult':assert set(finger.loc[a]).isdisjoint(finger.loc[b])
   if ds=='nba':assert df.loc[a,'season'].max()<df.loc[b,'season'].min()
  metrics=pd.read_csv(folder/'metrics.csv');pred=pd.read_csv(folder/'predictions.csv')
  for model,g in pred.groupby('model'):
   assert set(g.row_id)==set(va);assert np.allclose(g.actual,df.loc[g.row_id,('income_gt_50k' if ds=='adult' else 'SalePrice' if ds=='house' else 'pts' if ds=='nba' else 'math score')])
   y=g.actual.to_numpy();p=g.predicted.to_numpy()
   if ds=='adult':recomputed={'auc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'f1':f1_score(y,p>=.5)}
   else:recomputed={'rmse':np.sqrt(mean_squared_error(y,p)),'mae':mean_absolute_error(y,p),'r2':r2_score(y,p)}
   row=metrics.query('model==@model and partition=="validation"').iloc[0]
   for key,value in recomputed.items():
    delta=abs(row[key]-value);assert delta<1e-7,(ds,fold,model,key,delta);checks.append(dict(dataset=ds,fold=fold,model=model,metric=key,max_absolute_difference=delta))
  shap=pd.read_csv(folder/'shap.csv');cols=json.loads((folder.parent/'data_audit.json').read_text())['features']
  assert set(shap.row_id)==set(va);assert np.max(np.abs(shap[cols].sum(axis=1)+shap.base-shap.raw_prediction))<1e-5
  n=pred.loc[pred.model=='HGB_grid'].set_index('row_id').loc[shap.row_id].predicted.to_numpy()
  rebuilt=1/(1+np.exp(-shap.raw_prediction.to_numpy())) if ds=='adult' else shap.raw_prediction.to_numpy()
  assert np.allclose(n,rebuilt,atol=1e-7)
  for name in ['RF','HGB']:
   for method in ['grid','tpe']:
    s=json.loads((folder/f'{name}_{method}_search.json').read_text());assert len(s['rows'])==12
pd.DataFrame(checks).to_csv(ROOT/'codes/verification/research/metric_reconciliation.csv',index=False)
print(f'{len(checks)} independently reconstructed metrics agree; all outer/inner split and SHAP checks passed.')
