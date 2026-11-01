"""Aggregate the saved outer-fold metrics without selecting further hyperparameters."""
from pathlib import Path
import pandas as pd
import json
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'artifacts/research'
if __name__=='__main__':
 metrics=pd.concat([pd.read_csv(p) for p in DATA.glob('*/fold*/metrics.csv')],ignore_index=True)
 summary=[]
 for ds,q in metrics.groupby('dataset'):
  main='auc' if ds=='adult' else 'rmse'
  for name,g in q.query('partition=="validation"').groupby('model'):
   row=dict(dataset=ds,model=name,metric=main,mean=g[main].mean(),sd=g[main].std(),folds=g.fold.nunique())
   for col in ['auc','ap','brier','log_loss','f1','rmse','mae','r2']:
    if col in g and g[col].notna().any():row[col+'_mean']=g[col].mean();row[col+'_sd']=g[col].std()
   summary.append(row)
 summary=pd.DataFrame(summary);summary.to_csv(DATA/'summary.csv',index=False)
 tuning=[]
 for ds in ['house','adult','nba','students']:
  m='auc' if ds=='adult' else 'rmse';q=metrics.query('dataset==@ds and partition=="validation"')
  for name in ['RF','HGB']:
   for method in ['grid','tpe']:
    base=q.query('model==@name').set_index('fold')[m];tuned=q.loc[q.model==name+'_'+method].set_index('fold')[m]
    diff=tuned-base if ds=='adult' else base-tuned
    histories=[json.loads(p.read_text()) for p in (DATA/ds).glob(f'fold*/{name}_{method}_search.json')]
    tuning.append(dict(dataset=ds,model=name,method=method,improvement=diff.mean(),sd=diff.std(),total_seconds=sum(x['total_seconds'] for x in histories),mean_unique_candidates=sum(x['unique_candidates'] for x in histories)/len(histories)))
 pd.DataFrame(tuning).to_csv(DATA/'tuning_summary.csv',index=False)
 print(summary[['dataset','model','mean','sd']].to_string(index=False))
