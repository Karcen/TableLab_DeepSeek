"""Prespecified sample-size and tree-depth controls under the expanded recipe."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss,f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import prepare,make_model,model_predict,sha
from tablelab.research.experiment import load_data,subset,metric,dump

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
OUT=DATA/'diagnostics'


def main():
    threadpool_limits(1);OUT.mkdir(exist_ok=True)
    spec={'code_sha256':sha(Path(__file__)),'expanded_protocol_sha256':sha(DATA/'protocol.json'),
          'learning_fractions':[.25,.5,.75,1.],'learning_model':'HGB; fixed parameters selected in the full outer-training fold; refit all preprocessing on each smaller training subset',
          'subset':'Adult full-predictor groups; NBA earliest complete training seasons; other tasks fixed-seed permutation. No changes to validation rows.',
          'complexity_depths':[1,2,4,8,None],'tree_min_samples_leaf':2,'evaluation':'Both training and all outer-validation records; all predictions saved. Diagnostic controls, not additional leaderboard entries.'}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==spec
    else:dump(protocol,spec)
    all_metrics=[];reconciled=[]
    for ds in ['house','adult','nba','students']:
        frame,meta=load_data(ROOT,ds);frame=frame.set_index('_row_id',drop=False)
        for fold in range(3):
            folder=OUT/ds/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'complete.json').exists():
                base=DATA/ds/f'fold{fold}'/'Full'
                o=joblib.load(base/'prepared.joblib')['outer']
                train=frame.loc[o['train_ids']].reset_index(drop=True);valid=frame.loc[o['validation_ids']].reset_index(drop=True)
                params=json.loads((base/'HGB_grid/search.json').read_text())['best_params']
                configurations=[('learning',f,f'HGB',params) for f in [.25,.5,.75,1.]]
                configurations += [('complexity',depth,'Tree',{'max_depth':depth,'min_samples_leaf':2}) for depth in [1,2,4,8,None]]
                metrics=[];predictions=[];ids=[]
                for kind,value,name,pars in configurations:
                    tr=train.iloc[subset(train,ds,value)] if kind=='learning' else train
                    p=prepare(tr,valid,meta['features'],meta['target'],ds=='adult')
                    fit=make_model(name,pars,ds=='adult',42+fold*100,p['groups'])
                    fit.fit(p['X'],p['y'])
                    for part,X,y,rids in [('train',p['X'],tr[meta['target']].to_numpy(),p['train_ids']),('validation',p['V'],p['actual'],p['validation_ids'])]:
                        pp=np.asarray(model_predict(fit,X,ds=='adult',name),dtype=np.float64)*p['scale']+p['centre']
                        values=metric(np.asarray(y,dtype=np.float64),pp,ds=='adult')
                        key={'dataset':ds,'fold':fold,'kind':kind,'value':value if value is not None else -1,'partition':part,'n_train':len(tr),'n':len(y)}
                        metrics.append({**key,**values})
                        predictions.extend({**key,'row_id':rid,'actual':yy,'predicted':zz} for rid,yy,zz in zip(rids,y,pp))
                    ids.append({'kind':kind,'value':value,'train_row_ids':p['train_ids'],'validation_row_ids':p['validation_ids'],'params':pars})
                pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False,float_format='%.17g')
                pd.DataFrame(predictions).to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
                dump(folder/'splits_and_parameters.json',ids)
                dump(folder/'complete.json',{'fitted_controls':9,'protocol_sha256':sha(protocol),'metrics_sha256':sha(folder/'metrics.csv'),'predictions_sha256':sha(folder/'predictions.csv')})
            scores=pd.read_csv(folder/'metrics.csv',float_precision='round_trip')
            predictions=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
            all_metrics.append(scores)
            for (kind,value,part),q in predictions.groupby(['kind','value','partition']):
                assert q.row_id.is_unique
                actual=frame.loc[q.row_id,meta['target']].to_numpy()
                assert np.array_equal(actual,q.actual)
                y=q.actual.to_numpy();p=q.predicted.to_numpy()
                values={'auc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'f1':f1_score(y,p>=.5)} if ds=='adult' else {'rmse':np.sqrt(mean_squared_error(y,p)),'mae':mean_absolute_error(y,p),'r2':r2_score(y,p)}
                row=scores.query('kind == @kind and value == @value and partition == @part').iloc[0]
                for key,actual_metric in values.items():
                    error=abs(row[key]-actual_metric);assert error<1e-7
                    reconciled.append({'dataset':ds,'fold':fold,'kind':kind,'value':value,'partition':part,'metric':key,'absolute_difference':error})
            print(ds,fold,'diagnostic controls checked',flush=True)
    pd.concat(all_metrics,ignore_index=True).to_csv(DATA/'diagnostics.csv',index=False)
    report={'status':'passed','fitted_controls':108,'metric_reconstructions':len(reconciled),'max_absolute_metric_difference':max(r['absolute_difference'] for r in reconciled)}
    qa=ROOT/'codes/verification/research/expanded';pd.DataFrame(reconciled).to_csv(qa/'diagnostic_metric_reconciliation.csv',index=False)
    dump(qa/'diagnostic_checks.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
