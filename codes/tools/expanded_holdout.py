"""Explicitly disclosed Adult test extension; grid flows only, no test retuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,argparse,time
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog,SPECS
from tablelab.engine import predictor_fingerprints
from tablelab.research.expanded import MODELS,domains,prepare,search_model,model_predict,sha
from tablelab.research.experiment import splits,dump,metric
from run_adult_holdout import prepared_auc

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
OUT=DATA/'holdout_extension'


def specification():
    return {'models':MODELS,'method':'grid only; TPE remains an outer-CV comparison',
            'domains':{m:domains(m,True) for m in MODELS},'seed':42,'inner_cv':'three stratified fingerprint-group folds on full original training file',
            'selected_flow':'Highest Adult grid outer-CV mean; exact ties by model name; choice frozen before test prediction',
            'test_history':'Original test previously profiled and evaluated for the earlier 13 flows. This is a subsequent extension, not a new unseen blind test. No test-driven changes to families, hyperparameters or threshold.',
            'exclusion':'Exclude 26 original-test rows with full 14-predictor fingerprints shared with original train; retain within-test repeats',
            'primary_metric':'auc','secondary_metrics':['ap','brier','log_loss','f1'],'threshold':.5,
            'uncertainty':{'replicates':2000,'seed':20261102,'unit':'full-predictor fingerprint group','interval':'95% percentile, conditional on fitted models; descriptive paired intervals, not multiplicity adjusted'},
            'code_sha256':sha(Path(__file__)),'expanded_protocol_sha256':sha(DATA/'protocol.json'),
            'prior_holdout_protocol_sha256':sha(ROOT/'artifacts/holdout/adult/protocol.json')}


def train_one(name):
    threadpool_limits(1)
    folder=OUT/'fits'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():
        marker=json.loads((folder/'fit.json').read_text());assert marker['model_sha256']==sha(folder/'model.joblib');return name
    start=time.perf_counter();cache=joblib.load(OUT/'training_preparation.joblib')
    fit,params=search_model(name,'grid',cache,True,42,0,3,folder/'search.json')
    joblib.dump(fit,folder/'model.joblib',compress=1)
    dump(folder/'fit.json',{'model':name,'selected_params':params,'model_sha256':sha(folder/'model.joblib'),
                          'seconds':time.perf_counter()-start,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
    print('Extension training',name,f'{time.perf_counter()-start:.1f}s',flush=True)
    return name


def train():
    threadpool_limits(1)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.json'
    spec=specification()
    if protocol.exists():assert json.loads(protocol.read_text())['specification']==spec
    else:dump(protocol,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'specification':spec})
    frame,_=Catalog(ROOT/'data').load('adult')
    train=frame.query('_source_split == "train"').reset_index(drop=True)
    features=list(SPECS['adult'].defaults)
    cv=splits(train,'adult',3)
    if not (OUT/'training_preparation.joblib').exists():
        # Final refits receive only training data, including diagnostic logging.
        cache={'outer':prepare(train,train,features,'income_gt_50k',True),
               'inner':[prepare(train.iloc[a],train.iloc[b],features,'income_gt_50k',True) for a,b in cv]}
        joblib.dump(cache,OUT/'training_preparation.joblib',compress=1)
        dump(OUT/'inner_splits.json',[{'train_row_ids':train.iloc[a]._row_id.tolist(),'validation_row_ids':train.iloc[b]._row_id.tolist()} for a,b in cv])
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_one,m) for m in MODELS if m!='Mean']
        for future in as_completed(futures):future.result()
    lock={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),
          'models':{m:sha(OUT/'fits'/m/'model.joblib') for m in MODELS if m!='Mean'},'training_rows':len(train),'test_prediction_computed':False}
    if (OUT/'models_locked.json').exists():assert json.loads((OUT/'models_locked.json').read_text())['models']==lock['models']
    else:dump(OUT/'models_locked.json',lock)
    print('All twenty fitted models locked before any extension test predictions.',flush=True)


def evaluate():
    threadpool_limits(1)
    assert json.loads((OUT/'protocol.json').read_text())['specification']==specification()
    if (OUT/'complete.json').exists():
        print('Existing extension test retained; no repeated evaluation or refit.',flush=True);return
    checks=json.loads((ROOT/'codes/verification/research/expanded/checks.json').read_text());assert checks['status']=='passed'
    summary=pd.read_csv(DATA/'summary.csv')
    q=summary.query('dataset == "adult" and variant == "Full" and method in ["grid","fixed"]')
    choice=q.sort_values(['mean','model'],ascending=[False,True]).iloc[0]
    selection=OUT/'selection.json'
    selected={'chosen_before_test_prediction':choice.model,'cv_auc':float(choice['mean']),'cv_sd':float(choice.sd),
              'cv_summary_sha256':sha(DATA/'summary.csv'),'selected_at_utc':datetime.now(timezone.utc).isoformat()}
    if selection.exists():
        previous=json.loads(selection.read_text());assert previous['chosen_before_test_prediction']==choice.model and previous['cv_summary_sha256']==selected['cv_summary_sha256']
    else:dump(selection,selected)
    locked=json.loads((OUT/'models_locked.json').read_text())
    for name,h in locked['models'].items():assert sha(OUT/'fits'/name/'model.joblib')==h
    frame,entry=Catalog(ROOT/'data').load('adult')
    cols=[c for c in frame if not c.startswith('_') and c not in ['income','income_gt_50k']]
    fingerprint=predictor_fingerprints(frame[cols])
    training=frame._source_split=='train'
    candidate=(frame._source_split=='original_test') & ~fingerprint.isin(set(fingerprint[training]))
    test=frame.loc[candidate].copy();groups=fingerprint.loc[candidate].to_numpy()
    assert len(test)==16255 and len(np.unique(groups))==16249
    old=pd.read_csv(ROOT/'artifacts/holdout/adult/predictions.csv')
    assert set(test._row_id)==set(old.row_id)
    o=joblib.load(OUT/'training_preparation.joblib')['outer']
    Z=o['pre'].transform(test[list(SPECS['adult'].defaults)]).astype(np.float32)
    y=test.income_gt_50k.to_numpy(dtype=np.float64)
    pred=pd.DataFrame({'row_id':test._row_id.to_numpy(),'fingerprint':groups,'actual':y})
    scores=[]
    for name in MODELS:
        if name=='Mean':p=np.full(len(test),o['y'].mean(),dtype=np.float64)
        else:p=np.asarray(model_predict(joblib.load(OUT/'fits'/name/'model.joblib'),Z,True,name),dtype=np.float64)
        pred[name]=p;scores.append({'model':name,'n':len(y),**metric(y,p,True)})
    pred.to_csv(OUT/'predictions.csv',index=False,float_format='%.17g')
    funcs=[prepared_auc(y,pred[m].to_numpy()) for m in MODELS]
    _,codes=np.unique(groups,return_inverse=True);n_groups=int(codes.max()+1)
    rng=np.random.default_rng(20261102);samples=np.empty((2000,len(MODELS)))
    for b in range(2000):
        multiplicity=np.bincount(rng.integers(n_groups,size=n_groups),minlength=n_groups)
        weights=multiplicity[codes]
        for k,f in enumerate(funcs):samples[b,k]=f(weights)
    np.savez_compressed(OUT/'bootstrap_auc.npz',auc=samples,models=np.array(MODELS))
    scores=pd.DataFrame(scores).set_index('model')
    scores['auc_ci_low'],scores['auc_ci_high']=np.quantile(samples,[.025,.975],axis=0)
    scores['cv_auc']=q.set_index('model')['mean'];scores['cv_sd']=q.set_index('model').sd
    scores['test_rank']=scores.auc.rank(method='min',ascending=False).astype(int)
    scores.to_csv(OUT/'metrics.csv',float_format='%.17g')
    primary=MODELS.index(choice.model);pairs=[]
    for k,name in enumerate(MODELS):
        if k==primary:continue
        lo,hi=np.quantile(samples[:,primary]-samples[:,k],[.025,.975])
        pairs.append({'primary':choice.model,'comparator':name,'auc_difference':scores.loc[choice.model,'auc']-scores.loc[name,'auc'],'ci_low':lo,'ci_high':hi})
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False,float_format='%.17g')
    inner=sum(json.loads((OUT/'fits'/m/'search.json').read_text())['actual_inner_fits'] for m in MODELS if m!='Mean')
    dump(OUT/'complete.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'primary':choice.model,'test_rows':len(test),'test_groups':n_groups,
                             'inner_fits':inner,'final_fits':20,'protocol_sha256':sha(OUT/'protocol.json'),'lock_sha256':sha(OUT/'models_locked.json'),
                             'selection_sha256':sha(selection),'source_sha256':entry['sha256'],
                             'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='complete.json'}})
    print(scores.sort_values('auc',ascending=False).to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','evaluate'],required=True);args=parser.parse_args()
    train() if args.stage=='train' else evaluate()
