"""Independent checks for the disclosed Adult holdout extension."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog,SPECS
from tablelab.engine import predictor_fingerprints
from tablelab.research.expanded import sha,model_predict,MODELS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'codes/verification/research/expanded'
DATA=ROOT/'artifacts/expanded/holdout_extension'


def main():
    threadpool_limits(1)
    complete=json.loads((DATA/'complete.json').read_text())
    protocol=json.loads((DATA/'protocol.json').read_text())
    assert protocol['specification']['code_sha256']==sha(ROOT/'codes/tools/expanded_holdout.py')
    for p,h in complete['output_sha256'].items():assert sha(DATA/p)==h
    lock=json.loads((DATA/'models_locked.json').read_text());selection=json.loads((DATA/'selection.json').read_text())
    assert protocol['frozen_at_utc']<=lock['locked_at_utc']<=selection['selected_at_utc']<=complete['completed_at_utc']
    raw,entry=Catalog(ROOT/'data').load('adult');assert entry['sha256']==complete['source_sha256']
    train=raw.query('_source_split == "train"');test_raw=raw.query('_source_split == "original_test"')
    cols=[c for c in raw if not c.startswith('_') and c not in ['income','income_gt_50k']]
    joined=test_raw.merge(train[cols].drop_duplicates(),on=cols,how='left',indicator=True,validate='many_to_one')
    excluded=set(joined.loc[joined['_merge']=='both','_row_id']);included=set(joined.loc[joined['_merge']=='left_only','_row_id'])
    assert len(excluded)==26 and len(included)==16255
    p=pd.read_csv(DATA/'predictions.csv',float_precision='round_trip')
    assert p.row_id.is_unique and set(p.row_id)==included
    fp=predictor_fingerprints(raw.set_index('_row_id')[cols])
    assert np.array_equal(p.fingerprint,fp.loc[p.row_id])
    y=raw.set_index('_row_id').loc[p.row_id,'income_gt_50k'].to_numpy()
    assert np.array_equal(y,p.actual)
    cache=joblib.load(DATA/'training_preparation.joblib');o=cache['outer']
    assert set(o['train_ids'])==set(train._row_id)==set(o['validation_ids'])
    for fold in cache['inner']:
        assert set(fold['train_ids']).isdisjoint(fold['validation_ids'])
        assert set(fold['train_ids'])|set(fold['validation_ids'])==set(train._row_id)
        assert set(fp.loc[fold['train_ids']]).isdisjoint(fp.loc[fold['validation_ids']])
    summary=pd.read_csv(ROOT/'artifacts/expanded/summary.csv')
    choice=summary.query('dataset == "adult" and variant == "Full" and method in ["grid","fixed"]').sort_values(['mean','model'],ascending=[False,True]).iloc[0]
    assert choice.model==selection['chosen_before_test_prediction']==complete['primary']
    assert selection['cv_summary_sha256']==sha(ROOT/'artifacts/expanded/summary.csv')
    Z=o['pre'].transform(raw.set_index('_row_id').loc[p.row_id,list(SPECS['adult'].defaults)]).astype(np.float32)
    scores=pd.read_csv(DATA/'metrics.csv',float_precision='round_trip').set_index('model');checks=[]
    inner_fits=0
    for name in MODELS:
        if name=='Mean':model_p=np.full(len(p),train.income_gt_50k.mean())
        else:
            path=DATA/'fits'/name
            fit=json.loads((path/'fit.json').read_text());assert protocol['frozen_at_utc']<=fit['finished_at_utc']<=lock['locked_at_utc']
            assert sha(path/'model.joblib')==fit['model_sha256']==lock['models'][name]
            trace=json.loads((path/'search.json').read_text());assert trace['best_params']==fit['selected_params']
            inner_fits+=trace['actual_inner_fits']
            model_p=np.asarray(model_predict(joblib.load(path/'model.joblib'),Z,True,name),dtype=np.float64)
        assert np.allclose(model_p,p[name],rtol=0,atol=1e-13)
        values={'auc':roc_auc_score(y,p[name]),'ap':average_precision_score(y,p[name]),'brier':brier_score_loss(y,p[name]),'log_loss':log_loss(y,p[name]),'f1':f1_score(y,p[name]>=.5)}
        for key,value in values.items():
            error=abs(value-scores.loc[name,key]);assert error<1e-12
            checks.append({'model':name,'metric':key,'absolute_difference':error})
    samples=np.load(DATA/'bootstrap_auc.npz');assert list(samples['models'])==MODELS
    auc=samples['auc'];assert auc.shape==(2000,len(MODELS)) and np.isfinite(auc).all()
    lo,hi=np.quantile(auc,[.025,.975],axis=0)
    assert np.allclose(lo,scores.loc[MODELS,'auc_ci_low'],rtol=0,atol=1e-12)
    assert np.allclose(hi,scores.loc[MODELS,'auc_ci_high'],rtol=0,atol=1e-12)
    _,codes=np.unique(p.fingerprint,return_inverse=True);groups=int(codes.max()+1)
    assert groups==16249
    rng=np.random.default_rng(20261102);replicates={0,1,2,499,999,1499,1999}
    for b in range(2000):
        count=np.bincount(rng.integers(groups,size=groups),minlength=groups)
        if b in replicates:
            for k,name in enumerate(MODELS):assert abs(roc_auc_score(y,p[name],sample_weight=count[codes])-auc[b,k])<1e-12
    primary=MODELS.index(choice.model);pairs=pd.read_csv(DATA/'paired_differences.csv',float_precision='round_trip').set_index('comparator')
    for k,name in enumerate(MODELS):
        if k==primary:continue
        ci=np.quantile(auc[:,primary]-auc[:,k],[.025,.975])
        assert np.allclose(ci,pairs.loc[name,['ci_low','ci_high']].to_numpy(dtype=float),rtol=0,atol=1e-12)
        assert abs(pairs.loc[name,'auc_difference']-(scores.loc[choice.model,'auc']-scores.loc[name,'auc']))<1e-12
    report={'status':'passed','models':len(MODELS),'metrics_recomputed':len(checks),'max_absolute_metric_difference':max(r['absolute_difference'] for r in checks),
            'bootstrap_sklearn_comparisons':len(replicates)*len(MODELS),'training_rows':len(train),'test_rows':len(p),'excluded_rows':len(excluded),'test_groups':groups,
            'primary_chosen_from_cv':choice.model,'primary_confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,p[choice.model]>=.5).ravel().tolist(),
            'actual_inner_fits':inner_fits,'final_fits':20,'test_status':'subsequent extension, not a new blind test'}
    pd.DataFrame(checks).to_csv(OUT/'holdout_extension_reconciliation.csv',index=False)
    (OUT/'holdout_extension_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
