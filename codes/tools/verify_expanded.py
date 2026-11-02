"""Independently reconcile every expanded outer prediction and nested split."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import json,sys,argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,f1_score,mean_squared_error,mean_absolute_error,r2_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog
from tablelab.engine import predictor_fingerprints
from tablelab.research.expanded import sha,MODELS,SEQUENCE_MODELS,NEURAL,RECURRENT,domains,ARCH_ABLATIONS
from sklearn.model_selection import ParameterGrid
from expanded_paths import local_path,comparable_task

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
OUT=ROOT/'codes/verification/research/expanded'


def main(partial=False):
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((DATA/'protocol.json').read_text())['specification']
    for p,h in protocol['source_sha256'].items():
        assert sha(ROOT/p)==h,(p,'Frozen implementation changed')
    frames={ds:Catalog(ROOT/'data').load(ds)[0].set_index('_row_id',drop=False) for ds in ['house','adult','nba','students']}
    for ds,h in protocol['data_sha256'].items():
        assert sha(ROOT/'data/processed'/f'{ds}.csv')==h
    frames['adult']=frames['adult'].query('_source_split == "train"')
    frames['nba_sequence']=pd.read_csv(DATA/'nba_sequence/sequence_master.csv').set_index('_row_id',drop=False)
    seq=frames['nba_sequence'];raw=frames['nba']
    assert seq.index.is_unique
    # Independent one-to-one raw-column joins, rather than the builder's hash map.
    keys=['player_name','country','draft_year']
    r=raw.reset_index(drop=True)[['_row_id','season','age','player_height']+keys].copy()
    r['year']=r.season.str[:4].astype(int)
    duplicated=r.duplicated(keys+['year'],keep=False)
    clean=r.loc[~duplicated].copy()
    joined=clean.rename(columns={'_row_id':'target_id','age':'target_age','player_height':'target_height'})
    for lag in [3,2,1]:
        year=f'history_year{lag}';joined[year]=joined.year-lag
        history=clean[keys+['year','_row_id','age','player_height']].rename(columns={'year':year,'_row_id':f'history_id{lag}','age':f'history_age{lag}','player_height':f'history_height{lag}'})
        joined=joined.merge(history,on=keys+[year],how='inner',validate='one_to_one')
    assert int(duplicated.sum())==2 and len(clean)-len(joined)==6512
    ages=joined[[f'history_age{lag}' for lag in [3,2,1]]+['target_age']].to_numpy()
    heights=joined[[f'history_height{lag}' for lag in [3,2,1]]+['target_height']].to_numpy()
    continuity=((np.diff(ages,axis=1)>=0)&(np.diff(ages,axis=1)<=2)&(np.abs(np.diff(heights,axis=1))<=5)).all(axis=1)
    assert int((~continuity).sum())==102
    rebuilt=joined.loc[continuity].set_index('target_id')
    assert set(rebuilt.index)==set(seq.index)
    for lag in [3,2,1]:
        ids=seq[f'_history_id_lag{lag}']
        assert np.array_equal(ids,rebuilt.loc[seq.index,f'history_id{lag}'])
        history=raw.loc[ids].reset_index(drop=True)
        assert np.array_equal(history.season.str[:4].astype(int).to_numpy()+lag,seq.season.str[:4].astype(int).to_numpy())
        for field in ['age','player_height','player_weight','pts','reb','ast','gp']:
            assert np.allclose(history[field],seq[f'lag{lag}_{field}'],rtol=0,atol=1e-12)
    split_checks=[]
    target={'house':'SalePrice','adult':'income_gt_50k','nba':'pts','students':'math score','nba_sequence':'pts'}
    adult_cols=[c for c in frames['adult'] if not c.startswith('_') and c not in ['income','income_gt_50k']]
    fingerprints=predictor_fingerprints(frames['adult'][adult_cols])
    for ds,frame in frames.items():
        for fold in range(3):
            folder=DATA/ds/f'fold{fold}'
            split=pd.read_csv(folder/'split.csv')
            train=set(split.loc[split.partition=='train','row_id']);valid=set(split.loc[split.partition=='validation','row_id'])
            assert not train&valid and train|valid<=set(frame.index)
            if ds=='adult':
                assert set(fingerprints.loc[list(train)]).isdisjoint(fingerprints.loc[list(valid)])
            if ds.startswith('nba'):
                assert frame.loc[list(train),'season'].max()<frame.loc[list(valid),'season'].min()
            inner=json.loads((folder/'inner_splits.json').read_text())
            for k,s in enumerate(inner):
                a,b=set(s['train_row_ids']),set(s['validation_row_ids'])
                assert not a&b and a|b<=train
                if ds=='adult':assert set(fingerprints.loc[list(a)]).isdisjoint(fingerprints.loc[list(b)])
                if ds.startswith('nba'):assert frame.loc[list(a),'season'].max()<frame.loc[list(b),'season'].min()
            split_checks.append({'dataset':ds,'fold':fold,'n_train':len(train),'n_validation':len(valid),'inner_folds':len(inner),'passed':True})
    tasks=json.loads((DATA/'jobs.json').read_text())
    if partial:
        available=[]
        for t in tasks:
            marker=local_path(t['folder'])/(t['model']+'_'+t['method'])/'complete.json'
            if marker.exists() and 'precision_protocol_sha256' in json.loads(marker.read_text()):available.append(t)
        tasks=available
    precision=json.loads((DATA/'precision_protocol.json').read_text())
    assert precision['specification']['code_sha256']==sha(ROOT/'codes/tools/canonicalize_expanded.py')
    checks=[];actual_fits=0;requested=0;curves=0;model_hashes=0;verified_caches=set()
    for t in tasks:
        folder=local_path(t['folder'])/(t['model']+'_'+t['method'])
        done=json.loads((folder/'complete.json').read_text())
        assert done['precision_protocol_sha256']==sha(DATA/'precision_protocol.json')
        native=json.loads((folder/'native_precision/complete.json').read_text())
        assert native['model_sha256']==done['model_sha256'] and native['selected_params']==done['selected_params']
        assert native['prediction_sha256']==sha(folder/'native_precision/predictions.csv')
        assert comparable_task(done['task'])==comparable_task(t)
        for field,file in [('prediction_sha256','predictions.csv'),('metrics_sha256','metrics.csv')]:
            assert sha(folder/file)==done[field]
        if done['model_sha256']:
            assert sha(folder/'model.joblib')==done['model_sha256'];model_hashes+=1
        p=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
        outer=pd.read_csv(DATA/t['dataset']/f'fold{t["fold"]}'/'split.csv')
        ids=set(outer.loc[outer.partition=='validation','row_id'])
        assert p.row_id.is_unique and set(p.row_id)==ids
        y=frames[t['dataset']].loc[p.row_id,target[t['dataset']]].to_numpy()
        assert np.allclose(p.actual,y,rtol=0,atol=1e-12)
        z=p.predicted.to_numpy()
        if t['cls']:
            assert ((z>=0)&(z<=1)).all()
            values={'auc':roc_auc_score(y,z),'ap':average_precision_score(y,z),'brier':brier_score_loss(y,z),'log_loss':log_loss(y,z),'f1':f1_score(y,z>=.5)}
        else:
            values={'rmse':np.sqrt(mean_squared_error(y,z)),'mae':mean_absolute_error(y,z),'r2':r2_score(y,z)}
        scores=pd.read_csv(folder/'metrics.csv',float_precision='round_trip').query('partition == "validation"').iloc[0]
        for metric,value in values.items():
            error=abs(scores[metric]-value)
            assert error<1e-7,(t,metric,error)
            checks.append({'dataset':t['dataset'],'fold':t['fold'],'variant':t['variant'],'model':t['model'],'method':t['method'],'metric':metric,'absolute_difference':error})
        if t['method']!='fixed':
            trace=json.loads((folder/'search.json').read_text())
            candidates=list(ParameterGrid(domains(t['model'],t['cls'])))
            assert len(trace['rows'])==len(candidates)==trace['requested_evaluations']
            assert trace['best_params']==done['selected_params']
            assert done['selected_params'] in candidates
            rows=trace['rows'];best=max(r['score'] for r in rows)
            assert any(r['params']==done['selected_params'] and abs(r['score']-best)<1e-12 for r in rows)
            for r in rows:
                assert len(r['fold_scores'])==3 and np.isfinite(r['fold_scores']).all()
                assert abs(np.mean(r['fold_scores'])-r['score'])<1e-12
            if t['method']=='grid':
                assert {json.dumps(r['params'],sort_keys=True) for r in rows}=={json.dumps(p,sort_keys=True) for p in candidates}
            actual_fits+=trace['actual_inner_fits'];requested+=len(rows)
        if t['cache_path'] not in verified_caches:
            cache=joblib.load(local_path(t['cache_path']))
            assert set(cache['outer']['train_ids'])==set(outer.loc[outer.partition=='train','row_id'])
            assert set(cache['outer']['validation_ids'])==ids
            for inner in cache['inner']+[cache['outer']]:
                assert set(inner['train_ids']).isdisjoint(inner['validation_ids'])
                nums=list(inner['pre'].named_transformers_['num'].feature_names_in_) if 'num' in inner['pre'].named_transformers_ else []
                if nums:
                    original=frames[t['dataset']].loc[inner['train_ids'],nums]
                    assert np.allclose(inner['pre'].named_transformers_['num'].named_steps['impute'].statistics_,original.median(),rtol=0,atol=1e-10)
                    filled=original.fillna(original.median()).to_numpy(dtype=np.float64)
                    scaler=inner['pre'].named_transformers_['num'].named_steps['scale']
                    expected_scale=filled.std(axis=0);expected_scale[expected_scale==0]=1.
                    assert np.allclose(scaler.mean_,filled.mean(axis=0),rtol=1e-12,atol=1e-9)
                    assert np.allclose(scaler.scale_,expected_scale,rtol=1e-12,atol=1e-9)
                if 'cat' in inner['pre'].named_transformers_:
                    categorical=inner['pre'].named_transformers_['cat']
                    cat_cols=list(categorical.feature_names_in_)
                    encoder=categorical.named_steps['encode']
                    for col,levels in zip(cat_cols,encoder.categories_):
                        expected=set(frames[t['dataset']].loc[inner['train_ids'],col].fillna('__missing__'))
                        assert set(levels)==expected
                if t['cls']:
                    for a,b in inner['calibration_cv']:
                        a_ids=np.asarray(inner['train_ids'])[a];b_ids=np.asarray(inner['train_ids'])[b]
                        assert set(fingerprints.loc[a_ids]).isdisjoint(fingerprints.loc[b_ids])
                else:
                    original_y=frames[t['dataset']].loc[inner['train_ids'],target[t['dataset']]].to_numpy()
                    assert abs(inner['centre']-original_y.mean())<1e-10
                    assert abs(inner['scale']-original_y.std())<1e-10
            verified_caches.add(t['cache_path'])
        if (folder/'training_curve.csv').exists():
            curve=pd.read_csv(folder/'training_curve.csv');assert curve.epoch.tolist()==list(range(1,41))
            assert np.isfinite(curve.training_objective).all()
            epochs=curve.loc[curve.validation_eval_loss.notna(),'epoch'].tolist()
            assert epochs==[1,5,10,20,30,40]
            curves+=1
    trace_pairs=0
    keys={(t['dataset'],t['fold'],t['model'],t['method']):t for t in tasks if t['variant']=='Full' and t['method']!='fixed'}
    for (ds,fold,model,method),t in keys.items():
        if method!='grid' or (ds,fold,model,'tpe') not in keys:continue
        u=keys[(ds,fold,model,'tpe')]
        a=local_path(t['folder'])/(model+'_grid');b=local_path(u['folder'])/(model+'_tpe')
        grid=json.loads((a/'search.json').read_text());tpe=json.loads((b/'search.json').read_text())
        lookup={json.dumps(r['params'],sort_keys=True):r['fold_scores'] for r in grid['rows']}
        for r in tpe['rows']:
            assert np.allclose(lookup[json.dumps(r['params'],sort_keys=True)],r['fold_scores'],rtol=0,atol=1e-10)
        assert max(r['score'] for r in grid['rows'])+1e-10 >= max(r['score'] for r in tpe['rows'])
        if grid['best_params']==tpe['best_params']:
            p=pd.read_csv(a/'predictions.csv',float_precision='round_trip').set_index('row_id')
            q=pd.read_csv(b/'predictions.csv',float_precision='round_trip').set_index('row_id')
            assert np.allclose(p.predicted,q.loc[p.index].predicted,rtol=0,atol=1e-10)
        trace_pairs+=1
    report={'status':'partial_passed' if partial else 'passed','jobs':len(tasks),'validation_metrics_reconstructed':len(checks),'max_absolute_metric_difference':max(r['absolute_difference'] for r in checks),
            'outer_folds':len(split_checks),'inner_folds':sum(r['inner_folds'] for r in split_checks),'requested_candidate_evaluations':requested,
            'actual_inner_candidate_fits':actual_fits,'final_fitted_models':model_hashes,'neural_curves_checked':curves,'sequence_windows':len(seq),
            'paired_search_traces_checked':trace_pairs,'preprocessing_caches_checked':len(verified_caches),
            'sequence_reconstructed_by_raw_one_to_one_joins':True,
            'all_source_hashes_unchanged':True,'all_data_hashes_unchanged':True,'adult_original_test_used_in_nested_benchmark':False}
    suffix='_partial' if partial else ''
    pd.DataFrame(checks).to_csv(OUT/f'metric_reconciliation{suffix}.csv',index=False)
    pd.DataFrame(split_checks).to_csv(OUT/f'split_checks{suffix}.csv',index=False)
    (OUT/f'checks{suffix}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');main(parser.parse_args().partial)
