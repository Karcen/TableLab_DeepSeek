"""Aggregate expanded evidence; no fitting or outer-driven parameter changes."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from expanded_paths import local_path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'


def main():
    checks=json.loads((ROOT/'codes/verification/research/expanded/checks.json').read_text())
    assert checks['status']=='passed'
    frames=[];searches=[];curves=[];params=[];scales={}
    for path in sorted(DATA.glob('*/fold*/*/*/metrics.csv')):
        q=pd.read_csv(path)
        frames.append(q)
        done=json.loads((path.parent/'complete.json').read_text())
        t=done['task']
        item={k:t[k] for k in ['dataset','fold','variant','model','method']}
        params.append({**item,'parameters':json.dumps(done['selected_params'],sort_keys=True),'seconds':done['seconds'],'parameter_count':done['parameter_count']})
        trace_path=path.parent/'search.json'
        if trace_path.exists():
            trace=json.loads(trace_path.read_text())
            searches.append({**item,'candidate_evaluations':len(trace['rows']),'unique_candidates':trace['unique_candidates'],
                             'actual_inner_fits':trace['actual_inner_fits'],'seconds':done['seconds'],
                             'best_inner_score':max(r['score'] for r in trace['rows'])})
        if (path.parent/'training_curve.csv').exists():
            curve=pd.read_csv(path.parent/'training_curve.csv')
            cache_path=t['cache_path']
            if cache_path not in scales:scales[cache_path]=joblib.load(local_path(cache_path))['outer']['scale']
            for part in ['train','validation']:
                curve[part+'_display']=curve[part+'_eval_loss'] if t['cls'] else np.sqrt(curve[part+'_eval_loss'])*scales[cache_path]
            for key,value in item.items():curve[key]=value
            curves.append(curve)
    metrics=pd.concat(frames,ignore_index=True)
    metrics.to_csv(DATA/'metrics.csv',index=False)
    q=metrics.query('partition == "validation"')
    summary=[]
    for (ds,variant,model,method),g in q.groupby(['dataset','variant','model','method']):
        key='auc' if ds=='adult' else 'rmse'
        r={'dataset':ds,'variant':variant,'model':model,'method':method,'metric':key,'mean':g[key].mean(),'sd':g[key].std(),'folds':g.fold.nunique()}
        assert r['folds']==3
        for col in ['auc','ap','brier','log_loss','f1','rmse','mae','r2']:
            if col in g and g[col].notna().any():r[col+'_mean']=g[col].mean();r[col+'_sd']=g[col].std()
        summary.append(r)
    summary=pd.DataFrame(summary);summary.to_csv(DATA/'summary.csv',index=False)
    deltas=[]
    for (ds,model),g in q.query('method == "grid"').groupby(['dataset','model']):
        key='auc' if ds=='adult' else 'rmse'
        base=g.query('variant == "Full"').set_index('fold')
        for variant,v in g.query('variant != "Full"').groupby('variant'):
            reduced=v.set_index('fold')
            loss=base[key]-reduced[key] if ds=='adult' else reduced[key]-base[key]
            for fold,value in loss.items():
                deltas.append({'dataset':ds,'model':model,'variant':variant,'fold':fold,'loss':value,
                               'loss_percent':100*value/base.loc[fold,key],'full_score':base.loc[fold,key],'reduced_score':reduced.loc[fold,key]})
    pd.DataFrame(deltas).to_csv(DATA/'ablation_deltas.csv',index=False)
    pd.DataFrame(searches).to_csv(DATA/'search_summary.csv',index=False)
    pd.DataFrame(params).to_csv(DATA/'selected_parameters.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(DATA/'training_curves.csv',index=False)
    gains=[]
    full=q.query('variant == "Full"')
    for (ds,model),g in full.query('method != "fixed"').groupby(['dataset','model']):
        key='auc' if ds=='adult' else 'rmse'
        grid=g.query('method == "grid"').set_index('fold')[key]
        tpe=g.query('method == "tpe"').set_index('fold')[key]
        diff=tpe-grid if ds=='adult' else grid-tpe
        gains.append({'dataset':ds,'model':model,'tpe_improvement':diff.mean(),'sd':diff.std(),'folds':3})
    pd.DataFrame(gains).to_csv(DATA/'search_differences.csv',index=False)
    print(summary.query('variant == "Full"')[['dataset','model','method','mean','sd']].to_string(index=False))


if __name__=='__main__':main()
