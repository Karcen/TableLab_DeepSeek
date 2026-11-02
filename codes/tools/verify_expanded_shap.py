"""Reconstruct saved expanded predictions from all grouped SHAP values."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
if __name__=='__main__':
    rows=[]
    for ds in ['house','adult','nba','students']:
        for fold in range(3):
            folder=DATA/ds/f'fold{fold}'/'Full/HGB_grid'
            shap=pd.read_csv(folder/'shap.csv',float_precision='round_trip').set_index('row_id')
            pred=pd.read_csv(folder/'predictions.csv',float_precision='round_trip').set_index('row_id')
            fields=[c for c in shap if c not in ['base','raw_prediction']]
            assert shap.index.is_unique and set(shap.index)==set(pred.index)
            summed=shap[fields].sum(axis=1)+shap['base']
            add_error=float(np.max(np.abs(summed-shap.raw_prediction)))
            rebuilt=expit(summed) if ds=='adult' else summed
            prediction_error=float(np.max(np.abs(rebuilt-pred.loc[shap.index].predicted)))
            assert add_error<1e-5 and prediction_error<1e-5,(ds,fold,add_error,prediction_error)
            rows.append({'dataset':ds,'fold':fold,'n':len(shap),'max_additivity_error':add_error,'max_prediction_error':prediction_error})
    report={'status':'passed','total_explained':sum(r['n'] for r in rows),'folds':rows,'max_prediction_error':max(r['max_prediction_error'] for r in rows)}
    assert report['total_explained']==43893
    (ROOT/'codes/verification/research/expanded/shap_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
