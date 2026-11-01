"""Recompute the paper's statistics from the packaged TableLab project.

Usage: python codes/check_results.py [path_to_TableLab]
Requires pandas, numpy and scipy. No model is fitted or called.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def check(root: Path) -> dict:
    records = pd.read_csv(root / 'tables/real_data_baselines.csv')
    summary = pd.read_csv(root / 'tables/model_comparison.csv').set_index('case')
    matched_house_splits = []
    out = {}
    for (case, run), _ in records.groupby(['case', 'run_id']):
        folder = root / 'models/benchmarks' / run
        d = pd.read_csv(folder / 'validation_predictions.csv').set_index('row_id', verify_integrity=True)
        s = pd.read_csv(folder / 'split.csv').set_index('row_id', verify_integrity=True)
        assert set(s.partition) == {'train', 'interactive_validation'}, case
        assert set(d.index) == set(s.index[s.partition == 'interactive_validation']), case
        assert len(d) == summary.loc[case, 'n_validation'], case
        y = d.actual.to_numpy()
        if case.startswith('house_primary'):
            matched_house_splits.append(s.sort_index())
        if 'probability_1' in d:
            p = d.probability_1.to_numpy()
            pred = p >= 0.5
            np.testing.assert_array_equal(pred, d.predicted_class)
            tp, fp, fn = (int(np.sum(t)) for t in (pred & (y == 1), pred & (y == 0), ~pred & (y == 1)))
            n1, n0 = int(y.sum()), int(len(y) - y.sum())
            auc = (rankdata(p)[y == 1].sum() - n1 * (n1 + 1)/2) / (n1*n0)
            scores = {'f1': 2*tp/(2*tp+fp+fn), 'brier': np.mean((y-p)**2), 'roc_auc': auc, 'log_loss': -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))}
            counts = {'tp':tp, 'fp':fp, 'fn':fn}
        else:
            error = y - d.predicted.to_numpy()
            scores = {'mae':np.mean(abs(error)), 'rmse':np.sqrt(np.mean(error**2)), 'r2':1-np.sum(error**2)/np.sum((y-y.mean())**2)}
            counts = {}
        for k, value in scores.items():
            np.testing.assert_allclose(value,summary.at[case,k], rtol=1e-10, atol=1e-12, err_msg=f'{case}: {k}')
        out[case] = {'n':len(d), **{k:float(v) for k,v in scores.items()}, **counts}
    # Compare classroom fits (two inputs) with benchmark fits (six) only after ID alignment.
    for case in json.loads((root/'codes/verification/interface/classroom_calculations.json').read_text()):
        if case['dataset'] == 'house' and 'rmse_recomputed' in case:
            folder = root / 'models/classroom' / case['run_id']
            s = pd.read_csv(folder/'split.csv').set_index('row_id',verify_integrity=True)
            d = pd.read_csv(folder/'validation_predictions.csv').set_index('row_id',verify_integrity=True)
            assert set(d.index)==set(s.index[s.partition=='interactive_validation'])
            plan=json.loads((folder/'plan.json').read_text())
            assert set(plan['features'])=={'OverallQual','GrLivArea'} and not plan['filters']
            matched_house_splits.append(s.sort_index())
            rmse=float(np.sqrt(np.mean((d.actual-d.predicted)**2)))
            np.testing.assert_allclose(rmse,case['rmse_recomputed'],rtol=1e-10)
            out['house_two_'+plan['algorithm']]={'n':len(d),'rmse':rmse}
    assert len(matched_house_splits)==4
    for s in matched_house_splits[1:]:
        pd.testing.assert_frame_equal(s,matched_house_splits[0])
    h=pd.read_csv(root/'data/processed/house.csv',keep_default_na=False)
    y=h.loc[h.Neighborhood=='CollgCr','SalePrice']
    assert len(h)==1460 and len(y)==150 and y.sum()==29694866
    r=pd.read_csv(root/'tables/retrieval_checks.csv')
    assert len(r)==16 and (r['rank'].value_counts().to_dict()=={1:15,2:1})
    out['community_mean']={'n':len(y),'sum':int(y.sum()),'mean':float(y.mean())}
    out['retrieval']={'n':len(r),'rank1':15,'rank2':1,'mrr':float((1/r['rank']).mean())}
    return out

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    print(json.dumps(check(parser.parse_args().project),ensure_ascii=False,indent=2))
