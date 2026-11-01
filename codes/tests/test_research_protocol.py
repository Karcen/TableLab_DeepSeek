from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from tablelab.research.experiment import load_data,splits,pipeline,metric,preprocess
from tablelab.engine import predictor_fingerprints
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('ds',['house','adult','nba','students'])
def test_nested_splits_keep_groups_and_time(ds):
 df,_=load_data(ROOT,ds)
 for outer_train,outer_val in splits(df,ds,3):
  assert not set(outer_train)&set(outer_val)
  for data in [df,df.iloc[outer_train].reset_index(drop=True)]:
   for a,b in splits(data,ds,3):
    assert not set(a)&set(b)
    if ds=='nba':assert data.iloc[a].season.max()<data.iloc[b].season.min()
    if ds=='adult':
     cols=[c for c in data if not c.startswith('_') and c not in ('income','income_gt_50k')]
     fingerprints=predictor_fingerprints(data[cols])
     assert set(fingerprints.iloc[a]).isdisjoint(fingerprints.iloc[b])

def test_preprocessing_does_not_fit_validation_outlier():
 train=pd.DataFrame({'size':[1.,3.,np.nan],'area':['A','B','A']})
 pre=preprocess(train).fit(train)
 pre.transform(pd.DataFrame({'size':[1e9],'area':['unknown']}))
 assert pre.named_transformers_['num'].named_steps['impute'].statistics_[0]==2.
 assert pre.named_transformers_['num'].named_steps['scale'].mean_[0]==2.

def test_metrics_are_reconstructed_from_predictions():
 y=np.array([0.,2.,4.]);p=np.array([1.,1.,5.])
 assert metric(y,p,False)['rmse']==pytest.approx(1.)
