import json,hashlib,os,subprocess,sys
import numpy as np
import pandas as pd
import pytest
from tablelab.contracts import Plan,Filter
from tablelab.catalog import DataError
from tablelab.engine import apply_filters


def P(**kw):return Plan.model_validate(kw)

@pytest.mark.parametrize('op,expected', [('gt',[4]),('ge',[3,4]),('lt',[1]),('le',[1,3]),('eq',[3]),('ne',[1,4]),('in',[1,4]),('is_missing',[2]),('not_missing',[1,3,4])])
def test_filter_semantics(op,expected):
    d=pd.DataFrame({'id':[1,2,3,4],'x':[1,np.nan,3,4]})
    value=None if op in ('is_missing','not_missing') else [1,4] if op=='in' else 3
    assert apply_filters(d,[Filter(column='x',op=op,value=value)])['id'].tolist()==expected


def test_aggregate_uses_all_rows_and_full_export(engine,catalog):
    result=engine.execute(P(dataset='students',operation='aggregate',columns=['math score'],group_by=['gender'],metric='mean',limit=1))
    frame,_=catalog.load('students');export=pd.read_csv(engine.out/result['run_id']/'table.csv')
    assert len(export)==2 and len(result['table'])==1
    for _,row in export.iterrows():
        values=frame.loc[frame['gender']==row['gender'],'math score'].to_numpy()
        assert row['value']==pytest.approx(np.mean(values))
        assert row['n_rows']==len(values)


def test_rate_is_binary_mean_not_income_amount(engine,catalog):
    result=engine.execute(P(dataset='adult',operation='aggregate',columns=['income_gt_50k'],metric='mean'))
    d,_=catalog.load('adult');assert result['table'][0]['value']==pytest.approx(np.count_nonzero(d['income_gt_50k'])/len(d))
    with pytest.raises(DataError,match='不是数值'):engine.execute(P(dataset='adult',operation='aggregate',columns=['income'],metric='mean'))


def test_descriptive_missing_denominator(engine,catalog):
    r=engine.execute(P(dataset='house',operation='describe',columns=['GrLivArea']))
    d,_=catalog.load('house');row=r['table'][0]
    assert row['missing']==3 and row['n_valid']==len(d)-3
    assert row['mean']==pytest.approx(np.nanmean(d['GrLivArea']))


def test_correlation_complete_cases(engine,catalog):
    r=engine.execute(P(dataset='house',operation='correlation',columns=['GrLivArea','SalePrice']))
    d,_=catalog.load('house');a=d[['GrLivArea','SalePrice']].dropna().to_numpy()
    assert r['table'][0]['pearson_r']==pytest.approx(np.corrcoef(a.T)[0,1])
    assert r['table'][0]['n_complete']==len(a)

@pytest.mark.parametrize('payload,match',[
 ({'dataset':'house','operation':'describe','columns':["__import__('os').system('echo bad')"]},'字段不存在'),
 ({'dataset':'house','operation':'describe','filters':[{'column':'Id','op':'gt','value':999999}]},'没有记录'),
 ({'dataset':'house','operation':'describe','filters':[{'column':'Id','op':'gt','value':'30'}]},'必须是数值'),
 ({'dataset':'students','operation':'describe','group_by':['gender']},'不接受'),
 ({'dataset':'house','operation':'train','target':'SalePrice','features':['Id']},'标识符'),
 ({'dataset':'adult','operation':'train','target':'income_gt_50k','features':['income'],'algorithm':'logistic'},'标识符'),
 ({'dataset':'adult','operation':'train','target':'income_gt_50k','features':['age'],'algorithm':'linear'},'二元'),
 ({'dataset':'house','operation':'train','target':'SalePrice','features':['GrLivArea'],'algorithm':'logistic'},'连续'),
 ({'dataset':'nba','operation':'train','target':'pts','features':['age'],'split':'random'},'逐行随机'),
 ({'dataset':'students','operation':'train','target':'math score','features':['reading score'],'context':'prospective'},'尚不可用'),
 ({'dataset':'nba','operation':'train','target':'pts','features':['ast'],'split':'time','context':'prospective'},'尚不可用'),
 ({'dataset':'adult','operation':'train','target':'income_gt_50k','features':['education','education-num'],'algorithm':'logistic'},'重复编码'),
 ({'dataset':'students','operation':'train','target':'math score','features':['gender'],'split':'time'},'时间字段'),
 ({'dataset':'house','operation':'train','target':'SalePrice','features':['GrLivArea'],'filters':[{'column':'Id','op':'lt','value':40}]},'100 条')
])
def test_engine_rejects_unsupported_requests(engine,payload,match):
    with pytest.raises(DataError,match=match):engine.execute(P(**payload))

@pytest.mark.parametrize('dataset,target,features,algorithm,split',[
 ('house','SalePrice',['GrLivArea','OverallQual','Neighborhood'],'linear','random'),
 ('house','SalePrice',['GrLivArea','OverallQual'],'random_forest','time'),
 ('adult','income_gt_50k',['age','education-num','occupation'],'logistic','random'),
 ('adult','income_gt_50k',['age','hours-per-week'],'random_forest','random'),
 ('nba','pts',['age','player_height'],'linear','time'),
 ('nba','reb',['age','player_height'],'random_forest','group'),
 ('students','math score',['lunch','test preparation course'],'linear','random'),
 ('students','writing score',['reading score','gender'],'random_forest','random')])
def test_train_supported_paths(engine,catalog,dataset,target,features,algorithm,split):
    r=engine.execute(P(dataset=dataset,operation='train',target=target,features=features,algorithm=algorithm,split=split))
    assert r['status']=='ok' and r['source_kind']=='synthetic_test'
    assert all(np.isfinite(v) for v in r['metrics'].values())
    folder=engine.out/r['run_id'];part=pd.read_csv(folder/'split.csv');assert part['row_id'].is_unique
    raw,_=catalog.load(dataset);train=raw.set_index('_row_id').loc[part.loc[part['partition']=='train','row_id']]
    val=raw.set_index('_row_id').loc[part.loc[part['partition']=='interactive_validation','row_id']]
    pred=pd.read_csv(folder/'validation_predictions.csv')
    if dataset=='adult':
        assert set(train['_source_split'])=={'train'} and set(val['_source_split'])=={'train'}
        assert pred['probability_1'].between(0,1).all()
        assert r['metrics']['brier']==pytest.approx(np.mean((pred['actual']-pred['probability_1'])**2))
    else:
        assert r['metrics']['rmse']==pytest.approx(np.sqrt(np.mean((pred['actual']-pred['predicted'])**2)))
        assert r['metrics']['baseline_rmse']==pytest.approx(np.sqrt(np.mean((val[target]-train[target].mean())**2)))
    if split=='time':
        timefield='season' if dataset=='nba' else 'YrSold'
        assert train[timefield].max()<val[timefield].min()
    if split=='group':assert set(train['player_name']).isdisjoint(val['player_name'])
    assert (folder/'diagnostic.pdf').read_bytes().startswith(b'%PDF')
    assert (folder/'executor_source.py').exists()
    assert any('合成' in n for n in r['notes'])


def test_numeric_imputation_fit_only_on_training(engine,catalog):
    r=engine.execute(P(dataset='house',operation='train',target='SalePrice',features=['GrLivArea','OverallQual']))
    part=pd.read_csv(engine.out/r['run_id']/'split.csv');d,_=catalog.load('house')
    training=d.set_index('_row_id').loc[part.loc[part['partition']=='train','row_id']]
    stat=engine.models[r['model_id']].pipeline.named_steps['preprocess'].named_transformers_['num'].named_steps['impute'].statistics_
    assert np.allclose(stat,training[['GrLivArea','OverallQual']].median())


def test_prediction_with_new_record(engine):
    r=engine.execute(P(dataset='house',operation='train',target='SalePrice',features=['GrLivArea','OverallQual']))
    pred=engine.execute(P(dataset='house',operation='predict',model_id=r['model_id'],record={'GrLivArea':1800,'OverallQual':7}))
    assert np.isfinite(pred['table'][0]['prediction'])
    with pytest.raises(DataError,match='恰好包含'):engine.execute(P(dataset='house',operation='predict',model_id=r['model_id'],record={'GrLivArea':1800}))
    with pytest.raises(DataError,match='跨数据集'):engine.execute(P(dataset='students',operation='predict',model_id=r['model_id'],record={'GrLivArea':1800,'OverallQual':7}))


def test_source_unchanged_and_replay(engine,catalog):
    path=catalog.path('house');before=path.read_bytes()
    r=engine.execute(P(dataset='house',operation='aggregate',columns=['SalePrice'],metric='mean'))
    assert path.read_bytes()==before
    env={**os.environ,'TABLELAB_DATA_ROOT':str(catalog.root),'PYTHONPATH':os.getcwd()}
    completed=subprocess.run([sys.executable,str(engine.out/r['run_id']/'reproduce.py')],capture_output=True,text=True,env=env,check=True)
    replay=json.loads(completed.stdout);assert replay['table']==r['table']

@pytest.mark.parametrize('chart,columns,group_by', [('histogram',['SalePrice'],[]),('scatter',['GrLivArea','SalePrice'],[]),('bar',['SalePrice'],['Neighborhood'])])
def test_pdf_plot_export(engine,chart,columns,group_by):
    r=engine.execute(P(dataset='house',operation='plot',chart=chart,columns=columns,group_by=group_by))
    assert (engine.out/r['run_id']/'figure.pdf').read_bytes().startswith(b'%PDF')
