import io
import zipfile
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import pytest
from tablelab.prepare import parse_archive
from tablelab.catalog import DataError
from tablelab.contracts import Plan
from tablelab.engine import aggregate_frame, predictor_fingerprints


def test_duplicate_raw_header_is_rejected(tmp_path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:z.writestr('train.csv','Id,SalePrice,SalePrice\n1,10,10\n')
    with pytest.raises(DataError,match='重复字段'):parse_archive('house',buf.getvalue(),tmp_path)

@pytest.mark.parametrize('grouped',[True,False])
def test_all_missing_sum_is_missing(grouped):
    data=pd.DataFrame({'g':['A','A'],'x':[np.nan,np.nan]})
    result=aggregate_frame(data,Plan(dataset='house',operation='aggregate',columns=['x'],metric='sum',group_by=['g'] if grouped else []))
    assert pd.isna(result.value.iloc[0]) and result.n_valid.iloc[0]==0


def test_literal_none_is_not_missing(tmp_path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:z.writestr('train.csv','Id,SalePrice,MasVnrType\n1,100000,None\n')
    data=parse_archive('house',buf.getvalue(),tmp_path)
    assert data.MasVnrType.iloc[0]=='None'


def test_adult_duplicate_predictors_do_not_cross_split(engine,catalog):
    df,entry=catalog.load('adult')
    path=catalog.path('adult')
    train=df.loc[df._source_split.eq('train')].iloc[:5].copy()
    train['_row_id']=['adult:copy:'+str(i) for i in range(len(train))]
    combined=pd.concat([df,train],ignore_index=True)
    combined.to_csv(path,index=False)
    import hashlib
    manifest_path=catalog.root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['datasets']['adult']['rows']=len(combined)
    manifest['datasets']['adult']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result=engine.execute(Plan(dataset='adult',operation='train',target='income_gt_50k',features=['age','education-num'],algorithm='logistic'))
    parts=pd.read_csv(engine.out/result['run_id']/'split.csv')
    a=set(parts.loc[parts.partition.eq('train'),'predictor_fingerprint'])
    b=set(parts.loc[parts.partition.eq('interactive_validation'),'predictor_fingerprint'])
    assert a.isdisjoint(b)
    assert len(parts)==int(combined._source_split.eq('train').sum())


def test_replay_does_not_require_pythonpath(engine,catalog):
    result=engine.execute(Plan(dataset='house',operation='aggregate',columns=['SalePrice'],metric='mean'))
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['TABLELAB_DATA_ROOT']=str(catalog.root)
    run=subprocess.run([sys.executable,str(engine.out/result['run_id']/'reproduce.py')],env=env,check=True,capture_output=True,text=True)
    assert json.loads(run.stdout)['table']==result['table']
