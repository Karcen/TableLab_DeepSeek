from pathlib import Path
import hashlib,json,io,zipfile
import numpy as np
import pandas as pd
import pytest
from tablelab.catalog import Catalog,SPECS,DataError
from tablelab.prepare import parse_archive,member

@pytest.mark.parametrize('name',list(SPECS))
def test_registered_fixture_schema(catalog,name):
    df,meta=catalog.load(name)
    assert meta['source_kind']=='synthetic_test'
    assert df['_row_id'].is_unique
    assert set(SPECS[name].targets)<=set(df)
    assert catalog.schema(name)['source_kind']=='synthetic_test'

def test_missing_real_data_fails(tmp_path):
    with pytest.raises(DataError,match='不会自动替换'):Catalog(tmp_path).load('house')

def test_hash_tamper_fails(root):
    p=root/'processed/house.csv';p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(DataError,match='SHA-256'):Catalog(root).load('house')

def test_unregistered_dataset_fails(catalog):
    with pytest.raises(DataError):catalog.path('../secret')

def test_uci_parser_preserves_official_split(tmp_path):
    row='39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n'
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('adult.data',row);z.writestr('adult.test','|1x3 Cross validator\n'+row.rstrip()+'.\n')
    data=parse_archive('adult',out.getvalue(),tmp_path)
    assert len(data)==2 and data['income_gt_50k'].tolist()==[0,0]
    assert data['_source_split'].tolist()==['train','original_test']

def test_zip_paths_are_not_extracted(tmp_path):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('../../train.csv','Id,SalePrice,GrLivArea\n1,100000,1000\n')
    df=parse_archive('house',out.getvalue(),tmp_path)
    assert len(df)==1 and (tmp_path/'train.csv').exists()
    assert not (tmp_path.parent/'train.csv').exists()

def test_house_numeric_missing_and_category_na(tmp_path):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('train.csv','Id,SalePrice,LotFrontage,Alley,MSSubClass\n1,100000,NA,NA,20\n')
    df=parse_archive('house',out.getvalue(),tmp_path)
    assert pd.isna(df['LotFrontage'].iloc[0]) and df['Alley'].iloc[0]=='NA'
    assert df['MSSubClass'].iloc[0]=='20'

def test_malformed_student_score_fails(tmp_path):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('StudentsPerformance.csv','math score,reading score,writing score\n101,50,50\n')
    with pytest.raises(DataError,match='0 至 100'):parse_archive('students',out.getvalue(),tmp_path)
