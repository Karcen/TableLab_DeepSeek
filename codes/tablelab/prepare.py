"""仅准备阶段联网。也可导入在原始网站合法取得的 ZIP；不绕过 Kaggle 登录或条款。"""
import argparse
import csv
from pathlib import Path
import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests
from .catalog import SPECS, DataError
from .engine import write_json

URLS={
 'adult':'https://archive.ics.uci.edu/static/public/2/adult.zip',
 'nba':'https://www.kaggle.com/api/v1/datasets/download/justinas/nba-players-data',
 'students':'https://www.kaggle.com/api/v1/datasets/download/spscientist/students-performance-in-exams'}
ADULT_COLUMNS=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
HOUSE_NUMERIC=['Id','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice']
NBA_NUMERIC=['age','player_height','player_weight','gp','pts','reb','ast','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct']


def member(archive,name,required=True):
    found=[n for n in archive.namelist() if Path(n).name==name and not n.endswith('/')]
    if not found and not required: return None
    if len(found)!=1: raise DataError(f'压缩包必须且只能包含一个 {name}。')
    info=archive.getinfo(found[0])
    if info.file_size>100_000_000: raise DataError('单个数据文件超过 100 MB 资源上限。')
    return archive.read(found[0])


def parse_archive(dataset,archive_bytes,rawdir):
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        if dataset=='adult':
            frames=[]
            for name,split in [('adult.data','train'),('adult.test','original_test')]:
                data=member(z,name);(rawdir/name).write_bytes(data)
                df=pd.read_csv(io.BytesIO(data),names=ADULT_COLUMNS,sep=r',\s*',engine='python',comment='|',na_values=['?'],skip_blank_lines=True)
                df['income']=df['income'].str.strip().str.rstrip('.')
                if not set(df['income'].dropna())<={'<=50K','>50K'} or df['income'].isna().any(): raise DataError('Adult 收入标签不符合原始二元编码。')
                df['income_gt_50k']=df['income'].eq('>50K').astype(int)
                df['_source_split']=split;df['_row_id']=[f'adult:{split}:{i:08d}' for i in range(len(df))]
                frames.append(df)
            names=member(z,'adult.names',required=False)
            if names is not None: (rawdir/'adult.names').write_bytes(names)
            result=pd.concat(frames,ignore_index=True)
        else:
            filename={'house':'train.csv','nba':'all_seasons.csv','students':'StudentsPerformance.csv'}[dataset]
            data=member(z,filename);(rawdir/filename).write_bytes(data)
            header=next(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
            if len(header)!=len(set(header)): raise DataError('原始 CSV 有重复字段名；不允许自动重命名后继续。')
            result=pd.read_csv(io.BytesIO(data),keep_default_na=False,na_values=[''],low_memory=False)
            if dataset=='house':
                for name in ['test.csv','data_description.txt','sample_submission.csv']:
                    b=member(z,name,required=False)
                    if b is not None:(rawdir/name).write_bytes(b)
                numeric=[c for c in HOUSE_NUMERIC if c in result]
                for c in numeric: result[c]=pd.to_numeric(result[c].mask(result[c].eq('NA'),np.nan),errors='raise')
                if 'MSSubClass' in result: result['MSSubClass']=result['MSSubClass'].astype(str)
                if 'Id' not in result or not result['Id'].is_unique: raise DataError('房价训练集 Id 缺失或重复。')
            elif dataset=='nba':
                for c in NBA_NUMERIC:
                    if c in result: result[c]=pd.to_numeric(result[c],errors='raise')
                if not {'season','player_name'}<=set(result): raise DataError('NBA 缺少 season 或 player_name。')
                if not result['season'].astype(str).str.fullmatch(r'\d{4}-\d{2}').all(): raise DataError('NBA 赛季不是 YYYY-YY 格式。')
                for c in ['draft_year','draft_round','draft_number']:
                    if c in result: result[c]=result[c].astype('string')
            else:
                for c in SPECS['students'].targets:
                    result[c]=pd.to_numeric(result[c],errors='raise')
                    if not result[c].dropna().between(0,100).all(): raise DataError('学生分数超出 0 至 100。')
            result['_row_id']=[f'{dataset}:{i:08d}' for i in range(len(result))]
        if len(result)==0 or not set(SPECS[dataset].targets)<=set(result): raise DataError('数据为空或不符合所选数据集目标字段。')
        if result.columns.duplicated().any(): raise DataError('数据字段重复。')
        return result


def prepare_one(dataset,root,local_archive=None):
    root=Path(root);rawdir=root/'raw'/dataset;rawdir.mkdir(parents=True,exist_ok=True)
    if local_archive is not None:
        data=Path(local_archive).read_bytes(); origin='user_supplied_archive'
    else:
        if dataset=='house': raise DataError('房价数据需先从原比赛页面下载 ZIP，再用 --house-archive 指定。程序不代替你接受比赛条款。')
        response=requests.get(URLS[dataset],timeout=(20,180))
        if response.status_code in (401,403): raise DataError('来源要求登录或授权；请从原网站下载 ZIP，再用相应 --*-archive 参数导入。')
        response.raise_for_status();data=response.content;origin='downloaded_from_source'
    if len(data)>100_000_000 or not zipfile.is_zipfile(io.BytesIO(data)): raise DataError('收到的文件不是可接受的 ZIP；未使用网页错误信息冒充数据。')
    (rawdir/'source.zip').write_bytes(data)
    df=parse_archive(dataset,data,rawdir)
    out=root/'processed'/f'{dataset}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False,encoding='utf-8')
    entry={'source_kind':'course_dataset','provenance_status':('source_endpoint_download' if origin=='downloaded_from_source' else 'user_supplied_unverified'),'origin_mode':origin,'source_page':SPECS[dataset].source,
        'download_endpoint':URLS.get(dataset,'user supplied competition archive'),
        'archive_sha256':hashlib.sha256(data).hexdigest(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'rows':len(df),'columns':list(df.columns),'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'exact_duplicate_rows_excluding_identifiers':int(df.drop(columns=[c for c in ['_row_id','Id','Unnamed: 0'] if c in df]).duplicated().sum()),
        'license_note':'CC BY 4.0，引用 UCI Adult。' if dataset=='adult' else '遵守原 Kaggle 页面列示许可和适用比赛规则；本软件不改变或另授数据许可。'}
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format_version':1,'datasets':{}}
    manifest['datasets'][dataset]=entry;write_json(manifest_path,manifest)
    print(f'{dataset}: 已准备 {len(df)} 行；sha256={entry["sha256"]}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only',choices=['all',*SPECS],default='all');ap.add_argument('--root',default='data')
    for name in SPECS: ap.add_argument('--'+name+'-archive',type=Path)
    args=ap.parse_args();names=list(SPECS) if args.only=='all' else [args.only]
    if 'house' in names and args.house_archive is None: ap.error('准备全部数据需要 --house-archive，或先用 --only 分别准备其他数据集。')
    for name in names: prepare_one(name,args.root,getattr(args,name+'_archive'))

if __name__=='__main__': main()
