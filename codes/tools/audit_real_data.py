"""离线导入和审计课程数据副本。保留全部原始记录，不推断缺失来源。"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog
from tablelab.prepare import prepare_one
from tablelab.engine import write_json, predictor_fingerprints

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES={'house':'house-prices.zip','adult':'adult.zip','nba':'nba.zip','students':'students.zip'}
EXPECTED={'house':(1460,81),'adult':(48842,15),'nba':(12844,22),'students':(1000,8)}

def sha(data):return hashlib.sha256(data).hexdigest()

def csv_bytes(data, **kwargs):
    return pd.read_csv(io.BytesIO(data),keep_default_na=False,na_values=[''],**kwargs)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--incoming',type=Path,default=ROOT/'data/incoming')
    args=ap.parse_args();incoming=args.incoming.resolve()
    data_root=ROOT/'data';out=ROOT/'tables';out.mkdir(exist_ok=True)
    receipt=json.loads((incoming/'download_manifest.json').read_text('utf-8'))
    integrity=[];schemas=[];numeric=[];profiles=[];issues=[];row_flags=[];fig_sources=[]
    for dataset,archive in ARCHIVES.items():
        content=(incoming/archive).read_bytes()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            assert z.testzip() is None, f'{archive} ZIP CRC 不匹配'
            assert len(z.namelist())==len(set(z.namelist())), 'ZIP 文件名重复'
            for item in z.infolist():
                name=Path(item.filename)
                assert not name.is_absolute() and '..' not in name.parts, 'ZIP 有不安全路径'
                assert not item.flag_bits&1, '不接受加密压缩包'
                assert item.file_size<=100_000_000, '单文件超限'
            record=receipt['datasets'].get(dataset)
            if record is not None:
                assert sha(content)==record['archive_sha256'], f'{dataset} 与下载清单不符'
                assert len(content)==record['archive_bytes']
                integrity.append({'dataset':dataset,'object':archive,'sha256':sha(content),'bytes':len(content),'reference':'user_download_manifest','status':'match'})
                for member,expected in record['files'].items():
                    blob=z.read(member)
                    assert sha(blob)==expected['sha256'], f'{dataset}/{member} 指纹不符'
                    assert len(blob)==expected['bytes']
                    integrity.append({'dataset':dataset,'object':member,'sha256':sha(blob),'bytes':len(blob),'reference':'user_download_manifest','status':'match'})
            else:
                assert dataset=='house', f'{dataset} 下载记录缺失'
                blob=z.read('train.csv')
                integrity.append({'dataset':dataset,'object':archive,'sha256':sha(content),'bytes':len(content),'reference':'no_download_receipt','status':'recorded_not_source_authenticated'})
                if (incoming/'train.csv').is_file():
                    assert blob==(incoming/'train.csv').read_bytes(), '房价散文件与压缩包不同'
                    integrity.append({'dataset':dataset,'object':'train.csv','sha256':sha(blob),'bytes':len(blob),'reference':'uploaded_loose_csv','status':'byte_identical'})
        prepare_one(dataset,data_root,incoming/archive)
    manifest_path=data_root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    for dataset,entry in manifest['datasets'].items():
        entry['source_kind']='course_dataset'
        entry['provenance_status']='download_manifest_matched' if dataset in receipt['datasets'] else 'user_supplied_unverified'
        entry['source_note']=('上传文件及包内登记文件与用户下载清单 SHA-256 一致；清单不是第三方数字签名，未重新从原站逐字节认证。' if dataset in receipt['datasets'] else '房价为用户自行取得的 CSV；下载清单没有该条目，实际下载网址及原站逐行一致性未核实。不得标成官方原件。')
        entry['declared_source_page']=entry.pop('source_page')
        entry['classroom_use_only']=True
        entry['source_nature']=('课程指定数据；源站致谢指向 generated_data/exams，实际生成过程未核实' if dataset=='students' else '数据副本；记录真实性不由哈希证明')
        if dataset in receipt['datasets']:entry['download_receipt']=receipt['datasets'][dataset]
        if dataset=='students':entry['license_note']='核验到的 Kaggle 数据卡标为 Unknown；本项目不授予公开再分发许可。'
        if dataset=='house':entry['license_note']='实际下载来源及其许可未核实；仅随用户提供文件返回，不声明公开再分发许可。'
    write_json(manifest_path,manifest)
    cat=Catalog(data_root)
    for dataset in ARCHIVES:
        df,entry=cat.load(dataset)
        columns=[c for c in df if c not in ('_row_id','_source_split','income_gt_50k')]
        raw=df[columns]
        assert raw.shape==EXPECTED[dataset], f'{dataset} 当前交付副本结构与锁定范围不同；核查后更新协议，不要跳过。'
        assert df['_row_id'].is_unique and df['_row_id'].notna().all()
        measure=[c for c in columns if c not in ('Id','Unnamed: 0')]
        profiles.append({'dataset':dataset,'rows':len(df),'raw_columns':len(columns),'processed_columns':len(df.columns),'missing_cells':int(raw.isna().sum().sum()),'rows_with_any_missing':int(raw.isna().any(axis=1).sum()),'duplicate_full_records_excluding_ids':int(raw[measure].duplicated().sum()),'missing_target':int(df[{'house':'SalePrice','adult':'income_gt_50k','nba':'pts','students':'math score'}[dataset]].isna().sum()),'rows_removed':0,'provenance_status':entry['provenance_status']})
        for c in columns:
            v=df[c]
            schemas.append({'dataset':dataset,'column':c,'dtype':str(v.dtype),'n':len(v),'n_valid':int(v.count()),'missing':int(v.isna().sum()),'missing_fraction':float(v.isna().mean()),'unique_nonmissing':int(v.nunique()),'literal_None':int(v.eq('None').sum()),'literal_NA':int(v.eq('NA').sum())})
            if pd.api.types.is_numeric_dtype(v):
                assert not np.isinf(v.dropna().astype(float)).any(), f'{dataset}/{c} 无穷值'
                desc=v.describe(percentiles=[.01,.25,.5,.75,.99]).to_dict()
                numeric.append({'dataset':dataset,'column':c,**desc})
        if dataset=='house':
            assert df.Id.is_unique and df.Id.notna().all()
            assert df.SalePrice.gt(0).all()
            assert df.OverallQual.between(1,10).all()
            assert df.MoSold.between(1,12).all()
            issues.append({'dataset':dataset,'severity':'must disclose','finding':'没有房价来源下载记录或原始字段说明附件','count':1,'action':'保留 user_supplied_unverified；不宣称与 Kaggle 官方训练文件逐行一致'})
            issues.append({'dataset':dataset,'severity':'should disclose','finding':'缺少 LotFrontage','count':int(df.LotFrontage.isna().sum()),'action':'不全表删除；仅选作特征时在训练子集中拟合中位数插补'})
            for c,size in [('BsmtQual','TotalBsmtSF'),('FireplaceQu','Fireplaces'),('PoolQC','PoolArea'),('GarageType','GarageCars')]:
                missing=df[c].isna()
                for status,mask in [('blank_with_zero_size',missing & df[size].eq(0)),('blank_with_positive_size',missing & df[size].gt(0))]:
                    fig_sources.append({'dataset':dataset,'field':c,'cross_field':size,'status':status,'n':int(mask.sum()),'interpretation':'结构性缺省线索，不覆盖原值'})
            future=df.YearRemodAdd.gt(df.YrSold)
            for _,r in df.loc[future].iterrows():row_flags.append({'dataset':dataset,'row_id':r['_row_id'],'source_id':r.Id,'flag':'remodel_after_sale','detail':f'YearRemodAdd={r.YearRemodAdd}; YrSold={r.YrSold}','action':'保留并标记；默认模型不使用 YearRemodAdd'})
            for _,r in df.loc[df.GrLivArea.gt(4000)].iterrows():row_flags.append({'dataset':dataset,'row_id':r['_row_id'],'source_id':r.Id,'flag':'GrLivArea_above_4000','detail':f'GrLivArea={r.GrLivArea}; SalePrice={r.SalePrice}','action':'保留；阈值仅诊断，不用于删值或挑选指标'})
            issues.append({'dataset':dataset,'severity':'should disclose','finding':'改建年份晚于出售年份','count':int(future.sum()),'action':'标记但不推断更正值；默认特征未使用此字段'})
        elif dataset=='adult':
            train=df.loc[df._source_split.eq('train')];test=df.loc[df._source_split.eq('original_test')]
            assert (len(train),len(test))==(32561,16281)
            assert set(df.income.unique())=={'<=50K','>50K'}
            assert df.income_gt_50k.eq(df.income.eq('>50K').astype(int)).all()
            assert df['education-num'].between(1,16).all() and df['hours-per-week'].between(1,99).all()
            assert df.age.ge(17).all() and df.fnlwgt.gt(0).all()
            assert df.groupby('education')['education-num'].nunique().eq(1).all()
            xcols=[c for c in columns if c!='income']; hashes=predictor_fingerprints(df[xcols])
            trainhash=set(hashes.loc[train.index]); testhash=set(hashes.loc[test.index])
            duplicate=train[columns].duplicated(keep=False)
            dup_rows=train.loc[duplicate,['_row_id',*columns]]
            dup_rows.to_csv(out/'adult_duplicate_records.csv',index=False,encoding='utf-8-sig')
            overlap=sorted(trainhash&testhash)
            pd.DataFrame({'predictor_fingerprint':overlap}).to_csv(out/'adult_original_split_predictor_overlap.csv',index=False)
            issues.extend([
                {'dataset':dataset,'severity':'must fix for splitting','finding':'adult.data 内非目标原始字段完全相同的额外记录','count':int(hashes.loc[train.index].duplicated().sum()),'action':'记录全部保留；按非目标字段指纹分层分组，禁止跨训练与交互验证'},
                {'dataset':dataset,'severity':'must disclose','finding':'原始训练/测试文件共有的非目标字段指纹','count':len(overlap),'action':'未使用 adult.test 训练或报告其性能；未来最终测试需单独规定重合组处置'},
                {'dataset':dataset,'severity':'should disclose','finding':'样本权重存在但当前模型与描述统计未加权','count':len(df),'action':'比例仅代表上传样本；不外推为总体人口比例'}])
        elif dataset=='nba':
            assert df['Unnamed: 0'].tolist()==list(range(len(df)))
            assert df.season.str.fullmatch(r'\d{4}-\d{2}').all()
            assert df.player_height.gt(0).all() and df.player_weight.gt(0).all()
            assert (df[['pts','reb','ast','gp']]>=0).all().all()
            dup=df.duplicated(['player_name','season'],keep=False)
            df.loc[dup,['_row_id','player_name','season','age','player_height','player_weight','team_abbreviation']].to_csv(out/'nba_nonunique_name_season.csv',index=False,encoding='utf-8-sig')
            nkeys=len(df.loc[dup,['player_name','season']].drop_duplicates())
            issues.append({'dataset':dataset,'severity':'must disclose','finding':'姓名-赛季键不唯一','count':nkeys,'action':'8 条相关记录全部保留；不合并；group 指姓名分组，非验证后的唯一球员分组'})
            above=df.ts_pct.gt(1)
            df.loc[above,['_row_id','player_name','season','gp','pts','ts_pct']].to_csv(out/'nba_ts_above_one.csv',index=False,encoding='utf-8-sig')
            issues.append({'dataset':dataset,'severity':'should disclose','finding':'ts_pct 大于 1','count':int(above.sum()),'action':'真实投篮效率不是普通二项概率，不按 0-1 硬截断；无原始投篮次数，不能逐条重算'})
            issues.append({'dataset':dataset,'severity':'should disclose','finding':'college 为字面 None','count':int(df.college.eq('None').sum()),'action':'保留为类别，不用 pandas 默认缺失表吞掉'})
        else:
            for c in ['math score','reading score','writing score']:assert df[c].between(0,100).all()
            issues.append({'dataset':dataset,'severity':'must disclose','finding':'源站致谢指向数据生成页面，未核实真实抽样','count':len(df),'action':'称为课程样本；不用其备考关联支持教育因果结论；保留来源性质限制'})
    for name,rows in [('input_integrity',integrity),('data_profile',profiles),('column_quality',schemas),('numeric_profile',numeric),('quality_findings',issues),('row_flags',row_flags),('house_missing_semantics',fig_sources)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    report={'stage':'data_import_and_quality','status':'pass_with_documented_caveats','source_scope':'上传课程数据副本；不外发数据；哈希对照用户日志，不是官方数字签名','rows_removed':0,'data':profiles,'findings':issues,'utc':datetime.now(timezone.utc).isoformat()}
    write_json(ROOT/'verification/data_quality.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
