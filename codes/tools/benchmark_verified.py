"""固定协议下的课程数据评估；不使用 LLM、不调参、不报告盲测成绩。"""
from pathlib import Path
import sys
import json
import hashlib
import time
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog,SPECS,DataError
from tablelab.contracts import Plan
from tablelab.engine import Engine,write_json,runtime_fingerprint
from tablelab.retrieval import Retriever
from tablelab.offline import install_loopback_guard
ROOT=Path(__file__).resolve().parents[1]

def independent_metrics(pred,train_y,classification):
    y=pred.actual.to_numpy(float)
    if not classification:
        e=y-pred.predicted.to_numpy(float)
        return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),
                'r2':float(1-np.square(e).sum()/np.square(y-y.mean()).sum()),
                'baseline_rmse':float(np.sqrt(np.square(y-train_y.mean()).mean()))}
    prob=pred.probability_1.to_numpy(float);hat=(prob>=.5).astype(int)
    n1=int(y.sum());n0=len(y)-n1; ranks=rankdata(prob,method='average')
    tp=int(((y==1)&(hat==1)).sum());tn=int(((y==0)&(hat==0)).sum());fp=int(((y==0)&(hat==1)).sum());fn=int(((y==1)&(hat==0)).sum())
    order=np.argsort(-prob,kind='stable');yp=y[order];pp=prob[order]
    ends=np.r_[np.flatnonzero(pp[:-1]!=pp[1:]),len(pp)-1]
    cumtp=np.cumsum(yp)[ends];rec=cumtp/n1;prec=cumtp/(ends+1)
    clipped=np.clip(prob,np.finfo(float).eps,1-np.finfo(float).eps)
    return {'roc_auc':float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)),
        'average_precision':float(np.sum(np.diff(np.r_[0,rec])*prec)),
        'accuracy':float(np.mean(hat==y)),'balanced_accuracy':float(.5*(tp/n1+tn/n0)),
        'f1':float(2*tp/(2*tp+fp+fn)),'brier':float(np.square(y-prob).mean()),
        'log_loss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
        'baseline_brier':float(np.square(y-train_y.mean()).mean()),'validation_positive_rate':float(y.mean())}

def main():
    install_loopback_guard();cat=Catalog(ROOT/'data');engine=Engine(cat,ROOT/'artifacts/benchmarks')
    protocol=[]
    for dataset,spec in SPECS.items():
        algorithms=('logistic','random_forest') if dataset=='adult' else ('linear','random_forest')
        for algorithm in algorithms:
            protocol.append({'case':dataset+'_primary_'+algorithm,'role':'primary','plan':Plan(dataset=dataset,operation='train',target=spec.targets[0],features=list(spec.defaults),algorithm=algorithm,split='time' if dataset=='nba' else 'random',context='prospective').model_dump()})
    for dataset,split in [('house','time'),('nba','group')]:
        for algorithm in ('linear','random_forest'):
            protocol.append({'case':dataset+'_'+split+'_'+algorithm,'role':'split_sensitivity','plan':Plan(dataset=dataset,operation='train',target=SPECS[dataset].targets[0],features=list(SPECS[dataset].defaults),algorithm=algorithm,split=split,context='prospective').model_dump()})
    protocol_path=ROOT/'verification/benchmark_protocol.json'
    write_json(protocol_path,{'purpose':'固定模型与固定划分；主结果 8 个模型，划分敏感性 4 个；不根据结果选择配置','scope':'交互验证，不是最终盲测或大模型能力测试','seed':42,'cases':protocol})
    protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows=[];checks=[];summaries=[];seen_splits={}
    for case in protocol:
        plan=Plan.model_validate(case['plan']);df,entry=cat.load(plan.dataset)
        assert entry['source_kind']=='course_dataset'
        evidence=Retriever(cat,plan.dataset).search(' '.join([plan.target,*plan.features]))
        result=engine.execute(plan,'固定协议参数执行；没有调用自然语言模型',evidence)
        folder=engine.out/result['run_id']
        part=pd.read_csv(folder/'split.csv');pred=pd.read_csv(folder/'validation_predictions.csv')
        source=df.set_index('_row_id')
        training=source.loc[part.loc[part.partition.eq('train'),'row_id']]
        valid=source.loc[part.loc[part.partition.eq('interactive_validation'),'row_id']]
        assert len(part)==len(training)+len(valid) and part.row_id.is_unique
        assert set(training.index).isdisjoint(valid.index)
        assert pred.row_id.is_unique and set(pred.row_id)==set(valid.index)
        valid=valid.loc[pred.row_id]
        assert np.array_equal(pred.actual.to_numpy(),valid[plan.target].to_numpy())
        if plan.dataset=='adult':
            assert set(training._source_split)=={'train'} and set(valid._source_split)=={'train'}
            assert set(part.loc[part.partition.eq('train'),'predictor_fingerprint']).isdisjoint(part.loc[part.partition.eq('interactive_validation'),'predictor_fingerprint'])
        key=(plan.dataset,plan.split)
        split_hash=hashlib.sha256((folder/'split.csv').read_bytes()).hexdigest()
        if key in seen_splits:assert seen_splits[key]==split_hash, '比较算法没有使用相同划分'
        else:seen_splits[key]=split_hash
        if plan.split=='time':
            timecol='season' if plan.dataset=='nba' else 'YrSold'
            assert training[timecol].max()<valid[timecol].min()
        if plan.split=='group':assert set(training.player_name).isdisjoint(valid.player_name)
        verified=independent_metrics(pred,training[plan.target],plan.dataset=='adult')
        for metric,value in result['metrics'].items():
            diff=abs(value-verified[metric]);assert np.isclose(value,verified[metric],rtol=1e-10,atol=1e-10),(case['case'],metric,diff)
            checks.append({'case':case['case'],'metric':metric,'reported':value,'independent':verified[metric],'abs_difference':diff,'passed':True})
            rows.append({'case':case['case'],'role':case['role'],'dataset':plan.dataset,'target':plan.target,'algorithm':plan.algorithm,'requested_split':plan.split,'effective_split':result['provenance']['split']['kind'],'metric':metric,'value':value,'n_train':len(training),'n_validation':len(valid),'run_id':result['run_id'],'source_sha256':entry['sha256'],'protocol_sha256':protocol_sha})
        transform=json.loads((folder/'transformation_audit.json').read_text())
        for column,median in transform.get('imputation_medians',{}).items():
            expected=float(np.nanmedian(training[column].to_numpy(float)))
            assert np.isclose(median,expected)
        summaries.append({'case':case['case'],'role':case['role'],'dataset':plan.dataset,'algorithm':plan.algorithm,'requested_split':plan.split,'n_train':len(training),'n_validation':len(valid),'split':result['provenance']['split'],'metrics':result['metrics'],'run_id':result['run_id'],'notes':result['notes'],'seconds_environment_only':result['provenance']['elapsed_seconds'],'protocol_sha256':protocol_sha})
        print(case['case'],result['metrics'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'tables/real_data_baselines.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(ROOT/'tables/metric_reconciliation.csv',index=False,encoding='utf-8-sig')
    wide=pd.DataFrame(rows).pivot(index=['case','role','dataset','algorithm','requested_split','n_train','n_validation'],columns='metric',values='value').reset_index()
    wide.to_csv(ROOT/'tables/model_comparison.csv',index=False,encoding='utf-8-sig')
    write_json(ROOT/'verification/real_data_benchmark.json',{'status':'passed','model_fits':len(protocol),'independently_recomputed_metrics':len(checks),'blind_test_performance_evaluated':False,'llm_used':False,'runtime':runtime_fingerprint(),'cases':summaries})

if __name__=='__main__':main()
