"""Build the self-contained classroom recording from archived, real calculations.

With --calculate, execute the published manual plans and record their new run IDs.
Without it, rebuild HTML from the selected archived runs without fitting anything.
The recording does not call a language model or answer new questions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from tablelab.web import create_app
from tablelab.catalog import Catalog
from tablelab.engine import write_json


def cases():
    study=[('house','社区均价','CollgCr 社区房屋的平均成交价是多少？',dict(operation='aggregate',columns=['SalePrice'],filters=[{'column':'Neighborhood','op':'eq','value':'CollgCr'}],metric='mean'))]
    for a,title in [('linear','线性回归'),('random_forest','随机森林')]:
        study.append(('house',title,f'用全部房屋记录，以居住面积和整体质量为自变量，使用{title}预测成交价，随机划分训练与验证样本。',dict(operation='train',target='SalePrice',features=['GrLivArea','OverallQual'],algorithm=a,split='random')))
    study+= [('adult','分组比例','按教育程度比较年收入超过 5 万美元的样本比例。',dict(operation='aggregate',columns=['income_gt_50k'],group_by=['education'],metric='mean')),('adult','概率模型','用年龄、教育编码和每周工时建立逻辑回归，预测高收入概率。',dict(operation='train',target='income_gt_50k',features=['age','education-num','hours-per-week'],algorithm='logistic',split='random')),('nba','赛季比较','按赛季比较球员场均得分的均值，展示所有赛季。',dict(operation='aggregate',columns=['pts'],group_by=['season'],metric='mean',limit=100)),('nba','后续赛季','用年龄、身高和体重做线性回归，预测场均得分，按赛季划分验证集。',dict(operation='train',target='pts',features=['age','player_height','player_weight'],algorithm='linear',split='time')),('students','备考与成绩','按备考课程完成状态比较数学平均分。',dict(operation='aggregate',columns=['math score'],group_by=['test preparation course'],metric='mean')),('students','考前预测','只用考前信息，以午餐类别和父母教育程度做线性回归预测数学成绩。',dict(operation='train',target='math score',features=['lunch','parental level of education'],algorithm='linear',split='random',context='prospective'))]
    return study


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--calculate',action='store_true');args=cli.parse_args()
    os.chdir(ROOT); data=ROOT/'data'; runs=ROOT/'artifacts/classroom'; manifest=ROOT/'verification/interface/replay_runs.json'
    if args.calculate:
        paths=[]
        with TestClient(create_app(data,runs)) as client:
            for dataset,title,question,params in cases():
                plan={'dataset':dataset,**params};resp=client.post('/api/execute',json={'question':question,'plan':plan},headers={'X-TableLab':'local-ui'});resp.raise_for_status();r=resp.json()
                assert r['interaction_mode']=='manual_parameters' and r['status']=='ok'
                paths.append({'dataset':dataset,'title':title,'question':question,'run_id':r['run_id']});print(dataset,title,r['run_id'],flush=True)
        write_json(manifest,paths)
    cat=Catalog(data);ids=['house','adult','nba','students'];bundle={'schemas':{i:cat.schema(i) for i in ids},'status':{'model':{'ready':False},'datasets':[{'id':i,'verified':True} for i in ids]},'cases':{i:[] for i in ids}}
    for item in json.loads(manifest.read_text()):
        folder=runs/item['run_id'];r=json.loads((folder/'result.json').read_text());assert r['provenance']['sha256']==bundle['schemas'][item['dataset']]['sha256'];assert r['provenance']['question']==item['question'];assert r['interaction_mode']=='manual_parameters'
        result={k:r[k] for k in ('status','summary','plan','notes','metrics','table','run_id','interaction_mode','explanation') if k in r};exp={name:(folder/name).read_text(encoding='utf-8-sig') for name in ['table.csv','plan.json','evidence.json','executor_source.py']};bundle['cases'][item['dataset']].append({**item,'result':result,'evidence':json.loads((folder/'evidence.json').read_text()),'exports':exp})
    static=ROOT/'tablelab/static';index=(static/'index.html').read_text();css=(static/'style.css').read_text();js=(static/'app.js').read_text();payload=json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e')
    html=index.replace('<link rel="stylesheet" href="/static/style.css">','<style>'+css+'</style>').replace('<script defer src="/static/app.js"></script>','').replace('<title>TableLab｜机器学习课程大作业</title>','<title>TableLab｜机器学习课程大作业（课堂回放）</title>').replace('</body>','<script>window.TABLELAB_RECORDS='+payload+';</script><script>'+js+'</script></body>');(ROOT/'docs/classroom_replay.html').write_text(html)

if __name__=='__main__':main()
