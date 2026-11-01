"""真实课程数据上的 HTTP 接口检查；通过人工结构化参数执行，不冒充自然语言测试。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from tablelab.web import create_app
from tablelab.catalog import Catalog,SPECS
from tablelab.engine import write_json
from tablelab.offline import install_loopback_guard
ROOT=Path(__file__).resolve().parents[1]

def main():
    install_loopback_guard();cat=Catalog(ROOT/'data');app=create_app(ROOT/'data',ROOT/'artifacts/api_checks')
    checks=[]
    def check(label,condition):
        assert condition,label
        checks.append({'check':label,'passed':True})
    with TestClient(app) as client:
        headers={'X-TableLab':'local-ui'}
        status=client.get('/api/status').json()
        check('四份数据已通过载入检查',len(status['datasets'])==4 and all(x['verified'] for x in status['datasets']))
        for dataset,spec in SPECS.items():
            schema=client.get('/api/schema/'+dataset).json()
            check(dataset+':schema provenance',schema['source_kind']=='course_dataset' and bool(schema['source_note']))
            target=spec.targets[0];data,_=cat.load(dataset)
            r=client.post('/api/execute',headers=headers,json={'plan':{'dataset':dataset,'operation':'aggregate','columns':[target],'metric':'mean'},'question':'人工参数功能检查'}).json()
            check(dataset+':actual mean',np.isclose(r['table'][0]['value'],np.asarray(data[target],dtype=float).mean()))
            check(dataset+':manual mode recorded',r['interaction_mode']=='manual_parameters')
            check(dataset+':local evidence recorded',len(r['evidence'])>0)
            plan=json.loads((ROOT/'examples'/f'{dataset}_plan.json').read_text())
            trained=client.post('/api/execute',headers=headers,json={'plan':plan,'question':'人工参数重新训练检查'}).json()
            check(dataset+':training completed',trained.get('status')=='ok' and bool(trained['metrics']))
            file=client.get(f'/api/files/{trained["run_id"]}/diagnostic.pdf')
            check(dataset+':PDF real bytes',file.status_code==200 and file.content.startswith(b'%PDF'))
            audit=client.get(f'/api/files/{trained["run_id"]}/transformation_audit.json')
            check(dataset+':transformation audit download',audit.status_code==200 and audit.json()['n_train']>0)
            record={c:(None if pd.isna(data[c].iloc[0]) else data[c].iloc[0].item() if isinstance(data[c].iloc[0],np.generic) else data[c].iloc[0]) for c in plan['features']}
            p=client.post('/api/execute',headers=headers,json={'plan':{'dataset':dataset,'operation':'predict','model_id':trained['model_id'],'record':record}}).json()
            check(dataset+':single record prediction',p.get('status')=='ok')
        rejection=client.post('/api/execute',headers=headers,json={'plan':{'dataset':'students','operation':'train','target':'math score','features':['reading score'],'context':'prospective'}})
        check('考前使用同期考试成绩被拒绝',rejection.status_code==422)
        check('缺少本机写标记被拒绝',client.post('/api/execute',json={'plan':{'dataset':'house','operation':'describe'}}).status_code==403)
    write_json(ROOT/'verification/real_api_checks.json',{'status':'passed','scope':'FastAPI TestClient；上传课程数据；人工结构化参数；未测试自然语言理解','llm_status_observed':status['model'],'checks':checks})
    print(f'{len(checks)} actual-data API checks passed; no LLM generation evaluated')

if __name__=='__main__':main()
