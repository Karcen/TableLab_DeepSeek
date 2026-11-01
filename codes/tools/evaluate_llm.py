"""真实本地模型端到端验收。失败逐项记录并以非零状态退出，不以 mock 顶替。"""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
import pandas as pd
from tablelab.catalog import Catalog,DataError
from tablelab.llm import Ollama
from tablelab.retrieval import Retriever
from tablelab.engine import Engine,write_json
from tablelab.offline import install_loopback_guard


def require_course_source(entry):
    if entry['source_kind'] not in {'original', 'course_dataset'}:
        raise DataError('正式端到端验收只接受已载入的课程数据；不接受合成或未知来源标签。')


def main():
    install_loopback_guard();client=Ollama();model=client.check_lock();cat=Catalog(Path('data'))
    engine=Engine(cat,Path('artifacts/llm_acceptance'))
    cases=json.loads(Path('examples/acceptance_questions.json').read_text(encoding='utf-8'));rows=[]
    for case in cases:
        start=time.perf_counter();entry=cat.load(case['dataset'])[1]
        require_course_source(entry)
        row={'case_id':case['id'],'dataset':case['dataset'],'question':case['question'],'status':'failed'}
        try:
            evidence=Retriever(cat,case['dataset']).search(case['question'])
            plan,meta=client.plan(case['question'],cat.schema(case['dataset']),evidence)
            actual=plan.model_dump();row['actual_plan']=json.dumps(actual,ensure_ascii=False)
            mismatches=[]
            for key,value in case['expected'].items():
                actualvalue=actual[key]
                if key in ('features','columns','group_by'):same=set(value)==set(actualvalue)
                else:same=value==actualvalue
                if not same:mismatches.append(key)
            row['plan_matches']=not mismatches
            if mismatches:row['error']='与预定义需求不一致的字段：'+','.join(mismatches)
            else:
                result=engine.execute(plan,case['question'],evidence);row['status']='passed';row['run_id']=result.get('run_id','')
            row['model_digest']=meta['model_digest']
        except (ValueError,KeyError,requests.RequestException) as exc:row['error']=type(exc).__name__+': '+str(exc)
        row['elapsed_seconds']=time.perf_counter()-start;rows.append(row);print(row['case_id'],row['status'],row.get('error',''))
    out=Path('verification');out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'local_llm_acceptance.csv',index=False,encoding='utf-8-sig')
    write_json(out/'local_llm_acceptance.json',{'model_digest':model['digest'],'cases':rows,
        'scope':'预定义验收题，不证明任意新问题正确；用户仍应另外口头提出未准备过的问题。'})
    if any(r['status']!='passed' for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
