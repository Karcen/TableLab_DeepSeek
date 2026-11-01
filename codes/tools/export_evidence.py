"""将实际测试、数据来源与固定基准状态整理为交付证据。"""
from pathlib import Path
from collections import defaultdict
import csv,json,platform,sys,xml.etree.ElementTree as ET
from importlib.metadata import version
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import SPECS
ROOT=Path(__file__).resolve().parents[1]
LABELS={'tests.test_engine':'计算与建模','tests.test_contracts':'参数约束','tests.test_data':'数据导入与完整性','tests.test_api_offline':'接口与外连边界','tests.test_retrieval':'检索片段匹配','tests.test_v02':'真实数据审计后的回归测试'}

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    tests=list(ET.parse(ROOT/'verification/component_tests.xml').getroot().iter('testcase'))
    summaries=defaultdict(lambda:{'tests':0,'passed':0,'failed':0,'skipped':0})
    rows=[]
    for t in tests:
        status='failed' if t.find('failure') is not None or t.find('error') is not None else 'skipped' if t.find('skipped') is not None else 'passed'
        module=t.attrib['classname'];summaries[module]['tests']+=1;summaries[module][status]+=1
        rows.append({'module':module,'case':t.attrib['name'],'status':status,'seconds':t.attrib.get('time',''),'evidence_scope':'synthetic_component_test'})
    write_csv(ROOT/'tables/component_test_cases.csv',rows)
    counts=[{'module':k,'module_zh':LABELS[k],**v} for k,v in summaries.items()]
    write_csv(ROOT/'tables/component_test_summary.csv',counts)
    acceptance=json.loads((ROOT/'examples/acceptance_questions.json').read_text())
    write_csv(ROOT/'tables/llm_acceptance_plan.csv',[{'case_id':c['id'],'dataset':c['dataset'],'question':c['question'],'status':'NOT_RUN_REAL_MODEL','expected_plan':json.dumps(c['expected'],ensure_ascii=False)} for c in acceptance])
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    write_csv(ROOT/'tables/data_sources.csv',[{'dataset':k,'label':s.label,'declared_course_source':s.source,'provenance_status':manifest['datasets'][k]['provenance_status'],'source_note':manifest['datasets'][k]['source_note'],'targets':';'.join(s.targets),'license_status':'UCI CC BY 4.0' if k=='adult' else 'Unknown on source page; no new redistribution license' if k=='students' else 'Check original Kaggle dataset/competition terms'} for k,s in SPECS.items()])
    environment={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'packages':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','pydantic','fastapi','uvicorn','requests','pytest']}}
    (ROOT/'verification/environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2))
    passed=sum(x['passed'] for x in counts)
    macros=f'\\newcommand{{\\ComponentTests}}{{{len(tests)}}}\n\\newcommand{{\\ComponentPassed}}{{{passed}}}\n\\newcommand{{\\AcceptanceCases}}{{{len(acceptance)}}}\n'
    (ROOT/'docs/source/verification_numbers.tex').write_text(macros)
    benchmark=json.loads((ROOT/'verification/real_data_benchmark.json').read_text())
    api=json.loads((ROOT/'verification/real_api_checks.json').read_text())
    retrieval=json.loads((ROOT/'verification/retrieval_checks.json').read_text())
    status={'project_version':'0.2.0','date':'2026-10-30',
        'component_tests':{'total':len(tests),'passed':passed,'data':'synthetic fixtures','environment':'Linux x86_64'},
        'course_datasets_bundled':True,'dataset_quality_audit_run':True,
        'fixed_benchmark_models':benchmark['model_fits'],'independent_metric_checks':benchmark['independently_recomputed_metrics'],
        'actual_data_api_checks':len(api['checks']),'retrieval_fixed_cases':retrieval['n'],'retrieval_hits_at_6':retrieval['hits_at_6'],
        'house_original_source_authenticated':False,'student_real_world_sampling_verified':False,
        'model_weights_bundled':False,'real_llm_end_to_end_run':False,'mac_M4_tested':False,
        'physical_network_disconnect_tested':False,'external_skills_locally_installed':False,
        'browser_ui_status':json.loads((ROOT/'verification/real_browser_checks.json').read_text())['status'],
        'delivery_status':'Data and deterministic analysis verified with caveats; local LLM and Mac offline acceptance outstanding'}
    (ROOT/'verification/build_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(f'{len(tests)} tests; {passed} passed; {len(acceptance)} real-model cases not yet run')
if __name__=='__main__':main()
