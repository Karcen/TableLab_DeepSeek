"""Protocol tests use explicit fake model responses; no claim about Ollama accuracy."""
import json
import pytest
from tools import paired_questions as p
from tablelab.contracts import Plan

class Scripted:
    def __init__(self,rows): self.byq={r['question']:r for r in rows};self.calls=[]
    def check_lock(self): return {'name':'test-double','digest':'TEST-ONLY'}
    def plan(self, question, schema, evidence, previous, handle):
        arm=bool(evidence);r=self.byq[question]
        self.calls.append((r['id'],arm,None if previous is None else previous.message))
        if r['id']=='H01' and not arm:raise ValueError('模拟无效的模型方案')
        return Plan(dataset=r['dataset'],operation='unsupported',message='test:'+str(arm)),{'model_digest':'TEST-ONLY'}

def test_practice_and_empty_form():
    assert len(p.validate_bank(p.BANK/'practice.csv','practice'))==8
    with pytest.raises(ValueError,match='question 和 reference'):p.validate_bank(p.BANK/'class_questions.csv','class')

def test_independent_inputs_and_scoring(tmp_path):
    rows=p.validate_bank(p.BANK/'practice.csv','practice');client=Scripted(rows)
    dest=p.run(p.BANK/'practice.csv','practice',client,tmp_path)
    assert len(client.calls)==15 and ('H02',True,'test:True') in client.calls
    assert ('H02',False,'test:False') not in client.calls and ('N02',False,'test:False') in client.calls
    assert ('N02',True,'test:True') in client.calls
    key=p.read_json(dest/'audit/assignment.json')
    for area in p.SPECS:
        family=[r['id'] for r in rows if r['dataset']==area]
        assert {key[q]['A'] for q in family}==set(p.ARMS)
    review=p.read_csv(dest/'review.csv',p.RATING);assert len(review)==16
    with pytest.raises(ValueError,match='还没填完'): p.score(dest)
    for x in review:
        answer=p.read_json(dest/x['answer_file']);x['initial_ok']='1' if answer['status']=='unsupported' else '0';x['why']='测试评分，不是真实问答'
    p.write_csv(dest/'review.csv',p.RATING,review);s=p.score(dest)
    assert (s['both_correct'],s['only_with_retrieval'],s['only_without_retrieval'],s['both_wrong'])==(6,2,0,0)
    assert s['difference_percentage_points']==25
    p.write_csv(dest/'review.csv',p.RATING,review[:-1]);
    with pytest.raises(ValueError,match='缺少题目'):p.score(dest)

def test_references_not_sent_to_application_and_raw_preserved(tmp_path,monkeypatch):
    rows=p.validate_bank(p.BANK/'practice.csv','practice'); seen=[]
    def local_response(self,path,body=None,timeout=10):
        if path=='/api/tags':return {'models':[{'name':'qwen3:8b-q4_K_M','digest':'TEST'}]}
        payload=json.loads(body['messages'][-1]['content']);seen.append(payload);assert set(payload)=={'question','schema','retrieved_documents','previous_plan','model_id'}
        assert all(r['reference'] not in str(body) for r in rows)
        return {'message':{'content':Plan(dataset=payload['schema']['id'],operation='unsupported',message='测试响应').model_dump_json()},'done':True}
    monkeypatch.setattr(p.Ollama,'_request',local_response); lock=tmp_path/'lock.json';p.dump(lock,{'name':'qwen3:8b-q4_K_M','digest':'TEST'})
    path=p.run(p.BANK/'practice.csv','practice',p.Recorder(lock),tmp_path)
    assert len(seen)==16 and len(list((path/'audit').rglob('response.json')))==16
    for row in rows:
        pair=[x for x in seen if x['question']==row['question']];assert len(pair)==2 and pair[0]['schema']==pair[1]['schema'];assert sum(bool(x['retrieved_documents']) for x in pair)==1
